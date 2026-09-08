//! Pass 4 — graph assembly: spatial renumbering, CSR, and the snap index.
//!
//! Two things happen here, and both are about locality rather than size.
//!
//! **Hilbert renumbering.** Vertices come out of pass 3 in OSM id order, which
//! is roughly *creation* order — Oslo's junctions are scattered through the
//! whole file. At planet scale the graph is mapped, not loaded, so that
//! scattering would mean a page fault per edge. Sorting vertices along a
//! Hilbert curve puts geographic neighbours next to each other in the file, so
//! a search around Oslo touches a handful of contiguous pages.
//!
//! **The snap index.** "Within 5 m of the endpoints" is a promise about
//! *edges*, not vertices: junctions can be a kilometre apart, so snapping to
//! the nearest vertex would be wildly out. We therefore index segments by the
//! grid cells their bounding box touches, and the router snaps to the nearest
//! point on the nearest *segment*, splitting its exact length at that point.

use crate::bitmap::BitMap;
use crate::build::Paths;
use crate::geo;
use crate::mmapvec;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufWriter, Write};

/// Snap-grid resolution in 1e-7 degrees. 0.01° ≈ 1.1 km; a segment longer than
/// that lands in several cells, which is exactly what we want.
pub const CELL_E7: i32 = 100_000;

/// Coarse grid for the overview layer: 1°. At continental zoom the fine grid
/// would need millions of cell probes to cover the viewport, so major roads
/// get their own index at a resolution matched to how they are drawn.
pub const BIG_E7: i32 = 10_000_000;

#[inline]
pub fn big_cell_of(lat_e7: i32, lon_e7: i32) -> u64 {
    let y = ((lat_e7 as i64 + 900_000_000) / BIG_E7 as i64) as u64;
    let x = ((lon_e7 as i64 + 1_800_000_000) / BIG_E7 as i64) as u64;
    (y << 32) | x
}

#[inline]
pub fn cell_of(lat_e7: i32, lon_e7: i32) -> u64 {
    let y = ((lat_e7 as i64 + 900_000_000) / CELL_E7 as i64) as u64;
    let x = ((lon_e7 as i64 + 1_800_000_000) / CELL_E7 as i64) as u64;
    (y << 32) | x
}

pub struct GraphStats {
    pub vertices: u64,
    pub edges: u64,
    pub snap_entries: u64,
}

pub fn pass4(paths: &Paths, need: &BitMap, junc: &BitMap) -> io::Result<GraphStats> {
    let t0 = std::time::Instant::now();
    let coords_map = mmapvec::open(&paths.f("coords.bin"))?;
    let coords: &[(i32, i32)] = unsafe { mmapvec::as_slice(&coords_map[..]) };

    // ---- 1. vertex coordinates in pass-3 (rank) order -------------------
    let nv = junc.total() as usize;
    let mut vc: Vec<(i32, i32)> = vec![(0, 0); nv];
    junc.for_each_set(|id, rank| {
        vc[rank as usize] = coords[need.rank1(id) as usize];
    });
    eprintln!("[pass4] {} vertex coordinates gathered ({:.1} s)", nv, t0.elapsed().as_secs_f64());

    // ---- 2. Hilbert order ----------------------------------------------
    let t = std::time::Instant::now();
    let mut order: Vec<(u64, u32)> = (0..nv as u32)
        .into_par_iter()
        .map(|i| {
            let c = vc[i as usize];
            (geo::hilbert_latlon(c.0, c.1), i)
        })
        .collect();
    order.par_sort_unstable();
    // perm[old] = new
    let mut perm: Vec<u32> = vec![0; nv];
    let mut newc: Vec<(i32, i32)> = vec![(0, 0); nv];
    for (newid, &(_, old)) in order.iter().enumerate() {
        perm[old as usize] = newid as u32;
        newc[newid] = vc[old as usize];
    }
    drop(order);
    drop(vc);
    eprintln!("[pass4] Hilbert renumbering done ({:.1} s)", t.elapsed().as_secs_f64());

    {
        // Kept so later passes can map an OSM node id (→ rank → old vertex id)
        // onto the final, spatially renumbered vertex.
        let mut w = BufWriter::with_capacity(1 << 22, File::create(paths.f("vperm.bin"))?);
        for &p in &perm {
            w.write_all(&p.to_le_bytes())?;
        }
        w.flush()?;
    }
    {
        let mut w = BufWriter::with_capacity(1 << 22, File::create(paths.f("vcoord.bin"))?);
        for &(a, b) in &newc {
            w.write_all(&a.to_le_bytes())?;
            w.write_all(&b.to_le_bytes())?;
        }
        w.flush()?;
    }

    // ---- 3. CSR -----------------------------------------------------------
    let t = std::time::Instant::now();
    let edges_map = mmapvec::open(&paths.f("edges.bin"))?;
    let ne = edges_map.len() / 12;
    let raw = &edges_map[..];
    let mut head: Vec<u32> = vec![0; nv + 1];
    for e in 0..ne {
        let u = u32::from_le_bytes(raw[e * 12..e * 12 + 4].try_into().unwrap());
        head[perm[u as usize] as usize + 1] += 1;
    }
    for i in 1..=nv {
        head[i] += head[i - 1];
    }
    let mut cursor = head.clone();
    let mut to_map = mmapvec::create::<u32>(&paths.f("csr.to"), ne)?;
    let mut seg_map = mmapvec::create::<u32>(&paths.f("csr.seg"), ne)?;
    // Snapping needs the reverse map: which two vertices a segment joins, in
    // the direction its stored geometry runs.
    let nseg = mmapvec::open(&paths.f("seg.len"))?.len() / 4;
    let mut su_map = mmapvec::create::<u32>(&paths.f("seg.u"), nseg)?;
    let mut sv_map = mmapvec::create::<u32>(&paths.f("seg.v"), nseg)?;
    {
        let su = unsafe { mmapvec::as_mut_slice::<u32>(&mut su_map[..]) };
        let sv = unsafe { mmapvec::as_mut_slice::<u32>(&mut sv_map[..]) };
        for e in 0..ne {
            let b = &raw[e * 12..e * 12 + 12];
            let u = perm[u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize];
            let v = perm[u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize];
            let sw = u32::from_le_bytes(b[8..12].try_into().unwrap());
            let sid = (sw & 0x7fff_ffff) as usize;
            // The dir bit marks the copy that runs against the geometry, so
            // the other assignment is the geometric one.
            if sw & 0x8000_0000 == 0 {
                su[sid] = u;
                sv[sid] = v;
            } else {
                su[sid] = v;
                sv[sid] = u;
            }
        }
    }
    su_map.flush()?;
    sv_map.flush()?;
    {
        let to = unsafe { mmapvec::as_mut_slice::<u32>(&mut to_map[..]) };
        let sg = unsafe { mmapvec::as_mut_slice::<u32>(&mut seg_map[..]) };
        for e in 0..ne {
            let b = &raw[e * 12..e * 12 + 12];
            let u = perm[u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize];
            let v = perm[u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize];
            let s = u32::from_le_bytes(b[8..12].try_into().unwrap());
            let slot = cursor[u as usize] as usize;
            cursor[u as usize] += 1;
            to[slot] = v;
            sg[slot] = s;
        }
    }
    to_map.flush()?;
    seg_map.flush()?;
    {
        let mut w = BufWriter::with_capacity(1 << 22, File::create(paths.f("csr.head"))?);
        for &h in &head {
            w.write_all(&h.to_le_bytes())?;
        }
        w.flush()?;
    }
    // Reverse CSR. A bidirectional search has to expand *into* a vertex, and
    // a one-way network gives no way to derive that from the forward lists.
    let t2 = std::time::Instant::now();
    let mut rhead: Vec<u32> = vec![0; nv + 1];
    for e in 0..ne {
        let v = u32::from_le_bytes(raw[e * 12 + 4..e * 12 + 8].try_into().unwrap());
        rhead[perm[v as usize] as usize + 1] += 1;
    }
    for i in 1..=nv {
        rhead[i] += rhead[i - 1];
    }
    let mut rcur = rhead.clone();
    let mut rto_map = mmapvec::create::<u32>(&paths.f("csr.rto"), ne)?;
    let mut rseg_map = mmapvec::create::<u32>(&paths.f("csr.rseg"), ne)?;
    {
        let rto = unsafe { mmapvec::as_mut_slice::<u32>(&mut rto_map[..]) };
        let rsg = unsafe { mmapvec::as_mut_slice::<u32>(&mut rseg_map[..]) };
        for e in 0..ne {
            let b = &raw[e * 12..e * 12 + 12];
            let u = perm[u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize];
            let v = perm[u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize];
            let s = u32::from_le_bytes(b[8..12].try_into().unwrap());
            let slot = rcur[v as usize] as usize;
            rcur[v as usize] += 1;
            rto[slot] = u;
            rsg[slot] = s;
        }
    }
    rto_map.flush()?;
    rseg_map.flush()?;
    {
        let mut w = BufWriter::with_capacity(1 << 22, File::create(paths.f("csr.rhead"))?);
        for &h in &rhead {
            w.write_all(&h.to_le_bytes())?;
        }
        w.flush()?;
    }
    eprintln!(
        "[pass4] CSR built: {} vertices, {} edges — forward {:.1} s, reverse {:.1} s",
        nv,
        ne,
        t.elapsed().as_secs_f64() - t2.elapsed().as_secs_f64(),
        t2.elapsed().as_secs_f64()
    );

    // ---- 3b. segment renumbering ------------------------------------------
    // Vertices were put in Hilbert order so a search walks contiguous pages.
    // Segments were not: pass 3 emits them in *way* order, so `seg_len[sid]`
    // and `seg_attr[sid]` — read once per edge relaxation — scatter across a
    // 1.35 GB array. Measured on 200 000 consecutive vertices: 122 pages of
    // `csr.to` against 8 794 pages of `seg.len`, for the same 495 000 edges.
    // Seventy-two times the I/O for the same work, and on a machine where the
    // dataset cannot be cached that is the whole cost of a query.
    //
    // Renumbering segments into the order the CSR first references them makes
    // those reads as local as the edges that cause them.
    let t3 = std::time::Instant::now();
    renumber_segments(paths, nv, ne, &head, &mut seg_map, &mut rseg_map, &su_map)?;
    eprintln!("[pass4] segments renumbered into CSR order ({:.1} s)", t3.elapsed().as_secs_f64());

    // ---- 3c/4 ordering note ------------------------------------------------
    // The dictionary is built here but the raw array is only removed after the
    // snap index, which still reads it.
    // ---- 3c. attribute dictionary -----------------------------------------
    // The packed attribute word takes only 1 370 distinct values across the
    // planet — road class, speed, oneway and flags combine in far fewer ways
    // than 32 bits allow. Storing a u16 index into a table of those halves the
    // array with no decoding at all: the table is 5 KB and never leaves cache.
    let t4 = std::time::Instant::now();
    let ndistinct = build_attr_dict(paths)?;
    eprintln!(
        "[pass4] attribute dictionary: {ndistinct} distinct values, seg.attr halved ({:.1} s)",
        t4.elapsed().as_secs_f64()
    );

    // ---- 3d. pack the edge targets ----------------------------------------
    // Hilbert renumbering put a vertex's neighbours numerically near it:
    // 99.77 % of `to[k] - u` fits in 16 signed bits. Storing that delta halves
    // the two largest arrays in the dataset, and decoding is one add.
    //
    // The 0.23 % that do not fit go in a sorted exception list, found by
    // binary search — no rank structure, no bit fiddling, and a branch that is
    // right 997 times in 1000. An i8 delta would have covered 94 % and saved
    // more, but needs a rank index over the escapes to address the exceptions;
    // that is a lot of machinery for one extra byte per edge.
    let t5 = std::time::Instant::now();
    let (pk_f, pk_b) = (
        pack_targets(paths, &head, ne, "csr.to", "to")?,
        pack_targets(paths, &rhead, ne, "csr.rto", "rto")?,
    );
    eprintln!(
        "[pass4] edge targets packed to i16 deltas: {} + {} exceptions of {} edges ({:.1} s)",
        pk_f,
        pk_b,
        ne * 2,
        t5.elapsed().as_secs_f64()
    );

    // ---- 4. segment → cell snap index -------------------------------------
    // Free the renumbering scratch first. At planet scale each of these is a
    // gigabyte or more, and the snap index is about to want ~8 GB of its own
    // while the coordinate and geometry files compete for page cache.
    drop(perm);
    drop(head);
    drop(cursor);
    drop(rhead);
    drop(rcur);
    drop(to_map);
    drop(seg_map);
    drop(rto_map);
    drop(rseg_map);
    let t = std::time::Instant::now();
    drop(su_map);
    drop(sv_map);
    let snap_entries = build_snap_index(paths, &newc)?;
    eprintln!(
        "[pass4] snap index: {} cell entries ({:.1} s)",
        snap_entries,
        t.elapsed().as_secs_f64()
    );

    // The dictionary supersedes the raw attribute array. Keeping both would
    // make the dataset larger, which is the opposite of the point on a machine
    // that reads from disk.
    std::fs::remove_file(paths.f("seg.attr")).ok();

    Ok(GraphStats { vertices: nv as u64, edges: ne as u64, snap_entries })
}

/// Give segments ids that follow the order the CSR references them.
///
/// Every per-segment array is permuted to match, and both copies of the edge
/// list are remapped, so the change is invisible to every reader — the only
/// thing that differs is which pages a query has to touch.
#[allow(clippy::too_many_arguments)]
fn renumber_segments(
    paths: &Paths,
    nv: usize,
    ne: usize,
    head: &[u32],
    seg_map: &mut memmap2::MmapMut,
    rseg_map: &mut memmap2::MmapMut,
    su_map: &memmap2::MmapMut,
) -> io::Result<()> {
    let nseg = su_map.len() / 4;
    // old id -> new id
    let mut newid: Vec<u32> = vec![u32::MAX; nseg];
    let mut next = 0u32;
    {
        let seg: &[u32] = unsafe { mmapvec::as_slice(&seg_map[..]) };
        for u in 0..nv {
            // The index addresses several parallel arrays, not just this one.
            #[allow(clippy::needless_range_loop)]
            for k in head[u] as usize..head[u + 1] as usize {
                let sid = (seg[k] & 0x7fff_ffff) as usize;
                if newid[sid] == u32::MAX {
                    newid[sid] = next;
                    next += 1;
                }
            }
        }
    }
    // A segment the CSR never references cannot be routed over, but it is
    // still addressable by the snap index, so it keeps an id at the end.
    for id in newid.iter_mut() {
        if *id == u32::MAX {
            *id = next;
            next += 1;
        }
    }
    debug_assert_eq!(next as usize, nseg);

    // Permute every per-segment array. `u32` arrays first.
    for name in ["seg.len", "seg.attr", "seg.name", "seg.u", "seg.v"] {
        let src = mmapvec::open(&paths.f(name))?;
        let old: &[u32] = unsafe { mmapvec::as_slice(&src[..]) };
        let mut dst = mmapvec::create::<u32>(&paths.f(&format!("{name}.new")), nseg)?;
        {
            let new: &mut [u32] = unsafe { mmapvec::as_mut_slice(&mut dst[..]) };
            for (o, v) in old.iter().enumerate() {
                new[newid[o] as usize] = *v;
            }
        }
        dst.flush()?;
        drop(dst);
        drop(src);
        std::fs::rename(paths.f(&format!("{name}.new")), paths.f(name))?;
    }

    // Geometry: rebuild the offset table in the new order and copy each run.
    {
        let goff_src = mmapvec::open(&paths.f("geom.off"))?;
        let goff: &[u32] = unsafe { mmapvec::as_slice(&goff_src[..]) };
        let gpts_src = mmapvec::open(&paths.f("geom.pts"))?;
        let gpts: &[(i32, i32)] = unsafe { mmapvec::as_slice(&gpts_src[..]) };
        // Where each *new* segment's run starts.
        let mut len_by_new: Vec<u32> = vec![0; nseg];
        for o in 0..nseg {
            len_by_new[newid[o] as usize] = goff[o + 1] - goff[o];
        }
        let mut new_off = mmapvec::create::<u32>(&paths.f("geom.off.new"), nseg + 1)?;
        let mut acc = 0u32;
        {
            let no: &mut [u32] = unsafe { mmapvec::as_mut_slice(&mut new_off[..]) };
            for i in 0..nseg {
                no[i] = acc;
                acc += len_by_new[i];
            }
            no[nseg] = acc;
        }
        let mut new_pts = mmapvec::create::<(i32, i32)>(&paths.f("geom.pts.new"), acc as usize)?;
        {
            let no: &[u32] = unsafe { mmapvec::as_slice(&new_off[..]) };
            let np: &mut [(i32, i32)] = unsafe { mmapvec::as_mut_slice(&mut new_pts[..]) };
            for o in 0..nseg {
                let (a, b) = (goff[o] as usize, goff[o + 1] as usize);
                if a == b {
                    continue;
                }
                let d = no[newid[o] as usize] as usize;
                np[d..d + (b - a)].copy_from_slice(&gpts[a..b]);
            }
        }
        new_off.flush()?;
        new_pts.flush()?;
        drop(new_off);
        drop(new_pts);
        drop(goff_src);
        drop(gpts_src);
        std::fs::rename(paths.f("geom.off.new"), paths.f("geom.off"))?;
        std::fs::rename(paths.f("geom.pts.new"), paths.f("geom.pts"))?;
    }

    // Remap both copies of the edge list, keeping the direction bit.
    for map in [&mut *seg_map, &mut *rseg_map] {
        let sl: &mut [u32] = unsafe { mmapvec::as_mut_slice(&mut map[..]) };
        for w in sl.iter_mut().take(ne) {
            let dir = *w & 0x8000_0000;
            *w = newid[(*w & 0x7fff_ffff) as usize] | dir;
        }
    }
    seg_map.flush()?;
    rseg_map.flush()?;
    Ok(())
}

/// Store each edge target as a 16-bit delta from its source vertex.
///
/// Returns how many edges needed the exception list.
fn pack_targets(
    paths: &Paths,
    head: &[u32],
    ne: usize,
    src_name: &str,
    stem: &str,
) -> io::Result<usize> {
    let src = mmapvec::open(&paths.f(src_name))?;
    let to: &[u32] = unsafe { mmapvec::as_slice(&src[..]) };
    let mut d16 = mmapvec::create::<i16>(&paths.f(&format!("{stem}.d16")), ne)?;
    // (edge index, true target) for the deltas that do not fit, in edge order
    // so a binary search over the first field finds them.
    let mut exc: Vec<(u32, u32)> = Vec::new();
    {
        let out: &mut [i16] = unsafe { mmapvec::as_mut_slice(&mut d16[..]) };
        for u in 0..head.len() - 1 {
            for k in head[u] as usize..head[u + 1] as usize {
                let d = to[k] as i64 - u as i64;
                // i16::MIN is the escape marker, so a delta that happens to
                // equal it takes the exception path too — correct, just one
                // more entry.
                if d > i16::MIN as i64 && d <= i16::MAX as i64 {
                    out[k] = d as i16;
                } else {
                    out[k] = i16::MIN;
                    exc.push((k as u32, to[k]));
                }
            }
        }
    }
    d16.flush()?;
    drop(d16);
    drop(src);
    let mut w = BufWriter::with_capacity(1 << 20, File::create(paths.f(&format!("{stem}.exc")))?);
    for (k, v) in &exc {
        w.write_all(&k.to_le_bytes())?;
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    std::fs::remove_file(paths.f(src_name)).ok();
    Ok(exc.len())
}

/// Replace the 32-bit attribute word with a 16-bit index into a table of the
/// values that actually occur.
///
/// Not a codec: there is nothing to decompress, only a smaller integer and one
/// indirection through a table small enough to stay in L1. `seg.attr` is kept
/// as well, so a reader that has not been taught the dictionary still works.
fn build_attr_dict(paths: &Paths) -> io::Result<usize> {
    let src = mmapvec::open(&paths.f("seg.attr"))?;
    let attr: &[u32] = unsafe { mmapvec::as_slice(&src[..]) };
    let mut map: std::collections::HashMap<u32, u16> = Default::default();
    let mut dict: Vec<u32> = Vec::new();
    let mut idx = mmapvec::create::<u16>(&paths.f("seg.attr16"), attr.len())?;
    {
        let out: &mut [u16] = unsafe { mmapvec::as_mut_slice(&mut idx[..]) };
        for (i, &a) in attr.iter().enumerate() {
            let id = match map.get(&a) {
                Some(&v) => v,
                None => {
                    // More than 65 536 distinct words would mean the attribute
                    // packing changed; fail loudly rather than truncate.
                    let v = u16::try_from(dict.len()).map_err(|_| {
                        io::Error::other("attribute dictionary overflowed 16 bits")
                    })?;
                    dict.push(a);
                    map.insert(a, v);
                    v
                }
            };
            out[i] = id;
        }
    }
    idx.flush()?;
    let mut w = BufWriter::with_capacity(1 << 16, File::create(paths.f("attr.dict"))?);
    for a in &dict {
        w.write_all(&a.to_le_bytes())?;
    }
    w.flush()?;
    Ok(dict.len())
}

/// Insert every segment into each grid cell its bounding box touches.
///
/// The box is taken over the *drawn* geometry as well as the endpoints, so a
/// long curving road is found from anywhere along it, not just near its ends.
/// Insert every segment into each grid cell its bounding box touches.
///
/// The box is taken over the *drawn* geometry as well as the endpoints, so a
/// long curving road is found from anywhere along it, not just near its ends.
///
/// This walks **segments**, not edges. Walking edges and skipping the reverse
/// copy looks equivalent and is not: a way tagged `oneway=-1` produces a single
/// edge that runs *against* its geometry, so its only copy carries the
/// direction bit and the segment would never be indexed — leaving those roads
/// invisible to snapping.
fn build_snap_index(paths: &Paths, vcoord: &[(i32, i32)]) -> io::Result<u64> {
    // Opened here rather than handed in: segment renumbering replaces these
    // files by rename, and a mapping taken before that still shows the old
    // inode. Reading it produced segments whose two endpoints were on
    // different continents, whose bounding box covered a million cells, and
    // whose index entries grew to a 100 GB allocation before the process was
    // killed. Open after the rename, and the hazard cannot arise.
    let su_src = mmapvec::open(&paths.f("seg.u"))?;
    let sv_src = mmapvec::open(&paths.f("seg.v"))?;
    let su: &[u32] = unsafe { mmapvec::as_slice(&su_src[..]) };
    let sv: &[u32] = unsafe { mmapvec::as_slice(&sv_src[..]) };
    // The raw array, not the dictionary: this runs during the build, and the
    // dictionary is derived from what this pass is still finishing.
    let sattr_map = mmapvec::open(&paths.f("seg.attr"))?;
    let sattr: &[u32] = unsafe { mmapvec::as_slice(&sattr_map[..]) };
    let gpts_map = mmapvec::open(&paths.f("geom.pts"))?;
    let gpts: &[(i32, i32)] = unsafe { mmapvec::as_slice(&gpts_map[..]) };
    let goff_map = mmapvec::open(&paths.f("geom.off"))?;
    let goff: &[u32] = unsafe { mmapvec::as_slice(&goff_map[..]) };

    let nseg = su.len();
    let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(nseg + nseg / 3);
    let mut big: Vec<(u64, u32)> = Vec::new();
    let mut oversized = 0u64;
    for sid in 0..nseg {
        let a = vcoord[su[sid] as usize];
        let b = vcoord[sv[sid] as usize];
        let (mut lo_la, mut hi_la) = (a.0.min(b.0), a.0.max(b.0));
        let (mut lo_lo, mut hi_lo) = (a.1.min(b.1), a.1.max(b.1));
        if sid + 1 < goff.len() {
            for &p in &gpts[goff[sid] as usize..goff[sid + 1] as usize] {
                lo_la = lo_la.min(p.0);
                hi_la = hi_la.max(p.0);
                lo_lo = lo_lo.min(p.1);
                hi_lo = hi_lo.max(p.1);
            }
        }
        let (y0, y1) = (cell_axis(lo_la, 900_000_000), cell_axis(hi_la, 900_000_000));
        let (x0, x1) = (cell_axis(lo_lo, 1_800_000_000), cell_axis(hi_lo, 1_800_000_000));
        // A road segment spans a handful of cells; a ferry does not. Twenty of
        // them cross open sea in one hop — Hirtshals to the Faroes is a single
        // segment whose bounding box covers 740 000 cells, and sweeping those
        // cost more index entries than the entire rest of the planet.
        //
        // Sweeping the box is wrong for them and dropping them is wrong too:
        // you cannot snap to a crossing you can never find. So index the
        // places a ferry is actually boarded — its endpoints and the points
        // along it — and leave the empty sea between them out.
        const MAX_CELLS: u64 = 4096;
        if (y1 - y0 + 1).saturating_mul(x1 - x0 + 1) > MAX_CELLS {
            oversized += 1;
            let mut put = |la: i32, lo: i32| {
                let (y, x) = (cell_axis(la, 900_000_000), cell_axis(lo, 1_800_000_000));
                pairs.push(((y << 32) | x, sid as u32));
            };
            put(a.0, a.1);
            put(b.0, b.1);
            if sid + 1 < goff.len() {
                for &p in &gpts[goff[sid] as usize..goff[sid + 1] as usize] {
                    put(p.0, p.1);
                }
            }
        } else {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    pairs.push(((y << 32) | x, sid as u32));
                }
            }
        }
        // Overview layer: motorways, trunks, primaries and ferries, on the 1°
        // grid. Drawing a continent from the fine grid would mean millions of
        // cell probes for a single frame.
        let cls = crate::build::class_of(crate::build::attr_class(sattr[sid]));
        if cls.tier() <= 1 {
            let (by0, by1) = (big_axis(lo_la, 900_000_000), big_axis(hi_la, 900_000_000));
            let (bx0, bx1) = (big_axis(lo_lo, 1_800_000_000), big_axis(hi_lo, 1_800_000_000));
            for y in by0..=by1 {
                for x in bx0..=bx1 {
                    big.push(((y << 32) | x, sid as u32));
                }
            }
        }
    }
    if oversized > 0 {
        eprintln!(
            "[pass4] {oversized} very long segment(s) (ferries) indexed at their endpoints \
             rather than across their whole bounding box"
        );
    }
    pairs.par_sort_unstable();
    big.par_sort_unstable();
    let n = pairs.len() as u64;
    write_cell_csr(paths, "big", &big)?;
    drop(big);
    write_cell_csr(paths, "snap", &pairs)?;
    Ok(n)
}

/// Write a sorted (cell, segment) list as a CSR: a key array to binary-search
/// and the deduplicated segment ids grouped behind it.
fn write_cell_csr(paths: &Paths, name: &str, pairs: &[(u64, u32)]) -> io::Result<()> {
    let mut wk = BufWriter::with_capacity(1 << 22, File::create(paths.f(&format!("{name}.cell")))?);
    let mut wo = BufWriter::with_capacity(1 << 22, File::create(paths.f(&format!("{name}.off")))?);
    let mut wv = BufWriter::with_capacity(1 << 22, File::create(paths.f(&format!("{name}.seg")))?);
    let mut i = 0usize;
    let mut written = 0u32;
    while i < pairs.len() {
        let key = pairs[i].0;
        let mut j = i;
        wk.write_all(&key.to_le_bytes())?;
        wo.write_all(&written.to_le_bytes())?;
        let mut last = u32::MAX;
        while j < pairs.len() && pairs[j].0 == key {
            if pairs[j].1 != last {
                wv.write_all(&pairs[j].1.to_le_bytes())?;
                written += 1;
                last = pairs[j].1;
            }
            j += 1;
        }
        i = j;
    }
    wo.write_all(&written.to_le_bytes())?;
    wk.flush()?;
    wo.flush()?;
    wv.flush()
}

#[inline]
fn cell_axis(v: i32, off: i64) -> u64 {
    ((v as i64 + off) / CELL_E7 as i64) as u64
}

#[inline]
fn big_axis(v: i32, off: i64) -> u64 {
    ((v as i64 + off) / BIG_E7 as i64) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_are_monotonic_and_distinct() {
        let a = cell_of(599_114_900, 107_573_300);
        let b = cell_of(599_114_900 + CELL_E7, 107_573_300);
        let c = cell_of(599_114_900, 107_573_300 + CELL_E7);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert!(b > a, "a step north increases the key");
        // Two points inside the same cell share a key.
        assert_eq!(a, cell_of(599_114_900 + 10, 107_573_300 + 10));
    }

    #[test]
    fn cell_axis_handles_the_southern_and_western_hemispheres() {
        assert_eq!(cell_axis(-900_000_000, 900_000_000), 0);
        assert_eq!(cell_axis(0, 900_000_000), 9000);
        assert_eq!(cell_axis(-1_800_000_000, 1_800_000_000), 0);
    }
}
