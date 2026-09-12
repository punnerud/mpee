//! Overlay graph: collapse a region to the roads that enter it.
//!
//! A long route crosses thousands of regions it never actually drives into.
//! Today it settles every vertex in them anyway, because the search has no way
//! to know that a suburb reachable by three roads cannot be worth entering. An
//! overlay gives it that: each region is replaced, for through-traffic, by a
//! table of the shortest paths between its **boundary vertices** — the ones an
//! edge crosses into. The interior is never touched.
//!
//! This is Customizable Route Planning, and it is matcodec's gateway
//! structure applied to the graph rather than to a matrix: predict the region
//! by its few entrances, and keep the exact cost between them.
//!
//! # The partition is the whole game
//!
//! Measured on Norway, at the same 4 096-vertex target:
//!
//! ```text
//!                        boundary      overlay      regions with <=4 gateways
//!   geographic blocks    2.25 %        18.4 MB      0
//!   graph distance       2.00 %         5.4 MB      902, covering 14.6 %
//! ```
//!
//! Cutting along latitude and longitude slices through dense areas; growing
//! regions along the roads themselves finds the peninsulas, the valleys and
//! the estates behind one junction. Same target size, a third of the overlay.
//!
//! # Exactness
//!
//! A cell's table holds the shortest path between two boundary vertices
//! **using only edges inside that cell**. A real shortest path that leaves and
//! re-enters is not lost: the overlay search finds it by going out through the
//! boundary and back, exactly as the road does. So the two-level answer equals
//! the one-level answer, which `tests/overlay.rs` checks against the plain
//! router rather than assuming.

use crate::build::{attr_kmh, Paths};
use crate::dataset::Dataset;
use crate::overrides::Overrides;
use crate::mmapvec;
use crate::router::SEG_MASK;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Vertices per region. Measured sweet spot: smaller regions leave more
/// boundary (13 % at 256), larger ones make each cell's table grow as b².
pub const TARGET: usize = 4096;

/// The Earth's polar radius. Every geographic bound uses it rather than the
/// equatorial or a mean radius, because it is the smallest the planet gets:
/// a distance computed on it can never exceed the real one.
pub const R_POLAR: f64 = 6_356_752.3;

/// Subtracted from every chord before it becomes a bound.
///
/// `vxyz` is `f32`, so a coordinate of magnitude 6.4e6 carries up to a quarter
/// metre of rounding — under half a metre in the chord. Four metres is an order
/// of magnitude of headroom and costs a tenth of a second at motorway speed,
/// against bounds measured in hours. A lower bound that is occasionally a metre
/// too high is not a lower bound.
const XYZ_SLACK_M: f64 = 4.0;

/// The most gates one region may end up with.
///
/// A region's table is `2b²` entries and `b` restricted searches, so this is
/// the one number that bounds both the disk a rung costs and the work to warm
/// it. Uncapped, the upper rungs do not converge: on the planet the worst
/// region's gate count ran 505 -> 2302 -> 11685 -> 32166 as the ladder rose,
/// because each rung merges and merging concentrates gates on whatever is
/// already worst. The top was not a region at all but the road network's giant
/// connected component with a boundary drawn round it.
///
/// 2048 is measured, not chosen. Sweeping the planet, Lisboa-Warszawa:
///
/// | cap  | rungs | top rung  | table    | settled | cold fill     |
/// |------|-------|-----------|----------|---------|---------------|
/// | 256  | 4     | 1 719 392 | 15.70 GB | -       | -             |
/// | 512  | 5     |   612 408 | 16.36 GB | -       | -             |
/// | 1024 | 6     |   195 756 | 16.71 GB |  62 690 | 60 min        |
/// | 2048 | 3     |   167 697 | 12.29 GB |  73 459 | 29 min        |
/// | 4096 | 4     |    30 850 | 14.58 GB | -       | >2 h, gave up |
///
/// Below 1024 regions cannot grow enough to bury gates, so the ladder stops
/// collapsing — 256 leaves 1.7 million gates at the top. Above 2048 the rows
/// grow faster than the boundary shrinks: at 4096 a region holds 3898 gates and
/// a 121 MB row, and one corridor had not finished filling after two hours on
/// one core.
///
/// The surprise is that the ladder's *height* decides the query, not the top
/// rung's size. 2048 has a smaller top rung than 1024 and still settles 17 %
/// more vertices, because three rungs let a search climb less far than six. It
/// wins anyway: half the fill, a quarter less disk, 296 ms warm against 853 ms
/// for the partition this replaced.
pub const GATE_CAP: usize = 2048;

/// How many times a rung may cut its over-cap regions before giving up.
///
/// Each round quarters the target, so this reaches a target of `4^-8` of the
/// original — far past the point where anything still moves. The loop stops on
/// its own when a round changes nothing, which is the case that actually ends
/// it: a lower-level region already over the cap cannot be cut from above.
pub const CUT_ROUNDS: usize = 8;

/// Unreachable, in the same units the router uses (tenths of a second).
pub const UNREACHABLE: u32 = u32::MAX;

#[inline]
fn dur_ds(len_cm: u32, kmh: u16) -> u32 {
    ((len_cm as u64 * 36) / (kmh.max(1) as u64 * 100)).min(u32::MAX as u64) as u32
}

/// What one segment costs, with any local override applied.
///
/// `None` means the segment is closed and the edge does not exist.
///
/// The overlay has to see overrides for the same reason the router does, and
/// it is easy to miss why: a table entry is a *shortcut across a whole region*,
/// summarising paths the search never walks. If the table were computed from
/// the original weights, closing a road inside a region would change every
/// route that goes along it and no route that hops over it — the shortcut
/// would still quote a journey through a road that is shut.
#[inline(always)]
fn edge_ds_ov(ds: &Dataset, ovr: Option<&Overrides>, sid: usize) -> Option<u32> {
    let len = ds.seg_len[sid];
    let kmh = attr_kmh(ds.attr(sid));
    if let Some(o) = ovr {
        if o.touches(sid as u32) {
            return match o.effect(sid as u32) {
                Some(crate::overrides::Effect::Closed) => None,
                Some(crate::overrides::Effect::Speed(k)) => Some(dur_ds(len, k)),
                Some(crate::overrides::Effect::Penalty(f)) => {
                    Some(((dur_ds(len, kmh) as f32 * f) as u32).max(1))
                }
                None => Some(dur_ds(len, kmh)),
            };
        }
    }
    Some(dur_ds(len, kmh))
}

#[inline]
fn edge_ds(ds: &Dataset, sid: usize) -> u32 {
    dur_ds(ds.seg_len[sid], attr_kmh(ds.attr(sid)))
}

pub struct OverlayStats {
    pub cells: usize,
    pub boundary: usize,
    pub matrix_entries: u64,
    pub secs: f64,
}

/// Grow connected regions of roughly `target` vertices each.
///
/// Deliberately the cheapest partition that follows the graph: a BFS that
/// stops at the size cap and starts again from the first unassigned vertex.
/// It is not a minimum-cut partitioner, but it already beats cutting by
/// geography by 3.4× on overlay size, and it costs one pass.
fn partition(ds: &Dataset, target: usize) -> (Vec<u32>, usize) {
    let nv = ds.n_vertices();
    let mut cell = vec![u32::MAX; nv];
    let mut ncells = 0u32;
    let mut q: std::collections::VecDeque<u32> = Default::default();
    for s in 0..nv as u32 {
        if cell[s as usize] != u32::MAX {
            continue;
        }
        q.clear();
        q.push_back(s);
        cell[s as usize] = ncells;
        let mut n = 1usize;
        while let Some(u) = q.pop_front() {
            if n >= target {
                break;
            }
            // Undirected for partitioning: a one-way street still keeps its
            // two ends in the same region.
            for (head, fwd) in [(ds.head, true), (ds.rhead, false)] {
                let (a, b) = (head[u as usize] as usize, head[u as usize + 1] as usize);
                for k in a..b {
                    let v = if fwd { ds.target(u, k) } else { ds.rtarget(u, k) };
                    if cell[v as usize] == u32::MAX {
                        cell[v as usize] = ncells;
                        n += 1;
                        q.push_back(v);
                        if n >= target {
                            break;
                        }
                    }
                }
                if n >= target {
                    break;
                }
            }
        }
        ncells += 1;
    }
    (cell, ncells as usize)
}

/// A region this small collapses nothing worth keeping.
const SMALL: usize = 64;
/// How far past `target` an absorbing region may grow.
const SLACK: usize = 2;

#[inline]
fn find(par: &mut [u32], mut x: u32) -> u32 {
    while par[x as usize] != x {
        par[x as usize] = par[par[x as usize] as usize];
        x = par[x as usize];
    }
    x
}

/// Absorb stranded fragments into the region they hang off.
///
/// The BFS fills a region to `target`, stops, and abandons whatever was still
/// on its frontier. Those vertices get picked up later as regions of their
/// own — usually a single vertex. Measured on the planet before this pass:
/// 1 756 486 regions held exactly one vertex, and 1 685 456 of them *had
/// edges* (mean degree 1.10, so dead-end tips). Nearly 30 % of the boundary
/// graph was fragments like that, and every one of them is forced to be a
/// boundary vertex, because by definition all its edges leave its region.
///
/// Absorbing a tip is free in table terms. It brings no new boundary vertex
/// with it — it has nowhere else to go — so `b` is unchanged and `b²` with it.
/// The pass only ever removes boundary vertices.
///
/// Whole regions are merged rather than individual vertices, which keeps every
/// region connected: that is what lets the region-internal Dijkstra reach all
/// of a region from any of its boundary.
fn absorb_strays(ds: &Dataset, cell: &mut [u32], ncells: usize, target: usize) -> usize {

    let mut size = vec![0u32; ncells];
    for &c in cell.iter() {
        size[c as usize] += 1;
    }
    let mut par: Vec<u32> = (0..ncells as u32).collect();
    let cap = (target * SLACK) as u32;
    let mut merged = 0usize;

    // Rounds, because absorbing one fragment can leave its neighbour small
    // enough to be absorbed in turn. It converges quickly; four is generous.
    for _ in 0..4 {
        // Which vertices currently sit in a small region. Cheap to recompute,
        // and it keeps the edge walk proportional to the fragments rather than
        // to the whole planet.
        let mut small: Vec<u32> = Vec::new();
        for (u, &c0) in cell.iter().enumerate() {
            let c = find(&mut par, c0);
            if (size[c as usize] as usize) <= SMALL {
                small.push(u as u32);
            }
        }
        if small.is_empty() {
            break;
        }
        // (small region, neighbour region) once per crossing edge, so sorting
        // groups them and the longest run is the strongest attachment.
        let mut prop: Vec<(u32, u32)> = Vec::with_capacity(small.len() * 2);
        for &u in &small {
            let c = find(&mut par, cell[u as usize]);
            for (head, fwd) in [(ds.head, true), (ds.rhead, false)] {
                let (a, b) = (head[u as usize] as usize, head[u as usize + 1] as usize);
                for k in a..b {
                    let v = if fwd { ds.target(u, k) } else { ds.rtarget(u, k) };
                    let d = find(&mut par, cell[v as usize]);
                    if d != c {
                        prop.push((c, d));
                    }
                }
            }
        }
        if prop.is_empty() {
            break;
        }
        prop.sort_unstable();

        // Best neighbour per small region: the one it shares most edges with.
        let mut best: Vec<(u32, u32, u32)> = Vec::new(); // (region, neighbour, edges)
        let mut i = 0;
        while i < prop.len() {
            let c = prop[i].0;
            let (mut bd, mut bn) = (u32::MAX, 0u32);
            while i < prop.len() && prop[i].0 == c {
                let d = prop[i].1;
                let mut n = 0;
                while i < prop.len() && prop[i] == (c, d) {
                    n += 1;
                    i += 1;
                }
                if n > bn {
                    bn = n;
                    bd = d;
                }
            }
            if bd != u32::MAX {
                best.push((c, bd, bn));
            }
        }
        // Smallest first: the most stranded fragments get first claim on the
        // room that is left under the cap.
        best.sort_unstable_by_key(|&(c, _, _)| size[c as usize]);

        let mut any = false;
        for (c, d, _) in best {
            let (rc, rd) = (find(&mut par, c), find(&mut par, d));
            if rc == rd || size[rc as usize] + size[rd as usize] > cap {
                continue;
            }
            // Attach the fragment to the neighbour, not the other way round.
            par[rc as usize] = rd;
            size[rd as usize] += size[rc as usize];
            size[rc as usize] = 0;
            merged += 1;
            any = true;
        }
        if !any {
            break;
        }
    }

    // Renumber into a dense range.
    let mut remap = vec![u32::MAX; ncells];
    let mut next = 0u32;
    for c in cell.iter_mut() {
        let r = find(&mut par, *c) as usize;
        if remap[r] == u32::MAX {
            remap[r] = next;
            next += 1;
        }
        *c = remap[r];
    }
    eprintln!("[overlay] absorbed {merged} stranded regions, {ncells} -> {next}");
    next as usize
}

/// Lay per-region tables out as bytes, narrowing each region that can be.
///
/// A region's internal distances are bounded by its own diameter, so most
/// regions fit 16-bit entries and a few do not. Each declares its own width;
/// the check is constant while walking one region's row, so it costs nothing
/// in the loop and there is nothing to decompress. Regions start on a 4-byte
/// boundary so a wide one can still be read as `u32`.
///
/// Shared by both overlay levels: the second level stores exactly the same
/// shape of table, just over a graph whose vertices are the first level's
/// boundary.
#[allow(clippy::too_many_arguments)]
fn narrow_and_write(
    paths: &Paths,
    wide: &[u32],
    ohead: &[u64],
    bhead: &[u32],
    ncells: usize,
    mat_name: &str,
    head_name: &str,
    width_name: &str,
) -> io::Result<(usize, u64)> {
    let mut width = vec![4u8; ncells];
    let mut boff = vec![0u64; ncells + 1];
    let mut acc = 0u64;
    let mut narrowed = 0usize;
    for c in 0..ncells {
        let b = (bhead[c + 1] - bhead[c]) as usize;
        let n = b * b;
        let (a, e) = (ohead[c] as usize, ohead[c] as usize + n);
        // 0xFFFF is the narrow form's "unreachable", so a finite value must
        // stay below it.
        if n > 0 && wide[a..e].iter().all(|&v| v == UNREACHABLE || v < 0xFFFF) {
            width[c] = 2;
            narrowed += 1;
        }
        boff[c] = acc;
        acc += (n as u64) * width[c] as u64;
        acc = acc.div_ceil(4) * 4;
    }
    boff[ncells] = acc;
    let mut out = mmapvec::create::<u8>(&paths.f(mat_name), acc as usize)?;
    {
        let o: &mut [u8] = &mut out[..];
        for c in 0..ncells {
            let b = (bhead[c + 1] - bhead[c]) as usize;
            let n = b * b;
            let a = ohead[c] as usize;
            let mut p = boff[c] as usize;
            if width[c] == 2 {
                for &v in &wide[a..a + n] {
                    let x: u16 = if v == UNREACHABLE { 0xFFFF } else { v as u16 };
                    o[p..p + 2].copy_from_slice(&x.to_le_bytes());
                    p += 2;
                }
            } else {
                for &v in &wide[a..a + n] {
                    o[p..p + 4].copy_from_slice(&v.to_le_bytes());
                    p += 4;
                }
            }
        }
    }
    out.flush()?;
    write_u64(paths, head_name, &boff)?;
    std::fs::write(paths.f(width_name), &width)?;
    Ok((narrowed, acc))
}

/// Build the overlay for a finished dataset.
pub fn build(paths: &Paths, ds: &Dataset, target: usize) -> io::Result<OverlayStats> {
    build_with(paths, ds, target, false)
}

/// `lazy` writes level 0's shape and leaves its values to be filled on demand.
///
/// It is what makes an update possible without a rebuild. An eager table packs
/// each region at a width chosen from values that an update is about to
/// change, so a changed region cannot be written back into it — the new
/// numbers may not fit the space reserved for the old. A lazy level has no
/// such commitment: forgetting a region is clearing bits, and it is safe to do
/// while queries are running.
pub fn build_with(
    paths: &Paths,
    ds: &Dataset,
    target: usize,
    lazy: bool,
) -> io::Result<OverlayStats> {
    let t0 = std::time::Instant::now();
    let nv = ds.n_vertices();
    let (mut cell, ncells) = partition(ds, target);
    eprintln!("[overlay] {ncells} regions over {nv} vertices ({:.1} s)", t0.elapsed().as_secs_f64());
    let ncells = absorb_strays(ds, &mut cell, ncells, target);

    // Vertices grouped by cell, and the boundary subset of each.
    let mut vhead = vec![0u32; ncells + 1];
    for &c in &cell {
        vhead[c as usize + 1] += 1;
    }
    for i in 1..=ncells {
        vhead[i] += vhead[i - 1];
    }
    let mut cursor = vhead.clone();
    let mut vlist = vec![0u32; nv];
    for u in 0..nv as u32 {
        let c = cell[u as usize] as usize;
        vlist[cursor[c] as usize] = u;
        cursor[c] += 1;
    }

    // A vertex is on the boundary when an edge leaves its region — in either
    // direction, since the backward search has to enter through them too.
    let is_bnd: Vec<bool> = (0..nv)
        .into_par_iter()
        .map(|u| {
            let c = cell[u];
            let u32u = u as u32;
            for (head, fwd) in [(ds.head, true), (ds.rhead, false)] {
                let (a, b) = (head[u] as usize, head[u + 1] as usize);
                for k in a..b {
                    let v = if fwd { ds.target(u32u, k) } else { ds.rtarget(u32u, k) };
                    if cell[v as usize] != c {
                        return true;
                    }
                }
            }
            false
        })
        .collect();

    let mut bhead = vec![0u32; ncells + 1];
    let mut blist: Vec<u32> = Vec::new();
    for c in 0..ncells {
        bhead[c] = blist.len() as u32;
        // Sorted, so a vertex's index inside its cell is a binary search.
        for &u in &vlist[vhead[c] as usize..vhead[c + 1] as usize] {
            if is_bnd[u as usize] {
                blist.push(u);
            }
        }
        blist[bhead[c] as usize..].sort_unstable();
    }
    bhead[ncells] = blist.len() as u32;
    let nb = blist.len();

    // A region's internal distances are bounded by its own diameter, and 97.7 %
    // of them fit in 16 bits. Rather than pay 32 everywhere or lose exactness,
    // each region declares its own width: measured, 0.35 % of regions need the
    // wide form. The check is constant while walking one region's row, so it
    // costs nothing in the loop and there is nothing to decompress.
    //
    // Widths are decided after the tables are computed, so this first pass
    // lays them out at the wide width and a second pass narrows what it can.
    let mut ohead = vec![0u64; ncells + 1];
    for c in 0..ncells {
        let b = (bhead[c + 1] - bhead[c]) as u64;
        ohead[c + 1] = ohead[c] + b * b;
    }
    let entries = ohead[ncells];
    eprintln!(
        "[overlay] {nb} boundary vertices ({:.2} %), {entries} table entries ({:.2} GB at 32 bits)",
        nb as f64 / nv as f64 * 100.0,
        entries as f64 * 4.0 / 1e9
    );

    if lazy {
        let f = level_files(0);
        let lf = lazy_files(0);
        // A level opens eager when both of those exist, so switching to lazy
        // has to take them away.
        std::fs::remove_file(paths.f(&f[4])).ok();
        std::fs::remove_file(paths.f(&f[5])).ok();
        // The eager table packs `b²` per region and narrows it; the lazy one
        // holds `2b²` — both directions, side by side — so that a region's
        // block is self-contained and survives its neighbours changing size.
        let mut lohead = vec![0u64; ncells + 1];
        for c in 0..ncells {
            let b = (bhead[c + 1] - bhead[c]) as u64;
            lohead[c + 1] = lohead[c] + 2 * b * b;
        }
        write_u32(paths, &f[0], &cell)?;
        write_u32(paths, &f[1], &blist)?;
        write_u32(paths, &f[2], &bhead)?;
        write_u64(paths, &f[3], &lohead)?;
        write_u32(paths, &lf[0], &vhead)?;
        write_u32(paths, &lf[1], &vlist)?;
        std::fs::File::create(paths.f(&lf[2]))?.set_len(lohead[ncells] * 4)?;
        std::fs::File::create(paths.f(&lf[3]))?.set_len((2 * nb).div_ceil(8) as u64)?;
        // Last, so that a stamp never stands for a level that was not finished.
        write_lazy_stamp(
            &paths.f(&fmt_file(0)),
            ncells as u64,
            nb as u64,
            lohead[ncells],
        )?;
        // `ov.bounds` is a per-region cost summary the eager pass computes as
        // a by-product. Without a table there is nothing to summarise, so it
        // is written empty and the reader treats every region as unbounded.
        std::fs::File::create(paths.f("ov.bounds"))?.set_len((ncells * 8) as u64)?;
        eprintln!(
            "[overlay] lazy: {:.2} GB of table reserved, {} rows to fill on demand",
            entries as f64 * 8.0 / 1e9,
            2 * nb
        );
        return Ok(OverlayStats {
            cells: ncells,
            boundary: nb,
            matrix_entries: entries,
            secs: t0.elapsed().as_secs_f64(),
        });
    }
    let mut mat = mmapvec::create::<u32>(&paths.f("ov.mat.wide"), entries as usize)?;
    let matp = {
        let s: &mut [u32] = unsafe { mmapvec::as_mut_slice(&mut mat[..]) };
        crate::parallel::Scatter(s.as_mut_ptr(), s.len())
    };
    // Per-cell bounds, for deciding whether a region can help before its table
    // is read at all.
    let mut bounds = vec![(0u32, 0u32); ncells];

    let bounds_out: Vec<(usize, u32, u32)> = (0..ncells)
        .into_par_iter()
        .map(|c| {
            let (bs, be) = (bhead[c] as usize, bhead[c + 1] as usize);
            let b = be - bs;
            if b == 0 {
                return (c, UNREACHABLE, 0);
            }
            let cellv = &vlist[vhead[c] as usize..vhead[c + 1] as usize];
            // Cell-local index for the vertices of this region.
            let local = |v: u32| cellv.binary_search(&v).ok();
            let n = cellv.len();
            let mut dist = vec![UNREACHABLE; n];
            let mut touched: Vec<u32> = Vec::new();
            let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
            let (mut lo, mut hi) = (UNREACHABLE, 0u32);
            for (i, &src) in blist[bs..be].iter().enumerate() {
                for &t in &touched {
                    dist[t as usize] = UNREACHABLE;
                }
                touched.clear();
                heap.clear();
                let si = local(src).expect("boundary vertex is in its own cell");
                dist[si] = 0;
                touched.push(si as u32);
                heap.push(Reverse((0u32, si as u32)));
                while let Some(Reverse((d, li))) = heap.pop() {
                    if d > dist[li as usize] {
                        continue;
                    }
                    let u = cellv[li as usize];
                    let (a, e) = (ds.head[u as usize] as usize, ds.head[u as usize + 1] as usize);
                    for k in a..e {
                        let v = ds.target(u, k);
                        // Inside the cell only: a path that leaves is the
                        // overlay search's job, not this table's.
                        let Some(vi) = local(v) else { continue };
                        let sid = (ds.eseg[k] & SEG_MASK) as usize;
                        let nd = d.saturating_add(edge_ds(ds, sid));
                        if nd < dist[vi] {
                            if dist[vi] == UNREACHABLE {
                                touched.push(vi as u32);
                            }
                            dist[vi] = nd;
                            heap.push(Reverse((nd, vi as u32)));
                        }
                    }
                }
                for (j, &dst) in blist[bs..be].iter().enumerate() {
                    let d = local(dst).map(|x| dist[x]).unwrap_or(UNREACHABLE);
                    unsafe { matp.put(ohead[c] as usize + i * b + j, d) };
                    if i != j && d != UNREACHABLE {
                        lo = lo.min(d);
                        hi = hi.max(d);
                    }
                }
            }
            (c, lo, hi)
        })
        .collect();
    for (c, lo, hi) in bounds_out {
        bounds[c] = (lo, hi);
    }
    mat.flush()?;

    let (narrowed, byte_len) = {
        let wide: &[u32] = unsafe { mmapvec::as_slice(&mat[..]) };
        narrow_and_write(paths, wide, &ohead, &bhead, ncells, "ov.mat", "ov.head", "ov.width")?
    };
    drop(mat);
    std::fs::remove_file(paths.f("ov.mat.wide")).ok();
    eprintln!(
        "[overlay] {narrowed} of {ncells} regions fit 16-bit entries — table {:.2} GB ({:.2}x smaller)",
        byte_len as f64 / 1e9,
        entries as f64 * 4.0 / byte_len as f64
    );

    write_u32(paths, "cell.of", &cell)?;
    write_u32(paths, "cell.bnd", &blist)?;
    write_u32(paths, "cell.bhead", &bhead)?;
    {
        let mut w = BufWriter::with_capacity(1 << 20, std::fs::File::create(paths.f("ov.bounds"))?);
        for (lo, hi) in &bounds {
            w.write_all(&lo.to_le_bytes())?;
            w.write_all(&hi.to_le_bytes())?;
        }
        w.flush()?;
    }
    Ok(OverlayStats {
        cells: ncells,
        boundary: nb,
        matrix_entries: entries,
        secs: t0.elapsed().as_secs_f64(),
    })
}

fn write_u32(paths: &Paths, name: &str, v: &[u32]) -> io::Result<()> {
    let mut w = BufWriter::with_capacity(1 << 22, std::fs::File::create(paths.f(name))?);
    for x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()
}
fn write_u64(paths: &Paths, name: &str, v: &[u64]) -> io::Result<()> {
    let mut w = BufWriter::with_capacity(1 << 22, std::fs::File::create(paths.f(name))?);
    for x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()
}

// ---------------------------------------------------------------- reading

/// One level of the overlay.
///
/// Every level has the same shape, because a level *is* a road network: its
/// vertices are the level below's boundary vertices, its edges are that
/// level's table shortcuts plus the cut edges between its regions. That is why
/// the second level was built by the same restricted Dijkstra as the first —
/// and why there is no reason to stop at two.
/// Where a level keeps its numbers.
///
/// The shape of a level is cheap and the numbers are not — on the planet,
/// seconds against twenty minutes — so the two are stored apart and a level
/// may have its values either way.
pub enum Values {
    /// Every entry computed up front, narrowed to 2 or 4 bytes per region.
    Eager { mat: &'static [u8], width: &'static [u8] },
    /// Rows computed the first time they are asked for, and kept.
    ///
    /// The upper rungs of the ladder are why this exists. A rung's table grows
    /// as its regions get coarser — the planet's second rung came to 8.66 GB
    /// against the first's 5.07 — while a single route reads a handful of its
    /// rows. Materialising all of it is the waste; the file is sparse, so the
    /// disk holds exactly the rows somebody actually asked for.
    Lazy(Lazy),
}

/// A row cache backed by a sparse file.
///
/// The file is created at the table's full logical size but nothing is written
/// to it, so the filesystem allocates no blocks. A row is filled on its first
/// miss and a bit is set; from then on reading it is the same single indexed
/// load an eager table would be.
///
/// Rows are held in both directions. The search reads *rows* going forward and
/// *columns* going backward, and a column would touch `b` separate rows — so a
/// backward row is computed by running the region's Dijkstra on the reverse
/// graph instead, which is one row again.
pub struct Lazy {
    data: *mut u8,
    have: *mut u8,
    /// Rows in one direction; the backward bits start at this index.
    rows: usize,
    /// Region -> the vertex set its table is computed over.
    vhead: &'static [u32],
    vlist: &'static [u32],
    /// How often each member has been settled while filling this level's rows.
    ///
    /// Betweenness, as a by-product. Every row is a shortest-path tree over
    /// the region, so a member that keeps being settled is one the region's
    /// traffic goes through — and a cut should avoid it. Null when the dataset
    /// carries no counter, which costs a branch that predicts perfectly.
    ///
    /// Counted per *settle*, not per relaxation. Settles are roughly `1/b` as
    /// many, which keeps the atomic out of the innermost loop — the one where
    /// a binary search per relaxation already dominates.
    use_ct: *mut u32,
    _maps: Vec<memmap2::MmapMut>,
}

// The data and presence maps are only ever written through `fill_row`, which
// writes a whole row before setting its bit, and every writer computes the
// same bytes for the same row. A torn read is therefore impossible: a reader
// either sees the bit unset and computes the row itself, or sees it set and
// reads bytes that are already final.
unsafe impl Send for Lazy {}
unsafe impl Sync for Lazy {}

impl Lazy {
    /// Presence bits are atomic because rows are filled in parallel.
    ///
    /// Two threads setting different bits of the same byte with a plain
    /// read-modify-write would lose one of them — harmless in itself, since a
    /// lost bit only means the row is computed again, but the ordering matters
    /// more: a reader must not see the bit before it can see the row. So the
    /// bit is set with `Release` after the bytes are written, and read with
    /// `Acquire` before they are.
    #[inline]
    fn byte(&self, n: usize) -> &std::sync::atomic::AtomicU8 {
        unsafe { &*(self.have.add(n >> 3) as *const std::sync::atomic::AtomicU8) }
    }
    #[inline]
    fn bit(&self, row: usize, fwd: bool) -> bool {
        let n = if fwd { row } else { self.rows + row };
        self.byte(n).load(std::sync::atomic::Ordering::Acquire) >> (n & 7) & 1 == 1
    }
    #[inline]
    fn clear_bit(&self, row: usize, fwd: bool) {
        let n = if fwd { row } else { self.rows + row };
        self.byte(n).fetch_and(!(1 << (n & 7)), std::sync::atomic::Ordering::Release);
    }
    #[inline]
    fn set_bit(&self, row: usize, fwd: bool) {
        let n = if fwd { row } else { self.rows + row };
        self.byte(n).fetch_or(1 << (n & 7), std::sync::atomic::Ordering::Release);
    }
    /// Byte offset of one row's first entry.
    ///
    /// A region owns one contiguous block of `2b²` entries: the forward table,
    /// then the backward one. It used to be two halves of the whole level, with
    /// the backward rows offset by the level's total entry count — which meant
    /// that growing the level moved every backward row of every region, and made
    /// splitting impossible without rewriting the file. Self-contained blocks
    /// cost nothing and let a region keep its place while its neighbours change.
    #[inline]
    fn at(&self, base: u64, b: usize, i: usize, fwd: bool) -> usize {
        let e = base + if fwd { 0 } else { (b * b) as u64 } + (i * b) as u64;
        e as usize * 4
    }
    #[inline]
    fn get(&self, p: usize, j: usize) -> u32 {
        unsafe {
            let q = self.data.add(p + j * 4);
            u32::from_le_bytes([*q, *q.add(1), *q.add(2), *q.add(3)])
        }
    }
    #[inline]
    fn put(&self, p: usize, j: usize, v: u32) {
        unsafe {
            let q = self.data.add(p + j * 4);
            for (n, b) in v.to_le_bytes().into_iter().enumerate() {
                *q.add(n) = b;
            }
        }
    }
    /// Record that a member was settled. Relaxed ordering: this is a
    /// statistic, and losing an increment to a race costs nothing that
    /// matters — an approximate count is what a cut decision needs.
    #[inline]
    fn note_use(&self, at: usize) {
        if self.use_ct.is_null() {
            return;
        }
        unsafe {
            (*(self.use_ct.add(at) as *const std::sync::atomic::AtomicU32))
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Every member's count, in `vlist` order.
    /// The vertex set this level's tables are computed over, in row order.
    pub fn vlist_all(&self) -> &[u32] {
        self.vlist
    }

    pub fn use_ct_all(&self) -> &[u32] {
        if self.use_ct.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.use_ct, self.vlist.len()) }
    }

    /// Whether this level gathers use statistics.
    pub fn counts_use(&self) -> bool {
        !self.use_ct.is_null()
    }

    /// The use counts of one region's members.
    pub fn uses(&self, c: u32) -> &[u32] {
        if self.use_ct.is_null() {
            return &[];
        }
        let (a, b) = (self.vhead[c as usize] as usize, self.vhead[c as usize + 1] as usize);
        unsafe { std::slice::from_raw_parts(self.use_ct.add(a), b - a) }
    }

    /// How many rows have been computed, and how much of the file that is.
    pub fn filled(&self) -> (usize, usize) {
        let mut n = 0;
        for i in 0..(2 * self.rows).div_ceil(8) {
            n += self.byte(i * 8).load(std::sync::atomic::Ordering::Relaxed).count_ones() as usize;
        }
        (n, 2 * self.rows)
    }
}

/// One level of the overlay.
///
/// Every level has the same shape, because a level *is* a road network: its
/// vertices are the level below's boundary vertices, its edges are that
/// level's table shortcuts plus the cut edges between its regions. That is why
/// the second level was built by the same restricted Dijkstra as the first —
/// and why there is no reason to stop at two.
pub struct Level {
    /// Maps a region of the level below to a region of this one. At level 0
    /// the "level below" is the road graph itself, so this maps a vertex.
    of: &'static [u32],
    bnd: &'static [u32],
    bhead: &'static [u32],
    head: &'static [u64],
    /// Geographic box per region, empty when the dataset has none.
    geo: &'static [(i32, i32, i32, i32)],
    pub values: Values,
}

impl Level {
    pub fn regions(&self) -> usize {
        self.bhead.len().saturating_sub(1)
    }

    /// The region of this level that a region of the level below sits in.
    #[inline]
    pub fn parent(&self, c: u32) -> u32 {
        self.of[c as usize]
    }

    /// Boundary vertices of a region, ascending. They are graph vertex ids at
    /// every level — a level narrows *which* vertices are boundary, it never
    /// invents new ones.
    #[inline]
    pub fn boundary(&self, c: u32) -> &[u32] {
        let (a, b) = (self.bhead[c as usize] as usize, self.bhead[c as usize + 1] as usize);
        &self.bnd[a..b]
    }

    /// Every gate of the level, in region order — the array `boundary_start`
    /// indexes into.
    #[inline]
    pub fn boundary_all(&self) -> &[u32] {
        self.bnd
    }

    /// Where this region's gates begin in the level-wide boundary list, so a
    /// caller can index its own arrays by that position rather than by vertex
    /// id — which for the planet would mean 270 million slots to describe 6.5
    /// million gates.
    #[inline]
    pub fn boundary_start(&self, c: u32) -> usize {
        self.bhead[c as usize] as usize
    }

    #[inline]
    pub fn bindex(&self, c: u32, v: u32) -> Option<usize> {
        self.boundary(c).binary_search(&v).ok()
    }

    pub fn is_lazy(&self) -> bool {
        matches!(self.values, Values::Lazy(_))
    }

    /// Shortest time from the `i`-th to the `j`-th boundary vertex of region
    /// `c`, using only paths inside it.
    ///
    /// On an eager level the width branch is constant for a region, so walking
    /// one row predicts perfectly — and nothing is decompressed, only read at
    /// its own size. On a lazy one the row must already have been ensured.
    #[inline(always)]
    pub fn cost(&self, c: u32, i: usize, j: usize) -> u32 {
        self.cost_dir(c, i, j, true)
    }

    /// `fwd` reads the row from `i`; otherwise the cost *into* `i` from `j`.
    ///
    /// # Invariant
    /// On a lazy level the row must already have been filled, by
    /// [`Overlay::ensure_row`]. An unfilled row is a hole in a sparse file,
    /// which reads back as zeros — and zero here means *free travel*, the most
    /// dangerous wrong answer there is: a route through it looks not merely
    /// plausible but ideal. The search always ensures before it reads; the
    /// debug assertion is there so anything else that forgets says so loudly
    /// instead of quietly routing through nothing.
    #[inline(always)]
    pub fn cost_dir(&self, c: u32, i: usize, j: usize, fwd: bool) -> u32 {
        debug_assert!(
            match &self.values {
                Values::Lazy(z) => z.bit(self.bhead[c as usize] as usize + i, fwd),
                Values::Eager { .. } => true,
            },
            "row {i} of region {c} read before it was computed — see ensure_row"
        );
        let b = self.boundary(c).len();
        // `head` counts bytes for an eager level, because its regions store 2-
        // or 4-byte entries and the layout has to pack them; it counts entries
        // for a lazy one, which is always 4 bytes wide and needs no packing.
        let base = self.head[c as usize];
        match &self.values {
            Values::Eager { mat, width } => {
                // The transpose is not stored, so a backward read is the
                // mirrored entry of the same table.
                let (i, j) = if fwd { (i, j) } else { (j, i) };
                let k = i * b + j;
                if width[c as usize] == 2 {
                    let p = base as usize + k * 2;
                    let v = u16::from_le_bytes([mat[p], mat[p + 1]]);
                    if v == 0xFFFF {
                        UNREACHABLE
                    } else {
                        v as u32
                    }
                } else {
                    let p = base as usize + k * 4;
                    u32::from_le_bytes([mat[p], mat[p + 1], mat[p + 2], mat[p + 3]])
                }
            }
            Values::Lazy(z) => z.get(z.at(base, b, i, fwd), j),
        }
    }

    pub fn boundary_total(&self) -> usize {
        self.bnd.len()
    }

    /// Which region a global row index falls in.
    ///
    /// `bhead` is the prefix sum of rows per region, so this is one binary
    /// search — the same array the boundary lists are addressed through, used
    /// backwards.
    #[inline]
    pub fn region_of_row(&self, row: usize, nregions: usize) -> u32 {
        match self.bhead[..=nregions].binary_search(&(row as u32)) {
            // Several empty regions can share a boundary, so land on the last.
            Ok(mut i) => {
                while i < nregions && self.bhead[i + 1] as usize == row {
                    i += 1;
                }
                i as u32
            }
            Err(i) => (i - 1) as u32,
        }
    }

    /// First global row index of a region.
    #[inline]
    pub fn row_base(&self, c: u32) -> usize {
        self.bhead[c as usize] as usize
    }

    pub fn has_geo(&self) -> bool {
        !self.geo.is_empty()
    }

    /// A floor on how far a point is from anything in this region.
    ///
    /// The point is clamped into the region's box and measured on a sphere of
    /// the Earth's *polar* radius, which under-states the true geodesic — so
    /// the result is a genuine lower bound twice over. That is what lets it be
    /// used to skip work: a loose box prunes less but never wrongly, and two
    /// boxes may overlap without breaking anything.
    ///
    /// Returns 0 when the dataset carries no bounds, which disables every
    /// decision built on it rather than making a wrong one.
    pub fn geo_floor_m(&self, c: u32, lat_e7: i32, lon_e7: i32) -> f64 {
        if self.geo.is_empty() {
            return 0.0;
        }
        let (la0, la1, lo0, lo1) = self.geo[c as usize];
        if la0 == i32::MAX {
            return 0.0; // a region with no vertices bounds nothing
        }
        let clat = lat_e7.clamp(la0, la1) as f64 * 1e-7;
        let clon = lon_e7.clamp(lo0, lo1) as f64 * 1e-7;
        let (plat, plon) = (lat_e7 as f64 * 1e-7, lon_e7 as f64 * 1e-7);
        const R_POLAR: f64 = 6_356_752.3;
        let (dlat, dlon) = ((plat - clat).to_radians(), (plon - clon).to_radians());
        let a = (dlat / 2.0).sin().powi(2)
            + plat.to_radians().cos() * clat.to_radians().cos() * (dlon / 2.0).sin().powi(2);
        2.0 * R_POLAR * a.sqrt().clamp(0.0, 1.0).asin()
    }

    /// Bytes per entry in a region's table: 2 when its widest finite crossing
    /// fit 16 bits, 4 otherwise. A lazy level is always 4 — the narrowing needs
    /// every value of a region at once, which is exactly what it declines to
    /// compute.
    #[inline]
    pub fn width_of(&self, c: u32) -> u8 {
        match &self.values {
            Values::Eager { width, .. } => width[c as usize],
            Values::Lazy(_) => 4,
        }
    }
}

/// The file names one level's six arrays live under.
///
/// Level 0 keeps the names it was born with so a dataset built before there
/// were levels still opens; every level above is `l{n}.*`.
pub fn level_files(k: usize) -> [String; 6] {
    if k == 0 {
        ["cell.of", "cell.bnd", "cell.bhead", "ov.head", "ov.mat", "ov.width"]
            .map(|s| s.to_string())
    } else {
        let n = k + 1;
        [
            format!("l{n}.of"),
            format!("l{n}.bnd"),
            format!("l{n}.bhead"),
            format!("l{n}.head"),
            format!("l{n}.mat"),
            format!("l{n}.width"),
        ]
    }
}

/// The four extra arrays a lazily-valued level needs.
///
/// `vhead`/`vlist` are structure: the vertex set each region's table is
/// computed over, which an eager level throws away once the table exists but a
/// lazy one needs every time it fills a row. `lazy` is the sparse table and
/// `have` the presence bitmap.
pub fn lazy_files(k: usize) -> [String; 4] {
    let n = k + 1;
    if k == 0 {
        ["cell.vhead".into(), "cell.vlist".into(), "ov.lazy".into(), "ov.have".into()]
    } else {
        [
            format!("l{n}.vhead"),
            format!("l{n}.vlist"),
            format!("l{n}.lazy"),
            format!("l{n}.have"),
        ]
    }
}

/// The layout revision the lazy tables on disk are written in.
///
/// Revision 2 gives every region a self-contained `2b²` block: `b²` forward
/// entries followed by `b²` backward ones, both addressed from that region's
/// own `head`. Revision 1 stored `b²` per region and put every backward row in
/// a second, level-wide area starting at the level's total.
///
/// The two are indistinguishable by file size — a level holds `2Σb²` entries
/// either way — which is exactly why the revision has to be written down. Read
/// revision-1 bytes with the revision-2 formula and a region's backward block
/// lands on the *next* region's forward block. Every table read then returns
/// some other region's numbers, all of them valid `u32`s, and the router
/// believes them: the planet answered Paris-Berlin in 52 seconds and nothing
/// anywhere reported a fault.
pub const LAZY_LAYOUT: u32 = 2;

const STAMP_MAGIC: &[u8; 8] = b"MPEEOVL\0";
const STAMP_BYTES: usize = 40;

/// The file recording how a level's lazy table is laid out.
pub fn fmt_file(k: usize) -> String {
    if k == 0 {
        "ov.fmt".into()
    } else {
        format!("l{}.fmt", k + 1)
    }
}

/// Record the layout a freshly created lazy level is written in.
///
/// Written last, after the table and bitmap exist, so a stamp is never present
/// for a level that was not finished.
pub fn write_lazy_stamp(path: &Path, ncells: u64, nb: u64, entries: u64) -> io::Result<()> {
    let mut buf = Vec::with_capacity(STAMP_BYTES);
    buf.extend_from_slice(STAMP_MAGIC);
    buf.extend_from_slice(&LAZY_LAYOUT.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes()); // bytes per entry
    buf.extend_from_slice(&ncells.to_le_bytes());
    buf.extend_from_slice(&nb.to_le_bytes());
    buf.extend_from_slice(&entries.to_le_bytes());
    debug_assert_eq!(buf.len(), STAMP_BYTES);
    std::fs::write(path, &buf)
}

/// Check that a lazy level on disk is laid out the way this build reads it.
///
/// Two questions, and the cheap one is not the one that matters. The stamp
/// settles the revision in constant time — but the level that broke the planet
/// had no stamp to disagree with, and would have had a correct one had stamps
/// existed when it was built: its `head` was written by one build and its table
/// re-sized by a later one, so the two files were individually right and
/// jointly meaningless. Only the structural pass catches that, so it runs
/// whether or not a stamp is there.
///
/// Costs one sweep of `head` and `bhead` — a few milliseconds on the planet's
/// 1.1 million regions, once, against a table read that would otherwise be
/// wrong for the life of the process.
pub fn verify_lazy_layout(
    dir: &Path,
    k: usize,
    bhead: &[u32],
    head: &[u64],
    nb: usize,
) -> Result<(), String> {
    let lf = lazy_files(k);
    let name = |n: &str| format!("{}/{}", dir.display(), n);
    let lazy_len = match std::fs::metadata(dir.join(&lf[2])) {
        Ok(m) => m.len(),
        // No table at all: an eager level, or a level that simply is not
        // there. Not this function's business — the caller stops on its own.
        Err(_) => return Ok(()),
    };
    let ncells = bhead.len().saturating_sub(1);
    if head.len() != ncells + 1 {
        return Err(format!(
            "{}: {} regions in bhead but {} offsets in head",
            name(&lf[2]),
            ncells,
            head.len()
        ));
    }
    let cap = lazy_len / 4;

    // The stamp, when there is one. A mismatch here is the clean case: the
    // level says outright which revision it was written in.
    let fmt = dir.join(fmt_file(k));
    if let Ok(b) = std::fs::read(&fmt) {
        if b.len() < STAMP_BYTES || &b[..8] != STAMP_MAGIC {
            return Err(format!("{}: not an overlay stamp", name(&fmt_file(k))));
        }
        let rev = u32::from_le_bytes([b[8], b[9], b[10], b[11]]);
        if rev != LAZY_LAYOUT {
            return Err(format!(
                "{}: table is layout revision {rev}, this build reads revision {LAZY_LAYOUT} \
                 — rebuild the overlay for this level",
                name(&lf[2])
            ));
        }
        let g = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        for (got, want, what) in
            [(g(16), ncells as u64, "regions"), (g(24), nb as u64, "boundary vertices"), (g(32), cap, "table entries")]
        {
            if got != want {
                return Err(format!(
                    "{}: stamped with {got} {what}, found {want} — the level was rebuilt in part",
                    name(&fmt_file(k))
                ));
            }
        }
    }

    // The structural pass. Every region's block must lie inside the table, and
    // — when the offsets are still in region order, which they are until a
    // split appends out of line — must not run into the next region's.
    let sorted = head.windows(2).all(|w| w[0] <= w[1]);
    for c in 0..ncells {
        let b = (bhead[c + 1] - bhead[c]) as u64;
        let need = 2 * b * b;
        if head[c] + need > cap {
            return Err(format!(
                "{}: region {c} needs {need} entries at offset {} but the table holds {cap}",
                name(&lf[2]),
                head[c]
            ));
        }
        if sorted && head[c] + need > head[c + 1] {
            return Err(format!(
                "{}: region {c} has {b} boundary vertices and needs {need} entries, but only \
                 {} are free before region {} — the table is laid out {}b² per region, which is \
                 layout revision 1; this build reads revision {LAZY_LAYOUT}. Rebuild the overlay.",
                name(&lf[2]),
                head[c + 1] - head[c],
                c + 1,
                if head[c + 1] - head[c] == b * b { "" } else { "under 2" }
            ));
        }
    }

    // One bit per row per direction.
    let want_bits = (2 * nb).div_ceil(8) as u64;
    match std::fs::metadata(dir.join(&lf[3])) {
        Ok(m) if m.len() >= want_bits => {}
        Ok(m) => {
            return Err(format!(
                "{}: {} bytes of presence bits for {nb} rows, needs {want_bits}",
                name(&lf[3]),
                m.len()
            ))
        }
        Err(e) => return Err(format!("{}: {e}", name(&lf[3]))),
    }
    Ok(())
}

/// The use counter, parallel to `vlist`. Optional: a dataset without it works
/// exactly as before, it simply gathers no statistics.
pub fn use_file(k: usize) -> String {
    if k == 0 {
        "cell.use".into()
    } else {
        format!("l{}.use", k + 1)
    }
}

/// Map a lazy level's four arrays, the table and bitmap writable.
///
/// # Safety
/// The returned pointers stay valid because the mappings are moved into
/// `keep`, which the `Overlay` owns for as long as the pointers are used.
unsafe fn open_lazy(
    dir: &Path,
    k: usize,
    rows: usize,
    keep: &mut Vec<memmap2::Mmap>,
) -> Option<Lazy> {
    let f = lazy_files(k);
    let ro = |n: &str, keep: &mut Vec<memmap2::Mmap>| -> Option<&'static [u32]> {
        let mm = mmapvec::open(&dir.join(n)).ok()?;
        let s: &[u32] = mmapvec::as_slice(&mm[..]);
        let out = std::slice::from_raw_parts(s.as_ptr(), s.len());
        keep.push(mm);
        Some(out)
    };
    let rw = |n: &str| -> Option<memmap2::MmapMut> {
        let file = std::fs::OpenOptions::new().read(true).write(true).open(dir.join(n)).ok()?;
        memmap2::MmapOptions::new().map_mut(&file).ok()
    };
    let vhead = ro(&f[0], keep)?;
    let vlist = ro(&f[1], keep)?;
    let mut use_ct: *mut u32 = std::ptr::null_mut();
    let mut use_map: Option<memmap2::MmapMut> = None;
    if let Some(mut m) = rw(&use_file(k)) {
        if m.len() >= vlist.len() * 4 {
            use_ct = m.as_mut_ptr() as *mut u32;
            use_map = Some(m);
        }
    }
    let mut data = rw(&f[2])?;
    let mut have = rw(&f[3])?;
    let (dp, hp) = (data.as_mut_ptr(), have.as_mut_ptr());
    let mut maps = vec![data, have];
    if let Some(m) = use_map {
        maps.push(m);
    }
    Some(Lazy { data: dp, have: hp, rows, vhead, vlist, use_ct, _maps: maps })
}

/// How many levels a dataset could ever carry.
///
/// Was 8, on the assumption that each rung is coarser than the last by a large
/// factor so the ladder stays short. That assumption is what made the planet's
/// third rung merge 47 regions at once and end up with 1657-gate cliques. A
/// tall ladder of small aggregations is the cheaper shape: the rung above pays
/// for *edges*, and a table makes a region a complete graph on its gates, so
/// what matters is keeping the gate count per region low — not keeping the rung
/// count low. The shed rule ends the ladder when it stops paying; this is only
/// the ceiling that stops it running away.
pub const MAX_LEVELS: usize = 24;

pub struct Overlay {
    _maps: Vec<memmap2::Mmap>,
    levels: Vec<Level>,
    pub bounds: &'static [(u32, u32)],
}

/// What recomputing a row found.
///
/// The distinction between `Same` and `Changed` is the whole point. An update
/// that recomputes a region and finds every number where it was has learned
/// something worth acting on: nothing above that region can have moved because
/// of it. A structural invalidation throws that away before anyone looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Row {
    /// The region has no gates, so there is no row.
    Empty,
    /// Was absent; computed and stored.
    Written,
    /// Was there, and recomputing gives the same numbers.
    Same,
    /// Was there, the numbers have moved, and the new ones are stored.
    Changed,
}

impl Overlay {
    /// The mappings behind the region tables. These are the largest arrays in
    /// the dataset and the ones a cache budget mostly governs.
    pub fn maps(&self) -> &[memmap2::Mmap] {
        &self._maps
    }

    pub fn open(dir: &Path) -> io::Result<Overlay> {
        let mut keep = Vec::new();
        unsafe {
            let u32s = |n: &str, keep: &mut Vec<memmap2::Mmap>| -> Option<&'static [u32]> {
                let mm = mmapvec::open(&dir.join(n)).ok()?;
                let s: &[u32] = mmapvec::as_slice(&mm[..]);
                let out = std::slice::from_raw_parts(s.as_ptr(), s.len());
                keep.push(mm);
                Some(out)
            };
            let u64s = |n: &str, keep: &mut Vec<memmap2::Mmap>| -> Option<&'static [u64]> {
                let mm = mmapvec::open(&dir.join(n)).ok()?;
                let s: &[u64] = mmapvec::as_slice(&mm[..]);
                let out = std::slice::from_raw_parts(s.as_ptr(), s.len());
                keep.push(mm);
                Some(out)
            };
            let bytes = |n: &str, keep: &mut Vec<memmap2::Mmap>| -> Option<&'static [u8]> {
                let mm = mmapvec::open(&dir.join(n)).ok()?;
                let out = std::slice::from_raw_parts(mm.as_ptr(), mm.len());
                keep.push(mm);
                Some(out)
            };

            let mut levels: Vec<Level> = Vec::new();
            // Levels stop at the first one that is absent: a ladder with a gap
            // in it would let the query climb past a level it cannot descend.
            for k in 0..MAX_LEVELS {
                let f = level_files(k);
                let (Some(of), Some(bnd), Some(bhead), Some(head)) = (
                    u32s(&f[0], &mut keep),
                    u32s(&f[1], &mut keep),
                    u32s(&f[2], &mut keep),
                    u64s(&f[3], &mut keep),
                ) else {
                    break;
                };
                // A level stores its values one way or the other. Eager wins
                // when both are present, so a level can be filled in ahead of
                // time without the lazy files having to be removed first.
                let values = match (bytes(&f[4], &mut keep), bytes(&f[5], &mut keep)) {
                    (Some(mat), Some(width)) => Values::Eager { mat, width },
                    _ => {
                        // Checked before the first row is read, and hard rather
                        // than a fallback to the level below. A mis-laid-out
                        // table does not fail: it returns another region's
                        // numbers, which are valid costs, so the search settles
                        // and answers. Degrading quietly to a shorter ladder
                        // would hide exactly what this is here to surface.
                        if let Err(e) = verify_lazy_layout(dir, k, bhead, head, bnd.len()) {
                            return Err(io::Error::new(io::ErrorKind::InvalidData, e));
                        }
                        match open_lazy(dir, k, bnd.len(), &mut keep) {
                            Some(z) => Values::Lazy(z),
                            None => break,
                        }
                    }
                };
                let geo = match mmapvec::open(&dir.join(geo_file(k))) {
                    Ok(mm) => {
                        let g: &[(i32, i32, i32, i32)] = mmapvec::as_slice(&mm[..]);
                        let out = std::slice::from_raw_parts(g.as_ptr(), g.len());
                        keep.push(mm);
                        out
                    }
                    Err(_) => &[],
                };
                levels.push(Level { of, bnd, bhead, head, geo, values });
            }
            if levels.is_empty() {
                return Err(io::Error::new(io::ErrorKind::NotFound, "no overlay level 0"));
            }
            let bounds = {
                let mm = mmapvec::open(&dir.join("ov.bounds"))?;
                let s: &[(u32, u32)] = mmapvec::as_slice(&mm[..]);
                let out = std::slice::from_raw_parts(s.as_ptr(), s.len());
                keep.push(mm);
                out
            };
            Ok(Overlay { _maps: keep, levels, bounds })
        }
    }

    /// How many levels this dataset carries. Always at least one.
    #[inline]
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    #[inline]
    pub fn level(&self, k: usize) -> &Level {
        &self.levels[k]
    }

    /// The vertex-to-region map of level 0.
    pub fn cell_of(&self) -> &'static [u32] {
        self.levels[0].of
    }

    pub fn cells(&self) -> usize {
        self.levels[0].regions()
    }

    /// The level-0 region a vertex belongs to.
    #[inline]
    pub fn cell(&self, v: u32) -> u32 {
        self.levels[0].of[v as usize]
    }

    /// The region of level `k` that a vertex belongs to.
    ///
    /// The partition is nested, so this is a walk up the ladder rather than a
    /// per-level map of every vertex: level 0 places the vertex, and each
    /// level above places the region below it.
    #[inline]
    pub fn cell_at(&self, k: usize, v: u32) -> u32 {
        let mut c = self.levels[0].of[v as usize];
        for l in 1..=k {
            c = self.levels[l].of[c as usize];
        }
        c
    }

    /// Boundary vertices of a level-0 region, ascending.
    #[inline]
    pub fn boundary(&self, c: u32) -> &[u32] {
        self.levels[0].boundary(c)
    }

    #[inline]
    pub fn bindex(&self, c: u32, v: u32) -> Option<usize> {
        self.levels[0].bindex(c, v)
    }

    #[inline(always)]
    pub fn cost(&self, c: u32, i: usize, j: usize) -> u32 {
        self.levels[0].cost(c, i, j)
    }

    pub fn boundary_total(&self) -> usize {
        self.levels[0].boundary_total()
    }

    /// Whether the ladder goes above level 0.
    #[inline]
    pub fn has_l2(&self) -> bool {
        self.levels.len() > 1
    }

    /// Make sure one row of a region's table is there, computing and keeping
    /// it if the level stores its values lazily. A no-op on an eager level.
    ///
    /// The row is one restricted Dijkstra inside the region — the same one the
    /// eager build runs, just for a single source instead of all of them. It
    /// recurses: filling a row at level `k` reads the level below, which may
    /// itself be lazy and fill its own rows first. The recursion is bounded by
    /// the height of the ladder.
    ///
    /// `fwd` false builds the row on the *reverse* graph, which is the column
    /// the backward search wants. Storing it as a row is the whole point: a
    /// column of a lazily-filled table would touch `b` separate rows.
    pub fn ensure_row(
        &self,
        ds: &Dataset,
        ovr: Option<&Overrides>,
        k: usize,
        c: u32,
        i: usize,
        fwd: bool,
    ) {
        self.row_op(ds, ovr, k, c, i, fwd, false);
    }

    /// Recompute a row that is already there and say whether its numbers moved.
    ///
    /// Writes the new values when they differ, so the row is correct either way
    /// — this repairs and reports in one pass rather than inviting a caller to
    /// forget the repair. An absent row is computed and reported as `Written`,
    /// which a cascade must treat as "assume it moved": there is nothing to
    /// compare against.
    pub fn recheck_row(
        &self,
        ds: &Dataset,
        ovr: Option<&Overrides>,
        k: usize,
        c: u32,
        i: usize,
        fwd: bool,
    ) -> Row {
        self.row_op(ds, ovr, k, c, i, fwd, true)
    }

    /// One Dijkstra, two destinations for its answer.
    ///
    /// `compare` is what separates `ensure_row` from `recheck_row`. Sharing the
    /// search matters: the warm path runs this 16 million times, so a second
    /// copy of it would drift, and a per-row buffer to compare through would
    /// cost an allocation where today there is none.
    #[allow(clippy::too_many_arguments)]
    fn row_op(
        &self,
        ds: &Dataset,
        ovr: Option<&Overrides>,
        k: usize,
        c: u32,
        i: usize,
        fwd: bool,
        compare: bool,
    ) -> Row {
        let lvl = &self.levels[k];
        let Values::Lazy(z) = &lvl.values else { return Row::Empty };
        let row = lvl.bhead[c as usize] as usize + i;
        let had = z.bit(row, fwd);
        if had && !compare {
            return Row::Same;
        }
        let bnd = lvl.boundary(c);
        let b = bnd.len();
        if b == 0 {
            z.set_bit(row, fwd);
            return Row::Empty;
        }
        let (vs, ve) = (z.vhead[c as usize] as usize, z.vhead[c as usize + 1] as usize);
        let members = &z.vlist[vs..ve];

        let n = members.len();
        let mut dist = vec![UNREACHABLE; n];
        let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        let local = |v: u32| members.binary_search(&v).ok();
        // Parents, only when the level gathers statistics.
        //
        // Counting *settles* measures how many searches reached a member —
        // which under full warming is one per row for every member alike, a
        // flat signal that says nothing. Betweenness is how many shortest
        // paths run *through* a member, and that needs the tree.
        let track = z.counts_use();
        let mut par: Vec<u32> = if track { vec![u32::MAX; n] } else { Vec::new() };
        if let Some(si) = local(bnd[i]) {
            dist[si] = 0;
            heap.push(Reverse((0u32, si as u32)));
        }
        while let Some(Reverse((d, li))) = heap.pop() {
            if d > dist[li as usize] {
                continue;
            }
            // A settled member is one this region's traffic goes through. The
            // count is the raw material a cut decision needs, and taking it
            // here — once per settle rather than once per relaxation — keeps
            // it out of the loop that already dominates.
            z.note_use(vs + li as usize);
            let u = members[li as usize];
            // Level 0 walks the road graph itself; every level above walks the
            // one below, using its table for everything interior to a region
            // and real edges only where one is left.
            let cb = if k == 0 { u32::MAX } else { self.cell_at(k - 1, u) };
            if k > 0 {
                let below = k - 1;
                if let Some(i2) = self.levels[below].bindex(cb, u) {
                    self.ensure_row(ds, ovr, below, cb, i2, fwd);
                    let bl = self.levels[below].boundary(cb);
                    for (j2, &w) in bl.iter().enumerate() {
                        if j2 == i2 {
                            continue;
                        }
                        let step = self.levels[below].cost_dir(cb, i2, j2, fwd);
                        if step == UNREACHABLE {
                            continue;
                        }
                        if let Some(vi) = local(w) {
                            let nd = d.saturating_add(step);
                            if nd < dist[vi] {
                                dist[vi] = nd;
                                if track {
                                    par[vi] = li;
                                }
                                heap.push(Reverse((nd, vi as u32)));
                            }
                        }
                    }
                }
            }
            // The edges leaving `u`, in whichever direction this row is being
            // built. Anything leaving *this* region falls outside `members`
            // and is dropped, which is exactly the restriction. Above level 0
            // an edge that stays inside the region below is skipped too: its
            // table already accounts for it.
            let (head, segs) = if fwd { (ds.head, ds.eseg) } else { (ds.rhead, ds.rseg) };
            let (a, e) = (head[u as usize] as usize, head[u as usize + 1] as usize);
            for (n, &sg) in segs[a..e].iter().enumerate() {
                let t = a + n;
                let v = if fwd { ds.target(u, t) } else { ds.rtarget(u, t) };
                if k > 0 && self.cell_at(k - 1, v) == cb {
                    continue;
                }
                let Some(vi) = local(v) else { continue };
                let sid = (sg & SEG_MASK) as usize;
                let Some(w) = edge_ds_ov(ds, ovr, sid) else { continue };
                let nd = d.saturating_add(w);
                if nd < dist[vi] {
                    dist[vi] = nd;
                    if track {
                        par[vi] = li;
                    }
                    heap.push(Reverse((nd, vi as u32)));
                }
            }
        }
        // Betweenness: walk back from every boundary vertex to the source and
        // count each member the path runs through. `b` walks per row, rather
        // than an increment per relaxation, so the cost sits outside the loop
        // where a binary search already dominates.
        if track {
            for &dst in bnd.iter() {
                let Some(x) = local(dst) else { continue };
                if dist[x] == UNREACHABLE {
                    continue;
                }
                let mut at = par[x];
                while at != u32::MAX {
                    z.note_use(vs + at as usize);
                    at = par[at as usize];
                }
            }
        }
        let p = z.at(lvl.head[c as usize], b, i, fwd);
        if !had {
            for (j, &dst) in bnd.iter().enumerate() {
                z.put(p, j, local(dst).map(|x| dist[x]).unwrap_or(UNREACHABLE));
            }
            z.set_bit(row, fwd);
            return Row::Written;
        }
        // The row was there. Compare first, and only write when something
        // moved — a write that changes nothing still dirties a page, and on a
        // planet that is gigabytes of needless writeback.
        let mut moved = false;
        for (j, &dst) in bnd.iter().enumerate() {
            let want = local(dst).map(|x| dist[x]).unwrap_or(UNREACHABLE);
            if z.get(p, j) != want {
                moved = true;
                break;
            }
        }
        if !moved {
            return Row::Same;
        }
        for (j, &dst) in bnd.iter().enumerate() {
            z.put(p, j, local(dst).map(|x| dist[x]).unwrap_or(UNREACHABLE));
        }
        Row::Changed
    }

    /// A hash of everything a level-0 region's rows are computed *from*.
    ///
    /// A row is a deterministic function of the region's internal graph: which
    /// members there are, which edges run between them, and what those edges
    /// cost. Same inputs, same row. So an update that finds this unchanged has
    /// proved the rows cannot have moved — without computing one.
    ///
    /// That is the difference between asking and answering. Recomputing a
    /// region to see whether it changed costs `b` restricted searches; hashing
    /// its inputs costs one pass over its edges, which on the planet's first
    /// rung is about 47 times less. Nine days of edits recomputed 3 056 304
    /// level-0 rows to discover that none of them moved.
    ///
    /// Only edges whose far end is also a member are folded in. An edge leaving
    /// the region is not part of what the row is computed from — the search
    /// ignores it — so including it would make the hash differ over changes the
    /// rows cannot feel, which is safe but wasteful.
    ///
    /// Level 0 only. Above it, a region's inputs are the tables below, and
    /// whether *those* moved is what the cascade already establishes.
    pub fn input_hash(&self, ds: &Dataset, ovr: Option<&Overrides>, c: u32) -> u64 {
        let Values::Lazy(z) = &self.levels[0].values else { return 0 };
        let (vs, ve) = (z.vhead[c as usize] as usize, z.vhead[c as usize + 1] as usize);
        let members = &z.vlist[vs..ve];
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut fold = |x: u64| {
            for b in x.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x1000_0000_01b3);
            }
        };
        fold(members.len() as u64);
        for &u in members {
            let (a, e) = (ds.head[u as usize] as usize, ds.head[u as usize + 1] as usize);
            for j in a..e {
                let v = ds.target(u, j);
                if members.binary_search(&v).is_err() {
                    continue;
                }
                let sid = (ds.eseg[j] & SEG_MASK) as usize;
                // A closed edge folds in as a value no duration can take, so
                // closing a road and making it infinitely slow are not the same
                // input.
                fold(v as u64);
                fold(edge_ds_ov(ds, ovr, sid).map(|w| w as u64).unwrap_or(u64::MAX));
            }
        }
        h
    }

    /// Whether a row is already there. Always true on an eager level.
    #[inline]
    pub fn row_present(&self, k: usize, c: u32, i: usize, fwd: bool) -> bool {
        match &self.levels[k].values {
            Values::Eager { .. } => true,
            Values::Lazy(z) => z.bit(self.levels[k].bhead[c as usize] as usize + i, fwd),
        }
    }

    /// Forget every row of a region, so the next query recomputes them.
    ///
    /// This is what an update costs at a lazy level: a few bit clears. It is
    /// safe while the dataset is being served — the bits are atomic, and a
    /// query that finds one cleared simply computes the row again from the
    /// graph, which by then is the updated one. Nothing has to be locked and
    /// no reader sees a half-written table.
    ///
    /// Returns how many rows were dropped. An eager level cannot be
    /// invalidated this way and reports zero: its values live in a packed
    /// table whose per-region width was chosen from values that are about to
    /// change, so it needs rebuilding rather than forgetting.
    pub fn invalidate(&self, k: usize, c: u32) -> usize {
        let Values::Lazy(z) = &self.levels[k].values else { return 0 };
        let (a, b) = (
            self.levels[k].bhead[c as usize] as usize,
            self.levels[k].bhead[c as usize + 1] as usize,
        );
        let mut n = 0;
        for row in a..b {
            for fwd in [true, false] {
                if z.bit(row, fwd) {
                    z.clear_bit(row, fwd);
                    n += 1;
                }
            }
        }
        n
    }

    /// Drop every cached row that a set of overrides could have changed.
    ///
    /// A row is computed from edge weights, so an override that changes a
    /// weight makes every row of that region wrong — and a cached row has no
    /// way to notice. Without this the tables keep summarising the graph as it
    /// was before the road closed, and the failure is silent: the route stays
    /// plausible, just shorter than any real journey.
    ///
    /// The partition is nested, so each affected segment reaches exactly one
    /// region per level — a walk up the ladder rather than a search. A handful
    /// of closures therefore costs a handful of regions.
    ///
    /// Returns the rows dropped. Eager levels report none: their values live
    /// in a packed table whose per-region width was chosen from the old
    /// numbers, so they have to be rebuilt rather than forgotten.
    pub fn invalidate_for(&self, ds: &Dataset, ovr: &Overrides) -> usize {
        let mut seen: Vec<std::collections::BTreeSet<u32>> = vec![Default::default(); self.levels()];
        for sid in ovr.segments() {
            if sid as usize >= ds.seg_u.len() {
                continue;
            }
            for v in [ds.seg_u[sid as usize], ds.seg_v[sid as usize]] {
                if v as usize >= ds.n_vertices() {
                    continue;
                }
                let mut c = self.cell(v);
                for (k, set) in seen.iter_mut().enumerate() {
                    if k > 0 {
                        c = self.levels[k].parent(c);
                    }
                    set.insert(c);
                }
            }
        }
        let mut n = 0;
        for (k, set) in seen.iter().enumerate() {
            for &c in set {
                n += self.invalidate(k, c);
            }
        }
        n
    }

    /// Whether any level fills its rows on demand.
    pub fn any_lazy(&self) -> bool {
        self.levels.iter().any(|l| l.is_lazy())
    }

    /// How many rows of a lazy level have been filled, out of how many there
    /// are. `None` on an eager level, which has them all by construction.
    pub fn filled(&self, k: usize) -> Option<(usize, usize)> {
        match &self.levels[k].values {
            Values::Lazy(z) => Some(z.filled()),
            Values::Eager { .. } => None,
        }
    }

    /// Cheapest and dearest crossing of a level-0 region, for deciding whether
    /// it can help before its table is read.
    #[inline]
    pub fn bounds_of(&self, c: u32) -> (u32, u32) {
        self.bounds[c as usize]
    }
}

// ----------------------------------------------------------------- query

use crate::cachecap::CacheCap;
use crate::router::{seg_entries, seg_exits, Snap};
use crate::searchstate::SearchState;

/// Two-level search state.
///
/// Distances live in hash maps rather than arrays indexed by vertex. On a
/// planet the array form is 4.3 GB — eight times the whole memory budget this
/// is aimed at — while the overlay search touches tens of thousands of
/// boundary vertices, which a map holds in a few megabytes.
pub struct OverlayRouter {
    /// Refuse rather than grow past this many bytes of search state.
    ///
    /// The state is the part a machine must actually have; mapped table pages
    /// are a cache the OS can evict. Making the limit explicit turns "the
    /// process died" into "this query needs a coarser overlay than you built",
    /// which is a thing an operator can act on.
    pub budget_bytes: usize,
    state: [SearchState; 2],
    /// Scratch for the searches inside one region.
    lstate: SearchState,
    pub settled: u64,
    pub regions_entered: u64,
    /// How many settles were answered from a level-2 table — that is, how much
    /// of the route was crossed as whole groups of regions rather than one
    /// boundary vertex at a time.
    pub regions2_entered: u64,
    /// Settles dropped by the geometric bound before doing any work for them.
    pub pruned: u64,
    /// Parallel prefetch batches, and the rows they filled.
    pub batches: u64,
    pub batch_rows: u64,
    /// Rows the sequential loop still had to fill itself.
    pub late_rows: u64,
    /// Set when the last search stopped because it would have exceeded
    /// [`OverlayRouter::budget_bytes`].
    pub over_budget: bool,
    /// Optional ceiling on resident memory, enforced by dropping mapped pages
    /// mid-search. Set it to find out what the query costs on a machine that
    /// cannot hold the tables, rather than on the one that built them.
    pub cache: Option<CacheCap>,
    /// Whether to use the second level when the dataset has one. Turning it
    /// off is how the two-level answer gets checked against the one-level
    /// answer, which must be identical.
    pub use_l2: bool,
}

impl OverlayRouter {
    /// Bytes the two frontiers and the region scratch currently hold — the
    /// figure a memory budget is stated against.
    ///
    /// This is *capacity*, not occupancy, and deliberately so: a reused router
    /// keeps the tables it grew, which is the memory the machine has to have.
    pub fn bytes(&self) -> usize {
        self.state[0].bytes() + self.state[1].bytes() + self.lstate.bytes()
    }

    /// Just the two frontiers — the part that scales with the search rather
    /// than with a region.
    ///
    /// Worth separating once there are two levels: the frontier reaches far
    /// fewer vertices, while the region-local scratch still has to hold a
    /// whole level-1 region, so the scratch becomes the larger half. A budget
    /// wants [`OverlayRouter::bytes`]; a question about whether the queue is
    /// holding duplicates wants this.
    pub fn frontier_bytes(&self) -> usize {
        self.state[0].bytes() + self.state[1].bytes()
    }

    /// Check the resident-cache budget, if one is set.
    ///
    /// Called on a stride rather than every settle: reading the resident set
    /// is a syscall, and a few thousand vertices cannot page in enough to
    /// overshoot a budget that is stated in megabytes.
    #[inline]
    fn check_cache(&mut self) {
        const STRIDE: u64 = 4096;
        if self.settled.is_multiple_of(STRIDE) {
            if let Some(c) = self.cache.as_mut() {
                c.enforce();
            }
        }
    }

    /// Vertices the two frontiers have reached — the quantity the state is
    /// supposed to scale with.
    pub fn reached(&self) -> usize {
        self.state[0].reached() + self.state[1].reached()
    }
}

impl Default for OverlayRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlayRouter {
    pub fn new() -> OverlayRouter {
        OverlayRouter {
            budget_bytes: usize::MAX,
            state: [SearchState::new(), SearchState::new()],
            lstate: SearchState::new(),
            settled: 0,
            regions_entered: 0,
            regions2_entered: 0,
            pruned: 0,
            batches: 0,
            batch_rows: 0,
            late_rows: 0,
            over_budget: false,
            cache: None,
            use_l2: true,
        }
    }

    /// Dijkstra confined to one region, from `sources`, returning the cost to
    /// each of the region's boundary vertices.
    ///
    /// Membership is `cell_of[v] == c`, so no per-region vertex list has to be
    /// stored — on a planet that list would be another 1.08 GB for no
    /// information the partition does not already carry. Distances go in a
    /// map because a region holds a few thousand vertices, not 270 million.
    #[allow(clippy::too_many_arguments)] // a region-local search needs the region
    fn local(
        &mut self,
        ds: &Dataset,
        ovr: Option<&Overrides>,
        ov: &Overlay,
        c: u32,
        sources: &[(u32, u32)],
        fwd: bool,
        out: &mut Vec<(u32, u32)>,
    ) {
        self.lstate.clear();
        for &(v, c0) in sources {
            if ov.cell(v) == c {
                self.lstate.relax(v, c0, u32::MAX);
            }
        }
        while let Some((u, d)) = self.lstate.pop() {
            let (head, a_dir) = if fwd { (ds.head, true) } else { (ds.rhead, false) };
            let (a, e) = (head[u as usize] as usize, head[u as usize + 1] as usize);
            for k in a..e {
                let v = if a_dir { ds.target(u, k) } else { ds.rtarget(u, k) };
                if ov.cell(v) != c {
                    continue;
                }
                let segs = if a_dir { ds.eseg } else { ds.rseg };
                let sid = (segs[k] & SEG_MASK) as usize;
                if let Some(w) = edge_ds_ov(ds, ovr, sid) {
                    self.lstate.relax(v, d.saturating_add(w), u);
                }
            }
        }
        out.clear();
        for &b in ov.boundary(c) {
            if let Some(d) = self.lstate.dist_of(b) {
                out.push((b, d));
            }
        }
    }

    /// Cheapest overlay cost between two vertices, with no snapping involved.
    ///
    /// Exists so correctness can be checked against a plain Dijkstra without
    /// the partial-end-segment accounting in the way: any disagreement here is
    /// the overlay's, not the bookkeeping's.
    pub fn search_vertices(
        &mut self,
        ds: &Dataset,
        ov: &Overlay,
        s: u32,
        t: u32,
    ) -> Option<u32> {
        self.search_vertices_with(ds, None, ov, s, t)
    }

    /// As `search_vertices`, with local overrides applied.
    pub fn search_vertices_with(
        &mut self,
        ds: &Dataset,
        ovr: Option<&Overrides>,
        ov: &Overlay,
        s: u32,
        t: u32,
    ) -> Option<u32> {
        self.seed_and_run(ds, ovr, ov, &[(s, 0)], &[(t, 0)]).map(|(c, _)| c)
    }

    /// Cost of the cheapest overlay route, or `None` when there is none.
    ///
    /// Correctness rests on one observation: a shortest path leaving its
    /// region must cross a boundary vertex, and the prefix up to that first
    /// crossing stays inside. So seeding the overlay from the region-local
    /// costs to every boundary vertex loses nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &mut self,
        ds: &Dataset,
        ov: &Overlay,
        from: &Snap,
        to: &Snap,
    ) -> Option<(u32, Vec<u32>)> {
        self.search_with(ds, None, ov, from, to)
    }

    /// As `search`, with local overrides applied — to the region tables as
    /// well as to the edges, which is the whole point.
    #[allow(clippy::too_many_arguments)]
    pub fn search_with(
        &mut self,
        ds: &Dataset,
        ovr: Option<&Overrides>,
        ov: &Overlay,
        from: &Snap,
        to: &Snap,
    ) -> Option<(u32, Vec<u32>)> {
        let entries: Vec<(u32, u32)> = seg_exits(ds, from, ovr)
            .into_iter()
            .map(|(v, cm)| (v, dur_ds(cm as u32, attr_kmh(ds.attr(from.seg as usize)))))
            .collect();
        let exits: Vec<(u32, u32)> = seg_entries(ds, to, ovr)
            .into_iter()
            .map(|(v, cm)| (v, dur_ds(cm as u32, attr_kmh(ds.attr(to.seg as usize)))))
            .collect();
        self.seed_and_run(ds, ovr, ov, &entries, &exits)
    }

    fn seed_and_run(
        &mut self,
        ds: &Dataset,
        ovr: Option<&Overrides>,
        ov: &Overlay,
        entries: &[(u32, u32)],
        exits: &[(u32, u32)],
    ) -> Option<(u32, Vec<u32>)> {
        self.state[0].clear();
        self.state[1].clear();
        self.settled = 0;
        self.regions_entered = 0;
        self.regions2_entered = 0;
        self.pruned = 0;
        self.batches = 0;
        self.batch_rows = 0;
        self.late_rows = 0;
        self.over_budget = false;
        // Where each side is heading, and the fastest anything can travel. A
        // settled vertex whose straight-line distance to its own target already
        // costs more than the best path found cannot be on a better one, so
        // everything it would relax — including a lazy row worth milliseconds —
        // can be skipped. Sound because the bound never overstates.
        let vmax = ds.max_kmh();
        let aim = [
            exits.first().map(|&(v, _)| ds.vcoord[v as usize]),
            entries.first().map(|&(v, _)| ds.vcoord[v as usize]),
        ];

        let any_lazy = ov.any_lazy();
        let (pf_depth, pf_min) = (prefetch_depth(), prefetch_min());
        let mut ahead: Vec<u32> = Vec::new();
        let mut since_scan = usize::MAX; // scan on the first settle
        let mut scan_every = 1usize;
        let mut want: Vec<(usize, u32, usize)> = Vec::new();

        // Goal direction, when the dataset and the query allow it.
        //
        // A plain Dijkstra grows a ball; A* grows an ellipse toward the
        // target. The potential has to be *consistent*, not merely a valid
        // lower bound — `π(u) ≤ w(u,v) + π(v)` for every edge — or the search
        // settles vertices out of order and the answer is quietly short. The
        // chord gives that: `chord(u,t) ≤ chord(u,v) + chord(v,t)` by the
        // triangle inequality, `chord(u,v) ≤ len(u,v)` because a tunnel beats
        // a road, and `len/vmax ≤ len/speed` because `vmax` is the fastest
        // anything goes. It holds for table hops too, since a table entry is a
        // real shortest path inside its region.
        //
        // Bidirectionally the two halves must agree, or the meeting rule stops
        // meaning anything. The symmetric form does that: `p_f = (π_t − π_s +
        // C)/2` and `p_b = (π_s − π_t + C)/2`, so `p_f + p_b = C` everywhere,
        // and termination becomes `top_f + top_b ≥ best + C`. `C` exists to
        // keep both non-negative in `u32`, and `chord(s,t)/vmax` is large
        // enough by the triangle inequality.
        let astar = std::env::var("MPEE_ASTAR").is_ok() && aim[0].is_some() && aim[1].is_some();
        let cc: u32 = if astar {
            geo_floor_between(aim[0].unwrap(), aim[1].unwrap(), vmax)
        } else {
            0
        };
        // Both targets converted once, here, rather than inside the potential.
        // This is the whole point of storing `vxyz`: the loop that runs
        // hundreds of thousands of times must not contain a `sin`.
        let axyz = [
            aim[0].map(|(a, o)| xyz_of(a, o)).unwrap_or([0.0; 3]),
            aim[1].map(|(a, o)| xyz_of(a, o)).unwrap_or([0.0; 3]),
        ];
        let pot = |v: u32, side: usize| -> u32 {
            if !astar {
                return 0;
            }
            let to_own = geo_floor_xyz(ds, v, axyz[side], vmax);
            let to_other = geo_floor_xyz(ds, v, axyz[1 - side], vmax);
            (to_own + cc).saturating_sub(to_other) / 2
        };

        let mut out: Vec<(u32, u32)> = Vec::new();
        for (side, srcs, fwd) in [(0usize, entries, true), (1, exits, false)] {
            for &(v, c0) in srcs {
                let c = ov.cell(v);
                self.local(ds, ovr, ov, c, &[(v, c0)], fwd, &mut out);
                self.regions_entered += 1;
                for &(b, d) in out.iter() {
                    self.state[side].relax_pot(b, d, u32::MAX, || pot(b, side));
                }
            }
        }
        if self.state[0].peek().is_none() || self.state[1].peek().is_none() {
            return None;
        }
        // The level-2 regions holding an endpoint. *Both* searches stay at
        // level 1 inside either of them: the forward search still has to reach
        // t, and the backward one s, and that needs the detail a level-2 table
        // has collapsed away. There are only ever a handful, so a linear scan
        // beats anything with a hash.
        // MPEE_MAX_LEVEL caps how high the ladder is climbed. A rung is
        // expensive to warm and expensive to store, and whether it earns that
        // is a question about a particular dataset — this answers it without
        // rebuilding anything.
        let cap: Option<usize> =
            std::env::var("MPEE_MAX_LEVEL").ok().and_then(|v| v.parse().ok());
        let nlev = if self.use_l2 { ov.levels() } else { 1 };
        let nlev = match cap {
            Some(n) => nlev.min(n.max(1)),
            None => nlev,
        };
        let mut home: Vec<Vec<u32>> = vec![Vec::new(); nlev];
        for &(v, _) in entries.iter().chain(exits.iter()) {
            let mut c = ov.cell(v);
            for (k, h) in home.iter_mut().enumerate() {
                if k > 0 {
                    c = ov.level(k).parent(c);
                }
                if !h.contains(&c) {
                    h.push(c);
                }
            }
        }

        let mut best = UNREACHABLE;
        let mut meet = u32::MAX;
        loop {
            let tf = self.state[0].peek().unwrap_or(UNREACHABLE);
            let tb = self.state[1].peek().unwrap_or(UNREACHABLE);
            if tf == UNREACHABLE && tb == UNREACHABLE {
                break;
            }
            if tf.saturating_add(tb) >= best.saturating_add(cc) {
                break;
            }
            let side = if tf <= tb { 0usize } else { 1 };
            // Fill the rows this side is about to need, in parallel.
            //
            // The search itself is sequential — it cannot settle a vertex
            // before it has that vertex's row — but the queue already holds
            // the vertices it will settle next, so their rows can be computed
            // ahead of it. Each row is milliseconds of Dijkstra; paying for a
            // batch of them at once is the difference between one core and all
            // of them. Rows are idempotent and their presence bits atomic, so
            // two threads racing on the same row is waste, never damage.
            // The scan walks the ladder for every candidate, so doing it on
            // each settle costs more than the fills it schedules. How often it
            // is worth doing depends on how cold the cache is, and that
            // changes *during* the search as well as between runs: the more
            // rows are already there, the less a scan finds and the less it is
            // worth paying for. So the interval follows the hit rate — it
            // halves whenever a scan finds a full batch and doubles when it
            // does not, which makes a warm cache cost almost nothing to scan
            // and a cold one get scanned hard.
            if any_lazy && since_scan >= scan_every {
                since_scan = 0;
                self.state[side].peek_many(pf_depth, &mut ahead);
                want.clear();
                for &v in ahead.iter() {
                    let (lv, lc, li) = choose_level(ov, &home, v);
                    if let Some(i) = li {
                        if !ov.row_present(lv, lc, i, side == 0) {
                            want.push((lv, lc, i));
                        }
                    }
                }
                // Below a handful there is nothing for the threads to share.
                if want.len() >= pf_min {
                    let fwd = side == 0;
                    self.batches += 1;
                    self.batch_rows += want.len() as u64;
                    want.par_iter().for_each(|&(lv, lc, i)| ov.ensure_row(ds, ovr, lv, lc, i, fwd));
                    scan_every = (scan_every / 2).max(1);
                } else {
                    scan_every = (scan_every * 2).min(pf_depth);
                }
            }
            since_scan += 1;
            let Some((u, d)) = self.state[side].pop() else { break };
            self.settled += 1;
            // Checked on the settle, not on every relaxation: once per popped
            // vertex is often enough to stop, and costs nothing in the inner
            // loop.
            if self.bytes() > self.budget_bytes {
                self.over_budget = true;
                return None;
            }
            self.check_cache();
            if best != UNREACHABLE
                && aim[side].is_some()
                && d.saturating_add(geo_floor_xyz(ds, u, axyz[side], vmax)) >= best
            {
                self.pruned += 1;
                continue;
            }
            let (lv, lc, li) = choose_level(ov, &home, u);
            let (head, fwd) = if side == 0 { (ds.head, true) } else { (ds.rhead, false) };
            let (a, e) = (head[u as usize] as usize, head[u as usize + 1] as usize);
            if lv > 0 {
                self.regions2_entered += 1;
            }
            // 1. across the chosen region, boundary to boundary, from its table.
            if let Some(i) = li {
                if any_lazy && !ov.row_present(lv, lc, i, side == 0) {
                    self.late_rows += 1;
                }
                // The table is directional: the forward search wants row i,
                // the backward one column i. On a lazy level the two are
                // separate rows, computed on their own side's graph — which is
                // why this asks for a direction rather than transposing.
                let fwd_side = side == 0;
                ov.ensure_row(ds, ovr, lv, lc, i, fwd_side);
                let lvl = ov.level(lv);
                let bl = lvl.boundary(lc);
                for (j, &w) in bl.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    let step = lvl.cost_dir(lc, i, j, fwd_side);
                    if step == UNREACHABLE {
                        continue;
                    }
                    let nd = d.saturating_add(step);
                    if nd < best {
                        self.state[side].relax_pot(w, nd, u, || pot(w, side));
                    }
                }
            }
            // 2. the edges that leave it. Anything staying inside the region
            //    the table above has already accounted for.
            for k in a..e {
                let v = if fwd { ds.target(u, k) } else { ds.rtarget(u, k) };
                if ov.cell_at(lv, v) == lc {
                    continue;
                }
                let segs = if fwd { ds.eseg } else { ds.rseg };
                let sid = (segs[k] & SEG_MASK) as usize;
                let Some(w) = edge_ds_ov(ds, ovr, sid) else { continue };
                let nd = d.saturating_add(w);
                if nd < best {
                    self.state[side].relax_pot(v, nd, u, || pot(v, side));
                }
            }
            // Did the two frontiers meet here?
            if let (Some(x), Some(y)) = (self.state[0].dist_of(u), self.state[1].dist_of(u)) {
                if x.saturating_add(y) < best {
                    best = x + y;
                    meet = u;
                }
            }
        }
        if meet == u32::MAX {
            return None;
        }
        let mut fwd_path = vec![meet];
        let mut v = meet;
        while let Some(p) = self.state[0].parent_of(v) {
            fwd_path.push(p);
            v = p;
        }
        fwd_path.reverse();
        let mut v = meet;
        while let Some(p) = self.state[1].parent_of(v) {
            fwd_path.push(p);
            v = p;
        }
        Some((best, fwd_path))
    }
}

/// Plain forward Dijkstra, as the yardstick the overlay is measured against.
///
/// Deliberately the simplest correct implementation: full arrays, no
/// bidirectional bound, nothing shared with the code under test.
pub fn reference_cost(ds: &Dataset, s: u32, t: u32) -> (Option<u32>, u64) {
    reference_cost_with(ds, None, s, t)
}

/// The same yardstick, with local overrides applied — so an overlay answer
/// that respects an override can be checked against a plain search that does.
pub fn reference_cost_with(
    ds: &Dataset,
    ovr: Option<&Overrides>,
    s: u32,
    t: u32,
) -> (Option<u32>, u64) {
    let mut dist = vec![UNREACHABLE; ds.n_vertices()];
    let mut heap = BinaryHeap::new();
    let mut settled = 0u64;
    dist[s as usize] = 0;
    heap.push(Reverse((0u32, s)));
    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u as usize] {
            continue;
        }
        settled += 1;
        if u == t {
            return (Some(d), settled);
        }
        let (a, e) = (ds.head[u as usize] as usize, ds.head[u as usize + 1] as usize);
        for k in a..e {
            let v = ds.target(u, k);
            let sid = (ds.eseg[k] & SEG_MASK) as usize;
            let Some(w) = edge_ds_ov(ds, ovr, sid) else { continue };
            let nd = d.saturating_add(w);
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                heap.push(Reverse((nd, v)));
            }
        }
    }
    (None, settled)
}
// ---------------------------------------------------------------------------
// The ladder above level 0
// ---------------------------------------------------------------------------

/// Road vertices a level-1 region aims to cover, and the factor between rungs.
///
/// Measured rather than guessed. A first attempt fitted a straight line
/// through two points and concluded that coarser rungs do not pay; a third
/// point showed the exponent is not constant — the boundary falls *faster* the
/// coarser the level gets (`A^-0.111` from 215 to 16 384 vertices, `A^-0.396`
/// from there to 131 072). The query agreed: on Lisboa → Warszawa under a
/// 64 MB cache, a 4x rung settled 570 843 vertices in 30.5 s where a 32x rung
/// settled 261 605 in 16.9 s. So the rungs are wide.
pub const TARGET2: usize = TARGET * 32;
pub const STEP: usize = 32;

/// How far down the queue to look for rows worth filling ahead of the search,
/// and how many misses are worth starting threads for.
///
/// Both matter more than they look. The queue's head moves by one vertex per
/// settle, so a shallow look finds almost everything already filled and starts
/// a thread pool for a handful of rows — measured at 218 batches averaging ten
/// rows, which is mostly launch overhead. Looking deep and waiting for a real
/// batch turns the same total work into a few large parallel fills.
fn prefetch_depth() -> usize {
    std::env::var("MPEE_PREFETCH").ok().and_then(|v| v.parse().ok()).unwrap_or(4096)
}
fn prefetch_min() -> usize {
    std::env::var("MPEE_PREFETCH_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(64)
}

pub struct L2Stats {
    pub cells: usize,
    pub boundary: usize,
    pub entries: u64,
    pub secs: f64,
    /// Estimated seconds to fill every row of this level. See [`fill_estimate`].
    pub fill_secs: f64,
    /// Set when the level was shaped but not written, because filling it would
    /// have cost more than the caller was willing to spend.
    pub refused: bool,
}

/// A duration a person can read at a glance, across nine orders of magnitude.
fn human_secs(s: f64) -> String {
    if s < 1.0 {
        format!("{:.0} ms", s * 1000.0)
    } else if s < 120.0 {
        format!("{s:.1} s")
    } else if s < 7200.0 {
        format!("{:.1} min", s / 60.0)
    } else {
        format!("{:.1} h", s / 3600.0)
    }
}

/// Roughly how long filling a level's rows will take, in seconds.
///
/// A rung is cheap to *shape* and can be ruinous to *fill*, and the two are
/// not related — so a stopping rule that only asks how small the top gets will
/// happily build a level nobody can afford. This is what that rule was missing.
///
/// One row is a Dijkstra over the region's members, and each settled member
/// relaxes about as many neighbours as the level below has boundary vertices
/// per region. So the work is `Σ_regions (rows × members) × b_below`, and the
/// constant is measured rather than derived — on the planet, warming levels 1,
/// 2 and 3 came to 32, 78 and 95 million of these operations per second, so
/// 85 million is a fair middle.
///
/// Checked against the three levels it was calibrated on: level 2 predicted
/// 26 hours against 28 measured, level 3 predicted 106 against 190. It is an
/// order-of-magnitude instrument, which is all a refusal needs to be.
pub fn fill_estimate(bhead: &[u32], vhead: &[u32], b_below: f64) -> f64 {
    const OPS_PER_SEC: f64 = 85e6;
    let mut work = 0f64;
    for c in 0..bhead.len().saturating_sub(1) {
        let b = (bhead[c + 1] - bhead[c]) as f64;
        let m = (vhead[c + 1] - vhead[c]) as f64;
        work += b * m;
    }
    work * b_below.max(1.0) / OPS_PER_SEC
}

/// Fill one region's table: shortest time between each ordered pair of its
/// boundary vertices, using only paths that stay inside the region.
///
/// This is the unit of work the whole overlay is made of, at every level above
/// zero. It depends on nothing outside the region — not on its neighbours, not
/// on the rest of the level — which is what makes the build embarrassingly
/// parallel, and what will let a table be computed on demand and cached rather
/// than computed in full up front. The N×N of one region is the natural slice.
///
/// `members` is the region's vertex set, ascending: the boundary vertices of
/// the level below that fall inside it. `bnd` is the subset of those that this
/// level treats as boundary, also ascending. `out` is resized to `bnd² ` and
/// filled row by row, row `i` holding the costs *from* `bnd[i]`.
fn region_table(
    ds: &Dataset,
    ovr: Option<&Overrides>,
    ov: &Overlay,
    below: usize,
    members: &[u32],
    bnd: &[u32],
    out: &mut Vec<u32>,
) {
    let b = bnd.len();
    out.clear();
    out.resize(b * b, UNREACHABLE);
    let local = |v: u32| members.binary_search(&v).ok();
    let n = members.len();
    let mut dist = vec![UNREACHABLE; n];
    let mut touched: Vec<u32> = Vec::new();
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    for (i, &src) in bnd.iter().enumerate() {
        for &t in &touched {
            dist[t as usize] = UNREACHABLE;
        }
        touched.clear();
        heap.clear();
        let Some(si) = local(src) else { continue };
        dist[si] = 0;
        touched.push(si as u32);
        heap.push(Reverse((0u32, si as u32)));
        while let Some(Reverse((d, li))) = heap.pop() {
            if d > dist[li as usize] {
                continue;
            }
            let u = members[li as usize];
            let cb = ov.cell_at(below, u);
            let relax = |v: u32,
                         nd: u32,
                         dist: &mut Vec<u32>,
                         touched: &mut Vec<u32>,
                         heap: &mut BinaryHeap<Reverse<(u32, u32)>>| {
                if let Some(vi) = local(v) {
                    if nd < dist[vi] {
                        if dist[vi] == UNREACHABLE {
                            touched.push(vi as u32);
                        }
                        dist[vi] = nd;
                        heap.push(Reverse((nd, vi as u32)));
                    }
                }
            };
            // 1. across `u`'s region on the level below, from its table. Every
            //    boundary vertex of that region is inside this one too,
            //    because the partition is nested.
            if let Some(i1) = ov.level(below).bindex(cb, u) {
                let bl = ov.level(below).boundary(cb);
                for (j, &w) in bl.iter().enumerate() {
                    if j == i1 {
                        continue;
                    }
                    let step = ov.level(below).cost(cb, i1, j);
                    if step == UNREACHABLE {
                        continue;
                    }
                    relax(w, d.saturating_add(step), &mut dist, &mut touched, &mut heap);
                }
            }
            // 2. the cut edges leaving it. Those that also leave *this* region
            //    land outside `members` and are dropped by `local`, which is
            //    exactly the restriction.
            let (a, e) = (ds.head[u as usize] as usize, ds.head[u as usize + 1] as usize);
            for j in a..e {
                let v = ds.target(u, j);
                if ov.cell_at(below, v) == cb {
                    continue;
                }
                let sid = (ds.eseg[j] & SEG_MASK) as usize;
                let Some(w) = edge_ds_ov(ds, ovr, sid) else { continue };
                let nd = d.saturating_add(w);
                relax(v, nd, &mut dist, &mut touched, &mut heap);
            }
        }
        for (j, &dst) in bnd.iter().enumerate() {
            out[i * b + j] = local(dst).map(|x| dist[x]).unwrap_or(UNREACHABLE);
        }
    }
}

/// A level's shape, without any of its numbers.
///
/// This is the metric-independent half of the overlay, and the cheap half: on
/// the planet it is seconds where filling the tables is twenty minutes. It
/// depends only on the road network's topology — which regions there are,
/// which vertices are on their boundaries, how big each table has to be — and
/// not on how long anything takes to drive. Change the speeds, the vehicle, or
/// close a road, and this survives untouched while only the values are redone.
pub struct LevelStructure {
    pub lof: Vec<u32>,
    /// Region -> its vertex set, the level below's boundary vertices inside it.
    /// Persisted, because computing a table on demand needs it.
    pub vhead: Vec<u32>,
    pub vlist: Vec<u32>,
    pub bhead: Vec<u32>,
    pub blist: Vec<u32>,
    /// Entry offset of each region's table, in u32s.
    pub ohead: Vec<u64>,
    pub ncells: usize,
    pub entries: u64,
}

/// Everything the gate-driven growth reads about how gates face regions.
///
/// Built once per rung. The region graph says two regions touch; this says
/// through which gates, which is what lets a growing cell be told exactly how
/// many gates it would gain or bury by taking a neighbour — before it takes it.
struct GateIndex {
    /// Region -> its gates' positions, as `bstart[c]..bstart[c + 1]`.
    bstart: Vec<u32>,
    /// Gate position -> the regions it faces.
    nhead: Vec<u32>,
    nlist: Vec<u32>,
    /// Region -> the gates of *other* regions that face it.
    ihead: Vec<u32>,
    ilist: Vec<u32>,
    /// Gate position -> the region it belongs to.
    owner: Vec<u32>,
}

impl GateIndex {
    /// Gates the cell would gain by taking `d`, less the gates it would bury.
    ///
    /// Exact, and evaluated without committing, because the answer decides
    /// whether to commit at all. `outside[p]` is how many regions gate `p`
    /// still faces from outside the cell; a gate stops being a gate the moment
    /// that reaches zero.
    fn delta(&self, d: u32, lof: &[u32], id: u32, outside: &[u32]) -> i64 {
        let mut add = 0i64;
        for p in self.bstart[d as usize]..self.bstart[d as usize + 1] {
            let (a, b) = (self.nhead[p as usize] as usize, self.nhead[p as usize + 1] as usize);
            // `r != d` reads d as already inside, which is the case being asked
            // about. Its own gates that face only the cell and itself vanish.
            if self.nlist[a..b].iter().any(|&r| r != d && lof[r as usize] != id) {
                add += 1;
            }
        }
        let mut sub = 0i64;
        let (a, b) = (self.ihead[d as usize] as usize, self.ihead[d as usize + 1] as usize);
        for &p in &self.ilist[a..b] {
            if lof[self.owner[p as usize] as usize] == id && outside[p as usize] == 1 {
                sub += 1;
            }
        }
        add - sub
    }

    /// Take `d` into the cell, and report the change in its gate count.
    ///
    /// The burial pass has to run before `d` is marked as a member, or `d`'s own
    /// gates — which have no counter yet — would be mistaken for members whose
    /// counter needs decrementing.
    fn absorb(
        &self,
        d: u32,
        lof: &mut [u32],
        id: u32,
        outside: &mut [u32],
        touched: &mut Vec<u32>,
    ) -> i64 {
        let mut delta = 0i64;
        let (a, b) = (self.ihead[d as usize] as usize, self.ihead[d as usize + 1] as usize);
        for &p in &self.ilist[a..b] {
            if lof[self.owner[p as usize] as usize] == id {
                outside[p as usize] -= 1;
                if outside[p as usize] == 0 {
                    delta -= 1;
                }
            }
        }
        lof[d as usize] = id;
        for p in self.bstart[d as usize]..self.bstart[d as usize + 1] {
            let (a, b) = (self.nhead[p as usize] as usize, self.nhead[p as usize + 1] as usize);
            let out =
                self.nlist[a..b].iter().filter(|&&r| lof[r as usize] != id).count() as u32;
            outside[p as usize] = out;
            touched.push(p);
            if out > 0 {
                delta += 1;
            }
        }
        delta
    }

    /// How attractive `d` looks from the cell: how many of its gates already
    /// face inward, and how much traffic those carry. An ordering heuristic,
    /// not a decision — `delta` decides.
    fn score(&self, d: u32, lof: &[u32], id: u32, traffic: &Traffic, bnd: &[u32]) -> (u32, u64) {
        let (mut gain, mut busy) = (0u32, 0u64);
        for p in self.bstart[d as usize]..self.bstart[d as usize + 1] {
            let (a, b) = (self.nhead[p as usize] as usize, self.nhead[p as usize + 1] as usize);
            if self.nlist[a..b].iter().any(|&r| lof[r as usize] == id) {
                gain += 1;
                busy += traffic.get(bnd[p as usize]) as u64;
            }
        }
        (gain, busy)
    }
}

/// How often each road vertex has been settled, keyed by vertex id.
///
/// The use counters a level gathers are indexed by position in that level's
/// `vlist`, which is a property of the partition: rebuild the level and the
/// index means something else. Traffic is a property of the *road*, so it is
/// stored against the vertex and survives any repartition — which is the whole
/// point, because the cut that wants to know where the traffic runs is made
/// while the level is being rebuilt.
///
/// Sparse on purpose. Only 16 % of the planet's level-0 gates have ever been
/// settled, so the pairs are 14 MB where a dense array over every vertex would
/// be 1.08 GB to say almost nothing.
pub struct Traffic {
    pairs: &'static [(u32, u32)],
    _map: Option<memmap2::Mmap>,
}

impl Traffic {
    /// Empty when the dataset has no `traffic.bin`. Every caller degrades to
    /// its own second-best rule rather than failing, because a dataset that has
    /// never been queried genuinely has nothing to say about traffic.
    pub fn open(dir: &Path) -> Traffic {
        match mmapvec::open(&dir.join("traffic.bin")) {
            Ok(mm) => unsafe {
                let s: &[(u32, u32)] = mmapvec::as_slice(&mm[..]);
                let pairs = std::slice::from_raw_parts(s.as_ptr(), s.len());
                Traffic { pairs, _map: Some(mm) }
            },
            Err(_) => Traffic { pairs: &[], _map: None },
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    #[inline]
    pub fn get(&self, v: u32) -> u32 {
        match self.pairs.binary_search_by_key(&v, |&(a, _)| a) {
            Ok(i) => self.pairs[i].1,
            Err(_) => 0,
        }
    }
}

/// Fold every level's use counters into one vertex-keyed table.
///
/// Summed across levels rather than kept apart. A vertex that carries traffic
/// on the first rung and again on the third is busy twice over, and the cut
/// that has to avoid it does not care which rung noticed.
pub fn write_traffic(dir: &Path, ov: &Overlay) -> io::Result<(usize, u32)> {
    let mut acc: Vec<(u32, u32)> = Vec::new();
    for k in 0..ov.levels() {
        let Values::Lazy(z) = &ov.level(k).values else { continue };
        if !z.counts_use() {
            continue;
        }
        let (vl, ct) = (z.vlist_all(), z.use_ct_all());
        for (i, &c) in ct.iter().enumerate() {
            if c > 0 {
                acc.push((vl[i], c));
            }
        }
    }
    acc.sort_unstable();
    // Same vertex on several rungs: add, keep one row.
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(acc.len());
    for (v, c) in acc {
        match out.last_mut() {
            Some(l) if l.0 == v => l.1 = l.1.saturating_add(c),
            _ => out.push((v, c)),
        }
    }
    let peak = out.iter().map(|&(_, c)| c).max().unwrap_or(0);
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 8) };
    let tmp = dir.join("traffic.bin.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, dir.join("traffic.bin"))?;
    Ok((out.len(), peak))
}

fn level_structure(
    ds: &Dataset,
    ov: &Overlay,
    k: usize,
    target: usize,
    dir: &Path,
) -> LevelStructure {
    assert!(k >= 1, "level 0 is built by `build`");
    assert!(k <= ov.levels(), "level {k} would leave a gap in the ladder");
    let t0 = std::time::Instant::now();
    let below = k - 1;
    let nprev = ov.level(below).regions();
    let nv = ds.n_vertices();

    // Road vertices per region of the level below, so the partition grows by
    // the size that decides the work rather than by region count.
    let mut csize = vec![0u32; nprev];
    for v in 0..nv as u32 {
        csize[ov.cell_at(below, v) as usize] += 1;
    }

    // The region graph of the level below. Every one of its edges is a cut
    // edge, and every cut edge is incident to one of that level's boundary
    // vertices, so this walks the boundary rather than the planet.
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    // The same walk, one level finer: which *gate* faces which region. The
    // region graph says two regions touch; this says through which vertices, so
    // a growing cell can be told exactly how many gates absorbing a neighbour
    // would bury. Indexed by position in the level-below boundary list, not by
    // vertex id — 6.5 million slots on the planet rather than 270 million.
    let mut gpairs: Vec<(u32, u32)> = Vec::new();
    let nb_below = ov.level(below).boundary_total();
    let mut owner = vec![0u32; nb_below];
    for c in 0..nprev as u32 {
        let base = ov.level(below).boundary_start(c);
        for (i, &u) in ov.level(below).boundary(c).iter().enumerate() {
            owner[base + i] = c;
            for (head, fwd) in [(ds.head, true), (ds.rhead, false)] {
                let (a, b) = (head[u as usize] as usize, head[u as usize + 1] as usize);
                for j in a..b {
                    let v = if fwd { ds.target(u, j) } else { ds.rtarget(u, j) };
                    let d = ov.cell_at(below, v);
                    if d != c {
                        pairs.push((c, d));
                        pairs.push((d, c));
                        gpairs.push(((base + i) as u32, d));
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    gpairs.sort_unstable();
    gpairs.dedup();

    // Gate -> the regions it faces.
    let mut nhead = vec![0u32; nb_below + 1];
    for &(g, _) in &gpairs {
        nhead[g as usize + 1] += 1;
    }
    for i in 1..=nb_below {
        nhead[i] += nhead[i - 1];
    }
    let nlist: Vec<u32> = gpairs.iter().map(|&(_, d)| d).collect();
    // Region -> the gates that face it, which is the half that lets an absorbed
    // neighbour find the gates it just buried without rescanning the cell.
    let mut inv: Vec<(u32, u32)> = gpairs.iter().map(|&(g, d)| (d, g)).collect();
    drop(gpairs);
    inv.sort_unstable();
    let mut ihead = vec![0u32; nprev + 1];
    for &(d, _) in &inv {
        ihead[d as usize + 1] += 1;
    }
    for i in 1..=nprev {
        ihead[i] += ihead[i - 1];
    }
    let ilist: Vec<u32> = inv.iter().map(|&(_, g)| g).collect();
    drop(inv);
    let mut bstart = vec![0u32; nprev + 1];
    for (c, b) in bstart.iter_mut().take(nprev).enumerate() {
        *b = ov.level(below).boundary_start(c as u32) as u32;
    }
    bstart[nprev] = nb_below as u32;
    let gx = GateIndex { bstart, nhead, nlist, ihead, ilist, owner };
    let mut ahead = vec![0u32; nprev + 1];
    for &(c, _) in &pairs {
        ahead[c as usize + 1] += 1;
    }
    for i in 1..=nprev {
        ahead[i] += ahead[i - 1];
    }
    let adj: Vec<u32> = pairs.iter().map(|&(_, d)| d).collect();
    drop(pairs);

    // Same BFS growth as level 0, over regions instead of vertices.
    let mut lof = vec![u32::MAX; nprev];
    let mut ncells = 0u32;
    let mut q: std::collections::VecDeque<u32> = Default::default();
    // For the cut below: candidates ordered by (gates buried, traffic buried).
    let mut heap: std::collections::BinaryHeap<(u32, u64, u32)> = Default::default();
    let traffic = Traffic::open(dir);
    // `MPEE_CUT=busy` lets the traffic decide which neighbour joins, instead of
    // only breaking ties between neighbours that bury equally many gates.
    let busy_leads = std::env::var("MPEE_CUT").as_deref() == Ok("busy");
    let cap: usize = std::env::var("MPEE_GATE_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(GATE_CAP);
    // `MPEE_GROW=vertices` restores the growth this replaced: absorb by road
    // vertex count and repair whatever came out over the cap afterwards. Kept
    // so the two can be measured against each other on the same data.
    let by_gates = cap > 0 && std::env::var("MPEE_GROW").as_deref() != Ok("vertices");
    // How many regions of the level below one region here may swallow.
    //
    // The gate cap bounds a region's *table*; this bounds the *clique* the next
    // rung has to search. They are not the same limit, and the planet showed the
    // gap between them: from rung 2 to rung 3 the regions merged 47 to 1 — every
    // one of them small, 64 gates each, so nothing came near the 2048 cap — and
    // the result was a rung whose regions hold 1657 gates. A table turns a
    // region into a complete graph on its gates, so the rung above then relaxes
    // 1657 edges for every vertex it settles, against 2.4 in the road graph.
    // That single step is 542 billion relaxations, nine times the rest of the
    // ladder together.
    let agg: usize = std::env::var("MPEE_AGG_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(usize::MAX);
    if by_gates {
        // Grow on the quantity that actually costs: a region's table is `2b²`
        // entries and `b` searches to fill, so `b` is what has to be bounded.
        // The vertex target stays as a second limit, because a region with few
        // gates and a great many members is cheap to store and slow to fill.
        //
        // Growing *to* the cap rather than growing blindly and cutting back is
        // the whole difference. The repair it replaces re-grew an over-cap cell
        // at a quarter of the target — a guess that overshoots downward on
        // purpose — and left the typical region at 68 gates against a cap of
        // 1024, which is to say the cap was never what stopped it.
        let mut outside = vec![0u32; nb_below];
        let mut touched: Vec<u32> = Vec::new();
        let bnd_below = ov.level(below).boundary_all();
        for s in 0..nprev as u32 {
            if lof[s as usize] != u32::MAX {
                continue;
            }
            let id = ncells;
            let mut n = csize[s as usize] as usize;
            let mut nabs = 1usize;
            let mut g = gx.absorb(s, &mut lof, id, &mut outside, &mut touched);
            heap.clear();
            let (a, b) = (ahead[s as usize] as usize, ahead[s as usize + 1] as usize);
            for &d in &adj[a..b] {
                if lof[d as usize] == u32::MAX {
                    let (gain, busy) = gx.score(d, &lof, id, &traffic, bnd_below);
                    heap.push(if busy_leads {
                        (busy.min(u32::MAX as u64) as u32, gain as u64, d)
                    } else {
                        (gain, busy, d)
                    });
                }
            }
            while let Some((_, _, d)) = heap.pop() {
                if n >= target || nabs >= agg {
                    break;
                }
                if lof[d as usize] != u32::MAX {
                    continue;
                }
                // Two limits, and a candidate that breaks either is passed over
                // rather than ending the cell: the cell can still grow in some
                // other direction, which is what lets a region reach around a
                // dense knot instead of stopping at it.
                if n + csize[d as usize] as usize > target && n > 0 {
                    continue;
                }
                if g + gx.delta(d, &lof, id, &outside) > cap as i64 {
                    continue;
                }
                g += gx.absorb(d, &mut lof, id, &mut outside, &mut touched);
                n += csize[d as usize] as usize;
                nabs += 1;
                let (a, b) = (ahead[d as usize] as usize, ahead[d as usize + 1] as usize);
                for &e in &adj[a..b] {
                    if lof[e as usize] == u32::MAX {
                        let (gain, busy) = gx.score(e, &lof, id, &traffic, bnd_below);
                        heap.push(if busy_leads {
                            (busy.min(u32::MAX as u64) as u32, gain as u64, e)
                        } else {
                            (gain, busy, e)
                        });
                    }
                }
            }
            for &p in &touched {
                outside[p as usize] = 0;
            }
            touched.clear();
            ncells += 1;
        }
    } else {
        for s in 0..nprev as u32 {
            if lof[s as usize] != u32::MAX {
                continue;
            }
            q.clear();
            q.push_back(s);
            lof[s as usize] = ncells;
            let mut n = csize[s as usize] as usize;
            while let Some(c) = q.pop_front() {
                if n >= target {
                    break;
                }
                let (a, b) = (ahead[c as usize] as usize, ahead[c as usize + 1] as usize);
                for &d in &adj[a..b] {
                    if lof[d as usize] == u32::MAX {
                        lof[d as usize] = ncells;
                        n += csize[d as usize] as usize;
                        q.push_back(d);
                        if n >= target {
                            break;
                        }
                    }
                }
            }
            ncells += 1;
        }
    }

    // ---- The gate cap ----
    //
    // The growth above stops on road vertices, which is the wrong quantity. A
    // region's table costs `2b²` entries and `b` restricted searches to fill,
    // so it is the gate count that decides both disk and work — and nothing so
    // far has looked at it. On the planet's fourth rung the target is 134
    // million road vertices, and the BFS duly swallowed everything reachable:
    // one region spanning 13°E to 129°E, Africa through Europe to Asia, with
    // 32 166 gates and 8.28 GB of table it would never be able to fill.
    //
    // So: grow as before, then count what was actually built, and cut whatever
    // came out over the cap — recursively, over the same region graph, with a
    // smaller target each round. Islands inside islands, decided by measurement
    // rather than by a size guessed in advance.
    //
    // Counting is exact and deliberately so. An estimate from `Σb` of the
    // absorbed regions overshoots badly — merging level 2's regions into level
    // 3 took a mean of 376 gates down to 57, a factor of 6.6 — and a cap fed by
    // a 6.6x overestimate would shred good regions to reach a limit they were
    // never near.
    let cap: usize = std::env::var("MPEE_GATE_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(GATE_CAP);
    let count_gates = |lof: &[u32], n: usize| -> Vec<u32> {
        let mut g = vec![0u32; n];
        for c in 0..nprev as u32 {
            let cc = lof[c as usize];
            for &u in ov.level(below).boundary(c) {
                let mut out = false;
                for (head, fwd) in [(ds.head, true), (ds.rhead, false)] {
                    let (a, b) = (head[u as usize] as usize, head[u as usize + 1] as usize);
                    for j in a..b {
                        let v = if fwd { ds.target(u, j) } else { ds.rtarget(u, j) };
                        if lof[ov.cell_at(below, v) as usize] != cc {
                            out = true;
                            break;
                        }
                    }
                    if out {
                        break;
                    }
                }
                if out {
                    g[cc as usize] += 1;
                }
            }
        }
        g
    };
    if cap > 0 {
        let before = ncells;
        let mut ctarget: Vec<usize> = vec![target; ncells as usize];
        let mut worst_before = 0u32;
        let mut cuts = 0usize;
        for round in 0..CUT_ROUNDS {
            let g = count_gates(&lof, ncells as usize);
            if round == 0 {
                worst_before = g.iter().copied().max().unwrap_or(0);
            }
            let over: Vec<u32> =
                (0..ncells).filter(|&c| g[c as usize] as usize > cap).collect();
            if over.is_empty() {
                break;
            }
            // Members by region, once per round rather than once per cut: the
            // first round of the planet's third rung has a cell holding most of
            // a hemisphere, and scanning `nprev` for each is the difference
            // between seconds and minutes.
            let mut mem: Vec<Vec<u32>> = vec![Vec::new(); ncells as usize];
            for d in 0..nprev as u32 {
                mem[lof[d as usize] as usize].push(d);
            }
            let mut moved = false;
            for &c in &over {
                // A quarter each round. Gates on a planar network grow roughly
                // with the square root of area, so halving the gate count wants
                // about a quarter of the vertices — and overshooting downward
                // costs only another round, while overshooting upward costs a
                // region that is still over the cap.
                let t = (ctarget[c as usize] / 4).max(1);
                let members = std::mem::take(&mut mem[c as usize]);
                let mut fresh: std::collections::HashMap<u32, u32> = Default::default();
                let mut first = true;
                for &s in &members {
                    if fresh.contains_key(&s) {
                        continue;
                    }
                    let id = if first {
                        first = false;
                        c
                    } else {
                        ncells += 1;
                        ctarget.push(t);
                        cuts += 1;
                        ncells - 1
                    };
                    fresh.insert(s, id);
                    // Greedy, not breadth-first. Which neighbour joins decides
                    // where the cut lands, and a queue picks by arrival order —
                    // which is to say, arbitrarily. Two things are worth
                    // preferring, in this order:
                    //
                    // `gain` is how many of the candidate's gates already face
                    // this cell. Absorbing it turns those from gates into
                    // interior vertices, so a high gain is a cut that costs
                    // fewer gates — greedy conductance, the same rule
                    // `plan_splits_by_cost` uses.
                    //
                    // `busy` is how much traffic those gates carry. Among
                    // candidates that cost the same in gates, take the one that
                    // buries the busiest crossing, so the vertices left on the
                    // cut are the ones queries seldom cross. This is the
                    // betweenness the levels gather while filling rows, fed
                    // back into the partition that gathers it.
                    heap.clear();
                    heap.push((0u32, 0u64, s));
                    let mut n = 0usize;
                    while let Some((_, _, x)) = heap.pop() {
                        if fresh.get(&x).is_some_and(|&i| i != id) {
                            continue;
                        }
                        if x != s {
                            if n >= t {
                                break;
                            }
                            fresh.insert(x, id);
                        }
                        n += csize[x as usize] as usize;
                        if n >= t {
                            break;
                        }
                        let (a, b) = (ahead[x as usize] as usize, ahead[x as usize + 1] as usize);
                        for &d in &adj[a..b] {
                            // Only inside the cell being cut: this is a
                            // re-partition of one region, not of the level.
                            if lof[d as usize] != c || fresh.contains_key(&d) {
                                continue;
                            }
                            let (mut gain, mut busy) = (0u32, 0u64);
                            for &u in ov.level(below).boundary(d) {
                                let mut touches = false;
                                for (head, fwd) in [(ds.head, true), (ds.rhead, false)] {
                                    let (p, q2) =
                                        (head[u as usize] as usize, head[u as usize + 1] as usize);
                                    for j in p..q2 {
                                        let w = if fwd { ds.target(u, j) } else { ds.rtarget(u, j) };
                                        let e = ov.cell_at(below, w);
                                        if fresh.get(&e).is_some_and(|&i| i == id) {
                                            touches = true;
                                            break;
                                        }
                                    }
                                    if touches {
                                        break;
                                    }
                                }
                                if touches {
                                    gain += 1;
                                    busy += traffic.get(u) as u64;
                                }
                            }
                            // Which of the two leads is a real question and
                            // the answer is measured, not assumed: `gain` first
                            // spends the cap's own currency, `busy` first
                            // spends the query's.
                            heap.push(if busy_leads {
                                (busy.min(u32::MAX as u64) as u32, gain as u64, d)
                            } else {
                                (gain, busy, d)
                            });
                        }
                    }
                }
                ctarget[c as usize] = t;
                for (d, id) in fresh {
                    if lof[d as usize] != id {
                        moved = true;
                    }
                    lof[d as usize] = id;
                }
            }
            if !moved {
                // The cell is as cut as this rule can make it — a single
                // lower-level region already over the cap cannot be divided
                // here, only at the level that built it.
                break;
            }
        }
        if cuts > 0 {
            let g = count_gates(&lof, ncells as usize);
            eprintln!(
                "[level {k}] gate cap {cap}: {before} -> {ncells} regions, \
                 worst region {worst_before} -> {} gates",
                g.iter().copied().max().unwrap_or(0)
            );
        }
    }

    let ncells = ncells as usize;
    eprintln!(
        "[level {k}] {ncells} regions over {nprev} below ({:.1} s)",
        t0.elapsed().as_secs_f64()
    );

    // The vertex set each search runs over: every boundary vertex of the level
    // below that falls inside the region. A vertex is on the boundary of
    // exactly one region per level — its own — so there is nothing to dedupe.
    let mut inner: Vec<(u32, u32)> = Vec::new();
    for c in 0..nprev as u32 {
        let cc = lof[c as usize];
        for &u in ov.level(below).boundary(c) {
            inner.push((cc, u));
        }
    }
    inner.sort_unstable();
    let mut vhead = vec![0u32; ncells + 1];
    for &(c, _) in &inner {
        vhead[c as usize + 1] += 1;
    }
    for i in 1..=ncells {
        vhead[i] += vhead[i - 1];
    }
    let vlist: Vec<u32> = inner.iter().map(|&(_, u)| u).collect();
    drop(inner);

    // This level's boundary: a boundary vertex of the level below that has an
    // edge leaving this level's region as well.
    let mut bv: Vec<(u32, u32)> = Vec::new();
    for c in 0..nprev as u32 {
        let cc = lof[c as usize];
        for &u in ov.level(below).boundary(c) {
            let mut out = false;
            for (head, fwd) in [(ds.head, true), (ds.rhead, false)] {
                let (a, b) = (head[u as usize] as usize, head[u as usize + 1] as usize);
                for j in a..b {
                    let v = if fwd { ds.target(u, j) } else { ds.rtarget(u, j) };
                    if lof[ov.cell_at(below, v) as usize] != cc {
                        out = true;
                        break;
                    }
                }
                if out {
                    break;
                }
            }
            if out {
                bv.push((cc, u));
            }
        }
    }
    bv.sort_unstable();
    let mut bhead = vec![0u32; ncells + 1];
    for &(c, _) in &bv {
        bhead[c as usize + 1] += 1;
    }
    for i in 1..=ncells {
        bhead[i] += bhead[i - 1];
    }
    let blist: Vec<u32> = bv.iter().map(|&(_, u)| u).collect();
    drop(bv);
    let nb = blist.len();

    let mut ohead = vec![0u64; ncells + 1];
    for c in 0..ncells {
        let b = (bhead[c + 1] - bhead[c]) as u64;
        // Both directions, side by side, so a region's block is its own.
        ohead[c + 1] = ohead[c] + 2 * b * b;
    }
    let entries = ohead[ncells];
    eprintln!(
        "[level {k}] {nb} boundary vertices ({:.2} % of the level below's {}), \
         {entries} table entries ({:.2} GB at 32 bits)",
        nb as f64 / ov.level(below).boundary_total() as f64 * 100.0,
        ov.level(below).boundary_total(),
        entries as f64 * 4.0 / 1e9
    );


    LevelStructure { lof, vhead, vlist, bhead, blist, ohead, ncells, entries }
}

/// Build one more level on top of the levels a dataset already has.
///
/// Level 0 collapses a region to the roads that enter it. Every level above
/// does the same thing again, one rung up: it collapses a *group* of regions
/// to the roads that enter the group, so crossing a continent reads one table
/// entry per group rather than walking every boundary vertex on the way.
///
/// The construction is self-similar, which is what makes it cheap and what
/// makes it recursive: a level's table is built by the *same* restricted
/// Dijkstra as level 0's, run over the level below's overlay graph instead of
/// the road graph. Its vertices are that level's boundary vertices, its edges
/// are that level's table shortcuts plus the cut edges between its regions —
/// so a region holding four million road vertices is searched as a graph of a
/// few thousand. Islands inside islands: a city inside a country inside a
/// continent, cut by the graph rather than by borders.
///
/// The partition is nested — a region is a union of whole regions from the
/// level below — which is what lets a query pick a level per vertex, and why
/// each level's boundary is a *subset* of the one below rather than a new set
/// of vertices needing their own geometry.
pub fn build_level(
    paths: &Paths,
    ds: &Dataset,
    ovr: Option<&Overrides>,
    ov: &Overlay,
    k: usize,
    target: usize,
) -> io::Result<L2Stats> {
    assert!(k >= 1, "level 0 is built by `build`");
    assert!(k <= ov.levels(), "level {k} would leave a gap in the ladder");
    assert!(
        !ov.level(k - 1).is_lazy(),
        "level {k} cannot be built eagerly on a lazy level {}: the table fill is \
         parallel and would have every thread racing to fill the same rows below. \
         Build this level lazily too, or materialise the one below first.",
        k - 1
    );
    let t0 = std::time::Instant::now();
    let below = k - 1;
    let LevelStructure { lof, vhead, vlist, bhead, blist, ohead, ncells, entries } =
        level_structure(ds, ov, k, target, paths.dir());
    let nb = blist.len();

    // Same refusal as the lazy path. This guard was added there first and not
    // here, and the omission showed immediately: rebuilding Norway eagerly
    // produced rungs 2 through 8, every one of them with no boundary at all,
    // because nothing stopped a degenerate rung from being written. A rule that
    // exists on one of two paths is not a rule.
    if nb == 0 {
        eprintln!(
            "[level {k}] refused: the partition left no boundary at all — every region \
             is a whole connected component, so this rung has nothing to collapse"
        );
        return Ok(L2Stats {
            cells: ncells,
            boundary: 0,
            entries,
            secs: t0.elapsed().as_secs_f64(),
            fill_secs: 0.0,
            refused: true,
        });
    }

    let f = level_files(k);
    let wide_name = format!("{}.wide", f[4]);
    let mut mat = mmapvec::create::<u32>(&paths.f(&wide_name), entries as usize)?;
    let matp = {
        let s: &mut [u32] = unsafe { mmapvec::as_mut_slice(&mut mat[..]) };
        crate::parallel::Scatter(s.as_mut_ptr(), s.len())
    };

    (0..ncells).into_par_iter().for_each(|c| {
        let (bs, be) = (bhead[c] as usize, bhead[c + 1] as usize);
        let b = be - bs;
        if b == 0 {
            return;
        }
        let members = &vlist[vhead[c] as usize..vhead[c + 1] as usize];
        let mut out = Vec::new();
        region_table(ds, ovr, ov, below, members, &blist[bs..be], &mut out);
        for (n, &v) in out.iter().enumerate() {
            unsafe { matp.put(ohead[c] as usize + n, v) };
        }
    });
    mat.flush()?;

    let (narrowed, byte_len) = {
        let wide: &[u32] = unsafe { mmapvec::as_slice(&mat[..]) };
        narrow_and_write(paths, wide, &ohead, &bhead, ncells, &f[4], &f[3], &f[5])?
    };
    drop(mat);
    std::fs::remove_file(paths.f(&wide_name)).ok();
    eprintln!(
        "[level {k}] {narrowed} of {ncells} regions fit 16-bit entries — table {:.2} GB ({:.2}x smaller)",
        byte_len as f64 / 1e9,
        entries as f64 * 4.0 / byte_len.max(1) as f64
    );

    write_u32(paths, &f[0], &lof)?;
    write_u32(paths, &f[1], &blist)?;
    write_u32(paths, &f[2], &bhead)?;
    Ok(L2Stats {
        cells: ncells,
        boundary: nb,
        entries,
        secs: t0.elapsed().as_secs_f64(),
        fill_secs: 0.0,
        refused: false,
    })
}

/// Build a level's shape and leave its numbers to be computed on demand.
///
/// This is the half of CRP that the road network decides and the metric does
/// not, and on the planet it is seconds where filling the tables is twenty
/// minutes. What it writes instead of a table is a *sparse* file at the table's
/// full logical size: the filesystem allocates nothing until a row is written,
/// so the disk holds exactly the rows some query actually asked for.
///
/// The upper rungs are what this is for. The planet's second rung comes to
/// 8.66 GB eagerly — larger than the first rung's 5.07 GB, because a rung's
/// regions get coarser faster than their count falls, and a table is `b²`.
/// A single route reads a handful of its rows.
pub fn build_level_lazy(
    paths: &Paths,
    ds: &Dataset,
    ov: &Overlay,
    k: usize,
    target: usize,
    max_fill_secs: f64,
) -> io::Result<L2Stats> {
    assert!(k >= 1, "level 0 is built by `build`");
    assert!(k <= ov.levels(), "level {k} would leave a gap in the ladder");
    let t0 = std::time::Instant::now();
    let st = level_structure(ds, ov, k, target, paths.dir());
    let nb = st.blist.len();

    // A rung with no boundary at all is not a small rung, it is a degenerate
    // one. It happens when the target outgrows the map: at 4.3 billion road
    // vertices every region absorbs a whole connected component, and a
    // component has no edges leaving it by definition. There is nothing to
    // collapse and nothing to search, and the ladder's own stopping rule reads
    // the zero as success — "small enough to search directly" — so it has to be
    // caught here, before the level exists.
    if nb == 0 {
        eprintln!(
            "[level {k}] refused: the partition left no boundary at all — every region \
             is a whole connected component, so this rung has nothing to collapse"
        );
        return Ok(L2Stats {
            cells: st.ncells,
            boundary: 0,
            entries: st.entries,
            secs: t0.elapsed().as_secs_f64(),
            fill_secs: 0.0,
            refused: true,
        });
    }

    // What this rung will cost to warm, before a byte of it is written.
    let below = k - 1;
    let nprev = ov.level(below).regions().max(1);
    let b_below = ov.level(below).boundary_total() as f64 / nprev as f64;
    let fill_secs = fill_estimate(&st.bhead, &st.vhead, b_below);

    // Refuse before writing anything. A rung that cannot be warmed is worse
    // than no rung: a query climbing into it computes rows on demand, and on
    // the planet's fourth level one row is five seconds.
    if fill_secs > max_fill_secs {
        eprintln!(
            "[level {k}] refused: {} boundary vertices would take {} to fill \
             (limit {}) — the level below is the top",
            nb,
            human_secs(fill_secs),
            human_secs(max_fill_secs)
        );
        return Ok(L2Stats {
            cells: st.ncells,
            boundary: nb,
            entries: st.entries,
            secs: t0.elapsed().as_secs_f64(),
            fill_secs,
            refused: true,
        });
    }

    let f = level_files(k);
    let lf = lazy_files(k);
    // A level opens eager when both of those exist, so a rebuild from eager to
    // lazy has to take them away.
    std::fs::remove_file(paths.f(&f[4])).ok();
    std::fs::remove_file(paths.f(&f[5])).ok();

    write_u32(paths, &f[0], &st.lof)?;
    write_u32(paths, &f[1], &st.blist)?;
    write_u32(paths, &f[2], &st.bhead)?;
    write_u64(paths, &f[3], &st.ohead)?;
    write_u32(paths, &lf[0], &st.vhead)?;
    write_u32(paths, &lf[1], &st.vlist)?;

    // Four bytes an entry; both directions are already in `entries`. A lazy
    // table cannot narrow: the narrowing needs every value of a region at once,
    // which is exactly what it declines to compute.
    let bytes = st.entries * 4;
    let file = std::fs::File::create(paths.f(&lf[2]))?;
    file.set_len(bytes)?;
    drop(file);
    // One bit per row per direction. On the planet's second rung that is
    // 110 KB standing in for 8.66 GB.
    let bits = std::fs::File::create(paths.f(&lf[3]))?;
    bits.set_len((2 * nb).div_ceil(8) as u64)?;
    drop(bits);
    // Last, so that a stamp never stands for a level that was not finished.
    write_lazy_stamp(
        &paths.f(&fmt_file(k)),
        (st.bhead.len() - 1) as u64,
        nb as u64,
        st.entries,
    )?;
    // One count per member, for the betweenness a cut decision reads. Four
    // bytes per boundary vertex of the level below — 7 MB on the planet's
    // first rung, against the 10 GB of table it sits beside.
    let uses = std::fs::File::create(paths.f(&use_file(k)))?;
    uses.set_len((st.vlist.len() * 4) as u64)?;
    drop(uses);

    let on_disk = allocated(&paths.f(&lf[2]));
    eprintln!("[level {k}] estimated {} to fill every row", human_secs(fill_secs));
    eprintln!(
        "[level {k}] lazy: {:.2} GB of table reserved, {:.3} GB allocated, \
         {} rows to fill on demand",
        bytes as f64 / 1e9,
        on_disk as f64 / 1e9,
        2 * nb
    );
    Ok(L2Stats {
        cells: st.ncells,
        boundary: nb,
        entries: st.entries,
        secs: t0.elapsed().as_secs_f64(),
        fill_secs,
        refused: false,
    })
}

/// Blocks a file actually occupies, which for a sparse file is not its length.
fn allocated(p: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(p).map(|m| m.blocks() * 512).unwrap_or(0)
}

/// The level a settled vertex should be handled at, and its region there.
///
/// Climb as high as the ladder allows: the coarsest level whose region holds
/// neither endpoint and on whose boundary `u` sits, because higher means one
/// table read crosses more of the world. Two rules stop the climb and both are
/// one-way — a region holding an endpoint needs the detail a coarser table has
/// collapsed away, and a vertex interior to a region at one level is interior
/// at every level above it, since boundary sets only shrink upward.
#[inline]
fn choose_level(ov: &Overlay, home: &[Vec<u32>], u: u32) -> (usize, u32, Option<usize>) {
    let mut lc = ov.cell(u);
    let mut lv = 0usize;
    let mut li = ov.bindex(lc, u);
    let mut c = lc;
    for (k, h) in home.iter().enumerate().skip(1) {
        c = ov.level(k).parent(c);
        if h.contains(&c) {
            break;
        }
        match ov.level(k).bindex(c, u) {
            Some(i) => {
                lv = k;
                lc = c;
                li = Some(i);
            }
            None => break,
        }
    }
    (lv, lc, li)
}

/// A floor on the time still needed to get from `v` to a point, in tenths of
/// a second.
///
/// Straight-line distance over the fastest speed in the dataset: no road can
/// beat it, so it can never overstate what remains. Measured on a sphere of
/// the Earth's *polar* radius, which under-states the geodesic, so the bound
/// is conservative twice over.
///
/// This is the vertex's own position, not its region's box — and that is
/// deliberate. The box was the obvious thing to reach for, but the vertex is
/// strictly tighter and costs the same one distance calculation. It is also
/// still valid for everything a table hop jumps to: for any `w` reached from
/// `v`, the triangle inequality gives `dist(v,t) <= dist(v,w) + dist(w,t)`, so
/// a path through `w` is never cheaper than this bound allows.
#[inline]
pub fn geo_floor_ds(ds: &Dataset, v: u32, lat_e7: i32, lon_e7: i32, vmax_kmh: u16) -> u32 {
    geo_floor_xyz(ds, v, xyz_of(lat_e7, lon_e7), vmax_kmh)
}

/// A lat/lon as a point in space, metres from the Earth's centre.
///
/// The one place trigonometry is still unavoidable — but a query has two
/// targets and hundreds of thousands of settled vertices, so this runs twice
/// and the loop runs neither time.
#[inline]
pub fn xyz_of(lat_e7: i32, lon_e7: i32) -> [f64; 3] {
    let la = (lat_e7 as f64 * 1e-7).to_radians();
    let lo = (lon_e7 as f64 * 1e-7).to_radians();
    let c = la.cos();
    [R_POLAR * c * lo.cos(), R_POLAR * c * lo.sin(), R_POLAR * la.sin()]
}

#[inline]
fn vertex_xyz(ds: &Dataset, v: u32) -> [f64; 3] {
    match ds.vxyz.get(v as usize) {
        Some(p) => [p[0] as f64, p[1] as f64, p[2] as f64],
        // No stored table: derive it, which is exactly the trigonometry the
        // table exists to avoid. Correct either way, and the same number.
        None => {
            let (a, o) = ds.vcoord[v as usize];
            xyz_of(a, o)
        }
    }
}

/// Lower bound on travel time from a vertex to a point, in tenths of a second.
///
/// The measure is the *chord* — the straight line through the Earth — not the
/// great-circle arc. A tunnel is never longer than the surface route above it,
/// so this understates, which is what a bound must do; and it is three
/// subtractions and a square root, where the arc needs an `asin` and four
/// trigonometric calls.
///
/// What it gives up is small. On the planet the chord is 0.77 % under the arc
/// for Lisboa-Warszawa, and it cost 278 extra settled vertices out of 197 725 —
/// 0.14 % — against a bound that already understates by 0.34 % from using the
/// polar radius and is in any case divided by a global maximum speed no road
/// achieves.
///
/// The `t` argument is the point, not its coordinates, and that is the part
/// that mattered. Taking the target's conversion out of the caller's loop —
/// four trigonometric calls per settled vertex, for a value constant across the
/// query — is what took A* from 37 % slower than plain Dijkstra to 12 % faster.
/// Reading `vxyz` instead of deriving it changed nothing measurable.
///
/// Sound for a table hop as well as an edge: for any `w` reached from `v`, the
/// triangle inequality gives `dist(v,t) <= dist(v,w) + dist(w,t)`, so a path
/// through `w` is never cheaper than this allows.
#[inline]
pub fn geo_floor_xyz(ds: &Dataset, v: u32, t: [f64; 3], vmax_kmh: u16) -> u32 {
    let p = vertex_xyz(ds, v);
    let (dx, dy, dz) = (p[0] - t[0], p[1] - t[1], p[2] - t[2]);
    let m = (dx * dx + dy * dy + dz * dz).sqrt() - XYZ_SLACK_M;
    (m.max(0.0) / (vmax_kmh as f64 * 1000.0 / 3600.0) * 10.0) as u32
}

/// The floor between two coordinates, for the constant a symmetric potential
/// is offset by. Once per query, so it converts both ends itself.
#[inline]
pub fn geo_floor_between(a: (i32, i32), b: (i32, i32), vmax_kmh: u16) -> u32 {
    let (p, t) = (xyz_of(a.0, a.1), xyz_of(b.0, b.1));
    let (dx, dy, dz) = (p[0] - t[0], p[1] - t[1], p[2] - t[2]);
    let m = (dx * dx + dy * dy + dz * dz).sqrt() - XYZ_SLACK_M;
    (m.max(0.0) / (vmax_kmh as f64 * 1000.0 / 3600.0) * 10.0) as u32
}

/// The file holding one level's geographic bounds.
pub fn geo_file(k: usize) -> String {
    if k == 0 {
        "ov.geo".into()
    } else {
        format!("l{}.geo", k + 1)
    }
}

/// Four numbers per region: the box its vertices fall in.
///
/// Not a container and not an index — the router already finds a point's
/// region exactly, by snapping to the nearest road in 0.3 ms, and no geometry
/// could do better: two roads that cross without meeting (a lane under a
/// motorway) are at the same coordinate in different regions, so *no* shape
/// separates them. Measured on the planet, a rasterised outline is only 1.03x
/// purer than the rectangle at level 0 and 1.05x at level 3 — a real polygon
/// buys almost nothing over four numbers.
///
/// What the box is for is a lower bound. Distance to a box can never exceed
/// distance to something inside it, so a loose box prunes less but never lies,
/// and overlapping boxes stay sound. That is what makes it usable here at all.
///
/// Bounds are exact and computed bottom-up: a region's box is the union of the
/// boxes below it, so one pass over the vertices settles every level.
pub fn build_geo(paths: &Paths, ds: &Dataset, ov: &Overlay) -> io::Result<()> {
    let t0 = std::time::Instant::now();
    let mut per_level: Vec<Vec<(i32, i32, i32, i32)>> = Vec::new();
    const EMPTY: (i32, i32, i32, i32) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);

    let mut b0 = vec![EMPTY; ov.level(0).regions()];
    for (v, &(lat, lon)) in ds.vcoord.iter().enumerate() {
        let e = &mut b0[ov.cell(v as u32) as usize];
        e.0 = e.0.min(lat);
        e.1 = e.1.max(lat);
        e.2 = e.2.min(lon);
        e.3 = e.3.max(lon);
    }
    per_level.push(b0);
    for k in 1..ov.levels() {
        let mut b = vec![EMPTY; ov.level(k).regions()];
        for c in 0..ov.level(k - 1).regions() as u32 {
            let s = per_level[k - 1][c as usize];
            if s.0 == i32::MAX {
                continue;
            }
            let e = &mut b[ov.level(k).parent(c) as usize];
            e.0 = e.0.min(s.0);
            e.1 = e.1.max(s.1);
            e.2 = e.2.min(s.2);
            e.3 = e.3.max(s.3);
        }
        per_level.push(b);
    }
    for (k, b) in per_level.iter().enumerate() {
        let mut w =
            BufWriter::with_capacity(1 << 20, std::fs::File::create(paths.f(&geo_file(k)))?);
        for e in b {
            for x in [e.0, e.1, e.2, e.3] {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        w.flush()?;
    }
    eprintln!(
        "[geo] bounds for {} level(s) in {:.1} s",
        ov.levels(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Fill missing rows, in parallel, until a budget runs out.
///
/// The lazy cache warms itself as queries arrive, which is the right behaviour
/// for a machine in service but a poor way to meet the first user. This is the
/// other half: warming on purpose, on whatever machine is convenient, and
/// stopping whenever you like.
///
/// It is resumable by construction. The presence bitmap already records what
/// exists, so an interrupted warm-up simply leaves the rows it finished, and
/// the next one continues from there — there is no progress file to keep in
/// step with the data, because the data *is* the progress.
///
/// Rows are filled bottom-up. A row at one level is computed by reading the
/// level below, so warming the lower rungs first means the upper ones find
/// what they need already there instead of recursing for it.
pub fn warm(
    ds: &Dataset,
    ovr: Option<&Overrides>,
    ov: &Overlay,
    budget_rows: u64,
    budget: std::time::Duration,
) -> (u64, f64) {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    let t0 = std::time::Instant::now();
    let done = AtomicU64::new(0);
    let stop = AtomicBool::new(false);

    // Two pools, because quality of service is a property of a thread and a
    // pool has one of it. On Apple Silicon that is also the only way to choose
    // a core type: `utility` prefers the performance cluster, `background` is
    // confined to the efficiency one — so a fast pool and a slow pool is a
    // fast-core pool and a slow-core pool.
    //
    // Measured on this machine, warming the planet's upper rungs: one
    // performance core does 4.2 rows/s and the whole efficiency cluster at
    // background priority does 2.2, so the slow cores are worth having but
    // only as an addition. Scaling is sublinear — two performance cores give
    // 1.48x, not 2x — which says memory bandwidth binds as hard as cores do,
    // and is the reason adding efficiency threads may return less than their
    // 2.2 rows/s when they contend for the same bandwidth.
    let nfast: usize = std::env::var("MPEE_WARM_FAST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::env::var("RAYON_NUM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
        });
    let nslow: usize =
        std::env::var("MPEE_WARM_SLOW").ok().and_then(|v| v.parse().ok()).unwrap_or(0);

    let build = |n: usize, qos: &'static str| -> Option<rayon::ThreadPool> {
        if n == 0 {
            return None;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .start_handler(move |_| {
                crate::cachecap::set_thread_qos(qos);
            })
            .build()
            .ok()
    };
    let fast = build(nfast, "utility");
    let slow = build(nslow, "background");
    eprintln!("[warm] {nfast} on performance cores, {nslow} on efficiency cores");

    for k in 0..ov.levels() {
        if !ov.level(k).is_lazy() || stop.load(Ordering::Relaxed) {
            continue;
        }
        // The unit of work is one *row*, not one region.
        //
        // A cursor over regions self-balances until the end and then collapses:
        // the largest region holds thousands of rows, so whichever thread
        // takes it last is left grinding alone while eight others idle.
        // Measured on the planet's second rung, the final 11 103 rows ran on a
        // single thread at 99 % CPU with nine available.
        //
        // Rows inside a region are independent, and `bhead` is already the
        // prefix sum of rows per region — so a global row index becomes
        // (region, row) with one binary search, and the tail disappears.
        let n = ov.level(k).regions();
        let rows = ov.level(k).boundary_total();
        let next = AtomicUsize::new(0);
        // `broadcast` runs the drain loop once on *every* thread of the pool.
        // A parallel iterator would not: the work is a cursor, not a
        // collection, so there is nothing for rayon to split — an earlier
        // version used `split` and silently ran on one thread per pool, which
        // showed up as CPU pinned near 200 % no matter how many threads were
        // asked for.
        let work = |pool: &rayon::ThreadPool| {
            pool.broadcast(|_| loop {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                let r = next.fetch_add(1, Ordering::Relaxed);
                if r >= rows * 2 {
                    return;
                }
                let (row, fwd) = (r % rows, r < rows);
                let c = ov.level(k).region_of_row(row, n);
                let i = row - ov.level(k).row_base(c);
                if ov.row_present(k, c, i, fwd) {
                    continue;
                }
                ov.ensure_row(ds, ovr, k, c, i, fwd);
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if d >= budget_rows || t0.elapsed() >= budget {
                    stop.store(true, Ordering::Relaxed);
                    return;
                }
            });
        };
        std::thread::scope(|sc| {
            if let Some(f) = fast.as_ref() {
                sc.spawn(|| work(f));
            }
            if let Some(sl) = slow.as_ref() {
                sc.spawn(|| work(sl));
            }
        });
    }
    (done.load(Ordering::Relaxed), t0.elapsed().as_secs_f64())
}


// ---------------------------------------------------------------------------
// Splitting a region without rebuilding the level
// ---------------------------------------------------------------------------

/// Move part of a region into a new one, keeping every other region's rows.
///
/// A region with too many gates is not a region that refuses to decompose — it
/// is one that was cut in the wrong place. The partition grows to a *size*
/// target and takes the gate count as whatever falls out, which is why the
/// worst regions on the planet reach 9 883 gates while the median has 7. This
/// is the operation that fixes one of them without starting over.
///
/// # Why the rows survive
///
/// A row's place in the table is `head[c] + i·b`, and `head` is a plain offset
/// array — nothing requires it to be monotonic. So a region whose boundary is
/// unchanged keeps its offset and its rows, and only the regions that actually
/// moved get fresh space at the end of the file. The old space becomes garbage
/// until a compaction reclaims it, which is what a log-structured store does.
///
/// What *cannot* survive is the index. `bhead` is a prefix sum, so changing one
/// region's gate count shifts every entry after it — and the presence bitmap is
/// addressed through `bhead`, so its bits move too. Both are rebuilt whole,
/// which is cheap: on the planet's level 0 they are 31 MB against 5.6 GB of
/// table.
///
/// # What it does not do
///
/// It writes a new structure to `paths`; it does not modify the dataset being
/// read. Publishing is the caller's job, through the build catalogue — a
/// reader holds `cell.of` and the boundary lists, and changing those underneath
/// it would be inconsistent. On APFS the clone that makes this affordable costs
/// 0.146 s for 5.58 GB.
pub struct SplitPlan {
    /// Region to split, and the members to move out of it.
    pub cuts: Vec<(u32, Vec<u32>)>,
}

pub struct SplitStats {
    pub regions_before: usize,
    pub regions_after: usize,
    pub rows_kept: usize,
    pub rows_dropped: usize,
    pub gates_before: usize,
    pub gates_after: usize,
    pub secs: f64,
}

/// Apply splits to level `k` and write the new structure.
///
/// `k == 0` moves graph vertices between regions; above that it moves whole
/// regions of the level below, which is what keeps the partition nested.
pub fn split_level(
    paths: &Paths,
    ds: &Dataset,
    ov: &Overlay,
    k: usize,
    plan: &SplitPlan,
) -> io::Result<SplitStats> {
    let t0 = std::time::Instant::now();
    let lvl = ov.level(k);
    let n_before = lvl.regions();

    // 1. The new assignment. `of` maps a unit of the level below to a region;
    //    a moved unit gets a freshly appended id.
    let mut of: Vec<u32> = lvl.of.to_vec();
    let mut next = n_before as u32;
    let mut touched: std::collections::BTreeSet<u32> = Default::default();
    // Which region each fresh id came out of, so the level above can hand it
    // the same parent.
    let mut origin: Vec<(u32, u32)> = Vec::new();
    for (c, members) in &plan.cuts {
        if members.is_empty() {
            continue;
        }
        let fresh = next;
        next += 1;
        origin.push((fresh, *c));
        touched.insert(*c);
        touched.insert(fresh);
        for &u in members {
            // `members` are the level below's units — vertices at level 0.
            if of[u as usize] != *c {
                continue; // stale plan entry; the unit already moved
            }
            of[u as usize] = fresh;
        }
    }
    let n_after = next as usize;

    // 2. Every level above inherits the split.
    //
    //    A new region takes the parent of the one it came out of, so no level
    //    above gains a region — but the level directly above *addresses* its
    //    units by region id, so its `of` array has to grow to cover them. And
    //    every level above has this level's boundary as its members, which
    //    just changed, so each one's index is rebuilt. The plan called this
    //    free; it is cheap, which is not the same thing.
    let mut ofs: Vec<Vec<u32>> = vec![of];
    for l in (k + 1)..ov.levels() {
        let mut up = ov.level(l).of.to_vec();
        if l == k + 1 {
            up.resize(n_after, 0);
            for (fresh, from) in &origin {
                up[*fresh as usize] = ov.level(l).of[*from as usize];
            }
        }
        ofs.push(up);
    }

    let mut fresh: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
    let (bhead, blist, vhead, vlist) = rebuild_index_at(ds, ov, k, k, &ofs, n_after, &fresh);
    fresh.push((bhead.clone(), blist.clone()));

    // 3. Which regions kept exactly the boundary they had. Those keep their
    //    rows; everything else starts empty.
    //
    //    As splits are shaped today this is always true for an untouched
    //    region — dividing `c` into `{c, c'}` leaves every other region's
    //    vertex set alone, and a vertex is on its region's boundary by its own
    //    edges. The comparison is here for a plan that moves units *between*
    //    existing regions, where it would stop being true. Mutating it to
    //    `true` changes nothing today, which is why no test catches that.
    let mut keep: Vec<bool> = vec![false; n_after];
    for c in 0..n_before {
        if touched.contains(&(c as u32)) {
            continue;
        }
        let old = lvl.boundary(c as u32);
        let (a, b) = (bhead[c] as usize, bhead[c + 1] as usize);
        keep[c] = old == &blist[a..b];
    }

    // 4. Offsets: kept regions hold theirs, the rest are appended.
    let old_end = lvl.head[n_before];
    let mut head = vec![0u64; n_after + 1];
    let mut acc = old_end;
    for c in 0..n_after {
        let b = (bhead[c + 1] - bhead[c]) as u64;
        if keep[c] {
            head[c] = lvl.head[c];
        } else {
            head[c] = acc;
            acc += 2 * b * b;
        }
    }
    head[n_after] = acc;

    let (mut kept_rows, mut dropped_rows) = (0usize, 0usize);
    for c in 0..n_after {
        let b = (bhead[c + 1] - bhead[c]) as usize;
        if keep[c] {
            kept_rows += b;
        } else {
            dropped_rows += b;
        }
    }

    let gates_before: usize = lvl.bnd.len();
    let gates_after = blist.len();

    // 5. Write. The table file grows; its existing bytes are untouched, which
    //    is what lets the kept offsets still mean what they meant.
    let f = level_files(k);
    let lf = lazy_files(k);
    write_u32(paths, &f[0], &ofs[0])?;
    write_u32(paths, &f[1], &blist)?;
    write_u32(paths, &f[2], &bhead)?;
    write_u64(paths, &f[3], &head)?;
    write_u32(paths, &lf[0], &vhead)?;
    write_u32(paths, &lf[1], &vlist)?;
    {
        let file = std::fs::OpenOptions::new().read(true).write(true).open(paths.f(&lf[2]))?;
        file.set_len(acc * 4)?;
    }
    // 6. The presence bitmap, remapped. A kept region's rows are still there,
    //    but they are addressed through `bhead`, which has moved — so the bits
    //    move with them.
    let rows_after = blist.len();
    let mut have = vec![0u8; (2 * rows_after).div_ceil(8)];
    if let Values::Lazy(z) = &lvl.values {
        for c in 0..n_before {
            if !keep[c] {
                continue;
            }
            let (oa, ob) = (lvl.bhead[c] as usize, lvl.bhead[c + 1] as usize);
            let na = bhead[c] as usize;
            for (n, orow) in (oa..ob).enumerate() {
                for (dir, fwd) in [(0usize, true), (1, false)].into_iter() {
                    if z.bit(orow, fwd) {
                        let bit = dir * rows_after + na + n;
                        have[bit >> 3] |= 1 << (bit & 7);
                    }
                }
            }
        }
    }
    std::fs::write(paths.f(&lf[3]), &have)?;
    // The use counter is per member, and the member list changed.
    std::fs::write(paths.f(&use_file(k)), vec![0u8; vlist.len() * 4])?;
    // And the stamp, because a split changes every number in it: more regions,
    // more gates, a bigger table. Leaving the old one is not a stale comment,
    // it is a dataset that refuses to open — which is how this omission was
    // found, by a split test that could no longer read what it had just
    // written. A rule that holds on the build paths and not here is not a rule.
    write_lazy_stamp(
        &paths.f(&fmt_file(k)),
        (bhead.len() - 1) as u64,
        blist.len() as u64,
        acc,
    )?;

    // 7. And every level above, whose members are the boundary that just moved.
    //    Same rule for keeping rows: a region whose boundary is unchanged keeps
    //    its offset and its bits. In practice most of them are, because a split
    //    deep in the ladder rarely changes what crosses a coarse region's edge.
    for l in (k + 1)..ov.levels() {
        let up = ov.level(l);
        let nl = up.regions();
        let (ubhead, ublist, uvhead, uvlist) =
            rebuild_index_at(ds, ov, k, l, &ofs, nl, &fresh);
        fresh.push((ubhead.clone(), ublist.clone()));
        let mut ukeep = vec![false; nl];
        for c in 0..nl {
            let (a, b) = (ubhead[c] as usize, ubhead[c + 1] as usize);
            ukeep[c] = up.boundary(c as u32) == &ublist[a..b];
        }
        let mut uhead = vec![0u64; nl + 1];
        let mut uacc = up.head[nl];
        for c in 0..nl {
            let b = (ubhead[c + 1] - ubhead[c]) as u64;
            if ukeep[c] {
                uhead[c] = up.head[c];
            } else {
                uhead[c] = uacc;
                uacc += 2 * b * b;
            }
            if ukeep[c] {
                kept_rows += b as usize;
            } else {
                dropped_rows += b as usize;
            }
        }
        uhead[nl] = uacc;

        let uf = level_files(l);
        let ulf = lazy_files(l);
        write_u32(paths, &uf[0], &ofs[l - k])?;
        write_u32(paths, &uf[1], &ublist)?;
        write_u32(paths, &uf[2], &ubhead)?;
        write_u64(paths, &uf[3], &uhead)?;
        write_u32(paths, &ulf[0], &uvhead)?;
        write_u32(paths, &ulf[1], &uvlist)?;
        {
            let file =
                std::fs::OpenOptions::new().read(true).write(true).open(paths.f(&ulf[2]))?;
            file.set_len(uacc * 4)?;
        }
        let urows = ublist.len();
        let mut uhave = vec![0u8; (2 * urows).div_ceil(8)];
        if let Values::Lazy(z) = &up.values {
            for c in 0..nl {
                if !ukeep[c] {
                    continue;
                }
                let (oa, ob) = (up.bhead[c] as usize, up.bhead[c + 1] as usize);
                let na = ubhead[c] as usize;
                for (n, orow) in (oa..ob).enumerate() {
                    for (dir, fwd) in [(0usize, true), (1, false)].into_iter() {
                        if z.bit(orow, fwd) {
                            let bit = dir * urows + na + n;
                            uhave[bit >> 3] |= 1 << (bit & 7);
                        }
                    }
                }
            }
        }
        std::fs::write(paths.f(&ulf[3]), &uhave)?;
        std::fs::write(paths.f(&use_file(l)), vec![0u8; uvlist.len() * 4])?;
        write_lazy_stamp(
            &paths.f(&fmt_file(l)),
            (ubhead.len() - 1) as u64,
            ublist.len() as u64,
            uacc,
        )?;
    }

    Ok(SplitStats {
        regions_before: n_before,
        regions_after: n_after,
        rows_kept: kept_rows,
        rows_dropped: dropped_rows,
        gates_before,
        gates_after,
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// Boundary and member lists for level `target`, under new assignments.
///
/// `ofs[i]` is the new `of` array for level `base + i`; levels below `base` are
/// unchanged and read from `ov`.
#[allow(clippy::too_many_arguments)]
fn rebuild_index_at(
    ds: &Dataset,
    ov: &Overlay,
    base: usize,
    target: usize,
    ofs: &[Vec<u32>],
    ncells: usize,
    // The boundary lists already rebuilt for levels `base..target`. A level's
    // members are the level below's boundary, and below `target` that boundary
    // is exactly what a split has just changed — reading it from `ov` would
    // build the new level out of the old one's gates.
    fresh: &[(Vec<u32>, Vec<u32>)],
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let k = target;
    // The unit a vertex belongs to at `target`, walking up through whichever
    // assignment applies at each rung.
    let at = |v: u32| -> u32 {
        let mut c = ov.cell(v);
        if base == 0 {
            c = ofs[0][v as usize];
        }
        for l in 1..=k {
            c = if l >= base { ofs[l - base][c as usize] } else { ov.level(l).parent(c) };
        }
        c
    };
    // Members: at level 0 every vertex, above it the level below's boundary.
    let mut members: Vec<(u32, u32)> = Vec::new();
    if k == 0 {
        for v in 0..ds.n_vertices() as u32 {
            members.push((at(v), v));
        }
    } else if k > base {
        // The level below was rebuilt in this same pass; use its new gates.
        let (_, blist) = &fresh[k - 1 - base];
        for &u in blist {
            members.push((at(u), u));
        }
    } else {
        for c in 0..ov.level(k - 1).regions() as u32 {
            for &u in ov.level(k - 1).boundary(c) {
                members.push((at(u), u));
            }
        }
    }
    members.sort_unstable();
    let mut vhead = vec![0u32; ncells + 1];
    for &(c, _) in &members {
        vhead[c as usize + 1] += 1;
    }
    for i in 1..=ncells {
        vhead[i] += vhead[i - 1];
    }
    let vlist: Vec<u32> = members.iter().map(|&(_, u)| u).collect();

    // Boundary: a member with an edge leaving its region, either direction.
    let mut bnd: Vec<(u32, u32)> = Vec::new();
    for &(c, u) in &members {
        let mut out = false;
        for (head, fwd) in [(ds.head, true), (ds.rhead, false)] {
            let (a, b) = (head[u as usize] as usize, head[u as usize + 1] as usize);
            for j in a..b {
                let v = if fwd { ds.target(u, j) } else { ds.rtarget(u, j) };
                if at(v) != c {
                    out = true;
                    break;
                }
            }
            if out {
                break;
            }
        }
        if out {
            bnd.push((c, u));
        }
    }
    bnd.sort_unstable();
    let mut bhead = vec![0u32; ncells + 1];
    for &(c, _) in &bnd {
        bhead[c as usize + 1] += 1;
    }
    for i in 1..=ncells {
        bhead[i] += bhead[i - 1];
    }
    let blist: Vec<u32> = bnd.iter().map(|&(_, u)| u).collect();
    (bhead, blist, vhead, vlist)
}

/// Split the worst regions in half, by a breadth-first sweep over their members.
///
/// Placement is not the point here — a cut chosen by betweenness belongs to a
/// later stage — but a *connected* half is, because a region cut into two
/// disconnected pieces makes every path between them leave and re-enter, which
/// would flatter the gate count while making the routing worse. Growing one
/// half by BFS keeps both halves connected wherever the region itself is.
pub fn plan_splits(ds: &Dataset, ov: &Overlay, k: usize, max_gates: usize, limit: usize) -> SplitPlan {
    let lvl = ov.level(k);
    let mut worst: Vec<(usize, u32)> = (0..lvl.regions() as u32)
        .map(|c| (lvl.boundary(c).len(), c))
        .filter(|&(b, _)| b > max_gates)
        .collect();
    worst.sort_unstable_by_key(|&(b, _)| std::cmp::Reverse(b));
    worst.truncate(limit);

    let mut cuts = Vec::new();
    for (_, c) in worst {
        // Members of the region, in whatever form this level addresses them.
        let units: Vec<u32> = if k == 0 {
            let mut v: Vec<u32> = Vec::new();
            for (u, &x) in lvl.of.iter().enumerate() {
                if x == c {
                    v.push(u as u32);
                }
            }
            v
        } else {
            (0..lvl.of.len() as u32).filter(|&x| lvl.of[x as usize] == c).collect()
        };
        if units.len() < 4 {
            continue;
        }
        let half = units.len() / 2;
        let index: std::collections::HashSet<u32> = units.iter().copied().collect();
        // BFS from the first unit. At level 0 a unit is a vertex and adjacency
        // is the road graph; above it a unit is a region and two are adjacent
        // when a cut edge joins them, which is what `boundary` already knows.
        let mut seen: std::collections::HashSet<u32> = Default::default();
        let mut q: std::collections::VecDeque<u32> = Default::default();
        q.push_back(units[0]);
        seen.insert(units[0]);
        while let Some(x) = q.pop_front() {
            if seen.len() >= half {
                break;
            }
            let verts: Vec<u32> = if k == 0 {
                vec![x]
            } else {
                ov.level(k - 1).boundary(x).to_vec()
            };
            for u in verts {
                for (head, fwd) in [(ds.head, true), (ds.rhead, false)] {
                    let (a, b) = (head[u as usize] as usize, head[u as usize + 1] as usize);
                    for j in a..b {
                        let v = if fwd { ds.target(u, j) } else { ds.rtarget(u, j) };
                        let unit = if k == 0 {
                            v
                        } else {
                            let mut cc = ov.cell(v);
                            for l in 1..k {
                                cc = ov.level(l).parent(cc);
                            }
                            cc
                        };
                        if index.contains(&unit) && seen.insert(unit) {
                            q.push_back(unit);
                            if seen.len() >= half {
                                break;
                            }
                        }
                    }
                }
            }
        }
        if seen.len() > 1 && seen.len() < units.len() {
            cuts.push((c, seen.into_iter().collect()));
        }
    }
    SplitPlan { cuts }
}


/// Choose splits by cost, and place the cut where the traffic is not.
///
/// Two decisions, and they are separate.
///
/// **Which region.** A level's cost is `Σb²` on disk and roughly `Σb²` in the
/// work a query does, so a region's contribution is `b²` and the worst are
/// worth splitting first. On the planet's level 2 a thousand regions of 1.12
/// million carry every bit of the work.
///
/// **Where to cut.** This is what a plain bisection gets wrong: dividing a
/// region in half by breadth-first sweep raised the gate count by 18.9 %,
/// because the cut fell wherever the frontier happened to be — straight
/// through corridors. Every member the cut crosses becomes a new gate, and a
/// gate costs `b²`.
///
/// So the cut follows two signals at once. It grows a half greedily, always
/// taking the member that adds fewest new cut edges, and breaks ties toward
/// members the region's own shortest-path trees rarely run through — the
/// betweenness gathered while filling rows. Low traffic is where a cut is
/// cheap; high traffic is a corridor and cutting it makes every route through
/// it a gate.
pub fn plan_splits_by_cost(
    ds: &Dataset,
    ov: &Overlay,
    k: usize,
    max_gates: usize,
    limit: usize,
) -> SplitPlan {
    let lvl = ov.level(k);
    let mut worst: Vec<(usize, u32)> = (0..lvl.regions() as u32)
        .map(|c| (lvl.boundary(c).len(), c))
        .filter(|&(b, _)| b > max_gates)
        .collect();
    worst.sort_unstable_by_key(|&(b, _)| std::cmp::Reverse(b));
    worst.truncate(limit);

    let uses: &[u32] = match &lvl.values {
        Values::Lazy(z) if z.counts_use() => z.use_ct_all(),
        _ => &[],
    };

    let mut cuts = Vec::new();
    for (_, c) in worst {
        let units: Vec<u32> = (0..lvl.of.len() as u32)
            .filter(|&x| lvl.of[x as usize] == c)
            .collect();
        if units.len() < 4 {
            continue;
        }
        // Adjacency between units, and each unit's traffic.
        let index: std::collections::HashMap<u32, usize> =
            units.iter().enumerate().map(|(i, &u)| (u, i)).collect();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); units.len()];
        let mut traffic: Vec<u64> = vec![0; units.len()];
        for (i, &x) in units.iter().enumerate() {
            let verts: Vec<u32> =
                if k == 0 { vec![x] } else { ov.level(k - 1).boundary(x).to_vec() };
            for u in &verts {
                if !uses.is_empty() {
                    if let Some(p) = member_slot(ov, k, c, *u) {
                        traffic[i] += uses[p] as u64;
                    }
                }
                for (head, fwd) in [(ds.head, true), (ds.rhead, false)] {
                    let (a, b) = (head[*u as usize] as usize, head[*u as usize + 1] as usize);
                    for j in a..b {
                        let v = if fwd { ds.target(*u, j) } else { ds.rtarget(*u, j) };
                        let unit = unit_of(ov, k, v);
                        if let Some(&t) = index.get(&unit) {
                            if t != i {
                                adj[i].push(t);
                            }
                        }
                    }
                }
            }
        }

        // Grow one half. `gain` is how many of a unit's edges already point
        // inside — taking the highest keeps the cut small — and traffic breaks
        // the tie downward, so the boundary settles on quiet ground.
        let half = units.len() / 2;
        let mut inside = vec![false; units.len()];
        let mut gain = vec![0i64; units.len()];
        let seed = (0..units.len()).min_by_key(|&i| traffic[i]).unwrap_or(0);
        inside[seed] = true;
        let mut taken = 1usize;
        for &t in &adj[seed] {
            gain[t] += 1;
        }
        while taken < half {
            let mut best: Option<usize> = None;
            for i in 0..units.len() {
                if inside[i] || gain[i] == 0 {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some(b) => {
                        (gain[i], std::cmp::Reverse(traffic[i]))
                            > (gain[b], std::cmp::Reverse(traffic[b]))
                    }
                };
                if better {
                    best = Some(i);
                }
            }
            let Some(i) = best else { break };
            inside[i] = true;
            taken += 1;
            for &t in &adj[i] {
                gain[t] += 1;
            }
        }
        if taken > 1 && taken < units.len() {
            cuts.push((c, units.iter().enumerate().filter(|&(i, _)| inside[i]).map(|(_, &u)| u).collect()));
        }
    }
    SplitPlan { cuts }
}

/// The unit of level `k` a graph vertex belongs to.
fn unit_of(ov: &Overlay, k: usize, v: u32) -> u32 {
    let mut c = ov.cell(v);
    for l in 1..k {
        c = ov.level(l).parent(c);
    }
    if k == 0 {
        v
    } else {
        c
    }
}

/// Where a member sits in a region's use counter.
fn member_slot(ov: &Overlay, k: usize, c: u32, u: u32) -> Option<usize> {
    let Values::Lazy(z) = &ov.level(k).values else { return None };
    let (a, b) = (z.vhead[c as usize] as usize, z.vhead[c as usize + 1] as usize);
    z.vlist[a..b].binary_search(&u).ok().map(|i| a + i)
}

#[cfg(test)]
mod fill_tests {
    use super::fill_estimate;

    /// The estimate is calibrated, so it has to keep matching what was
    /// measured — a refusal threshold set against numbers that have quietly
    /// drifted would either build a level nobody can warm or refuse one that
    /// is fine.
    ///
    /// The planet's three rungs, as measured while warming them:
    ///
    /// | level | Σ(rows × members) | boundary per region below | measured |
    /// |---|---|---|---|
    /// | 1 | 4.93e9 | 5.2 | 13 min |
    /// | 2 | 9.32e9 | 856 | 28 h |
    /// | 3 | 7.24e9 | 4500 | 190 h |
    #[test]
    fn the_estimate_stays_within_an_order_of_magnitude_of_the_planet() {
        // One region holding all the work reproduces a given Σ(rows×members).
        let check = |work: f64, b_below: f64, measured_h: f64, name: &str| {
            let rows = 1_000u32;
            let members = (work / rows as f64) as u32;
            let bhead = [0u32, rows];
            let vhead = [0u32, members];
            let got = fill_estimate(&bhead, &vhead, b_below) / 3600.0;
            let ratio = got / measured_h;
            assert!(
                (0.2..5.0).contains(&ratio),
                "{name}: estimated {got:.2} h against {measured_h:.2} h measured \
                 ({ratio:.2}x) — the calibration has drifted"
            );
        };
        check(4.93e9, 5.2, 13.0 / 60.0, "level 1");
        check(9.32e9, 856.0, 28.0, "level 2");
        check(7.24e9, 4500.0, 190.0, "level 3");
    }

    #[test]
    fn an_empty_level_costs_nothing_rather_than_dividing_by_zero() {
        assert_eq!(fill_estimate(&[0u32], &[0u32], 0.0), 0.0);
        assert_eq!(fill_estimate(&[0u32, 0], &[0u32, 0], 5.0), 0.0);
    }
}
