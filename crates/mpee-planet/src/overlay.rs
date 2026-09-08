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

/// Unreachable, in the same units the router uses (tenths of a second).
pub const UNREACHABLE: u32 = u32::MAX;

#[inline]
fn dur_ds(len_cm: u32, kmh: u16) -> u32 {
    ((len_cm as u64 * 36) / (kmh.max(1) as u64 * 100)).min(u32::MAX as u64) as u32
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
pub struct Level {
    /// Maps a region of the level below to a region of this one. At level 0
    /// the "level below" is the road graph itself, so this maps a vertex.
    of: &'static [u32],
    bnd: &'static [u32],
    bhead: &'static [u32],
    head: &'static [u64],
    width: &'static [u8],
    mat: &'static [u8],
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

    #[inline]
    pub fn bindex(&self, c: u32, v: u32) -> Option<usize> {
        self.boundary(c).binary_search(&v).ok()
    }

    /// Shortest time from the `i`-th to the `j`-th boundary vertex of region
    /// `c`, using only paths inside it.
    ///
    /// The width branch is constant for a region, so walking one row predicts
    /// perfectly — and nothing is decompressed, only read at its own size.
    #[inline(always)]
    pub fn cost(&self, c: u32, i: usize, j: usize) -> u32 {
        let b = self.boundary(c).len();
        let base = self.head[c as usize] as usize;
        let k = i * b + j;
        if self.width[c as usize] == 2 {
            let p = base + k * 2;
            let v = u16::from_le_bytes([self.mat[p], self.mat[p + 1]]);
            if v == 0xFFFF {
                UNREACHABLE
            } else {
                v as u32
            }
        } else {
            let p = base + k * 4;
            u32::from_le_bytes([self.mat[p], self.mat[p + 1], self.mat[p + 2], self.mat[p + 3]])
        }
    }

    pub fn boundary_total(&self) -> usize {
        self.bnd.len()
    }

    /// Bytes per entry in a region's table: 2 when its widest finite crossing
    /// fit 16 bits, 4 otherwise.
    #[inline]
    pub fn width_of(&self, c: u32) -> u8 {
        self.width[c as usize]
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

/// How many levels a dataset could ever carry. Each one is coarser than the
/// last by a large factor, so the ladder is short even for a planet.
pub const MAX_LEVELS: usize = 8;

pub struct Overlay {
    _maps: Vec<memmap2::Mmap>,
    levels: Vec<Level>,
    pub bounds: &'static [(u32, u32)],
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
                let (Some(of), Some(bnd), Some(bhead), Some(head), Some(mat), Some(width)) = (
                    u32s(&f[0], &mut keep),
                    u32s(&f[1], &mut keep),
                    u32s(&f[2], &mut keep),
                    u64s(&f[3], &mut keep),
                    bytes(&f[4], &mut keep),
                    bytes(&f[5], &mut keep),
                ) else {
                    break;
                };
                levels.push(Level { of, bnd, bhead, head, width, mat });
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
    fn local(
        &mut self,
        ds: &Dataset,
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
                self.lstate.relax(v, d.saturating_add(edge_ds(ds, sid)), u);
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
        self.seed_and_run(ds, ov, &[(s, 0)], &[(t, 0)]).map(|(c, _)| c)
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
        let entries: Vec<(u32, u32)> = seg_exits(ds, from, None)
            .into_iter()
            .map(|(v, cm)| (v, dur_ds(cm as u32, attr_kmh(ds.attr(from.seg as usize)))))
            .collect();
        let exits: Vec<(u32, u32)> = seg_entries(ds, to, None)
            .into_iter()
            .map(|(v, cm)| (v, dur_ds(cm as u32, attr_kmh(ds.attr(to.seg as usize)))))
            .collect();
        self.seed_and_run(ds, ov, &entries, &exits)
    }

    fn seed_and_run(
        &mut self,
        ds: &Dataset,
        ov: &Overlay,
        entries: &[(u32, u32)],
        exits: &[(u32, u32)],
    ) -> Option<(u32, Vec<u32>)> {
        self.state[0].clear();
        self.state[1].clear();
        self.settled = 0;
        self.regions_entered = 0;
        self.regions2_entered = 0;
        self.over_budget = false;
        let mut out: Vec<(u32, u32)> = Vec::new();
        for (side, srcs, fwd) in [(0usize, entries, true), (1, exits, false)] {
            for &(v, c0) in srcs {
                let c = ov.cell(v);
                self.local(ds, ov, c, &[(v, c0)], fwd, &mut out);
                self.regions_entered += 1;
                for &(b, d) in out.iter() {
                    self.state[side].relax(b, d, u32::MAX);
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
        let nlev = if self.use_l2 { ov.levels() } else { 1 };
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
            if tf.saturating_add(tb) >= best {
                break;
            }
            let side = if tf <= tb { 0usize } else { 1 };
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
            // Climb as high as the ladder allows. Use the coarsest level whose
            // region holds neither endpoint and on whose boundary `u` sits:
            // higher means one table read crosses more of the world. Two rules
            // stop the climb, and both are one-way — a region holding an
            // endpoint needs the detail a coarser table has collapsed away,
            // and a vertex interior to a region at one level is interior at
            // every level above it, because boundary sets only shrink upward.
            let mut lc = ov.cell(u);
            let mut lv = 0usize;
            let mut li = ov.bindex(lc, u);
            {
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
            }
            let (head, fwd) = if side == 0 { (ds.head, true) } else { (ds.rhead, false) };
            let (a, e) = (head[u as usize] as usize, head[u as usize + 1] as usize);
            if lv > 0 {
                self.regions2_entered += 1;
            }
            // 1. across the chosen region, boundary to boundary, from its table.
            if let Some(i) = li {
                let lvl = ov.level(lv);
                let bl = lvl.boundary(lc);
                for (j, &w) in bl.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    // The table is directional: the forward search reads row
                    // i, the backward one reads column i.
                    let step = if side == 0 { lvl.cost(lc, i, j) } else { lvl.cost(lc, j, i) };
                    if step == UNREACHABLE {
                        continue;
                    }
                    let nd = d.saturating_add(step);
                    if nd < best {
                        self.state[side].relax(w, nd, u);
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
                let nd = d.saturating_add(edge_ds(ds, sid));
                if nd < best {
                    self.state[side].relax(v, nd, u);
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
            let nd = d.saturating_add(edge_ds(ds, sid));
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

pub struct L2Stats {
    pub cells: usize,
    pub boundary: usize,
    pub entries: u64,
    pub secs: f64,
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
                let nd = d.saturating_add(edge_ds(ds, sid));
                relax(v, nd, &mut dist, &mut touched, &mut heap);
            }
        }
        for (j, &dst) in bnd.iter().enumerate() {
            out[i * b + j] = local(dst).map(|x| dist[x]).unwrap_or(UNREACHABLE);
        }
    }
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
    ov: &Overlay,
    k: usize,
    target: usize,
) -> io::Result<L2Stats> {
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
    for c in 0..nprev as u32 {
        for &u in ov.level(below).boundary(c) {
            for (head, fwd) in [(ds.head, true), (ds.rhead, false)] {
                let (a, b) = (head[u as usize] as usize, head[u as usize + 1] as usize);
                for j in a..b {
                    let v = if fwd { ds.target(u, j) } else { ds.rtarget(u, j) };
                    let d = ov.cell_at(below, v);
                    if d != c {
                        pairs.push((c, d));
                        pairs.push((d, c));
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
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
        ohead[c + 1] = ohead[c] + b * b;
    }
    let entries = ohead[ncells];
    eprintln!(
        "[level {k}] {nb} boundary vertices ({:.2} % of the level below's {}), \
         {entries} table entries ({:.2} GB at 32 bits)",
        nb as f64 / ov.level(below).boundary_total() as f64 * 100.0,
        ov.level(below).boundary_total(),
        entries as f64 * 4.0 / 1e9
    );

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
        region_table(ds, ov, below, members, &blist[bs..be], &mut out);
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
    Ok(L2Stats { cells: ncells, boundary: nb, entries, secs: t0.elapsed().as_secs_f64() })
}
