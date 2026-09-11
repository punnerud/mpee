//! Pass 3 — junction contraction.
//!
//! This is the step that turns "a map" into "a distance graph". Every OSM way
//! is cut at its junction nodes; the run of shape points between two junctions
//! becomes one **segment** carrying
//!
//!   * its exact length, integrated over the *full-resolution* geometry in
//!     `f64` and stored once as centimetres — so a route's distance is an
//!     integer sum with no float drift and no dependence on how coarsely we
//!     later draw it;
//!   * a simplified polyline for drawing, stored as raw `i32` coordinate
//!     pairs — the whole dataset fits the budget uncompressed, so drawing a
//!     route is an mmap slice, not a varint decode;
//!   * the way attributes (class, speed, oneway, toll).
//!
//! Dropping the shape points from the *graph* is what makes the planet fit:
//! Norway goes from 15.5 M road nodes to 1.9 M vertices, and the length is
//! unaffected because it was measured before the simplification.

use crate::bitmap::BitMap;
use crate::build::{attr_oneway, Paths};
use crate::geo;
use crate::mmapvec;
use crate::varint::{get_bytes, get_i, get_u};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub struct ContractCfg {
    /// Douglas–Peucker tolerance for the drawn line, in metres.
    pub simplify_m: f64,
}

impl Default for ContractCfg {
    fn default() -> Self {
        ContractCfg { simplify_m: 3.0 }
    }
}

/// One contracted segment, as produced by a worker thread.
struct Seg {
    u_node: u64,
    v_node: u64,
    attr: u32,
    len_cm: u32,
    /// Intermediate points of the drawn line (endpoints excluded — those are
    /// the two vertices and are already in the coordinate table).
    geom: Vec<(i32, i32)>,
    raw_pts: u32,
    name: Vec<u8>,
}

pub struct ContractOut {
    pub segments: u64,
    pub directed_edges: u64,
    pub geom_points: u64,
    pub names: u64,
    pub dropped_incomplete: u64,
    /// Full-resolution shape points seen, for the retention ratio.
    pub raw_points: u64,
}

const MISSING: i32 = i32::MIN;

pub fn pass3(
    paths: &Paths,
    need: &BitMap,
    junc: &BitMap,
    cfg: &ContractCfg,
) -> io::Result<ContractOut> {
    let coords_map = mmapvec::open(&paths.f("coords.bin"))?;
    let coords: &[(i32, i32)] = unsafe { mmapvec::as_slice(&coords_map[..]) };
    let ways_map = mmapvec::open(&paths.f("ways.bin"))?;
    let ways: &[u8] = &ways_map[..];

    let mut w_len = BufWriter::with_capacity(1 << 22, File::create(paths.f("seg.len"))?);
    let mut w_attr = BufWriter::with_capacity(1 << 22, File::create(paths.f("seg.attr"))?);
    let mut w_name = BufWriter::with_capacity(1 << 22, File::create(paths.f("seg.name"))?);
    let mut w_geom = BufWriter::with_capacity(1 << 23, File::create(paths.f("geom.pts"))?);
    let mut w_gidx = BufWriter::with_capacity(1 << 22, File::create(paths.f("geom.off"))?);
    let mut w_edge = BufWriter::with_capacity(1 << 23, File::create(paths.f("edges.bin"))?);
    let mut w_pool = BufWriter::with_capacity(1 << 22, File::create(paths.f("name.pool"))?);
    let mut w_poff = BufWriter::with_capacity(1 << 20, File::create(paths.f("name.off"))?);

    // The way index: which OSM way each run of segments came from, and what it
    // looked like. Segments from one way are produced consecutively, so a
    // cumulative count is enough to address them — no per-segment back
    // pointer, and no sort.
    let mut w_wid = BufWriter::with_capacity(1 << 22, File::create(paths.f("way.id"))?);
    let mut w_whash = BufWriter::with_capacity(1 << 22, File::create(paths.f("way.hash"))?);
    let mut w_wtopo = BufWriter::with_capacity(1 << 22, File::create(paths.f("way.topo"))?);
    let mut w_whead = BufWriter::with_capacity(1 << 22, File::create(paths.f("way.head"))?);
    w_whead.write_all(&0u32.to_le_bytes())?;

    let mut names: HashMap<Vec<u8>, u32> = HashMap::with_capacity(1 << 20);
    let mut name_bytes = 0u64;
    w_poff.write_all(&0u64.to_le_bytes())?;

    let mut out = ContractOut {
        segments: 0,
        directed_edges: 0,
        geom_points: 0,
        names: 0,
        dropped_incomplete: 0,
        raw_points: 0,
    };

    // Slice the way stream into batches so the expensive part (length
    // integration + Douglas–Peucker over ~1.15 billion points on a planet)
    // runs on every core, while id assignment and name interning stay
    // sequential and therefore deterministic.
    const BATCH: usize = 200_000;
    let mut p = 0usize;
    let mut batch: Vec<(usize, usize)> = Vec::with_capacity(BATCH);
    let total = ways.len();
    let t0 = std::time::Instant::now();
    let mut last_report = std::time::Instant::now();

    while p < total {
        batch.clear();
        while p < total && batch.len() < BATCH {
            let start = p;
            skip_way(ways, &mut p);
            batch.push((start, p));
        }
        let produced: Vec<Vec<Seg>> = batch
            .par_iter()
            .map(|&(s, e)| contract_way(&ways[s..e], coords, need, junc, cfg))
            .collect();

        for (bi, segs) in produced.into_iter().enumerate() {
            let (wid, wattr, wtopo) = crate::build::way_hashes(&ways[batch[bi].0..batch[bi].1]);
            w_wid.write_all(&wid.to_le_bytes())?;
            w_whash.write_all(&wattr.to_le_bytes())?;
            w_wtopo.write_all(&wtopo.to_le_bytes())?;
            for sg in segs {
                if sg.u_node == u64::MAX {
                    out.dropped_incomplete += 1;
                    continue;
                }
                let sid = out.segments as u32;
                // One offset per segment, in *points*: geometry of segment s is
                // pts[off[s] .. off[s+1]]. Random access with no scanning.
                w_gidx.write_all(&(out.geom_points as u32).to_le_bytes())?;
                w_len.write_all(&sg.len_cm.to_le_bytes())?;
                w_attr.write_all(&sg.attr.to_le_bytes())?;
                let nid = if sg.name.is_empty() {
                    u32::MAX
                } else if let Some(&id) = names.get(&sg.name) {
                    id
                } else {
                    let id = out.names as u32;
                    w_pool.write_all(&sg.name)?;
                    name_bytes += sg.name.len() as u64;
                    w_poff.write_all(&name_bytes.to_le_bytes())?;
                    names.insert(sg.name.clone(), id);
                    out.names += 1;
                    id
                };
                w_name.write_all(&nid.to_le_bytes())?;
                for &(la, lo) in &sg.geom {
                    w_geom.write_all(&la.to_le_bytes())?;
                    w_geom.write_all(&lo.to_le_bytes())?;
                }
                out.geom_points += sg.geom.len() as u64;
                out.raw_points += sg.raw_pts as u64;

                let u = junc.rank1(sg.u_node) as u32;
                let v = junc.rank1(sg.v_node) as u32;
                // dir bit 31 of the segment word: set when the edge runs
                // against the stored geometry direction.
                match attr_oneway(sg.attr) {
                    1 => {
                        write_edge(&mut w_edge, u, v, sid)?;
                        out.directed_edges += 1;
                    }
                    2 => {
                        write_edge(&mut w_edge, v, u, sid | 0x8000_0000)?;
                        out.directed_edges += 1;
                    }
                    _ => {
                        write_edge(&mut w_edge, u, v, sid)?;
                        write_edge(&mut w_edge, v, u, sid | 0x8000_0000)?;
                        out.directed_edges += 2;
                    }
                }
                out.segments += 1;
            }
            // One entry per way, after its segments: `head[w]..head[w+1]` is
            // the run this way produced. Ways that produced none — dropped for
            // missing coordinates — get an empty run rather than disappearing,
            // so the index stays aligned with `way.id`.
            w_whead.write_all(&(out.segments as u32).to_le_bytes())?;
        }
        if last_report.elapsed().as_secs() >= 10 {
            last_report = std::time::Instant::now();
            let frac = p as f64 / total as f64;
            let el = t0.elapsed().as_secs_f64();
            eprintln!(
                "  [pass3] {:.1} % — {} segments, {} edges, {:.2} GB geom — {:.0} s, ~{:.0} s left",
                frac * 100.0,
                out.segments,
                out.directed_edges,
                out.geom_points as f64 * 8.0 / 1e9,
                el,
                el / frac.max(1e-9) * (1.0 - frac)
            );
        }
    }
    // Trailing entry so the last segment has an end offset.
    w_gidx.write_all(&(out.geom_points as u32).to_le_bytes())?;

    for w in [
        &mut w_len as &mut dyn Write,
        &mut w_attr,
        &mut w_name,
        &mut w_edge,
        &mut w_wid,
        &mut w_whash,
        &mut w_wtopo,
        &mut w_whead,
    ] {
        w.flush()?;
    }
    w_geom.flush()?;
    w_gidx.flush()?;
    w_pool.flush()?;
    w_poff.flush()?;
    eprintln!(
        "[pass3] {} segments, {} directed edges, {} street names ({:.1} s)",
        out.segments,
        out.directed_edges,
        out.names,
        t0.elapsed().as_secs_f64()
    );
    eprintln!(
        "[pass3] geometry: {} of {} shape points kept at {} m ({:.1} %) = {:.2} GB + {:.2} GB index",
        out.geom_points,
        out.raw_points,
        cfg.simplify_m,
        out.geom_points as f64 / out.raw_points.max(1) as f64 * 100.0,
        out.geom_points as f64 * 8.0 / 1e9,
        (out.segments + 1) as f64 * 4.0 / 1e9
    );
    if out.dropped_incomplete > 0 {
        eprintln!(
            "[pass3] {} segments dropped for missing node coordinates (normal in a cut extract)",
            out.dropped_incomplete
        );
    }
    // Last, after the index is whole: a stamp saying which revision its hashes
    // are in, so a reader never has to guess.
    // The way count comes from the index itself rather than a counter, so the
    // stamp can never disagree with what was written.
    let n_ways = std::fs::metadata(paths.f("way.id")).map(|m| m.len() / 8).unwrap_or(0);
    crate::build::write_way_stamp(&paths.f(crate::build::way_fmt_file()), n_ways)?;
    Ok(out)
}

#[inline]
fn write_edge(w: &mut impl Write, u: u32, v: u32, seg: u32) -> io::Result<()> {
    w.write_all(&u.to_le_bytes())?;
    w.write_all(&v.to_le_bytes())?;
    w.write_all(&seg.to_le_bytes())
}

/// Advance `p` past one way record without decoding it.
fn skip_way(b: &[u8], p: &mut usize) {
    get_u(b, p); // id
    get_u(b, p); // attr
    let l = get_u(b, p) as usize;
    *p += l; // name
    let l = get_u(b, p) as usize;
    *p += l; // ref
    let n = get_u(b, p) as usize;
    for _ in 0..n {
        get_u(b, p);
    }
}

fn contract_way(
    rec: &[u8],
    coords: &[(i32, i32)],
    need: &BitMap,
    junc: &BitMap,
    cfg: &ContractCfg,
) -> Vec<Seg> {
    let mut p = 0usize;
    let _id = get_u(rec, &mut p);
    let attr = get_u(rec, &mut p) as u32;
    let name = get_bytes(rec, &mut p);
    let rref = get_bytes(rec, &mut p);
    // A motorway carries `ref=E6` and usually no `name` at all, so storing the
    // name alone leaves the longest stretches of a route blank — and makes
    // them unnameable in a local override. Falling back to the ref costs
    // nothing (it shares the same pool) and is how people refer to these roads
    // anyway.
    let name = if name.is_empty() { rref.to_vec() } else { name.to_vec() };
    let n = get_u(rec, &mut p) as usize;
    let mut refs: Vec<u64> = Vec::with_capacity(n);
    let mut acc = 0i64;
    for _ in 0..n {
        acc += get_i(rec, &mut p);
        refs.push(acc as u64);
    }

    let mut out = Vec::new();
    let mut pts: Vec<(i32, i32)> = Vec::with_capacity(64);
    let mut keep: Vec<u32> = Vec::new();
    let mut start = 0usize;
    let mut i = 1usize;
    while i < refs.len() {
        let is_junction = junc.get(refs[i]) || i == refs.len() - 1;
        if !is_junction {
            i += 1;
            continue;
        }
        // Materialise this run's coordinates.
        pts.clear();
        let mut complete = true;
        for &r in &refs[start..=i] {
            if !need.get(r) {
                complete = false;
                break;
            }
            let c = coords[need.rank1(r) as usize];
            if c.0 == MISSING || (c.0 == 0 && c.1 == 0) {
                complete = false;
                break;
            }
            pts.push(c);
        }
        if !complete || pts.len() < 2 {
            out.push(incomplete());
        } else if refs[start] == refs[i] {
            // A closed run returns to its own vertex; it can never be on a
            // shortest path, so it is not worth an edge.
        } else {
            // Exact length first, on every original point.
            let len_m = geo::polyline_len_m(&pts);
            let len_cm = (len_m * 100.0).round().clamp(1.0, u32::MAX as f64) as u32;
            // Then the cheap drawing copy.
            geo::simplify(&pts, cfg.simplify_m, &mut keep);
            let geom: Vec<(i32, i32)> = keep
                .iter()
                .copied()
                .filter(|&k| k != 0 && k as usize != pts.len() - 1)
                .map(|k| pts[k as usize])
                .collect();
            out.push(Seg {
                u_node: refs[start],
                v_node: refs[i],
                attr,
                len_cm,
                geom,
                raw_pts: (pts.len() - 2) as u32,
                name: name.clone(),
            });
        }
        start = i;
        i += 1;
    }
    out
}

fn incomplete() -> Seg {
    Seg {
        u_node: u64::MAX,
        v_node: 0,
        attr: 0,
        len_cm: 0,
        geom: Vec::new(),
        raw_pts: 0,
        name: Vec::new(),
    }
}

pub fn read_u64s(path: &Path) -> io::Result<Vec<u64>> {
    let raw = std::fs::read(path)?;
    Ok(raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn geometry_offsets_delimit_each_segment() {
        // off[s]..off[s+1] must carve the point array into the per-segment runs.
        let counts = [3usize, 0, 5, 1];
        let mut off = Vec::new();
        let mut acc = 0u32;
        for &c in &counts {
            off.push(acc);
            acc += c as u32;
        }
        off.push(acc);
        assert_eq!(off, vec![0, 3, 3, 8, 9]);
        for (s, &c) in counts.iter().enumerate() {
            assert_eq!((off[s + 1] - off[s]) as usize, c);
        }
    }
}
