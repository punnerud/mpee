//! Isochrones / isodistances: the area reachable from a point within a time
//! (or distance) budget, as GeoJSON-ready polygons.
//!
//! Pipeline:
//!   1. Bounded one-to-all Dijkstra from the snapped origin over the CH's
//!      augmented graph. ALL edges are relaxed (originals + shortcuts —
//!      shortcuts are never shorter than the paths they stand for, and every
//!      original edge is present, so labels equal base-graph distances) and
//!      the frontier stops at the largest requested limit.
//!   2. Rasterise reached nodes into a small lat/lon grid (cell ≈ `cell_deg`).
//!   3. March the binary mask into closed rings (outer boundaries of the
//!      filled cell regions) per contour value.
//!
//! The polygons are cell-resolution approximations — the same trade Valhalla
//! makes (their default `generalize` smooths a grid contour too). Choose the
//! cell size by zoom level; 0.0015° ≈ 150 m works for city-scale displays.

use crate::ch::ContractionHierarchy;

/// One contour band: every point of `rings` is reachable within `limit`.
/// Rings are closed (first == last), in (lat, lon).
pub struct IsochroneBand {
    pub limit: f32,
    pub rings: Vec<Vec<(f32, f32)>>,
}

/// Compute isochrone bands from `src_internal` (CH-internal id). `limits`
/// must be ascending; `metric_dist` switches the budget from seconds to
/// metres (requires a dual-channel CH; falls back to duration otherwise).
pub fn isochrone(
    ch: &ContractionHierarchy,
    coords_of_internal: impl Fn(u32) -> (f32, f32),
    src_internal: u32,
    limits: &[f32],
    cell_deg: f32,
    metric_dist: bool,
) -> Vec<IsochroneBand> {
    if limits.is_empty() {
        return Vec::new();
    }
    let max_limit = *limits.last().unwrap();
    let n = ch.graph_fwd.n;

    // ── 1. Bounded Dijkstra over all out-edges (originals + shortcuts). ──
    let use_dist = metric_dist && ch.edge_dist_fwd.len() == ch.graph_fwd.m();
    let mut cost: Vec<f32> = vec![f32::INFINITY; n];
    let mut heap: Vec<HItem> = Vec::with_capacity(4096);
    let mut reached: Vec<u32> = Vec::with_capacity(8192);
    cost[src_internal as usize] = 0.0;
    push(&mut heap, 0.0, src_internal);
    while let Some(HItem { d, v: u }) = pop(&mut heap) {
        if d > cost[u as usize] {
            continue;
        }
        reached.push(u);
        let s = ch.graph_fwd.head[u as usize] as usize;
        let e = ch.graph_fwd.head[u as usize + 1] as usize;
        for k in s..e {
            let w = ch.graph_fwd.edge_to[k];
            let step = if use_dist { ch.edge_dist_fwd[k] } else { ch.graph_fwd.edge_w[k] };
            let nd = d + step;
            if nd <= max_limit && nd < cost[w as usize] {
                cost[w as usize] = nd;
                push(&mut heap, nd, w);
            }
        }
    }

    // ── 2+3. Per contour: rasterise + trace rings. ──
    // Shared bounding box across contours (from the largest reach).
    let mut min_lat = f32::INFINITY;
    let mut max_lat = f32::NEG_INFINITY;
    let mut min_lon = f32::INFINITY;
    let mut max_lon = f32::NEG_INFINITY;
    for &u in &reached {
        let (la, lo) = coords_of_internal(u);
        min_lat = min_lat.min(la);
        max_lat = max_lat.max(la);
        min_lon = min_lon.min(lo);
        max_lon = max_lon.max(lo);
    }
    if !min_lat.is_finite() {
        return limits.iter().map(|&l| IsochroneBand { limit: l, rings: Vec::new() }).collect();
    }
    let cell = cell_deg.max(1e-5);
    // +3: one cell padding each side so rings never touch the array edge.
    let cols = (((max_lon - min_lon) / cell).ceil() as usize + 3).max(3);
    let rows = (((max_lat - min_lat) / cell).ceil() as usize + 3).max(3);

    limits
        .iter()
        .map(|&limit| {
            let mut mask = vec![false; rows * cols];
            for &u in &reached {
                if cost[u as usize] > limit {
                    continue;
                }
                let (la, lo) = coords_of_internal(u);
                let r = (((la - min_lat) / cell) as usize + 1).min(rows - 2);
                let c = (((lo - min_lon) / cell) as usize + 1).min(cols - 2);
                mask[r * cols + c] = true;
            }
            let rings_rc = trace_rings(&mask, rows, cols);
            let rings = rings_rc
                .into_iter()
                .map(|ring| {
                    ring.into_iter()
                        .map(|(r, c)| {
                            // Cell-corner grid coordinates → lat/lon. Corner
                            // (r, c) sits at the cell boundary lattice; undo
                            // the +1 padding offset.
                            (
                                min_lat + (r as f32 - 1.0) * cell,
                                min_lon + (c as f32 - 1.0) * cell,
                            )
                        })
                        .collect()
                })
                .collect();
            IsochroneBand { limit, rings }
        })
        .collect()
}

/// Trace the boundary rings of a binary cell mask. Walks the lattice of cell
/// corners; an edge between two corners is part of a ring when it separates a
/// filled cell from an empty one. Returns closed rings (first == last) in
/// (row, col) corner coordinates.
///
/// This is the segment-chaining form of marching squares restricted to a
/// binary field — exact for the mask (no interpolation needed).
fn trace_rings(mask: &[bool], rows: usize, cols: usize) -> Vec<Vec<(usize, usize)>> {
    let filled = |r: isize, c: isize| -> bool {
        r >= 0 && c >= 0 && (r as usize) < rows && (c as usize) < cols
            && mask[r as usize * cols + c as usize]
    };
    // Collect boundary segments as directed edges between corner points,
    // oriented so the filled cell is on the LEFT (counter-clockwise outer
    // rings, clockwise holes — GeoJSON consumers treat both fine for display).
    use std::collections::HashMap;
    let mut next: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    let mut n_segments = 0usize;
    for r in 0..rows as isize {
        for c in 0..cols as isize {
            if !filled(r, c) {
                continue;
            }
            let (ru, cu) = (r as usize, c as usize);
            // For each empty 4-neighbour, emit the separating corner edge.
            // Corner lattice: cell (r,c) has corners (r,c)..(r+1,c+1).
            if !filled(r - 1, c) {
                next.entry((ru, cu)).or_default().push((ru, cu + 1)); // bottom edge, →
                n_segments += 1;
            }
            if !filled(r + 1, c) {
                next.entry((ru + 1, cu + 1)).or_default().push((ru + 1, cu)); // top edge, ←
                n_segments += 1;
            }
            if !filled(r, c - 1) {
                next.entry((ru + 1, cu)).or_default().push((ru, cu)); // left edge, ↓
                n_segments += 1;
            }
            if !filled(r, c + 1) {
                next.entry((ru, cu + 1)).or_default().push((ru + 1, cu + 1)); // right edge, ↑
                n_segments += 1;
            }
        }
    }
    // Chain segments into rings.
    let mut rings = Vec::new();
    let mut used = 0usize;
    while used < n_segments {
        // Take any remaining start.
        let Some((&start, _)) = next.iter().find(|(_, v)| !v.is_empty()) else {
            break;
        };
        let mut ring = vec![start];
        let mut cur = start;
        loop {
            let Some(outs) = next.get_mut(&cur) else { break };
            let Some(nxt) = outs.pop() else { break };
            used += 1;
            ring.push(nxt);
            cur = nxt;
            if cur == start {
                break;
            }
        }
        if ring.len() >= 4 && ring.first() == ring.last() {
            rings.push(ring);
        }
    }
    rings
}

// Local 4-ary heap (same shape as ch.rs's — kept private there).
#[derive(Clone, Copy)]
struct HItem {
    d: f32,
    v: u32,
}

fn push(h: &mut Vec<HItem>, d: f32, v: u32) {
    let mut i = h.len();
    h.push(HItem { d, v });
    while i > 0 {
        let p = (i - 1) >> 2;
        if h[p].d <= h[i].d {
            break;
        }
        h.swap(p, i);
        i = p;
    }
}

fn pop(h: &mut Vec<HItem>) -> Option<HItem> {
    let n = h.len();
    if n == 0 {
        return None;
    }
    let top = h[0];
    let last = h.pop().unwrap();
    if n == 1 {
        return Some(top);
    }
    h[0] = last;
    let len = h.len();
    let mut i = 0usize;
    loop {
        let first = 4 * i + 1;
        if first >= len {
            break;
        }
        let mut s = first;
        let mut sd = h[first].d;
        for c in (first + 1)..(first + 4).min(len) {
            if h[c].d < sd {
                s = c;
                sd = h[c].d;
            }
        }
        if sd >= h[i].d {
            break;
        }
        h.swap(i, s);
        i = s;
    }
    Some(top)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_cell_ring() {
        // One filled cell → one square ring of 5 points (closed).
        let rows = 3;
        let cols = 3;
        let mut mask = vec![false; 9];
        mask[1 * 3 + 1] = true;
        let rings = trace_rings(&mask, rows, cols);
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0].len(), 5);
        assert_eq!(rings[0].first(), rings[0].last());
    }

    #[test]
    fn two_blobs_two_rings() {
        // Two diagonal cells (not 4-connected) → two rings.
        let rows = 4;
        let cols = 4;
        let mut mask = vec![false; 16];
        mask[1 * 4 + 1] = true;
        mask[2 * 4 + 2] = true;
        let rings = trace_rings(&mask, rows, cols);
        assert_eq!(rings.len(), 2);
    }

    #[test]
    fn rectangle_single_ring() {
        // A 2×3 solid block → one outer ring with 2*(2+3)+1 = 11 points.
        let rows = 4;
        let cols = 5;
        let mut mask = vec![false; rows * cols];
        for r in 1..3 {
            for c in 1..4 {
                mask[r * cols + c] = true;
            }
        }
        let rings = trace_rings(&mask, rows, cols);
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0].len(), 11);
    }

    #[test]
    fn donut_has_outer_and_hole() {
        // 3×3 block with the centre empty → outer ring + hole ring.
        let rows = 5;
        let cols = 5;
        let mut mask = vec![false; rows * cols];
        for r in 1..4 {
            for c in 1..4 {
                mask[r * cols + c] = true;
            }
        }
        mask[2 * cols + 2] = false;
        let rings = trace_rings(&mask, rows, cols);
        assert_eq!(rings.len(), 2);
    }
}
