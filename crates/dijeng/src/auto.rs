//! Adaptive SSSP — picks an algorithm automatically based on graph properties.
//!
//! Heuristic:
//!   * avg_deg < 1.5 (path graphs, trees): 4-ary heap Dijeng. Bucket
//!     algorithms have too much overhead per node when each bucket has only
//!     one vertex.
//!   * 1.5 ≤ avg_deg < 4 (road networks, sparse trees): delta_stepping.
//!     Robust to variable edge weights (light/heavy split). The Duan variant
//!     has a known correctness bug on some road network graphs.
//!   * avg_deg ≥ 4 (dense random/power-law/grid): duan_inspired. Best
//!     performance on regular graphs with uniformly distributed weights.

use crate::delta_step::delta_stepping;
use crate::dijeng::dijeng_4ary;
use crate::duan::duan_inspired;
use crate::graph::CsrGraph;

pub fn sssp_auto(g: &CsrGraph, src: u32) -> Vec<f32> {
    let n = g.n;
    let m = g.m();
    let avg_deg = m as f32 / n.max(1) as f32;

    if avg_deg < 1.5 {
        return dijeng_4ary(g, src);
    }

    // Estimate avg edge weight via stride sampling — the first edges in CSR
    // are often from a local area and not representative.
    let target_sample = 4096usize;
    let stride = (m / target_sample).max(1);
    let mut s = 0.0f64;
    let mut c = 0u64;
    let mut i = 0usize;
    while i < m {
        s += g.edge_w[i] as f64;
        c += 1;
        i += stride;
    }
    let avg_w: f32 = if c == 0 { 1.0 } else { (s / c as f64) as f32 };
    let delta = (avg_w / avg_deg).max(1e-4);

    if avg_deg < 4.0 {
        // Robust choice for road networks and similar sparse graphs.
        return delta_stepping(g, src, delta);
    }

    duan_inspired(g, src, 4.0 * delta)
}
