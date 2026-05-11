//! Adaptiv SSSP — velger algoritme automatisk basert på graf-egenskaper.
//!
//! Heuristikk:
//!   * avg_deg < 1.5 (path-grafer, trær): 4-ary heap Dijkstra. Bucket-
//!     algoritmer har for mye overhead per node når hver bucket bare har
//!     én vertex.
//!   * 1.5 ≤ avg_deg < 4 (road networks, sparse trær): delta_stepping.
//!     Robust mot variable kantvekter (light/heavy split). Duan-varianten
//!     har en kjent korrekthetsbug på enkelte road-network-grafer.
//!   * avg_deg ≥ 4 (tette random/power-law/grid): duan_inspired. Best
//!     ytelse på regulære grafer med uniformt fordelte vekter.

use crate::delta_step::delta_stepping;
use crate::dijkstra::dijkstra_4ary;
use crate::duan::duan_inspired;
use crate::graph::CsrGraph;

pub fn sssp_auto(g: &CsrGraph, src: u32) -> Vec<f32> {
    let n = g.n;
    let m = g.m();
    let avg_deg = m as f32 / n.max(1) as f32;

    if avg_deg < 1.5 {
        return dijkstra_4ary(g, src);
    }

    // Estimate avg edge weight via stride-sampling — de første kantene i CSR
    // er ofte fra et lokalt område og ikke representative.
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
        // Robust valg for road networks og lignende sparse grafer.
        return delta_stepping(g, src, delta);
    }

    duan_inspired(g, src, 4.0 * delta)
}
