//! Ekstra synthetic graph generators.
//!
//! `gen_rmat` — Recursive MATrix model (Graph500 default a=0.57, b=0.19,
//! c=0.19, d=0.05). Skala-fri, har power-law-degenfordeling og er
//! standard akademisk benchmark. 2^s noder, edge_factor·2^s kanter.

use crate::graph::{CsrGraph, Rng};

/// Standard Graph500 RMAT-parametere.
pub const RMAT_A: f64 = 0.57;
pub const RMAT_B: f64 = 0.19;
pub const RMAT_C: f64 = 0.19;
pub const RMAT_D: f64 = 0.05;

/// `scale` = log2(n). Antall noder = 2^scale. Antall kanter ~ edge_factor * n.
pub fn gen_rmat(scale: u32, edge_factor: usize, seed: u64) -> CsrGraph {
    let n = 1usize << scale;
    let m_target = n * edge_factor;
    let mut rng = Rng(seed | 1);

    // Cumulative probabilities for binary search-style quadrant pick.
    let p_ab = RMAT_A + RMAT_B;
    let p_abc = p_ab + RMAT_C;

    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(m_target);
    for _ in 0..m_target {
        let mut u = 0u64;
        let mut v = 0u64;
        for i in 0..scale {
            // Pick quadrant.
            let r = (rng.next_u64() as f64) / (u64::MAX as f64);
            let bit_u: u64;
            let bit_v: u64;
            if r < RMAT_A {
                bit_u = 0;
                bit_v = 0;
            } else if r < p_ab {
                bit_u = 0;
                bit_v = 1;
            } else if r < p_abc {
                bit_u = 1;
                bit_v = 0;
            } else {
                bit_u = 1;
                bit_v = 1;
            }
            u |= bit_u << i;
            v |= bit_v << i;
        }
        if u != v {
            let w = rng.next_f32().max(1e-6);
            edges.push((u as u32, v as u32, w));
        }
    }
    CsrGraph::from_edges(n, &edges)
}
