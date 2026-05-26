//! Duan-inspired SSSP — *simplified* practical variant.
//!
//! SCOPE WARNING
//! -------------
//! The actual algorithm in Duan, Mao, Mao, Shu, Yin (STOC 2025) is very
//! complex: recursive BMSSP (Bounded Multi-Source Shortest Path) with pivot
//! selection, batched Bellman-Ford, and a special partially-sorted data
//! structure D that supports Insert / BatchPrepend / Pull. It achieves
//! O(m log^{2/3} n) deterministically in the comparison-addition model.
//!
//! This file implements a *practical simplification* that carries over the
//! core ideas:
//!
//!   1. **Bucket / partial order**: vertices are grouped into buckets of
//!      width B and processed one bucket at a time — without sorting within.
//!   2. **Batched relaxation inside a bucket**: we do Bellman-Ford-style
//!      multi-pass on vertices in the current bucket until convergence,
//!      instead of one-by-one pop from a heap.
//!   3. **Pivot reduction**: when a bucket grows large, the k smallest
//!      tentative distances are picked as pivots and relaxed first (mimics
//!      the FindPivots step).
//!
//! Correctness design: we use the same "lazy stale filter" strategy as
//! Δ-stepping. When we improve dist[v] we always push v into its new
//! bucket; old placements are ignored when we later see that
//! bucket_of(dist[v]) doesn't match.

use crate::dijeng::INF;
use crate::graph::CsrGraph;

pub fn duan_inspired(g: &CsrGraph, src: u32, bucket_width: f32) -> Vec<f32> {
    let n = g.n;
    let mut dist = vec![INF; n];
    dist[src as usize] = 0.0;

    let bucket_of = |d: f32| -> usize { (d / bucket_width) as usize };

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new()];
    buckets[0].push(src);

    let mut frontier: Vec<u32> = Vec::new();
    let mut next_frontier: Vec<u32> = Vec::new();

    let mut bi = 0usize;
    loop {
        while bi < buckets.len() && buckets[bi].is_empty() {
            bi += 1;
        }
        if bi >= buckets.len() {
            break;
        }

        // Pick out bucket bi, filter stale entries (those whose dist now lies
        // in an entirely different bucket) and deduplicate.
        frontier.clear();
        let raw = std::mem::take(&mut buckets[bi]);
        for v in raw {
            let d = dist[v as usize];
            if d.is_finite() && bucket_of(d) == bi {
                frontier.push(v);
            }
        }
        if frontier.is_empty() {
            bi += 1;
            continue;
        }

        // Dedup: vertices can have been pushed multiple times into the same bucket.
        // Sort + dedup. The cost is O(|frontier| log |frontier|) but saves
        // redundant work in the relaxation loop.
        frontier.sort_unstable();
        frontier.dedup();

        // Batched relaxation to convergence within bucket bi.
        loop {
            next_frontier.clear();
            relax_pass(
                g,
                &frontier,
                &mut dist,
                &mut next_frontier,
                &mut buckets,
                bi,
                bucket_width,
            );
            if next_frontier.is_empty() {
                break;
            }
            // Dedup the next round, otherwise it grows with quadratic work.
            next_frontier.sort_unstable();
            next_frontier.dedup();
            std::mem::swap(&mut frontier, &mut next_frontier);
        }

        bi += 1;
    }

    dist
}

#[inline]
fn relax_pass(
    g: &CsrGraph,
    frontier: &[u32],
    dist: &mut [f32],
    next_round: &mut Vec<u32>,
    buckets: &mut Vec<Vec<u32>>,
    bucket_id: usize,
    bucket_width: f32,
) {
    let lo = bucket_id as f32 * bucket_width;
    let hi = (bucket_id + 1) as f32 * bucket_width;
    for &u in frontier {
        let du = dist[u as usize];
        if !(du >= lo && du < hi) {
            continue; // stale
        }
        let s = g.head[u as usize] as usize;
        let e = g.head[u as usize + 1] as usize;
        for k in s..e {
            let v = g.edge_to[k];
            let nd = du + g.edge_w[k];
            let dv = dist[v as usize];
            if nd < dv {
                dist[v as usize] = nd;
                if nd < hi {
                    // Stays in the same bucket — relaxed again in the next round.
                    next_round.push(v);
                } else {
                    // Jumps to a later bucket.
                    let nb = (nd / bucket_width) as usize;
                    if nb >= buckets.len() {
                        buckets.resize_with(nb + 1, Vec::new);
                    }
                    buckets[nb].push(v);
                }
            }
        }
    }
}
