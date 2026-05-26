//! Bidirectional Dijeng for ren point-to-point shortest path.
//!
//! Searches from src and dst simultaneously: forward in the original graph,
//! backward in the transpose. Stops when top_f.dist + top_b.dist >= best
//! (optimal cutoff). On road networks this gives 2–3× speedup vs full SSSP.
//!
//! Requires a transpose function for directed graphs. For undirected
//! (asym-CSR with edges in both directions) g_fwd == g_bwd is fine.

use crate::dijeng::INF;
use crate::graph::CsrGraph;

/// Transponerer en CSR (rebuild med (v, u, w) for hver original (u, v, w)).
/// O(n + m) tid, O(n + m) ekstra minne.
pub fn transpose(g: &CsrGraph) -> CsrGraph {
    let (g, _) = transpose_with_dist(g, &[]);
    g
}

/// Like `transpose`, but also permutes a parallel per-edge channel (e.g.
/// distance-in-metres alongside duration-in-seconds). Pass an empty slice
/// for `edge_dist` to skip the secondary channel.
pub fn transpose_with_dist(g: &CsrGraph, edge_dist: &[f32]) -> (CsrGraph, Vec<f32>) {
    let n = g.n;
    let m = g.m();
    let has_dist = edge_dist.len() == m;
    let mut deg = vec![0u32; n + 1];
    for k in 0..m {
        deg[g.edge_to[k] as usize + 1] += 1;
    }
    for i in 1..=n {
        deg[i] += deg[i - 1];
    }
    let head_vec = deg.clone();
    let mut edge_to = vec![0u32; m];
    let mut edge_w = vec![0.0f32; m];
    let mut new_dist: Vec<f32> = if has_dist { vec![0.0; m] } else { Vec::new() };
    let mut cursor = head_vec.clone();
    for u in 0..n {
        let s = g.head[u] as usize;
        let e = g.head[u + 1] as usize;
        for k in s..e {
            let v = g.edge_to[k] as usize;
            let idx = cursor[v] as usize;
            edge_to[idx] = u as u32;
            edge_w[idx] = g.edge_w[k];
            if has_dist {
                new_dist[idx] = edge_dist[k];
            }
            cursor[v] += 1;
        }
    }
    (
        CsrGraph {
            n,
            head: head_vec.into(),
            edge_to: edge_to.into(),
            edge_w: edge_w.into(),
        },
        new_dist,
    )
}

/// Returns the shortest distance between src and dst, or None if unreachable.
pub fn bidir_dijeng(
    g_fwd: &CsrGraph,
    g_bwd: &CsrGraph,
    src: u32,
    dst: u32,
) -> Option<f32> {
    if src == dst {
        return Some(0.0);
    }
    let n = g_fwd.n;
    let mut dist_f = vec![INF; n];
    let mut dist_b = vec![INF; n];
    dist_f[src as usize] = 0.0;
    dist_b[dst as usize] = 0.0;

    let mut hf: Vec<HeapItem> = Vec::new();
    let mut hb: Vec<HeapItem> = Vec::new();
    push(&mut hf, 0.0, src);
    push(&mut hb, 0.0, dst);

    let mut best = INF;

    loop {
        let tf = hf.first().map(|h| h.dist).unwrap_or(INF);
        let tb = hb.first().map(|h| h.dist).unwrap_or(INF);
        // Termination: ingen kortere vei kan bli funnet.
        if tf + tb >= best {
            break;
        }
        if hf.is_empty() && hb.is_empty() {
            break;
        }

        // Velg retning: den med minste top.
        if tf <= tb && !hf.is_empty() {
            let HeapItem { dist: d, v: u } = pop(&mut hf).unwrap();
            if d > dist_f[u as usize] {
                continue;
            }
            let s = g_fwd.head[u as usize] as usize;
            let e = g_fwd.head[u as usize + 1] as usize;
            for k in s..e {
                let v = g_fwd.edge_to[k];
                let nd = d + g_fwd.edge_w[k];
                if nd < dist_f[v as usize] {
                    dist_f[v as usize] = nd;
                    push(&mut hf, nd, v);
                    let db = dist_b[v as usize];
                    if db.is_finite() {
                        let total = nd + db;
                        if total < best {
                            best = total;
                        }
                    }
                }
            }
        } else if !hb.is_empty() {
            let HeapItem { dist: d, v: u } = pop(&mut hb).unwrap();
            if d > dist_b[u as usize] {
                continue;
            }
            let s = g_bwd.head[u as usize] as usize;
            let e = g_bwd.head[u as usize + 1] as usize;
            for k in s..e {
                let v = g_bwd.edge_to[k];
                let nd = d + g_bwd.edge_w[k];
                if nd < dist_b[v as usize] {
                    dist_b[v as usize] = nd;
                    push(&mut hb, nd, v);
                    let df = dist_f[v as usize];
                    if df.is_finite() {
                        let total = nd + df;
                        if total < best {
                            best = total;
                        }
                    }
                }
            }
        } else {
            break;
        }
    }

    if best.is_finite() {
        Some(best)
    } else {
        None
    }
}

#[derive(Clone, Copy)]
struct HeapItem {
    dist: f32,
    v: u32,
}

#[inline]
fn push(heap: &mut Vec<HeapItem>, dist: f32, v: u32) {
    let mut i = heap.len();
    heap.push(HeapItem { dist, v });
    while i > 0 {
        let parent = (i - 1) >> 2; // 4-ary
        if heap[parent].dist <= heap[i].dist {
            break;
        }
        heap.swap(parent, i);
        i = parent;
    }
}

#[inline]
fn pop(heap: &mut Vec<HeapItem>) -> Option<HeapItem> {
    let n = heap.len();
    if n == 0 {
        return None;
    }
    let top = heap[0];
    let last = heap.pop().unwrap();
    if n == 1 {
        return Some(top);
    }
    heap[0] = last;
    let len = heap.len();
    let mut i = 0usize;
    loop {
        let first = 4 * i + 1;
        if first >= len {
            break;
        }
        let mut s = first;
        let mut sd = heap[first].dist;
        let last_c = (first + 4).min(len);
        for c in (first + 1)..last_c {
            let dc = heap[c].dist;
            if dc < sd {
                s = c;
                sd = dc;
            }
        }
        if sd >= heap[i].dist {
            break;
        }
        heap.swap(i, s);
        i = s;
    }
    Some(top)
}
