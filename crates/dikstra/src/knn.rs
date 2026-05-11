//! K nearest neighbours on graph distance.
//!
//! Given a road graph `g` and a customer node-id set, produces for each
//! customer its K nearest other customers in network distance (not
//! great-circle). The algorithm is **plain Dijkstra with early termination**
//! on the original (uncontracted) graph — we cannot use the CH-augmented
//! graph here because shortcuts skip intermediate customers and would
//! mis-rank neighbours.
//!
//! Output is `Vec<Vec<(u32, f32, f32)>>`. Row `i` is the K nearest
//! customers of `customers[i]`, sorted by duration ascending. Each entry is
//! `(neighbour_index_in_input, dur_s, dist_m)` — the index refers back to
//! position in the input `customers` slice so the consumer can look up the
//! actual node id / coordinate.
//!
//! Complexity per source: O(K × n_graph / customer_density × log) — for
//! 50k customers in a 1.16M-node graph (London) and K=160, the search
//! settles ~3700 graph nodes before hitting 160 customers (~740 µs per
//! source on Apple M3 Pro). Parallel over 11 cores: ~3-5 s total.
//!
//! Memory: per worker thread holds `2 × n × 4 B` Dijkstra state arrays;
//! output is `customers.len() × K × 12 B` (= 96 MB for 50k × 160).
//!
//! Use case: granular VRP solvers (Toth-Vigo style) restrict local search
//! to each customer's K nearest neighbours. The N × K table replaces the
//! N × N distance matrix for the hot path; cold-path lookups (depot ↔
//! customer, occasional cross-route) can still go through `ch::query`.

use crate::dijkstra::INF;
use crate::graph::CsrGraph;
use rayon::prelude::*;

/// 4-ary min-heap entry, sorted ascending by `d`.
#[derive(Clone, Copy)]
struct HItem {
    d: f32,
    v: u32,
}

#[inline]
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

#[inline]
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
        let last_c = (first + 4).min(len);
        for c in (first + 1)..last_c {
            let dc = h[c].d;
            if dc < sd {
                s = c;
                sd = dc;
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

/// Per-worker reusable scratch state.
struct KnnScratch {
    dist_dur: Vec<f32>,
    dist_dist: Vec<f32>,
    touched: Vec<u32>,
    heap: Vec<HItem>,
}

impl KnnScratch {
    fn new(n: usize, has_dist: bool) -> Self {
        Self {
            dist_dur: vec![INF; n],
            dist_dist: if has_dist { vec![INF; n] } else { Vec::new() },
            touched: Vec::with_capacity(8192),
            heap: Vec::with_capacity(2048),
        }
    }
}

/// Compute K nearest neighbours for each customer.
///
/// * `g` — the road graph (use `pp.graph`, the original reordered graph;
///   passing a CH-augmented graph will mis-rank because shortcuts skip
///   intermediate customers).
/// * `customers` — list of node IDs to use as both source and destination
///   set. Output row `i` corresponds to source `customers[i]`.
/// * `k` — number of neighbours to return per source. Output row may be
///   shorter than `k` if the source is in an isolated component.
/// * `edge_dist` — optional per-edge distance (metres), parallel to
///   `g.edge_w`. When `None`, the `dist_m` field of each entry is 0.0.
///
/// Returns `Vec<Vec<(u32, f32, f32)>>` of length `customers.len()`. Each
/// inner Vec is the K nearest customers sorted by duration ascending;
/// entries are `(neighbour_index_in_customers, dur_s, dist_m)`. The source
/// itself is **not** included in its own neighbour list.
pub fn knn_matrix(
    g: &CsrGraph,
    customers: &[u32],
    k: usize,
    edge_dist: Option<&[f32]>,
) -> Vec<Vec<(u32, f32, f32)>> {
    let n = g.n;
    let n_customers = customers.len();
    if n_customers == 0 || k == 0 {
        return Vec::new();
    }

    // Map node-id → its index in `customers` (or u32::MAX if not a customer).
    let mut is_customer: Vec<u32> = vec![u32::MAX; n];
    for (i, &id) in customers.iter().enumerate() {
        assert!((id as usize) < n, "customer id {id} out of range (n={n})");
        is_customer[id as usize] = i as u32;
    }
    let is_customer = &is_customer; // share read-only across threads

    let has_dist = match edge_dist {
        Some(ed) => {
            assert_eq!(
                ed.len(),
                g.edge_w.len(),
                "edge_dist length must match g.edge_w"
            );
            true
        }
        None => false,
    };
    let ed = edge_dist.unwrap_or(&[]);

    customers
        .par_iter()
        .enumerate()
        .map_init(
            || KnnScratch::new(n, has_dist),
            |scratch, (src_idx, &src)| {
                // Reset state touched by previous source.
                for &v in &scratch.touched {
                    scratch.dist_dur[v as usize] = INF;
                    if has_dist {
                        scratch.dist_dist[v as usize] = INF;
                    }
                }
                scratch.touched.clear();
                scratch.heap.clear();

                let mut result: Vec<(u32, f32, f32)> = Vec::with_capacity(k);
                scratch.dist_dur[src as usize] = 0.0;
                if has_dist {
                    scratch.dist_dist[src as usize] = 0.0;
                }
                scratch.touched.push(src);
                push(&mut scratch.heap, 0.0, src);

                while let Some(HItem { d, v: u }) = pop(&mut scratch.heap) {
                    if d > scratch.dist_dur[u as usize] {
                        continue;
                    }
                    let cust_idx = is_customer[u as usize];
                    if cust_idx != u32::MAX && cust_idx as usize != src_idx {
                        let dist_m = if has_dist {
                            scratch.dist_dist[u as usize]
                        } else {
                            0.0
                        };
                        result.push((cust_idx, d, dist_m));
                        if result.len() >= k {
                            break;
                        }
                    }

                    let start = g.head[u as usize] as usize;
                    let end = g.head[u as usize + 1] as usize;
                    for k_edge in start..end {
                        let w = g.edge_to[k_edge];
                        let nd = d + g.edge_w[k_edge];
                        if nd < scratch.dist_dur[w as usize] {
                            if scratch.dist_dur[w as usize] == INF {
                                scratch.touched.push(w);
                            }
                            scratch.dist_dur[w as usize] = nd;
                            if has_dist {
                                scratch.dist_dist[w as usize] =
                                    scratch.dist_dist[u as usize] + ed[k_edge];
                            }
                            push(&mut scratch.heap, nd, w);
                        }
                    }
                }
                // result is already in ascending duration order (Dijkstra
                // settle order); no extra sort needed.
                result
            },
        )
        .collect()
}

/// Convenience: same as [`knn_matrix`] but flattens the output to a single
/// `Vec<(u32, f32, f32)>` of length `customers.len() × k`. Row `i` lives at
/// `[i*k .. i*k + k]`; unused trailing slots (when a source has fewer than
/// `k` reachable customers) are filled with `(u32::MAX, f32::INFINITY,
/// f32::INFINITY)`. Cache-friendly for solvers that index `granular[i][j]`.
pub fn knn_matrix_flat(
    g: &CsrGraph,
    customers: &[u32],
    k: usize,
    edge_dist: Option<&[f32]>,
) -> Vec<(u32, f32, f32)> {
    let rows = knn_matrix(g, customers, k, edge_dist);
    let mut flat = vec![(u32::MAX, f32::INFINITY, f32::INFINITY); customers.len() * k];
    for (i, row) in rows.into_iter().enumerate() {
        for (j, entry) in row.into_iter().enumerate() {
            flat[i * k + j] = entry;
        }
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CsrGraph;

    /// Tiny graph: 0 ↔ 1 ↔ 2 ↔ 3 ↔ 4, weight 1.0 each. All five are customers.
    fn line_graph() -> (CsrGraph, Vec<f32>) {
        let edges: Vec<(u32, u32, f32, f32)> = vec![
            (0, 1, 1.0, 10.0),
            (1, 0, 1.0, 10.0),
            (1, 2, 1.0, 10.0),
            (2, 1, 1.0, 10.0),
            (2, 3, 1.0, 10.0),
            (3, 2, 1.0, 10.0),
            (3, 4, 1.0, 10.0),
            (4, 3, 1.0, 10.0),
        ];
        let (g, edge_dist) = CsrGraph::from_edges_with_dist(5, &edges);
        (g, edge_dist)
    }

    #[test]
    fn k_nearest_on_line() {
        let (g, ed) = line_graph();
        let customers: Vec<u32> = vec![0, 1, 2, 3, 4];
        let knn = knn_matrix(&g, &customers, 2, Some(&ed));
        // Customer 0 (node 0): nearest is node 1 (dur=1, dist=10), next is node 2 (dur=2, dist=20).
        assert_eq!(knn[0].len(), 2);
        assert_eq!(knn[0][0].0, 1);
        assert!((knn[0][0].1 - 1.0).abs() < 1e-6);
        assert!((knn[0][0].2 - 10.0).abs() < 1e-6);
        assert_eq!(knn[0][1].0, 2);
        assert!((knn[0][1].1 - 2.0).abs() < 1e-6);
        // Customer 2 (middle): nearest are 1 and 3, both at dur=1.
        let n0 = knn[2][0].0;
        let n1 = knn[2][1].0;
        assert!((n0 == 1 && n1 == 3) || (n0 == 3 && n1 == 1));
    }

    #[test]
    fn no_self_in_neighbours() {
        let (g, ed) = line_graph();
        let customers: Vec<u32> = vec![0, 1, 2, 3, 4];
        let knn = knn_matrix(&g, &customers, 4, Some(&ed));
        for (i, row) in knn.iter().enumerate() {
            for &(idx, _, _) in row {
                assert_ne!(idx as usize, i, "src {i} appeared in its own neighbours");
            }
        }
    }

    #[test]
    fn sorted_ascending_by_dur() {
        let (g, ed) = line_graph();
        let customers: Vec<u32> = vec![0, 1, 2, 3, 4];
        let knn = knn_matrix(&g, &customers, 5, Some(&ed));
        for row in &knn {
            for w in row.windows(2) {
                assert!(w[0].1 <= w[1].1, "row not sorted: {row:?}");
            }
        }
    }

    #[test]
    fn isolated_node_returns_short_row() {
        // 4 customers, only 0↔1 and 2↔3 connected. Customer 0's neighbours
        // should be only {1}, not include 2 or 3.
        let edges: Vec<(u32, u32, f32, f32)> = vec![
            (0, 1, 1.0, 10.0),
            (1, 0, 1.0, 10.0),
            (2, 3, 1.0, 10.0),
            (3, 2, 1.0, 10.0),
        ];
        let (g, ed) = CsrGraph::from_edges_with_dist(4, &edges);
        let customers: Vec<u32> = vec![0, 1, 2, 3];
        let knn = knn_matrix(&g, &customers, 3, Some(&ed));
        assert_eq!(knn[0].len(), 1);
        assert_eq!(knn[0][0].0, 1);
        assert_eq!(knn[2].len(), 1);
        assert_eq!(knn[2][0].0, 3);
    }

    #[test]
    fn flat_layout_padding() {
        let (g, ed) = line_graph();
        // K=3 but customer 0 has only 4 neighbours; ask for K=10 to force
        // padding.
        let customers: Vec<u32> = vec![0, 1, 2, 3, 4];
        let flat = knn_matrix_flat(&g, &customers, 10, Some(&ed));
        assert_eq!(flat.len(), 5 * 10);
        // Customer 0 has 4 real neighbours (1, 2, 3, 4), rest are sentinels.
        for j in 4..10 {
            let (idx, dur, dist) = flat[j];
            assert_eq!(idx, u32::MAX);
            assert!(dur.is_infinite());
            assert!(dist.is_infinite());
        }
    }
}
