//! Compressed sparse row (CSR) graph and synthetic generators.
//!
//! CSR layout is cache-friendly: for each vertex u, all (target, weight) pairs
//! sit contiguously in memory, and traversing N(u) is a single linear scan.

use crate::buffer::Buffer;
use std::time::Instant;

pub struct CsrGraph {
    pub n: usize,
    pub head: Buffer<u32>,    // length n+1; edges of u are head[u]..head[u+1]
    pub edge_to: Buffer<u32>, // length m; target of each edge
    pub edge_w: Buffer<f32>,  // length m; weight of each edge
}

impl CsrGraph {
    pub fn m(&self) -> usize {
        self.edge_to.len()
    }

    /// Build CSR from a vec of (u, v, w) edges (already directed, may include both
    /// directions for an undirected graph).
    pub fn from_edges(n: usize, edges: &[(u32, u32, f32)]) -> Self {
        let mut deg = vec![0u32; n + 1];
        for &(u, _, _) in edges {
            deg[u as usize + 1] += 1;
        }
        for i in 1..=n {
            deg[i] += deg[i - 1];
        }
        let head = deg.clone();
        let m = edges.len();
        let mut edge_to = vec![0u32; m];
        let mut edge_w = vec![0.0f32; m];
        let mut cursor = head.clone();
        for &(u, v, w) in edges {
            let idx = cursor[u as usize] as usize;
            edge_to[idx] = v;
            edge_w[idx] = w;
            cursor[u as usize] += 1;
        }
        CsrGraph {
            n,
            head: head.into(),
            edge_to: edge_to.into(),
            edge_w: edge_w.into(),
        }
    }

    /// Like `from_edges`, but each edge carries a second f32 channel (e.g.
    /// distance in metres alongside weight = duration in seconds). Returns
    /// the graph plus a parallel `edge_dist[k]` slot for the same edge index.
    pub fn from_edges_with_dist(
        n: usize,
        edges: &[(u32, u32, f32, f32)],
    ) -> (Self, Vec<f32>) {
        let mut deg = vec![0u32; n + 1];
        for &(u, _, _, _) in edges {
            deg[u as usize + 1] += 1;
        }
        for i in 1..=n {
            deg[i] += deg[i - 1];
        }
        let head = deg.clone();
        let m = edges.len();
        let mut edge_to = vec![0u32; m];
        let mut edge_w = vec![0.0f32; m];
        let mut edge_dist = vec![0.0f32; m];
        let mut cursor = head.clone();
        for &(u, v, w, d) in edges {
            let idx = cursor[u as usize] as usize;
            edge_to[idx] = v;
            edge_w[idx] = w;
            edge_dist[idx] = d;
            cursor[u as usize] += 1;
        }
        (
            CsrGraph {
                n,
                head: head.into(),
                edge_to: edge_to.into(),
                edge_w: edge_w.into(),
            },
            edge_dist,
        )
    }
}

/// xorshift64 PRNG. We avoid pulling in `rand` to keep dependencies at zero.
pub struct Rng(pub u64);
impl Rng {
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // 24 random bits in [0,1).
        ((self.next_u64() >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
    }
    #[inline]
    pub fn range(&mut self, n: u32) -> u32 {
        // Lemire-style unbiased range; cheap enough for benchmarks.
        let m = (self.next_u32() as u64) * (n as u64);
        (m >> 32) as u32
    }
}

/// Random sparse directed graph: each of n vertices has `avg_deg` outgoing edges
/// to uniformly random other vertices. Weights uniform in (0, 1].
pub fn gen_random_sparse(n: usize, avg_deg: usize, seed: u64) -> CsrGraph {
    let mut rng = Rng(seed | 1);
    let m = n * avg_deg;
    let mut edges = Vec::with_capacity(m);
    for u in 0..n as u32 {
        for _ in 0..avg_deg {
            let v = rng.range(n as u32);
            if v == u {
                continue;
            }
            let w = rng.next_f32().max(1e-6);
            edges.push((u, v, w));
        }
    }
    CsrGraph::from_edges(n, &edges)
}

/// Random sparse graph with exponentially distributed edge weights (scale = mean).
/// These weights have larger variance than uniform — stresses bucket-based
/// algorithms that are sensitive to heavy edges.
pub fn gen_random_exp_weights(n: usize, avg_deg: usize, mean: f32, seed: u64) -> CsrGraph {
    let mut rng = Rng(seed | 1);
    let m = n * avg_deg;
    let mut edges = Vec::with_capacity(m);
    for u in 0..n as u32 {
        for _ in 0..avg_deg {
            let v = rng.range(n as u32);
            if v == u {
                continue;
            }
            // Inverse-CDF: w = -mean * ln(1 - U), U in (0,1).
            let mut uu = rng.next_f32();
            if uu < 1e-7 { uu = 1e-7; }
            if uu > 1.0 - 1e-7 { uu = 1.0 - 1e-7; }
            let w = (-mean * (1.0 - uu).ln()).max(1e-6);
            edges.push((u, v, w));
        }
    }
    CsrGraph::from_edges(n, &edges)
}

/// Power-law (preferential-attachment-style) graph: nodes with higher ID have
/// more edges; produces heavy "hubs". Models realistic networks (web,
/// social graphs) better than uniform random.
pub fn gen_power_law(n: usize, m_edges_per_node: usize, seed: u64) -> CsrGraph {
    let mut rng = Rng(seed | 1);
    let mut edges = Vec::with_capacity(n * m_edges_per_node * 2);
    // Start with a small clique.
    let init = (m_edges_per_node + 1).min(n);
    for u in 0..init {
        for v in 0..init {
            if u != v {
                let w = rng.next_f32().max(1e-6);
                edges.push((u as u32, v as u32, w));
            }
        }
    }
    // Cumulative degree vector for probability weighting.
    let mut deg_running: Vec<u32> = vec![(init - 1) as u32; init];
    for u in init..n {
        // Pick `m_edges_per_node` targets with probability proportional to degree.
        let total: u64 = deg_running.iter().map(|&d| d as u64).sum();
        for _ in 0..m_edges_per_node {
            // Roulette selection.
            let r = (rng.next_u64() % total.max(1)) as u64;
            let mut acc = 0u64;
            let mut target = 0usize;
            for (i, &d) in deg_running.iter().enumerate() {
                acc += d as u64;
                if acc > r {
                    target = i;
                    break;
                }
            }
            let w = rng.next_f32().max(1e-6);
            edges.push((u as u32, target as u32, w));
            edges.push((target as u32, u as u32, w));
            deg_running[target] += 1;
        }
        deg_running.push(m_edges_per_node as u32);
    }
    CsrGraph::from_edges(n, &edges)
}

/// Path graph 0 → 1 → 2 → ... → n-1 with uniform weight. Worst case
/// for bucket-based algorithms: each bucket has only one vertex, so there is
/// no batch advantage and Bellman-Ford passes do no extra work.
pub fn gen_path(n: usize, seed: u64) -> CsrGraph {
    let mut rng = Rng(seed | 1);
    let mut edges = Vec::with_capacity(n);
    for u in 0..(n as u32 - 1) {
        let w = rng.next_f32().max(1e-6);
        edges.push((u, u + 1, w));
    }
    CsrGraph::from_edges(n, &edges)
}

/// 2D grid graph (undirected, stored as both directions): vertex (i,j) connected
/// to its 4 neighbours with random weight in (0,1]. Useful "geometric" workload
/// where Dijkstra's diameter is O(sqrt(n)).
pub fn gen_grid(side: usize, seed: u64) -> CsrGraph {
    let n = side * side;
    let mut rng = Rng(seed | 1);
    let mut edges = Vec::with_capacity(n * 4);
    let idx = |i: usize, j: usize| (i * side + j) as u32;
    for i in 0..side {
        for j in 0..side {
            let u = idx(i, j);
            if i + 1 < side {
                let v = idx(i + 1, j);
                let w = rng.next_f32().max(1e-6);
                edges.push((u, v, w));
                edges.push((v, u, w));
            }
            if j + 1 < side {
                let v = idx(i, j + 1);
                let w = rng.next_f32().max(1e-6);
                edges.push((u, v, w));
                edges.push((v, u, w));
            }
        }
    }
    CsrGraph::from_edges(n, &edges)
}

pub fn time_it<F: FnOnce() -> R, R>(label: &str, f: F) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    let secs = t.elapsed().as_secs_f64();
    println!("  {:<28} {:>10.3} ms", label, secs * 1000.0);
    (r, secs)
}
