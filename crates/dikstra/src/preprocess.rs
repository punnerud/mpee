//! Preprocessing for SSSP.
//!
//! To valgfrie preprocesserings-trinn:
//!
//!   1. **Vertex reorder (BFS-orden)**: numererer vertices slik at en vertex
//!      og dens nære naboer ligger i samme cache-linje. Reduserer L2/L3-
//!      cache-misses i alle SSSP-algoritmer. Typisk gevinst 1.2–1.5×.
//!
//!   2. **Edge partition på `delta`**: for hver vertex sortere kantene slik
//!      at LIGHT (w ≤ delta) ligger først, HEAVY etterpå. Δ-stepping kan
//!      hoppe direkte til den partisjonen som er aktuell for hver phase, og
//!      sparer en branch per kant.
//!
//! Resultatet er en ny `CsrGraph` + en `light_count[u]`-tabell. Algoritmer
//! kan ta hint om light_count for å skippe branchen.
//!
//! Hele resultatet kan caches til disk (`cache_pp.rs`) for instant cold-
//! start.

use crate::buffer::Buffer;
use crate::graph::CsrGraph;
use std::collections::VecDeque;

pub struct Preprocessed {
    /// Reordered graph.
    pub graph: CsrGraph,
    /// Per-edge distance in metres, in the same order as `graph.edge_to`.
    /// Empty if the input had no distance channel.
    pub edge_dist: Buffer<f32>,
    /// `light_count[u]` = antall light-kanter (w ≤ delta) blant adjacency
    /// til u. Kantene `[head[u] .. head[u]+light_count[u]]` er light;
    /// `[head[u]+light_count[u] .. head[u+1]]` er heavy.
    pub light_count: Buffer<u32>,
    /// Permutasjon: `new_id[old_id]` = ny id for hver gammel.
    pub new_id: Buffer<u32>,
    /// Δ-verdien som ble brukt for partisjonering (0.0 hvis ikke partisjonert).
    pub delta_used: f32,
}

/// BFS fra vertex 0 (eller en vertex med høyest grad) gir en orden hvor
/// nære naboer havner i sammen-rang i den nye nummereringen.
/// Argumenter: `partition_delta` = Some(d) hvis vi skal partisjonere kantene
/// på light/heavy med delta=d. None betyr at light_count fylles med 0.
/// `edge_dist`: parallel per-edge distance channel (empty slice if absent).
/// If non-empty it is reordered in lockstep with `edge_w`.
pub fn preprocess(
    g: &CsrGraph,
    partition_delta: Option<f32>,
    edge_dist: &[f32],
) -> Preprocessed {
    let n = g.n;
    let new_id = bfs_reorder(g);
    let has_dist = edge_dist.len() == g.m();

    // Bygg ny CSR med reordered vertices.
    let mut new_head = vec![0u32; n + 1];
    for u in 0..n {
        let nu = new_id[u] as usize;
        let deg = g.head[u + 1] - g.head[u];
        new_head[nu + 1] = deg;
    }
    for i in 1..=n {
        new_head[i] += new_head[i - 1];
    }

    let m = g.m();
    let mut new_to: Vec<u32> = vec![0; m];
    let mut new_w: Vec<f32> = vec![0.0; m];
    let mut new_dist: Vec<f32> = if has_dist { vec![0.0; m] } else { Vec::new() };
    let mut light_count: Vec<u32> = vec![0; n];

    // Fyll inn — ev. partisjonere på light/heavy per vertex.
    for u in 0..n {
        let nu = new_id[u] as usize;
        let s = g.head[u] as usize;
        let e = g.head[u + 1] as usize;
        let target_start = new_head[nu] as usize;

        if let Some(delta) = partition_delta {
            let mut lc = 0u32;
            let mut write = target_start;
            for k in s..e {
                if g.edge_w[k] <= delta {
                    new_to[write] = new_id[g.edge_to[k] as usize];
                    new_w[write] = g.edge_w[k];
                    if has_dist {
                        new_dist[write] = edge_dist[k];
                    }
                    write += 1;
                    lc += 1;
                }
            }
            for k in s..e {
                if g.edge_w[k] > delta {
                    new_to[write] = new_id[g.edge_to[k] as usize];
                    new_w[write] = g.edge_w[k];
                    if has_dist {
                        new_dist[write] = edge_dist[k];
                    }
                    write += 1;
                }
            }
            light_count[nu] = lc;
        } else {
            for (i, k) in (s..e).enumerate() {
                new_to[target_start + i] = new_id[g.edge_to[k] as usize];
                new_w[target_start + i] = g.edge_w[k];
                if has_dist {
                    new_dist[target_start + i] = edge_dist[k];
                }
            }
        }
    }

    Preprocessed {
        graph: CsrGraph {
            n,
            head: new_head.into(),
            edge_to: new_to.into(),
            edge_w: new_w.into(),
        },
        edge_dist: new_dist.into(),
        light_count: light_count.into(),
        new_id: new_id.into(),
        delta_used: partition_delta.unwrap_or(0.0),
    }
}

/// BFS fra den vertex med høyest grad. Returnerer permutasjon
/// `new_id[old] -> new`.
fn bfs_reorder(g: &CsrGraph) -> Vec<u32> {
    let n = g.n;
    // Start fra den med høyest grad — det gir en mer "kompakt" front for
    // road networks og sosiale grafer.
    let mut start = 0u32;
    let mut max_deg = 0u32;
    for u in 0..n {
        let d = g.head[u + 1] - g.head[u];
        if d > max_deg {
            max_deg = d;
            start = u as u32;
        }
    }

    let mut new_id = vec![u32::MAX; n];
    let mut queue: VecDeque<u32> = VecDeque::with_capacity(n);
    let mut next_id: u32 = 0;
    new_id[start as usize] = next_id;
    next_id += 1;
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        let s = g.head[u as usize] as usize;
        let e = g.head[u as usize + 1] as usize;
        for k in s..e {
            let v = g.edge_to[k];
            if new_id[v as usize] == u32::MAX {
                new_id[v as usize] = next_id;
                next_id += 1;
                queue.push_back(v);
            }
        }
    }

    // Vertices ikke nådd via BFS: gi dem ID-er på slutten.
    for u in 0..n {
        if new_id[u] == u32::MAX {
            new_id[u] = next_id;
            next_id += 1;
        }
    }

    new_id
}

/// Re-mapper en kilde-vertex etter reordering.
pub fn remap_src(pp: &Preprocessed, src: u32) -> u32 {
    pp.new_id[src as usize]
}
