//! Contraction Hierarchies (CH) — Geisberger et al. 2008-stil.
//!
//! Two-phase approach:
//!
//!   * **Preprocessing**: assign each vertex a `rank` (lowest first). For
//!     each vertex being contracted, check every pair of (incoming, outgoing)
//!     neighbors and add a *shortcut* edge if there is no existing shorter
//!     "witness" path between them that avoids the contracted vertex.
//!
//!   * **Query**: bidirectional Dijeng on the augmented graph, but
//!     relax **only** edges that go upward in the hierarchy (from low rank to
//!     high). This limits the search dramatically — typically <1000 nodes each
//!     way on road networks.
//!
//! This is a CORRECT, not necessarily highly tuned implementation. We
//! prioritize mathematical correctness (verified against full Dijeng) before
//! any performance tuning.

use crate::buffer::Buffer;
use crate::dijeng::INF;
use crate::graph::CsrGraph;
#[cfg(feature = "native")]
use crate::paged::{ChLayout, PagedMmap, TouchBuf};
#[cfg(feature = "native")]
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

/// Default hops limit in witness search. Lower → faster preprocessing,
/// more shortcuts. Higher → slower, fewer shortcuts.
const WITNESS_HOPS_LIMIT: u32 = 5;

/// Stall-on-demand master switch (A/B: set DIJENG_NO_STALL=1 at process start,
/// or toggle programmatically with [`set_stall`] — used by `ch_verify` to
/// compare both modes in one process). A relaxed atomic load per query is
/// noise next to the stall scans themselves.
static STALL_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static STALL_ENV_APPLIED: std::sync::Once = std::sync::Once::new();

#[inline]
fn stall_enabled() -> bool {
    STALL_ENV_APPLIED.call_once(|| {
        if std::env::var("DIJENG_NO_STALL").is_ok() {
            STALL_ON.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    STALL_ON.load(std::sync::atomic::Ordering::Relaxed)
}

/// Override the stall-on-demand switch (returns the previous value).
pub fn set_stall(enabled: bool) -> bool {
    STALL_ENV_APPLIED.call_once(|| {});
    STALL_ON.swap(enabled, std::sync::atomic::Ordering::Relaxed)
}

/// Stall-on-demand check for a node `u` popped at distance `d` in a search
/// whose distance array is `dist`: if some already-reached HIGHER-ranked
/// neighbour `x` with an edge x→u (in this search's direction) proves
/// `dist[x] + w(x→u) < d`, then `d` is not the start of any shortest
/// up-then-down path and u's expansion can be skipped entirely.
///
/// The edges "into u from higher rank" in the forward search are exactly u's
/// UPWARD arcs in the *transposed* graph (and vice versa) — so the caller
/// passes the opposite-direction graph. Untouched neighbours sit at INF and
/// `INF + w` is never `< d`, so no touched-tracking is needed here. The
/// pruning is exact (classic CH stall-on-demand, Geisberger et al. 2008):
/// distances are unchanged, only provably non-tight expansions are skipped.
#[inline]
fn stalled(
    opposite: &CsrGraph,
    up_count_opp: &[u32],
    dist: &[f32],
    u: u32,
    d: f32,
) -> bool {
    let s = opposite.head[u as usize] as usize;
    let e = s + up_count_opp[u as usize] as usize;
    for k in s..e {
        let x = opposite.edge_to[k];
        if dist[x as usize] + opposite.edge_w[k] < d {
            return true;
        }
    }
    false
}

/// The fully built Contraction Hierarchy.
pub struct ContractionHierarchy {
    /// Augmented forward graph (originals + shortcuts), with every vertex's
    /// out-edges sorted so "upward" (rank[v] > rank[u]) comes first.
    pub graph_fwd: CsrGraph,
    /// The transposed augmented graph.
    pub graph_bwd: CsrGraph,
    /// Number of *upward* edges per vertex in `graph_fwd`. The remaining
    /// edges of u (after `up_count_fwd[u]`) are "downward" and must never
    /// be relaxed in a forward search.
    pub up_count_fwd: Buffer<u32>,
    /// Likewise for backward.
    pub up_count_bwd: Buffer<u32>,
    /// `rank[v]` = position in the contraction order (0 = first, n-1 = last).
    /// In a rank-ordered (SSSPCH1B) layout, `rank[v] = n-1-v`.
    pub rank: Buffer<u32>,
    /// For every shortcut edge: the original vertex ID of the contracted
    /// midpoint (in the *internal* numbering — same as `edge_to`). `u32::MAX`
    /// for original (non-shortcut) edges. Co-allocated with `graph_fwd.edge_to`.
    pub via_fwd: Buffer<u32>,
    pub via_bwd: Buffer<u32>,
    /// Per-edge distance (metres). Length equals `graph_fwd.edge_to.len()`.
    /// For original edges this is the OSM haversine distance; for shortcut
    /// edges it's the sum of sub-shortcut distances. Empty when the input
    /// graph carried no distance channel.
    pub edge_dist_fwd: Buffer<f32>,
    pub edge_dist_bwd: Buffer<f32>,
    /// Maps a vertex ID from the *input* CSR graph to the internal
    /// rank-ordered ID used by `query`. For SSSPCH1A caches and freshly
    /// built CHs that haven't been reordered, this is the identity.
    pub perm: Buffer<u32>,
}

/// Build CH from a (typically directed) CSR graph. Returns the augmented
/// hierarchy. `edge_dist` is an optional parallel-to-`g.edge_w` slice
/// carrying a second metric (e.g. distance in metres while `g.edge_w` is
/// duration in seconds). Pass `&[]` to omit; if non-empty it's threaded
/// through shortcuts so `ch::matrix` can return both metrics in one MMM.
pub fn build(g: &CsrGraph) -> ContractionHierarchy {
    build_with_dist(g, &[])
}

pub fn build_with_dist(g: &CsrGraph, edge_dist: &[f32]) -> ContractionHierarchy {
    let n = g.n;
    let t_total = Instant::now();
    let has_dist = edge_dist.len() == g.m();

    // Internal working graphs. Tuple = (target, weight=duration, dist, via).
    // `dist` is meaningful iff has_dist; otherwise it's left at 0.
    let mut fwd: Vec<Vec<(u32, f32, f32, u32)>> = vec![Vec::new(); n];
    let mut bwd: Vec<Vec<(u32, f32, f32, u32)>> = vec![Vec::new(); n];
    for u in 0..n {
        let s = g.head[u] as usize;
        let e = g.head[u + 1] as usize;
        for k in s..e {
            let v = g.edge_to[k];
            let w = g.edge_w[k];
            let d = if has_dist { edge_dist[k] } else { 0.0 };
            fwd[u].push((v, w, d, u32::MAX));
            bwd[v as usize].push((u as u32, w, d, u32::MAX));
        }
    }
    let mut contracted: Vec<bool> = vec![false; n];
    let mut rank = vec![0u32; n];

    let mut wstate = WitnessState::new(n);

    let edge_diff = use_edge_diff_order();
    let mut deleted_neighbors: Vec<u32> = vec![0; n];

    let mut pq: BinaryHeap<(Reverse<i32>, u32)> = BinaryHeap::with_capacity(n);
    let t_init = Instant::now();
    let mut current_priority = vec![0i32; n];
    if edge_diff {
        // The initial edge-difference pass runs one witness-limited simulation
        // per vertex — embarrassingly parallel over the read-only base graph.
        use rayon::prelude::*;
        let contracted_ro = &contracted;
        let fwd_ro = &fwd;
        let bwd_ro = &bwd;
        let dn_ro = &deleted_neighbors;
        current_priority = (0..n)
            .into_par_iter()
            .map_init(
                || WitnessState::new(n),
                |ws, v| edge_diff_priority(fwd_ro, bwd_ro, contracted_ro, dn_ro, v as u32, ws),
            )
            .collect();
        for v in 0..n {
            pq.push((Reverse(current_priority[v]), v as u32));
        }
        println!(
            "[ch] initial edge-difference priority (parallel): {:.2} s",
            t_init.elapsed().as_secs_f64()
        );
    } else {
        for v in 0..n {
            let p = degree_priority(&fwd, &bwd, v as u32);
            current_priority[v] = p;
            pq.push((Reverse(p), v as u32));
        }
        println!(
            "[ch] initial degree-priority: {:.2} s",
            t_init.elapsed().as_secs_f64()
        );
    }

    let mut next_rank: u32 = 0;
    let mut total_shortcuts: u64 = 0;
    let mut last_progress = Instant::now();
    let mut last_progress_rank: u32 = 0;
    let mut neighbor_set: Vec<u32> = Vec::with_capacity(64);

    while let Some((Reverse(p_old), v)) = pq.pop() {
        if contracted[v as usize] {
            continue;
        }
        if p_old != current_priority[v as usize] {
            continue;
        }
        let new_shortcuts = contract_vertex(&mut fwd, &mut bwd, &contracted, v, &mut wstate);
        contracted[v as usize] = true;
        rank[v as usize] = next_rank;
        next_rank += 1;
        total_shortcuts += new_shortcuts as u64;

        neighbor_set.clear();
        for &(u, _, _, _) in &bwd[v as usize] {
            if !contracted[u as usize] {
                neighbor_set.push(u);
            }
        }
        for &(u, _, _, _) in &fwd[v as usize] {
            if !contracted[u as usize] {
                neighbor_set.push(u);
            }
        }
        neighbor_set.sort_unstable();
        neighbor_set.dedup();
        for &u in &neighbor_set {
            deleted_neighbors[u as usize] += 1;
            let p_u = if edge_diff {
                edge_diff_priority(&fwd, &bwd, &contracted, &deleted_neighbors, u, &mut wstate)
            } else {
                degree_priority(&fwd, &bwd, u)
            };
            current_priority[u as usize] = p_u;
            pq.push((Reverse(p_u), u));
        }

        if last_progress.elapsed().as_secs_f64() > 3.0 {
            let pct = next_rank as f64 / n as f64 * 100.0;
            let rate = (next_rank - last_progress_rank) as f64
                / last_progress.elapsed().as_secs_f64();
            println!(
                "[ch] {:.1}%  rank={}/{}  shortcuts={}  ({:.0} v/s)",
                pct, next_rank, n, total_shortcuts, rate
            );
            last_progress = Instant::now();
            last_progress_rank = next_rank;
        }
    }
    let preprocess_secs = t_total.elapsed().as_secs_f64();
    println!(
        "[ch] preprocessing done: {:.1} s, {} shortcuts ({:.2}× original m)",
        preprocess_secs,
        total_shortcuts,
        (total_shortcuts as f64 + g.m() as f64) / g.m() as f64
    );

    // ---- Build augmented CSR from fwd[] and bwd[] ----
    // For each vertex sort so that upward edges (rank[target] > rank[u])
    // come FIRST. up_count_fwd[u] = number of upward edges.
    let m_aug: usize = fwd.iter().map(|v| v.len()).sum();

    let mut head_fwd = vec![0u32; n + 1];
    let mut head_bwd = vec![0u32; n + 1];
    for u in 0..n {
        head_fwd[u + 1] = fwd[u].len() as u32;
        head_bwd[u + 1] = bwd[u].len() as u32;
    }
    for u in 1..=n {
        head_fwd[u] += head_fwd[u - 1];
        head_bwd[u] += head_bwd[u - 1];
    }

    let mut edge_to_fwd = vec![0u32; m_aug];
    let mut edge_w_fwd = vec![0.0f32; m_aug];
    let mut edge_dist_fwd_vec: Vec<f32> = if has_dist { vec![0.0; m_aug] } else { Vec::new() };
    let mut via_fwd = vec![u32::MAX; m_aug];
    let mut up_count_fwd = vec![0u32; n];

    for u in 0..n {
        let mut adj = std::mem::take(&mut fwd[u]);
        let ru = rank[u];
        adj.sort_by_key(|&(v, _, _, _)| !(rank[v as usize] > ru));
        let off = head_fwd[u] as usize;
        let mut up = 0u32;
        for (i, &(v, w, d, via)) in adj.iter().enumerate() {
            edge_to_fwd[off + i] = v;
            edge_w_fwd[off + i] = w;
            if has_dist {
                edge_dist_fwd_vec[off + i] = d;
            }
            via_fwd[off + i] = via;
            if rank[v as usize] > ru {
                up += 1;
            }
        }
        up_count_fwd[u] = up;
    }

    let mut edge_to_bwd = vec![0u32; m_aug];
    let mut edge_w_bwd = vec![0.0f32; m_aug];
    let mut edge_dist_bwd_vec: Vec<f32> = if has_dist { vec![0.0; m_aug] } else { Vec::new() };
    let mut via_bwd = vec![u32::MAX; m_aug];
    let mut up_count_bwd = vec![0u32; n];

    for u in 0..n {
        let mut adj = std::mem::take(&mut bwd[u]);
        let ru = rank[u];
        adj.sort_by_key(|&(v, _, _, _)| !(rank[v as usize] > ru));
        let off = head_bwd[u] as usize;
        let mut up = 0u32;
        for (i, &(v, w, d, via)) in adj.iter().enumerate() {
            edge_to_bwd[off + i] = v;
            edge_w_bwd[off + i] = w;
            if has_dist {
                edge_dist_bwd_vec[off + i] = d;
            }
            via_bwd[off + i] = via;
            if rank[v as usize] > ru {
                up += 1;
            }
        }
        up_count_bwd[u] = up;
    }

    // ---- Rank-ordered renumbering (SSSPCH1B layout) ----
    //
    // Renumber vertices so that new_id 0 has the highest rank and new_id n-1
    // the lowest. After this, every CSR section starts with the data for
    // the topmost vertices in the hierarchy. Since CH queries traverse
    // upward, those bytes are touched on essentially every query — placing
    // them at low file offsets means they cluster on the same OS pages and
    // an LRU-bounded page cache naturally keeps them resident.
    let t_reorder = Instant::now();
    let mut perm: Vec<u32> = (0..n as u32).collect();
    perm.sort_by_key(|&v| std::cmp::Reverse(rank[v as usize]));
    // perm[new_id] = old_id;  inv[old_id] = new_id
    let mut inv = vec![0u32; n];
    for (new_id, &old_id) in perm.iter().enumerate() {
        inv[old_id as usize] = new_id as u32;
    }

    let mut new_head_fwd = vec![0u32; n + 1];
    let mut new_head_bwd = vec![0u32; n + 1];
    let mut new_up_count_fwd = vec![0u32; n];
    let mut new_up_count_bwd = vec![0u32; n];
    let mut new_rank = vec![0u32; n];
    for new_id in 0..n {
        let old_id = perm[new_id] as usize;
        let deg_f = head_fwd[old_id + 1] - head_fwd[old_id];
        let deg_b = head_bwd[old_id + 1] - head_bwd[old_id];
        new_head_fwd[new_id + 1] = deg_f;
        new_head_bwd[new_id + 1] = deg_b;
        new_up_count_fwd[new_id] = up_count_fwd[old_id];
        new_up_count_bwd[new_id] = up_count_bwd[old_id];
        new_rank[new_id] = rank[old_id];
    }
    for u in 1..=n {
        new_head_fwd[u] += new_head_fwd[u - 1];
        new_head_bwd[u] += new_head_bwd[u - 1];
    }

    let mut new_edge_to_fwd = vec![0u32; m_aug];
    let mut new_edge_w_fwd = vec![0.0f32; m_aug];
    let mut new_via_fwd = vec![u32::MAX; m_aug];
    let mut new_edge_to_bwd = vec![0u32; m_aug];
    let mut new_edge_w_bwd = vec![0.0f32; m_aug];
    let mut new_via_bwd = vec![u32::MAX; m_aug];
    let mut new_edge_dist_fwd: Vec<f32> = if has_dist { vec![0.0; m_aug] } else { Vec::new() };
    let mut new_edge_dist_bwd: Vec<f32> = if has_dist { vec![0.0; m_aug] } else { Vec::new() };

    for new_id in 0..n {
        let old_id = perm[new_id] as usize;
        let s_old = head_fwd[old_id] as usize;
        let e_old = head_fwd[old_id + 1] as usize;
        let s_new = new_head_fwd[new_id] as usize;
        for (i, k) in (s_old..e_old).enumerate() {
            new_edge_to_fwd[s_new + i] = inv[edge_to_fwd[k] as usize];
            new_edge_w_fwd[s_new + i] = edge_w_fwd[k];
            if has_dist {
                new_edge_dist_fwd[s_new + i] = edge_dist_fwd_vec[k];
            }
            let via = via_fwd[k];
            new_via_fwd[s_new + i] = if via == u32::MAX {
                u32::MAX
            } else {
                inv[via as usize]
            };
        }

        let s_old = head_bwd[old_id] as usize;
        let e_old = head_bwd[old_id + 1] as usize;
        let s_new = new_head_bwd[new_id] as usize;
        for (i, k) in (s_old..e_old).enumerate() {
            new_edge_to_bwd[s_new + i] = inv[edge_to_bwd[k] as usize];
            new_edge_w_bwd[s_new + i] = edge_w_bwd[k];
            if has_dist {
                new_edge_dist_bwd[s_new + i] = edge_dist_bwd_vec[k];
            }
            let via = via_bwd[k];
            new_via_bwd[s_new + i] = if via == u32::MAX {
                u32::MAX
            } else {
                inv[via as usize]
            };
        }
    }

    println!(
        "[ch] rank-ordered layout: {:.2} s",
        t_reorder.elapsed().as_secs_f64()
    );

    ContractionHierarchy {
        graph_fwd: CsrGraph {
            n,
            head: new_head_fwd.into(),
            edge_to: new_edge_to_fwd.into(),
            edge_w: new_edge_w_fwd.into(),
        },
        graph_bwd: CsrGraph {
            n,
            head: new_head_bwd.into(),
            edge_to: new_edge_to_bwd.into(),
            edge_w: new_edge_w_bwd.into(),
        },
        up_count_fwd: new_up_count_fwd.into(),
        up_count_bwd: new_up_count_bwd.into(),
        rank: new_rank.into(),
        via_fwd: new_via_fwd.into(),
        via_bwd: new_via_bwd.into(),
        edge_dist_fwd: new_edge_dist_fwd.into(),
        edge_dist_bwd: new_edge_dist_bwd.into(),
        perm: inv.into(),
    }
}

/// After a rank-ordered build (`SSSPCH1B`), the *internal* vertex ID for an
/// original CSR vertex `old_id` is `n-1-rank[old_id]`. This holds because
/// `rank[perm[i]] = n-1-i` (highest rank at new_id 0). Callers that build
/// the CH from CSR can use this to translate (src, dst) pairs from CSR-IDs
/// to CH-internal IDs.
///
/// For caches loaded from disk, callers should remember the permutation
/// from build-time, or recompute via `argsort_by(rank, descending)`.
#[inline]
pub fn old_to_new_id(rank_of_old: u32, n: usize) -> u32 {
    (n as u32) - 1 - rank_of_old
}

/// Reusable scratch buffers for `query_with_path_into`. Allocate once per
/// worker thread (e.g. via rayon's `map_init`) and reuse across calls — for
/// large matrices this avoids `O(N_cells × n)` allocator pressure that
/// otherwise serialises parallel workers.
pub struct PathScratch {
    dist_f: Vec<f32>,
    dist_b: Vec<f32>,
    parent_f: Vec<(u32, u32)>,
    parent_b: Vec<(u32, u32)>,
    /// Indices into dist_f/parent_f that have been touched and must be reset
    /// to INF / (MAX, MAX) before the next call. Cheap sparse reset.
    touched_f: Vec<u32>,
    touched_b: Vec<u32>,
    hf: Vec<HItem>,
    hb: Vec<HItem>,
    /// Output path (caller may copy or borrow).
    pub path: Vec<u32>,
}

impl PathScratch {
    pub fn new(n: usize) -> Self {
        Self {
            dist_f: vec![INF; n],
            dist_b: vec![INF; n],
            parent_f: vec![(u32::MAX, u32::MAX); n],
            parent_b: vec![(u32::MAX, u32::MAX); n],
            touched_f: Vec::with_capacity(2048),
            touched_b: Vec::with_capacity(2048),
            hf: Vec::with_capacity(1024),
            hb: Vec::with_capacity(1024),
            path: Vec::with_capacity(2048),
        }
    }
    fn reset(&mut self) {
        for &v in &self.touched_f {
            self.dist_f[v as usize] = INF;
            self.parent_f[v as usize] = (u32::MAX, u32::MAX);
        }
        for &v in &self.touched_b {
            self.dist_b[v as usize] = INF;
            self.parent_b[v as usize] = (u32::MAX, u32::MAX);
        }
        self.touched_f.clear();
        self.touched_b.clear();
        self.hf.clear();
        self.hb.clear();
        self.path.clear();
    }
}

/// CH p2p query with full path. Returns (distance, path) where path is the
/// sequence of *internal* CH vertices (shortcuts unpacked recursively).
/// Allocates internally; for hot loops use `query_with_path_into`.
pub fn query_with_path(
    ch: &ContractionHierarchy,
    src: u32,
    dst: u32,
) -> Option<(f32, Vec<u32>)> {
    let mut scratch = PathScratch::new(ch.graph_fwd.n);
    let dist = query_with_path_into(ch, src, dst, &mut scratch)?;
    Some((dist, std::mem::take(&mut scratch.path)))
}

/// Same as `query_with_path` but reuses caller-provided scratch buffers.
/// On return the path is in `scratch.path`.
pub fn query_with_path_into(
    ch: &ContractionHierarchy,
    src: u32,
    dst: u32,
    scratch: &mut PathScratch,
) -> Option<f32> {
    scratch.reset();
    if src == dst {
        scratch.path.push(src);
        return Some(0.0);
    }
    {
        let dist_f = &mut scratch.dist_f;
        let dist_b = &mut scratch.dist_b;
        let parent_f = &mut scratch.parent_f;
        let parent_b = &mut scratch.parent_b;
        let touched_f = &mut scratch.touched_f;
        let touched_b = &mut scratch.touched_b;
        let hf = &mut scratch.hf;
        let hb = &mut scratch.hb;

        dist_f[src as usize] = 0.0;
        touched_f.push(src);
        dist_b[dst as usize] = 0.0;
        touched_b.push(dst);
        push(hf, 0.0, src);
        push(hb, 0.0, dst);

        let mut best = INF;
        let mut meet: u32 = u32::MAX;
        let stall = stall_enabled();

        loop {
            let tf = hf.first().map(|h| h.d).unwrap_or(INF);
            let tb = hb.first().map(|h| h.d).unwrap_or(INF);
            if tf >= best && tb >= best {
                break;
            }
            if hf.is_empty() && hb.is_empty() {
                break;
            }

            if tf <= tb && !hf.is_empty() {
                let HItem { d, v: u } = pop(hf).unwrap();
                if d > dist_f[u as usize] {
                    continue;
                }
                let total = d + dist_b[u as usize];
                if total < best {
                    best = total;
                    meet = u;
                }
                if d >= best {
                    continue;
                }
                if stall && stalled(&ch.graph_bwd, &ch.up_count_bwd, dist_f, u, d) {
                    continue;
                }
                let s = ch.graph_fwd.head[u as usize] as usize;
                let up_end = s + ch.up_count_fwd[u as usize] as usize;
                for k in s..up_end {
                    let v = ch.graph_fwd.edge_to[k];
                    let nd = d + ch.graph_fwd.edge_w[k];
                    if nd < dist_f[v as usize] {
                        if dist_f[v as usize] == INF {
                            touched_f.push(v);
                        }
                        dist_f[v as usize] = nd;
                        parent_f[v as usize] = (u, k as u32);
                        push(hf, nd, v);
                    }
                }
            } else if !hb.is_empty() {
                let HItem { d, v: u } = pop(hb).unwrap();
                if d > dist_b[u as usize] {
                    continue;
                }
                let total = d + dist_f[u as usize];
                if total < best {
                    best = total;
                    meet = u;
                }
                if d >= best {
                    continue;
                }
                if stall && stalled(&ch.graph_fwd, &ch.up_count_fwd, dist_b, u, d) {
                    continue;
                }
                let s = ch.graph_bwd.head[u as usize] as usize;
                let up_end = s + ch.up_count_bwd[u as usize] as usize;
                for k in s..up_end {
                    let v = ch.graph_bwd.edge_to[k];
                    let nd = d + ch.graph_bwd.edge_w[k];
                    if nd < dist_b[v as usize] {
                        if dist_b[v as usize] == INF {
                            touched_b.push(v);
                        }
                        dist_b[v as usize] = nd;
                        parent_b[v as usize] = (u, k as u32);
                        push(hb, nd, v);
                    }
                }
            } else {
                break;
            }
        }

        if !best.is_finite() {
            return None;
        }

        assemble_path(ch, &scratch.parent_f, &scratch.parent_b, src, dst, meet, &mut scratch.path);
        Some(best)
    }
}

/// Assemble the unpacked vertex path src → meet → dst from the two parent
/// arrays a bidirectional search produced. Appends into `out`.
fn assemble_path(
    ch: &ContractionHierarchy,
    parent_f: &[(u32, u32)],
    parent_b: &[(u32, u32)],
    src: u32,
    dst: u32,
    meet: u32,
    out: &mut Vec<u32>,
) {
    // ---- Forward path: src → meet ----
    let mut fwd_edges: Vec<(u32, u32, u32)> = Vec::with_capacity(64);
    let mut cur = meet;
    while cur != src && cur != u32::MAX {
        let (prev, edge_idx) = parent_f[cur as usize];
        if prev == u32::MAX {
            break;
        }
        fwd_edges.push((prev, edge_idx, cur));
        cur = prev;
    }
    fwd_edges.reverse();

    out.push(src);
    for &(from, idx, to) in &fwd_edges {
        unpack_edge_fwd(ch, from, idx as usize, to, out);
        out.push(to);
    }

    // ---- Backward path: meet → dst ----
    let mut cur = meet;
    let mut tmp: Vec<u32> = Vec::with_capacity(64);
    while cur != dst && cur != u32::MAX {
        let (prev, edge_idx) = parent_b[cur as usize];
        if prev == u32::MAX {
            break;
        }
        tmp.clear();
        unpack_edge_bwd(ch, prev, edge_idx as usize, cur, &mut tmp);
        tmp.reverse();
        out.extend_from_slice(&tmp);
        out.push(prev);
        cur = prev;
    }
}

/// Alternative routes via the PLATEAU/via-node method (Abraham et al.):
/// every vertex settled in both directions of the bidirectional search is a
/// candidate "via"; its route is the best src→via→dst path. Candidates are
/// kept when they are (a) not much longer than the optimum
/// (`total ≤ best × (1 + max_stretch)`) and (b) sufficiently different from
/// every already-accepted route (edge sharing ≤ `max_share`).
///
/// Returns `(distance, path)` per route, best first — `alts + 1` routes max.
pub fn query_alternatives(
    ch: &ContractionHierarchy,
    src: u32,
    dst: u32,
    scratch: &mut PathScratch,
    alts: usize,
    max_stretch: f32,
    max_share: f32,
) -> Vec<(f32, Vec<u32>)> {
    // Re-run the search, collecting every double-settled vertex. (Same loop as
    // query_with_path_into; collecting on the existing path would cost the hot
    // route query a branch per pop, so alternatives pay for their own search.)
    scratch.reset();
    if src == dst || alts == 0 {
        return match query_with_path_into(ch, src, dst, scratch) {
            Some(d) => vec![(d, scratch.path.clone())],
            None => Vec::new(),
        };
    }
    let mut meets: Vec<(u32, f32)> = Vec::new();
    let mut best = INF;
    let mut best_meet = u32::MAX;
    {
        let dist_f = &mut scratch.dist_f;
        let dist_b = &mut scratch.dist_b;
        let parent_f = &mut scratch.parent_f;
        let parent_b = &mut scratch.parent_b;
        let touched_f = &mut scratch.touched_f;
        let touched_b = &mut scratch.touched_b;
        let hf = &mut scratch.hf;
        let hb = &mut scratch.hb;
        dist_f[src as usize] = 0.0;
        touched_f.push(src);
        dist_b[dst as usize] = 0.0;
        touched_b.push(dst);
        push(hf, 0.0, src);
        push(hb, 0.0, dst);
        let stall = stall_enabled();
        // Search slightly past optimality so longer-but-different candidates
        // surface: keep popping until the frontier exceeds best × (1+stretch).
        loop {
            let tf = hf.first().map(|h| h.d).unwrap_or(INF);
            let tb = hb.first().map(|h| h.d).unwrap_or(INF);
            let horizon = if best.is_finite() { best * (1.0 + max_stretch) } else { INF };
            if tf >= horizon && tb >= horizon {
                break;
            }
            if hf.is_empty() && hb.is_empty() {
                break;
            }
            if tf <= tb && !hf.is_empty() {
                let HItem { d, v: u } = pop(hf).unwrap();
                if d > dist_f[u as usize] {
                    continue;
                }
                let total = d + dist_b[u as usize];
                if total.is_finite() {
                    meets.push((u, total));
                    if total < best {
                        best = total;
                        best_meet = u;
                    }
                }
                if d >= horizon {
                    continue;
                }
                if stall && stalled(&ch.graph_bwd, &ch.up_count_bwd, dist_f, u, d) {
                    continue;
                }
                let s = ch.graph_fwd.head[u as usize] as usize;
                let up_end = s + ch.up_count_fwd[u as usize] as usize;
                for k in s..up_end {
                    let v = ch.graph_fwd.edge_to[k];
                    let nd = d + ch.graph_fwd.edge_w[k];
                    if nd < dist_f[v as usize] {
                        if dist_f[v as usize] == INF {
                            touched_f.push(v);
                        }
                        dist_f[v as usize] = nd;
                        parent_f[v as usize] = (u, k as u32);
                        push(hf, nd, v);
                    }
                }
            } else if !hb.is_empty() {
                let HItem { d, v: u } = pop(hb).unwrap();
                if d > dist_b[u as usize] {
                    continue;
                }
                let total = d + dist_f[u as usize];
                if total.is_finite() {
                    meets.push((u, total));
                    if total < best {
                        best = total;
                        best_meet = u;
                    }
                }
                if d >= horizon {
                    continue;
                }
                if stall && stalled(&ch.graph_fwd, &ch.up_count_fwd, dist_b, u, d) {
                    continue;
                }
                let s = ch.graph_bwd.head[u as usize] as usize;
                let up_end = s + ch.up_count_bwd[u as usize] as usize;
                for k in s..up_end {
                    let v = ch.graph_bwd.edge_to[k];
                    let nd = d + ch.graph_bwd.edge_w[k];
                    if nd < dist_b[v as usize] {
                        if dist_b[v as usize] == INF {
                            touched_b.push(v);
                        }
                        dist_b[v as usize] = nd;
                        parent_b[v as usize] = (u, k as u32);
                        push(hb, nd, v);
                    }
                }
            } else {
                break;
            }
        }
    }
    if !best.is_finite() {
        return Vec::new();
    }

    // Best route first.
    let mut routes: Vec<(f32, Vec<u32>)> = Vec::with_capacity(alts + 1);
    let mut path = Vec::new();
    assemble_path(ch, &scratch.parent_f, &scratch.parent_b, src, dst, best_meet, &mut path);
    let edge_set = |p: &[u32]| -> std::collections::HashSet<(u32, u32)> {
        p.windows(2).map(|w| (w[0], w[1])).collect()
    };
    let mut accepted_edges = vec![edge_set(&path)];
    routes.push((best, path));

    // Candidate vias by total, deduped.
    meets.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    meets.dedup_by_key(|m| m.0);
    let limit = best * (1.0 + max_stretch);
    for &(via, total) in &meets {
        if routes.len() > alts {
            break;
        }
        if via == best_meet || total > limit {
            continue;
        }
        let mut p = Vec::new();
        assemble_path(ch, &scratch.parent_f, &scratch.parent_b, src, dst, via, &mut p);
        if p.len() < 2 {
            continue;
        }
        let es = edge_set(&p);
        let too_similar = accepted_edges.iter().any(|acc| {
            let common = es.iter().filter(|e| acc.contains(e)).count();
            (common as f32) / (es.len().max(1) as f32) > max_share
        });
        if too_similar {
            continue;
        }
        accepted_edges.push(es);
        routes.push((total, p));
    }
    routes
}

/// Find the edge `from → target` in the half of `from`'s adjacency where the
/// CH invariant says it must live: a shortcut's midpoint has LOWER rank than
/// both endpoints, so `from → via` is a DOWNWARD edge of `from` and
/// `via → to` is an UPWARD edge of `via`. Scanning only the right section
/// halves the unpack probes. Falls back to the other section defensively
/// (never expected; keeps unpacking correct even if an edge was misfiled).
#[inline]
fn find_edge(g: &CsrGraph, up_count: &[u32], from: u32, target: u32, upward: bool) -> Option<usize> {
    let s = g.head[from as usize] as usize;
    let e = g.head[from as usize + 1] as usize;
    let mid = s + up_count[from as usize] as usize;
    let (first, second) = if upward { (s..mid, mid..e) } else { (mid..e, s..mid) };
    for k in first {
        if g.edge_to[k] == target {
            return Some(k);
        }
    }
    for k in second {
        if g.edge_to[k] == target {
            return Some(k);
        }
    }
    None
}

/// Recursively unpack a forward shortcut. Appends INTERMEDIATE vertices
/// (excluding endpoints) to `out`.
fn unpack_edge_fwd(
    ch: &ContractionHierarchy,
    from: u32,
    edge_idx: usize,
    to: u32,
    out: &mut Vec<u32>,
) {
    let via = ch.via_fwd[edge_idx];
    if via == u32::MAX {
        // original edge — no intermediate
        return;
    }
    // The shortcut went from→via→to: from→via is downward, via→to is upward.
    if let Some(k) = find_edge(&ch.graph_fwd, &ch.up_count_fwd, from, via, false) {
        unpack_edge_fwd(ch, from, k, via, out);
    }
    out.push(via);
    if let Some(k) = find_edge(&ch.graph_fwd, &ch.up_count_fwd, via, to, true) {
        unpack_edge_fwd(ch, via, k, to, out);
    }
}

fn unpack_edge_bwd(
    ch: &ContractionHierarchy,
    from: u32,
    edge_idx: usize,
    to: u32,
    out: &mut Vec<u32>,
) {
    let via = ch.via_bwd[edge_idx];
    if via == u32::MAX {
        return;
    }
    if let Some(k) = find_edge(&ch.graph_bwd, &ch.up_count_bwd, from, via, false) {
        unpack_edge_bwd(ch, from, k, via, out);
    }
    out.push(via);
    if let Some(k) = find_edge(&ch.graph_bwd, &ch.up_count_bwd, via, to, true) {
        unpack_edge_bwd(ch, via, k, to, out);
    }
}

/// CH p2p query with batched page-cache tracking. Stages page accesses into
/// `buf` and commits them once at the end of the query. Use this in hot
/// loops (especially under rayon parallelism) — it cuts shard-lock
/// acquisitions from O(touches/query) to O(N_SHARDS/query).
#[cfg(feature = "native")]
pub fn query_paged_buf(
    ch: &ContractionHierarchy,
    layout: &ChLayout,
    pm: &PagedMmap,
    buf: &mut TouchBuf,
    src: u32,
    dst: u32,
) -> Option<f32> {
    let r = run_query_paged(ch, layout, pm, buf, src, dst);
    pm.commit(buf);
    r
}

/// CH p2p query that *also* touches the page-cache tracker on every read,
/// enabling memory-budgeted operation. Algorithmically identical to
/// `query()`; only the access bookkeeping differs.
///
/// Allocates a fresh `TouchBuf` per call. For batch workloads use
/// `query_paged_buf` with a reusable buffer.
#[cfg(feature = "native")]
pub fn query_paged(
    ch: &ContractionHierarchy,
    layout: &ChLayout,
    pm: &PagedMmap,
    src: u32,
    dst: u32,
) -> Option<f32> {
    let mut buf = TouchBuf::new();
    query_paged_buf(ch, layout, pm, &mut buf, src, dst)
}

#[cfg(feature = "native")]
fn run_query_paged(
    ch: &ContractionHierarchy,
    layout: &ChLayout,
    pm: &PagedMmap,
    buf: &mut TouchBuf,
    src: u32,
    dst: u32,
) -> Option<f32> {
    if src == dst {
        return Some(0.0);
    }
    let n = ch.graph_fwd.n;
    let mut dist_f = vec![INF; n];
    let mut dist_b = vec![INF; n];
    dist_f[src as usize] = 0.0;
    dist_b[dst as usize] = 0.0;

    let mut hf: Vec<HItem> = Vec::new();
    let mut hb: Vec<HItem> = Vec::new();
    push(&mut hf, 0.0, src);
    push(&mut hb, 0.0, dst);

    let mut best = INF;

    loop {
        let tf = hf.first().map(|h| h.d).unwrap_or(INF);
        let tb = hb.first().map(|h| h.d).unwrap_or(INF);
        if tf >= best && tb >= best {
            break;
        }
        if hf.is_empty() && hb.is_empty() {
            break;
        }

        if tf <= tb && !hf.is_empty() {
            let HItem { d, v: u } = pop(&mut hf).unwrap();
            if d > dist_f[u as usize] {
                continue;
            }
            pm.stage(buf, layout.head_fwd_off + (u as usize) * 4, 8);
            pm.stage(buf, layout.up_count_fwd_off + (u as usize) * 4, 4);

            let total = d + dist_b[u as usize];
            if total < best {
                best = total;
            }
            if d >= best {
                continue;
            }
            let s = ch.graph_fwd.head[u as usize] as usize;
            let up_end = s + ch.up_count_fwd[u as usize] as usize;
            let n_edges = up_end - s;
            if n_edges > 0 {
                pm.stage(buf, layout.edge_to_fwd_off + s * 4, n_edges * 4);
                pm.stage(buf, layout.edge_w_fwd_off + s * 4, n_edges * 4);
            }
            for k in s..up_end {
                let v = ch.graph_fwd.edge_to[k];
                let nd = d + ch.graph_fwd.edge_w[k];
                if nd < dist_f[v as usize] {
                    dist_f[v as usize] = nd;
                    push(&mut hf, nd, v);
                }
            }
        } else if !hb.is_empty() {
            let HItem { d, v: u } = pop(&mut hb).unwrap();
            if d > dist_b[u as usize] {
                continue;
            }
            pm.stage(buf, layout.head_bwd_off + (u as usize) * 4, 8);
            pm.stage(buf, layout.up_count_bwd_off + (u as usize) * 4, 4);

            let total = d + dist_f[u as usize];
            if total < best {
                best = total;
            }
            if d >= best {
                continue;
            }
            let s = ch.graph_bwd.head[u as usize] as usize;
            let up_end = s + ch.up_count_bwd[u as usize] as usize;
            let n_edges = up_end - s;
            if n_edges > 0 {
                pm.stage(buf, layout.edge_to_bwd_off + s * 4, n_edges * 4);
                pm.stage(buf, layout.edge_w_bwd_off + s * 4, n_edges * 4);
            }
            for k in s..up_end {
                let v = ch.graph_bwd.edge_to[k];
                let nd = d + ch.graph_bwd.edge_w[k];
                if nd < dist_b[v as usize] {
                    dist_b[v as usize] = nd;
                    push(&mut hb, nd, v);
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

/// CH many-to-many query: returns an `M × N` matrix of durations.
///
/// Algorithm (standard "bucket" MMM by Knopp et al.):
///
///   1. **Forward sweep over sources.** For each source `s`, run an
///      upward-only Dijeng in `graph_fwd` from `s` and store
///      `(s_idx, dist)` in every reached vertex's bucket.
///
///   2. **Backward sweep over destinations.** For each destination `t`,
///      run an upward-only Dijeng in `graph_bwd` from `t`. At every
///      reached vertex `v`, consult `buckets[v]`: each `(s_idx, d_s)` pair
///      gives a candidate `s → v → t` of length `d_s + d_t`; keep the
///      minimum per `(s_idx, t_idx)` cell.
///
/// Cost: roughly `O((M + N) · √n + M · N)` — far below `M · N` independent
/// p2p queries, since each upward sweep visits the same `O(√n)` core
/// regardless of who's doing the searching.
///
/// Variant of `matrix` that also returns per-cell distances if the CH
/// carries an `edge_dist_fwd`/`edge_dist_bwd` channel. The bucket
/// algorithm is the same — every visited vertex stores `(src_idx, dur, dist)`
/// instead of just `(src_idx, dur)` so we accumulate both metrics in a
/// single sweep. Result memory: `2 × M × N × f32`. For pure duration use
/// `matrix` (50 % less memory).
#[cfg(feature = "native")]
pub fn matrix_with_dist(
    ch: &ContractionHierarchy,
    srcs: &[u32],
    dsts: &[u32],
) -> (Vec<f32>, Vec<f32>) {
    let n = ch.graph_fwd.n;
    let n_src = srcs.len();
    let n_dst = dsts.len();
    let mut out_dur = vec![INF; n_src * n_dst];
    let mut out_dist = vec![INF; n_src * n_dst];
    if n_src == 0 || n_dst == 0 {
        return (out_dur, out_dist);
    }
    let has_dist = ch.edge_dist_fwd.len() == ch.graph_fwd.m()
        && ch.edge_dist_bwd.len() == ch.graph_bwd.m();
    if !has_dist {
        return (matrix(ch, srcs, dsts), out_dist);
    }

    // -------- Forward sweep (parallel across sources) --------
    // Each rayon worker owns its own scratch (dist arrays, heap, touched) and
    // its own accumulator of (vertex, s_idx, dur, dist) records. After the
    // parallel phase we radix-style scatter into a CSR-shaped bucket array.
    // Stall-on-demand (Knopp et al. use it in the MMM sweeps too): a stalled
    // vertex is on no shortest up-down path, so it gets NO bucket entry and no
    // expansion — shrinking both the search and, crucially, the bucket lists
    // the backward sweep has to scan.
    let stall = stall_enabled();
    type FwdEntry = (u32, u32, f32, f32);
    let entries: Vec<FwdEntry> = srcs
        .par_iter()
        .enumerate()
        .fold(
            || {
                (
                    vec![INF; n], // dist_dur
                    vec![INF; n], // dist_dist
                    Vec::<u32>::with_capacity(2048),
                    Vec::<HItem>::with_capacity(1024),
                    Vec::<FwdEntry>::with_capacity(8192),
                )
            },
            |(mut dist_dur, mut dist_dist, mut touched, mut heap, mut acc),
             (s_idx, &src)| {
                for &v in &touched {
                    dist_dur[v as usize] = INF;
                    dist_dist[v as usize] = INF;
                }
                touched.clear();
                heap.clear();

                dist_dur[src as usize] = 0.0;
                dist_dist[src as usize] = 0.0;
                touched.push(src);
                push(&mut heap, 0.0, src);

                while let Some(HItem { d, v: u }) = pop(&mut heap) {
                    if d > dist_dur[u as usize] {
                        continue;
                    }
                    if stall && stalled(&ch.graph_bwd, &ch.up_count_bwd, &dist_dur, u, d) {
                        continue;
                    }
                    acc.push((u, s_idx as u32, d, dist_dist[u as usize]));
                    let s = ch.graph_fwd.head[u as usize] as usize;
                    let up_end = s + ch.up_count_fwd[u as usize] as usize;
                    for k in s..up_end {
                        let w = ch.graph_fwd.edge_to[k];
                        let nd = d + ch.graph_fwd.edge_w[k];
                        let ndist = dist_dist[u as usize] + ch.edge_dist_fwd[k];
                        if nd < dist_dur[w as usize] {
                            if dist_dur[w as usize] == INF {
                                touched.push(w);
                            }
                            dist_dur[w as usize] = nd;
                            dist_dist[w as usize] = ndist;
                            push(&mut heap, nd, w);
                        }
                    }
                }
                (dist_dur, dist_dist, touched, heap, acc)
            },
        )
        .map(|(_, _, _, _, acc)| acc)
        .reduce(Vec::new, |mut a, b| {
            if a.is_empty() {
                b
            } else {
                a.reserve(b.len());
                a.extend(b);
                a
            }
        });

    // -------- Build CSR-shaped bucket layout --------
    // bucket_head[u..u+1] indexes bucket_data; one contiguous slice per vertex
    // means the backward sweep gets cache-friendly reads.
    let mut bucket_head = vec![0u32; n + 1];
    for &(u, _, _, _) in &entries {
        bucket_head[u as usize + 1] += 1;
    }
    for i in 0..n {
        bucket_head[i + 1] += bucket_head[i];
    }
    let total = entries.len();
    let mut bucket_data: Vec<(u32, f32, f32)> = vec![(0, 0.0, 0.0); total];
    let mut cursor: Vec<u32> = bucket_head[..n].to_vec();
    for &(u, s_idx, dur, dist) in &entries {
        let pos = cursor[u as usize] as usize;
        cursor[u as usize] += 1;
        bucket_data[pos] = (s_idx, dur, dist);
    }
    drop(entries);
    drop(cursor);

    // -------- Backward sweep (parallel across destinations) --------
    // Each thread handles a slice of destinations. For its t_idx values, it
    // writes only to cells `s * n_dst + t_idx` — disjoint across threads, so
    // the unsafe pointer writes are race-free. Pointers are passed as `usize`
    // (Send+Sync) and reinterpreted inside the closure.
    let dur_addr = out_dur.as_mut_ptr() as usize;
    let dist_addr = out_dist.as_mut_ptr() as usize;

    dsts.par_iter().enumerate().for_each_init(
        || {
            (
                vec![INF; n], // dist_dur
                vec![INF; n], // dist_dist
                Vec::<u32>::with_capacity(2048),
                Vec::<HItem>::with_capacity(1024),
            )
        },
        |(dist_dur, dist_dist, touched, heap), (t_idx, &dst)| {
            for &v in touched.iter() {
                dist_dur[v as usize] = INF;
                dist_dist[v as usize] = INF;
            }
            touched.clear();
            heap.clear();

            dist_dur[dst as usize] = 0.0;
            dist_dist[dst as usize] = 0.0;
            touched.push(dst);
            push(heap, 0.0, dst);

            while let Some(HItem { d, v: u }) = pop(heap) {
                if d > dist_dur[u as usize] {
                    continue;
                }
                if stall && stalled(&ch.graph_fwd, &ch.up_count_fwd, dist_dur, u, d) {
                    continue;
                }
                let head_u = bucket_head[u as usize] as usize;
                let tail_u = bucket_head[u as usize + 1] as usize;
                let dist_u_dist = dist_dist[u as usize];
                let dur_base = dur_addr as *mut f32;
                let dist_base = dist_addr as *mut f32;
                for &(s_idx, d_s, dist_s) in &bucket_data[head_u..tail_u] {
                    let cell = (s_idx as usize) * n_dst + t_idx;
                    let total_dur = d_s + d;
                    unsafe {
                        let dur_p = dur_base.add(cell);
                        if total_dur < *dur_p {
                            *dur_p = total_dur;
                            *dist_base.add(cell) = dist_s + dist_u_dist;
                        }
                    }
                }
                let s = ch.graph_bwd.head[u as usize] as usize;
                let up_end = s + ch.up_count_bwd[u as usize] as usize;
                for k in s..up_end {
                    let w = ch.graph_bwd.edge_to[k];
                    let nd = d + ch.graph_bwd.edge_w[k];
                    let ndist = dist_dist[u as usize] + ch.edge_dist_bwd[k];
                    if nd < dist_dur[w as usize] {
                        if dist_dur[w as usize] == INF {
                            touched.push(w);
                        }
                        dist_dur[w as usize] = nd;
                        dist_dist[w as usize] = ndist;
                        push(heap, nd, w);
                    }
                }
            }
        },
    );

    (out_dur, out_dist)
}

/// Chunked-srcs variant of `matrix_with_dist` with bounded peak RAM. Splits
/// the source list into batches of `src_chunk` and processes each batch
/// independently: forward sweep over the batch, backward sweep over all
/// destinations, callback fires with the batch's `K × n_dst` row-major
/// duration + distance blocks, then memory is released before the next batch.
///
/// Use this when the full `n_src × n_dst × 8` byte output won't fit in RAM —
/// e.g. 50k × 50k = 25 GB on a 36 GB box. Peak RAM per batch is roughly
/// `src_chunk × n_dst × 8` plus working state (~2 GB on London).
///
/// `on_chunk(s_start, s_end, dur, dist)` receives the block for sources
/// `[s_start, s_end)` over all destinations. `dur.len() == dist.len()
/// == (s_end - s_start) * dsts.len()` row-major.
///
/// Cost overhead vs `matrix_with_dist`: each batch re-runs the per-dst
/// backward Dijeng (cheap on CH, ~1 ms each). Bucket scans are unchanged
/// in total work.
#[cfg(feature = "native")]
pub fn matrix_with_dist_chunked<F>(
    ch: &ContractionHierarchy,
    srcs: &[u32],
    dsts: &[u32],
    src_chunk: usize,
    mut on_chunk: F,
) where
    F: FnMut(usize, usize, &[f32], &[f32]),
{
    let n = ch.graph_fwd.n;
    let n_src = srcs.len();
    let n_dst = dsts.len();
    if n_src == 0 || n_dst == 0 {
        return;
    }
    let has_dist = ch.edge_dist_fwd.len() == ch.graph_fwd.m()
        && ch.edge_dist_bwd.len() == ch.graph_bwd.m();
    assert!(
        has_dist,
        "matrix_with_dist_chunked requires a dual-channel CH (SSSPCH1D)"
    );
    let chunk = src_chunk.max(1).min(n_src);
    let stall = stall_enabled();

    let mut s_start = 0;
    while s_start < n_src {
        let s_end = (s_start + chunk).min(n_src);
        let k = s_end - s_start;
        let batch_srcs = &srcs[s_start..s_end];

        // -------- Forward sweep over this batch --------
        type FwdEntry = (u32, u32, f32, f32);
        let entries: Vec<FwdEntry> = batch_srcs
            .par_iter()
            .enumerate()
            .fold(
                || {
                    (
                        vec![INF; n],
                        vec![INF; n],
                        Vec::<u32>::with_capacity(2048),
                        Vec::<HItem>::with_capacity(1024),
                        Vec::<FwdEntry>::with_capacity(8192),
                    )
                },
                |(mut dist_dur, mut dist_dist, mut touched, mut heap, mut acc),
                 (local_s_idx, &src)| {
                    for &v in &touched {
                        dist_dur[v as usize] = INF;
                        dist_dist[v as usize] = INF;
                    }
                    touched.clear();
                    heap.clear();

                    dist_dur[src as usize] = 0.0;
                    dist_dist[src as usize] = 0.0;
                    touched.push(src);
                    push(&mut heap, 0.0, src);

                    while let Some(HItem { d, v: u }) = pop(&mut heap) {
                        if d > dist_dur[u as usize] {
                            continue;
                        }
                        if stall && stalled(&ch.graph_bwd, &ch.up_count_bwd, &dist_dur, u, d) {
                            continue;
                        }
                        acc.push((u, local_s_idx as u32, d, dist_dist[u as usize]));
                        let s = ch.graph_fwd.head[u as usize] as usize;
                        let up_end = s + ch.up_count_fwd[u as usize] as usize;
                        for k in s..up_end {
                            let w = ch.graph_fwd.edge_to[k];
                            let nd = d + ch.graph_fwd.edge_w[k];
                            let ndist = dist_dist[u as usize] + ch.edge_dist_fwd[k];
                            if nd < dist_dur[w as usize] {
                                if dist_dur[w as usize] == INF {
                                    touched.push(w);
                                }
                                dist_dur[w as usize] = nd;
                                dist_dist[w as usize] = ndist;
                                push(&mut heap, nd, w);
                            }
                        }
                    }
                    (dist_dur, dist_dist, touched, heap, acc)
                },
            )
            .map(|(_, _, _, _, acc)| acc)
            .reduce(Vec::new, |mut a, b| {
                if a.is_empty() {
                    b
                } else {
                    a.reserve(b.len());
                    a.extend(b);
                    a
                }
            });

        // CSR-shaped buckets over this batch's entries.
        let mut bucket_head = vec![0u32; n + 1];
        for &(u, _, _, _) in &entries {
            bucket_head[u as usize + 1] += 1;
        }
        for i in 0..n {
            bucket_head[i + 1] += bucket_head[i];
        }
        let total = entries.len();
        let mut bucket_data: Vec<(u32, f32, f32)> = vec![(0, 0.0, 0.0); total];
        let mut cursor: Vec<u32> = bucket_head[..n].to_vec();
        for &(u, s_idx, dur, dist) in &entries {
            let pos = cursor[u as usize] as usize;
            cursor[u as usize] += 1;
            bucket_data[pos] = (s_idx, dur, dist);
        }
        drop(entries);
        drop(cursor);

        // Allocate the batch output: k × n_dst row-major.
        let mut out_dur = vec![INF; k * n_dst];
        let mut out_dist = vec![INF; k * n_dst];
        let dur_addr = out_dur.as_mut_ptr() as usize;
        let dist_addr = out_dist.as_mut_ptr() as usize;

        // Backward sweep over all destinations in parallel.
        dsts.par_iter().enumerate().for_each_init(
            || {
                (
                    vec![INF; n],
                    vec![INF; n],
                    Vec::<u32>::with_capacity(2048),
                    Vec::<HItem>::with_capacity(1024),
                )
            },
            |(dist_dur, dist_dist, touched, heap), (t_idx, &dst)| {
                for &v in touched.iter() {
                    dist_dur[v as usize] = INF;
                    dist_dist[v as usize] = INF;
                }
                touched.clear();
                heap.clear();

                dist_dur[dst as usize] = 0.0;
                dist_dist[dst as usize] = 0.0;
                touched.push(dst);
                push(heap, 0.0, dst);

                while let Some(HItem { d, v: u }) = pop(heap) {
                    if d > dist_dur[u as usize] {
                        continue;
                    }
                    if stall && stalled(&ch.graph_fwd, &ch.up_count_fwd, dist_dur, u, d) {
                        continue;
                    }
                    let head_u = bucket_head[u as usize] as usize;
                    let tail_u = bucket_head[u as usize + 1] as usize;
                    let dist_u_dist = dist_dist[u as usize];
                    let dur_base = dur_addr as *mut f32;
                    let dist_base = dist_addr as *mut f32;
                    for &(local_s, d_s, dist_s) in &bucket_data[head_u..tail_u] {
                        let cell = (local_s as usize) * n_dst + t_idx;
                        let total_dur = d_s + d;
                        unsafe {
                            let dur_p = dur_base.add(cell);
                            if total_dur < *dur_p {
                                *dur_p = total_dur;
                                *dist_base.add(cell) = dist_s + dist_u_dist;
                            }
                        }
                    }
                    let s = ch.graph_bwd.head[u as usize] as usize;
                    let up_end = s + ch.up_count_bwd[u as usize] as usize;
                    for k in s..up_end {
                        let w = ch.graph_bwd.edge_to[k];
                        let nd = d + ch.graph_bwd.edge_w[k];
                        let ndist = dist_dist[u as usize] + ch.edge_dist_bwd[k];
                        if nd < dist_dur[w as usize] {
                            if dist_dur[w as usize] == INF {
                                touched.push(w);
                            }
                            dist_dur[w as usize] = nd;
                            dist_dist[w as usize] = ndist;
                            push(heap, nd, w);
                        }
                    }
                }
            },
        );

        on_chunk(s_start, s_end, &out_dur, &out_dist);
        // out_dur, out_dist, bucket_* drop here.
        s_start = s_end;
    }
}

/// Inputs are CH-internal vertex IDs (the same numbering `query` uses).
/// Returns `result[s_idx * dsts.len() + t_idx]`.
#[cfg(feature = "native")]
pub fn matrix(ch: &ContractionHierarchy, srcs: &[u32], dsts: &[u32]) -> Vec<f32> {
    let n = ch.graph_fwd.n;
    let n_src = srcs.len();
    let n_dst = dsts.len();
    let mut out = vec![INF; n_src * n_dst];
    if n_src == 0 || n_dst == 0 {
        return out;
    }
    let stall = stall_enabled();

    // Forward sweep parallelised across sources; same fold-then-scatter
    // strategy as `matrix_with_dist`.
    type FwdEntry = (u32, u32, f32);
    let entries: Vec<FwdEntry> = srcs
        .par_iter()
        .enumerate()
        .fold(
            || {
                (
                    vec![INF; n],
                    Vec::<u32>::with_capacity(2048),
                    Vec::<HItem>::with_capacity(1024),
                    Vec::<FwdEntry>::with_capacity(8192),
                )
            },
            |(mut dist, mut touched, mut heap, mut acc), (s_idx, &src)| {
                for &v in &touched {
                    dist[v as usize] = INF;
                }
                touched.clear();
                heap.clear();

                dist[src as usize] = 0.0;
                touched.push(src);
                push(&mut heap, 0.0, src);

                while let Some(HItem { d, v: u }) = pop(&mut heap) {
                    if d > dist[u as usize] {
                        continue;
                    }
                    if stall && stalled(&ch.graph_bwd, &ch.up_count_bwd, &dist, u, d) {
                        continue;
                    }
                    acc.push((u, s_idx as u32, d));
                    let s = ch.graph_fwd.head[u as usize] as usize;
                    let up_end = s + ch.up_count_fwd[u as usize] as usize;
                    for k in s..up_end {
                        let w = ch.graph_fwd.edge_to[k];
                        let nd = d + ch.graph_fwd.edge_w[k];
                        if nd < dist[w as usize] {
                            if dist[w as usize] == INF {
                                touched.push(w);
                            }
                            dist[w as usize] = nd;
                            push(&mut heap, nd, w);
                        }
                    }
                }
                (dist, touched, heap, acc)
            },
        )
        .map(|(_, _, _, acc)| acc)
        .reduce(Vec::new, |mut a, b| {
            if a.is_empty() {
                b
            } else {
                a.reserve(b.len());
                a.extend(b);
                a
            }
        });

    // CSR-shaped bucket layout for cache-friendly backward sweep.
    let mut bucket_head = vec![0u32; n + 1];
    for &(u, _, _) in &entries {
        bucket_head[u as usize + 1] += 1;
    }
    for i in 0..n {
        bucket_head[i + 1] += bucket_head[i];
    }
    let total = entries.len();
    let mut bucket_data: Vec<(u32, f32)> = vec![(0, 0.0); total];
    let mut cursor: Vec<u32> = bucket_head[..n].to_vec();
    for &(u, s_idx, dur) in &entries {
        let pos = cursor[u as usize] as usize;
        cursor[u as usize] += 1;
        bucket_data[pos] = (s_idx, dur);
    }
    drop(entries);
    drop(cursor);

    // Backward sweep parallelised across destinations; each thread writes
    // only to cells with its t_idx, so the unsafe pointer writes are
    // race-free.
    let out_addr = out.as_mut_ptr() as usize;
    dsts.par_iter().enumerate().for_each_init(
        || {
            (
                vec![INF; n],
                Vec::<u32>::with_capacity(2048),
                Vec::<HItem>::with_capacity(1024),
            )
        },
        |(dist, touched, heap), (t_idx, &dst)| {
            for &v in touched.iter() {
                dist[v as usize] = INF;
            }
            touched.clear();
            heap.clear();

            dist[dst as usize] = 0.0;
            touched.push(dst);
            push(heap, 0.0, dst);

            while let Some(HItem { d, v: u }) = pop(heap) {
                if d > dist[u as usize] {
                    continue;
                }
                if stall && stalled(&ch.graph_fwd, &ch.up_count_fwd, dist, u, d) {
                    continue;
                }
                let head_u = bucket_head[u as usize] as usize;
                let tail_u = bucket_head[u as usize + 1] as usize;
                let out_base = out_addr as *mut f32;
                for &(s_idx, d_s) in &bucket_data[head_u..tail_u] {
                    let cell = (s_idx as usize) * n_dst + t_idx;
                    let total = d_s + d;
                    unsafe {
                        let p = out_base.add(cell);
                        if total < *p {
                            *p = total;
                        }
                    }
                }
                let s = ch.graph_bwd.head[u as usize] as usize;
                let up_end = s + ch.up_count_bwd[u as usize] as usize;
                for k in s..up_end {
                    let w = ch.graph_bwd.edge_to[k];
                    let nd = d + ch.graph_bwd.edge_w[k];
                    if nd < dist[w as usize] {
                        if dist[w as usize] == INF {
                            touched.push(w);
                        }
                        dist[w as usize] = nd;
                        push(heap, nd, w);
                    }
                }
            }
        },
    );

    out
}

/// CH p2p query: bidirectional Dijeng som kun relaxer upward-kanter.
pub fn query(ch: &ContractionHierarchy, src: u32, dst: u32) -> Option<f32> {
    if src == dst {
        return Some(0.0);
    }
    let n = ch.graph_fwd.n;
    let mut dist_f = vec![INF; n];
    let mut dist_b = vec![INF; n];
    dist_f[src as usize] = 0.0;
    dist_b[dst as usize] = 0.0;

    let mut hf: Vec<HItem> = Vec::new();
    let mut hb: Vec<HItem> = Vec::new();
    push(&mut hf, 0.0, src);
    push(&mut hb, 0.0, dst);

    let mut best = INF;
    let stall = stall_enabled();

    loop {
        let tf = hf.first().map(|h| h.d).unwrap_or(INF);
        let tb = hb.first().map(|h| h.d).unwrap_or(INF);
        // Optimal stop: no shorter meeting can occur.
        if tf >= best && tb >= best {
            break;
        }
        if hf.is_empty() && hb.is_empty() {
            break;
        }

        // Pick direction: smallest top.
        if tf <= tb && !hf.is_empty() {
            let HItem { d, v: u } = pop(&mut hf).unwrap();
            if d > dist_f[u as usize] {
                continue;
            }
            // Check meeting: has u been reached in backward too?
            let total = d + dist_b[u as usize];
            if total < best {
                best = total;
            }
            // Stop if this direction's top ≥ best.
            if d >= best {
                continue;
            }
            if stall && stalled(&ch.graph_bwd, &ch.up_count_bwd, &dist_f, u, d) {
                continue;
            }
            // Relax only upward edges: the first up_count[u] edges.
            let s = ch.graph_fwd.head[u as usize] as usize;
            let up_end = s + ch.up_count_fwd[u as usize] as usize;
            for k in s..up_end {
                let v = ch.graph_fwd.edge_to[k];
                let nd = d + ch.graph_fwd.edge_w[k];
                if nd < dist_f[v as usize] {
                    dist_f[v as usize] = nd;
                    push(&mut hf, nd, v);
                }
            }
        } else if !hb.is_empty() {
            let HItem { d, v: u } = pop(&mut hb).unwrap();
            if d > dist_b[u as usize] {
                continue;
            }
            let total = d + dist_f[u as usize];
            if total < best {
                best = total;
            }
            if d >= best {
                continue;
            }
            if stall && stalled(&ch.graph_fwd, &ch.up_count_fwd, &dist_b, u, d) {
                continue;
            }
            let s = ch.graph_bwd.head[u as usize] as usize;
            let up_end = s + ch.up_count_bwd[u as usize] as usize;
            for k in s..up_end {
                let v = ch.graph_bwd.edge_to[k];
                let nd = d + ch.graph_bwd.edge_w[k];
                if nd < dist_b[v as usize] {
                    dist_b[v as usize] = nd;
                    push(&mut hb, nd, v);
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

/// Distance-only p2p query on reused scratch: the same bidirectional upward
/// search as `query_with_path_into` (incl. stall-on-demand) but with no
/// parent tracking and no path unpacking — the cheapest exact point-to-point
/// call for consumers that only need the cost (the VRP solver, the matrix
/// broker's probe path, the HTTP API's `?dist_only`). Uses the same
/// `PathScratch` so callers can mix the two per query.
pub fn query_dist_into(
    ch: &ContractionHierarchy,
    src: u32,
    dst: u32,
    scratch: &mut PathScratch,
) -> Option<f32> {
    scratch.reset();
    if src == dst {
        return Some(0.0);
    }
    let dist_f = &mut scratch.dist_f;
    let dist_b = &mut scratch.dist_b;
    let touched_f = &mut scratch.touched_f;
    let touched_b = &mut scratch.touched_b;
    let hf = &mut scratch.hf;
    let hb = &mut scratch.hb;

    dist_f[src as usize] = 0.0;
    touched_f.push(src);
    dist_b[dst as usize] = 0.0;
    touched_b.push(dst);
    push(hf, 0.0, src);
    push(hb, 0.0, dst);

    let mut best = INF;
    let stall = stall_enabled();

    loop {
        let tf = hf.first().map(|h| h.d).unwrap_or(INF);
        let tb = hb.first().map(|h| h.d).unwrap_or(INF);
        if tf >= best && tb >= best {
            break;
        }
        if hf.is_empty() && hb.is_empty() {
            break;
        }

        if tf <= tb && !hf.is_empty() {
            let HItem { d, v: u } = pop(hf).unwrap();
            if d > dist_f[u as usize] {
                continue;
            }
            let total = d + dist_b[u as usize];
            if total < best {
                best = total;
            }
            if d >= best {
                continue;
            }
            if stall && stalled(&ch.graph_bwd, &ch.up_count_bwd, dist_f, u, d) {
                continue;
            }
            let s = ch.graph_fwd.head[u as usize] as usize;
            let up_end = s + ch.up_count_fwd[u as usize] as usize;
            for k in s..up_end {
                let v = ch.graph_fwd.edge_to[k];
                let nd = d + ch.graph_fwd.edge_w[k];
                if nd < dist_f[v as usize] {
                    if dist_f[v as usize] == INF {
                        touched_f.push(v);
                    }
                    dist_f[v as usize] = nd;
                    push(hf, nd, v);
                }
            }
        } else if !hb.is_empty() {
            let HItem { d, v: u } = pop(hb).unwrap();
            if d > dist_b[u as usize] {
                continue;
            }
            let total = d + dist_f[u as usize];
            if total < best {
                best = total;
            }
            if d >= best {
                continue;
            }
            if stall && stalled(&ch.graph_fwd, &ch.up_count_fwd, dist_b, u, d) {
                continue;
            }
            let s = ch.graph_bwd.head[u as usize] as usize;
            let up_end = s + ch.up_count_bwd[u as usize] as usize;
            for k in s..up_end {
                let v = ch.graph_bwd.edge_to[k];
                let nd = d + ch.graph_bwd.edge_w[k];
                if nd < dist_b[v as usize] {
                    if dist_b[v as usize] == INF {
                        touched_b.push(v);
                    }
                    dist_b[v as usize] = nd;
                    push(hb, nd, v);
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

// =============================================================================
// Internals: contraction, witness search, priority
// =============================================================================

/// Witness search state (reused to avoid allocation per call).
struct WitnessState {
    dist: Vec<f32>,
    hops: Vec<u32>,
    timestamp: Vec<u64>,
    current_ts: u64,
    heap: Vec<HItem3>,
    touched: Vec<u32>, // to reset only the touched ones
}

#[derive(Clone, Copy)]
struct HItem3 {
    d: f32,
    v: u32,
    h: u32,
}

impl WitnessState {
    fn new(n: usize) -> Self {
        Self {
            dist: vec![f32::INFINITY; n],
            hops: vec![0; n],
            timestamp: vec![0; n],
            current_ts: 0,
            heap: Vec::new(),
            touched: Vec::new(),
        }
    }

    /// Returns true if there's a path from `src` to `target` (avoiding `exclude`)
    /// of length ≤ `limit`, using at most `hops_limit` edges.
    fn search(
        &mut self,
        fwd: &[Vec<(u32, f32, f32, u32)>],
        contracted: &[bool],
        src: u32,
        target: u32,
        exclude: u32,
        limit: f32,
        hops_limit: u32,
    ) -> bool {
        // Reset only the touched cells.
        for &v in &self.touched {
            self.timestamp[v as usize] = self.current_ts; // will be < current_ts+1
        }
        self.touched.clear();
        self.current_ts += 1;
        let ts = self.current_ts;
        self.heap.clear();

        self.dist[src as usize] = 0.0;
        self.hops[src as usize] = 0;
        self.timestamp[src as usize] = ts;
        self.touched.push(src);
        self.heap.push(HItem3 {
            d: 0.0,
            v: src,
            h: 0,
        });
        let last = self.heap.len() - 1;
        sift_up_h3(&mut self.heap, last);

        while let Some(top) = pop_h3(&mut self.heap) {
            if top.d > limit {
                return false; // sorted by dist; none can be smaller.
            }
            if top.v == target {
                return true;
            }
            // Stale (slipped through due to lazy delete).
            if self.timestamp[top.v as usize] != ts || top.d > self.dist[top.v as usize] {
                continue;
            }
            if top.h >= hops_limit {
                continue;
            }
            for &(w, ew, _, _via) in &fwd[top.v as usize] {
                if w == exclude || contracted[w as usize] {
                    continue;
                }
                let nd = top.d + ew;
                if nd > limit {
                    continue;
                }
                let prev = if self.timestamp[w as usize] == ts {
                    self.dist[w as usize]
                } else {
                    f32::INFINITY
                };
                if nd < prev {
                    self.dist[w as usize] = nd;
                    self.hops[w as usize] = top.h + 1;
                    if self.timestamp[w as usize] != ts {
                        self.timestamp[w as usize] = ts;
                        self.touched.push(w);
                    }
                    self.heap.push(HItem3 {
                        d: nd,
                        v: w,
                        h: top.h + 1,
                    });
                    let last = self.heap.len() - 1;
                    sift_up_h3(&mut self.heap, last);
                    if w == target && nd <= limit {
                        // Ikke sleng tilbake umiddelbart; la heap bekrefte at det
                        // er korteste — men for witness-test holder det at vi vet
                        // det er en lovlig vei innen limits.
                        return true;
                    }
                }
            }
        }
        false
    }
}

fn sift_up_h3(h: &mut Vec<HItem3>, mut i: usize) {
    while i > 0 {
        let p = (i - 1) >> 2;
        if h[p].d <= h[i].d {
            break;
        }
        h.swap(p, i);
        i = p;
    }
}
fn pop_h3(h: &mut Vec<HItem3>) -> Option<HItem3> {
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

/// Contract v: for hvert par (u, w) av in/out-naboer, sjekk om vi trenger
/// shortcut. Legg til shortcuts in-place i `fwd`/`bwd`. Returnerer antall
/// nye shortcuts.
/// Dry-run of `contract_vertex`: count the shortcuts contraction of `v` would
/// add, without mutating the graph. Used by the edge-difference ordering.
/// (May differ marginally from the real count because shortcuts added mid-
/// contraction are invisible to later witness searches here — fine for a
/// priority heuristic.)
fn simulate_contract(
    fwd: &[Vec<(u32, f32, f32, u32)>],
    bwd: &[Vec<(u32, f32, f32, u32)>],
    contracted: &[bool],
    v: u32,
    wstate: &mut WitnessState,
) -> usize {
    let v_us = v as usize;
    let mut added = 0usize;
    for &(u, wuv, _, _) in &bwd[v_us] {
        if u == v || contracted[u as usize] {
            continue;
        }
        for &(w, wvw, _, _) in &fwd[v_us] {
            if w == v || w == u || contracted[w as usize] {
                continue;
            }
            let shortcut_w = wuv + wvw;
            let mut existing_better = false;
            for &(t, ew, _, _) in &fwd[u as usize] {
                if t == w && ew <= shortcut_w {
                    existing_better = true;
                    break;
                }
            }
            if existing_better {
                continue;
            }
            if wstate.search(fwd, contracted, u, w, v, shortcut_w, WITNESS_HOPS_LIMIT) {
                continue;
            }
            added += 1;
        }
    }
    added
}

/// Edge-difference ordering priority (RoutingKit/OSRM-style): how much does
/// contracting `v` grow the graph, weighted, plus a spatial-uniformity term.
///   ED        = simulated shortcuts − active degree (edges removed)
///   deleted   = #already-contracted neighbours (spreads contraction evenly,
///               avoiding deep "towers" in one region)
/// Lower = contract earlier. The witness-limited simulation makes this far
/// more accurate than the degree product, at the cost of running a small
/// witness search per (re-)evaluation — the lazy-update loop keeps the total
/// number of evaluations near-linear.
fn edge_diff_priority(
    fwd: &[Vec<(u32, f32, f32, u32)>],
    bwd: &[Vec<(u32, f32, f32, u32)>],
    contracted: &[bool],
    deleted_neighbors: &[u32],
    v: u32,
    wstate: &mut WitnessState,
) -> i32 {
    let shortcuts = simulate_contract(fwd, bwd, contracted, v, wstate) as i32;
    let din = bwd[v as usize]
        .iter()
        .filter(|&&(u, _, _, _)| !contracted[u as usize])
        .count() as i32;
    let dout = fwd[v as usize]
        .iter()
        .filter(|&&(u, _, _, _)| !contracted[u as usize])
        .count() as i32;
    4 * (2 * shortcuts - (din + dout)) + deleted_neighbors[v as usize] as i32
}

/// Which node-ordering heuristic the CH build uses. Default is the
/// edge-difference ordering (measured: fewer shortcuts, smaller query search
/// space); `DIJENG_CH_ORDER=degree` restores the old degree product.
fn use_edge_diff_order() -> bool {
    match std::env::var("DIJENG_CH_ORDER") {
        Ok(s) => s != "degree",
        Err(_) => true,
    }
}

fn contract_vertex(
    fwd: &mut Vec<Vec<(u32, f32, f32, u32)>>,
    bwd: &mut Vec<Vec<(u32, f32, f32, u32)>>,
    contracted: &[bool],
    v: u32,
    wstate: &mut WitnessState,
) -> usize {
    let v_us = v as usize;
    let bwd_v = bwd[v_us].clone();
    let fwd_v = fwd[v_us].clone();

    let mut max_out_w: f32 = 0.0;
    for &(_, w, _, _) in &fwd_v {
        if w > max_out_w {
            max_out_w = w;
        }
    }

    let mut added = 0usize;
    for &(u, wuv, duv, _) in &bwd_v {
        if u == v || contracted[u as usize] {
            continue;
        }
        let limit_global = wuv + max_out_w;
        let _ = limit_global;
        for &(w, wvw, dvw, _) in &fwd_v {
            if w == v || w == u || contracted[w as usize] {
                continue;
            }
            let shortcut_w = wuv + wvw;
            let shortcut_d = duv + dvw;
            let mut existing_better = false;
            for &(t, ew, _, _) in &fwd[u as usize] {
                if t == w && ew <= shortcut_w {
                    existing_better = true;
                    break;
                }
            }
            if existing_better {
                continue;
            }
            let has_witness = wstate.search(
                fwd,
                contracted,
                u,
                w,
                v,
                shortcut_w,
                WITNESS_HOPS_LIMIT,
            );
            if has_witness {
                continue;
            }
            update_or_add(&mut fwd[u as usize], w, shortcut_w, shortcut_d, v);
            update_or_add(&mut bwd[w as usize], u, shortcut_w, shortcut_d, v);
            added += 1;
        }
    }
    added
}

/// Hvis det allerede finnes en kant fra source til target, og den nye vekten
/// er bedre, oppdater. Ellers legg til ny kant. `dist` is the parallel
/// distance metric carried alongside `w` (e.g. metres while `w` is seconds).
fn update_or_add(adj: &mut Vec<(u32, f32, f32, u32)>, target: u32, w: f32, dist: f32, via: u32) {
    for e in adj.iter_mut() {
        if e.0 == target {
            if w < e.1 {
                e.1 = w;
                e.2 = dist;
                e.3 = via;
            }
            return;
        }
    }
    adj.push((target, w, dist, via));
}

/// Lett degree-basert priority — ingen witness search. Brukes for initial
/// ordering og oppdatering av naboer (rask).
fn degree_priority(
    fwd: &[Vec<(u32, f32, f32, u32)>],
    bwd: &[Vec<(u32, f32, f32, u32)>],
    v: u32,
) -> i32 {
    // Heuristic: in-degree × out-degree is a good estimate of the number of
    // potential shortcuts (worst case). We want to contract first where
    // this number is lowest.
    let din = bwd[v as usize].len() as i32;
    let dout = fwd[v as usize].len() as i32;
    din * dout - (din + dout)
}

// `priority` (edge-difference) was an alternative ordering heuristic — kept
// historically but not used since `degree_priority` performs well enough on
// road networks. Removed during the dual-channel refactor.

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
