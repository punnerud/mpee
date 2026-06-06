//! Granular neighborhoods (Toth & Vigo).
//!
//! For each matrix location we precompute the indices of the K nearest
//! locations (by duration). Local-search operators consult this table to
//! skip moves whose target is far from the source — those almost never
//! improve a route, and discarding them up front is the dominant LS
//! speedup on instances of N ≥ 200.
//!
//! Default `K = 20` is the value commonly cited in the Toth-Vigo literature
//! and gives a good speed/quality trade. Lower K → faster but more local
//! optima; higher K → closer to vanilla LS.

use crate::matrix::Matrix;
use crate::problem::{Problem, TimeWindow};

#[derive(Debug, Clone)]
pub struct Granular {
    pub k: usize,
    /// Flat: `near[i * k + r]` is the r-th nearest location to `i`.
    /// Padded with `i` itself when fewer than k other locations exist.
    near: Vec<u32>,
    /// Number of valid neighbors stored for each `i` (≤ k).
    counts: Vec<u32>,
    n: usize,
}

impl Granular {
    pub fn build(matrix: &Matrix, k: usize) -> Self {
        let n = matrix.n;
        let k_eff = k.min(n.saturating_sub(1)).max(1);
        let mut near: Vec<u32> = vec![0u32; n * k_eff];
        let mut counts: Vec<u32> = vec![0u32; n];
        let mut buf: Vec<(i32, u32)> = Vec::with_capacity(n);
        for i in 0..n {
            buf.clear();
            for j in 0..n {
                if j == i { continue; }
                buf.push((matrix.durations[i * n + j], j as u32));
            }
            // Partial sort to k_eff nearest.
            if buf.len() > k_eff {
                buf.select_nth_unstable_by_key(k_eff - 1, |x| x.0);
                buf.truncate(k_eff);
            }
            buf.sort_unstable_by_key(|x| x.0);
            for (r, &(_, j)) in buf.iter().enumerate() {
                near[i * k_eff + r] = j;
            }
            counts[i] = buf.len() as u32;
        }
        Self { k: k_eff, near, counts, n }
    }

    /// Time-window-aware granular neighbourhood (Vidal et al. 2013, the proximity
    /// PyVRP uses). Instead of raw duration, the proximity of an ordered pair
    /// (i→j) adds penalties for the *minimum* waiting time and time warp implied
    /// by serving j right after i:
    ///
    /// ```text
    /// prox(i,j) = dur(i,j)
    ///           + W_WAIT · max(0, early[j] − dur(i,j) − service[i] − late[i])
    ///           + W_WARP · max(0, early[i] + service[i] + dur(i,j) − late[j])
    /// ```
    ///
    /// with `W_WAIT = 0.2`, `W_WARP = 1.0` (PyVRP defaults), symmetrised via
    /// `min(P, Pᵀ)`. So neighbours are clients that are *temporally* compatible,
    /// not merely close — the lever that helps time-windowed (R/RC) instances.
    /// For a window-less problem all penalties collapse to 0 and this is
    /// byte-identical to [`Granular::build`] (pure distance), so CVRP behaviour is
    /// unchanged. Depot locations get no neighbours and are never neighbours
    /// (matching PyVRP). The `prize` reward term is omitted — our default prize is
    /// a huge mandatory-sentinel that would swamp the metric.
    pub fn build_tw(matrix: &Matrix, k: usize, problem: &Problem) -> Self {
        const W_WAIT: f64 = 0.2;
        const W_WARP: f64 = 1.0;
        let n = matrix.n;
        let k_eff = k.min(n.saturating_sub(1)).max(1);

        // Per-location time-window data. Defaults (no client at a location ⇒ a
        // depot): the universal window + no service, and `is_client = false`.
        let mut early = vec![0i64; n];
        let mut late = vec![TimeWindow::FOREVER.end; n];
        let mut service = vec![0i64; n];
        let mut is_client = vec![false; n];
        let mut set_loc = |loc: Option<usize>, tws: &[TimeWindow], svc: i64| {
            if let Some(li) = loc {
                if li < n {
                    is_client[li] = true;
                    service[li] = svc;
                    if let Some(w) = tws.first() {
                        early[li] = w.start;
                        late[li] = w.end;
                    }
                }
            }
        };
        for j in &problem.jobs {
            set_loc(j.location.index, &j.time_windows, j.service);
        }
        for s in &problem.shipments {
            set_loc(s.pickup.location.index, &s.pickup.time_windows, s.pickup.service);
            set_loc(s.delivery.location.index, &s.delivery.time_windows, s.delivery.service);
        }

        let prox = |i: usize, j: usize| -> f64 {
            let d = matrix.durations[i * n + j] as f64;
            let min_wait = early[j] as f64 - d - service[i] as f64 - late[i] as f64;
            let min_warp = early[i] as f64 + service[i] as f64 + d - late[j] as f64;
            d + W_WAIT * min_wait.max(0.0) + W_WARP * min_warp.max(0.0)
        };

        let mut near: Vec<u32> = vec![0u32; n * k_eff];
        let mut counts: Vec<u32> = vec![0u32; n];
        let mut buf: Vec<(f64, u32)> = Vec::with_capacity(n);
        for i in 0..n {
            // Depots have no neighbours (Vidal/PyVRP).
            if !is_client[i] {
                counts[i] = 0;
                continue;
            }
            buf.clear();
            for j in 0..n {
                if j == i || !is_client[j] {
                    continue; // clients do not neighbour depots
                }
                // Symmetrise: min(prox(i,j), prox(j,i)).
                let p = prox(i, j).min(prox(j, i));
                buf.push((p, j as u32));
            }
            if buf.len() > k_eff {
                buf.select_nth_unstable_by(k_eff - 1, |a, b| a.0.total_cmp(&b.0));
                buf.truncate(k_eff);
            }
            buf.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            for (r, &(_, j)) in buf.iter().enumerate() {
                near[i * k_eff + r] = j;
            }
            counts[i] = buf.len() as u32;
        }
        Self { k: k_eff, near, counts, n }
    }

    /// Iterator over the K nearest matrix indices to `i`.
    pub fn neighbors(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let cnt = self.counts[i] as usize;
        let off = i * self.k;
        self.near[off..off + cnt].iter().map(|&v| v as usize)
    }

    pub fn n(&self) -> usize { self.n }
    pub fn k(&self) -> usize { self.k }

    /// Build directly from MMM's `knn_matrix_flat` output.
    ///
    /// Input layout: `flat[i*k .. i*k+k]` is the K nearest entries for
    /// location `i`, sorted ascending by duration. Each entry is
    /// `(neighbor_idx, dur_s, dist_m)`. Padding for isolated components
    /// uses `neighbor_idx == u32::MAX` (we treat those as the tail and
    /// stop counting there).
    ///
    /// `n` = number of locations (rows in `flat / k`).
    /// `k` = K used when generating the K-NN.
    ///
    /// Skips its own neighbor automatically (MMM excludes self, but we
    /// guard against it as a defensive check).
    pub fn from_knn_flat(flat: &[(u32, f32, f32)], n: usize, k: usize) -> Self {
        assert_eq!(flat.len(), n * k, "from_knn_flat: flat.len()={} != n*k={}", flat.len(), n * k);
        let k_eff = k.max(1);
        let mut near: Vec<u32> = vec![0u32; n * k_eff];
        let mut counts: Vec<u32> = vec![0u32; n];
        for i in 0..n {
            let row_off = i * k;
            let mut cnt = 0u32;
            for r in 0..k {
                let (nbr, _dur, _dist) = flat[row_off + r];
                if nbr == u32::MAX { break; }
                if nbr as usize == i { continue; }
                near[i * k_eff + cnt as usize] = nbr;
                cnt += 1;
            }
            // Pad remaining slots with self-index (existing convention).
            for r in cnt as usize..k_eff {
                near[i * k_eff + r] = i as u32;
            }
            counts[i] = cnt;
        }
        Self { k: k_eff, near, counts, n }
    }

    /// Same as `from_knn_flat` but accepts the row-major nested format
    /// that `knn_matrix` (non-flat) returns: `Vec<Vec<(idx, dur, dist)>>`.
    pub fn from_knn_rows(rows: &[Vec<(u32, f32, f32)>], k: usize) -> Self {
        let n = rows.len();
        let k_eff = k.max(1);
        let mut near: Vec<u32> = vec![0u32; n * k_eff];
        let mut counts: Vec<u32> = vec![0u32; n];
        for (i, row) in rows.iter().enumerate() {
            let mut cnt = 0u32;
            for &(nbr, _, _) in row.iter().take(k) {
                if nbr == u32::MAX { break; }
                if nbr as usize == i { continue; }
                near[i * k_eff + cnt as usize] = nbr;
                cnt += 1;
            }
            for r in cnt as usize..k_eff {
                near[i * k_eff + r] = i as u32;
            }
            counts[i] = cnt;
        }
        Self { k: k_eff, near, counts, n }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_knn_flat_basic() {
        // 4 locations, K=2.
        // Location 0's two nearest are 1, 2.
        // Location 1's two nearest are 0, 2.
        // Location 2's two nearest are 0, 1.
        // Location 3 is isolated — has 0 valid neighbors (sentinel padding).
        let flat = vec![
            (1, 1.0, 100.0), (2, 2.0, 200.0),     // i=0
            (0, 1.0, 100.0), (2, 1.5, 150.0),     // i=1
            (0, 2.0, 200.0), (1, 1.5, 150.0),     // i=2
            (u32::MAX, f32::INFINITY, f32::INFINITY), (u32::MAX, f32::INFINITY, f32::INFINITY),  // i=3
        ];
        let g = Granular::from_knn_flat(&flat, 4, 2);
        assert_eq!(g.n(), 4);
        assert_eq!(g.k(), 2);
        let nbrs0: Vec<usize> = g.neighbors(0).collect();
        assert_eq!(nbrs0, vec![1, 2]);
        let nbrs3: Vec<usize> = g.neighbors(3).collect();
        assert_eq!(nbrs3, Vec::<usize>::new());  // isolated component
    }

    #[test]
    fn from_knn_flat_skips_self() {
        // Defensive: even though MMM excludes self, verify we drop it if present.
        let flat = vec![
            (0, 0.0, 0.0), (1, 1.0, 100.0),  // i=0, includes self at start
            (1, 0.0, 0.0), (0, 1.0, 100.0),  // i=1, includes self at start
        ];
        let g = Granular::from_knn_flat(&flat, 2, 2);
        let nbrs0: Vec<usize> = g.neighbors(0).collect();
        assert_eq!(nbrs0, vec![1]);  // self skipped
    }

    #[test]
    fn from_knn_rows_matches_flat() {
        let rows = vec![
            vec![(1u32, 1.0f32, 100.0f32), (2, 2.0, 200.0)],
            vec![(0, 1.0, 100.0), (2, 1.5, 150.0)],
            vec![(0, 2.0, 200.0), (1, 1.5, 150.0)],
        ];
        let g = Granular::from_knn_rows(&rows, 2);
        let nbrs1: Vec<usize> = g.neighbors(1).collect();
        assert_eq!(nbrs1, vec![0, 2]);
    }
}
