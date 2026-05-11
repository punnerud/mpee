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
