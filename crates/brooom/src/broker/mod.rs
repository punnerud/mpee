//! Cost-aware matrix broker (Stage A).
//!
//! When the travel-time matrix comes from a **paid/limited** provider, buying
//! the full N×N is wasteful: the solver only ever reads each node's K-nearest
//! (granular neighbourhood) plus depot rows. This broker wraps any
//! [`CellSource`] and:
//!
//!  1. computes a free Haversine prior to RANK candidates,
//!  2. buys EXACTLY the "skeleton" cells the solver will read — each node's
//!     `K_buy = granular_k + margin` nearest, the depot rows/cols, and a set of
//!     farthest-point landmark rows/cols (exact → no quality loss on what local
//!     search consults),
//!  3. DERIVES every other (long-range) cell with a min-plus bridge over the
//!     bought landmark rows/cols — `base(i,j) = minₗ d(i,l)+d(l,j)` — falling
//!     back to the Haversine estimate where no landmark path exists,
//!  4. gates derivation behind a triangle-inequality sanity check on the bought
//!     cells; if the data is badly non-metric it keeps the Haversine fill.
//!
//! The broker is itself a [`MatrixSource`], so it drops into
//! [`crate::solver::build_matrix`] transparently. Stage B adds a persistent
//! cell DB + frequency prune; Stage C/D add PySpell pricing + a Google provider.

use std::collections::HashSet;
use std::sync::Mutex;

use crate::error::Result;
use crate::matrix::{CellRequest, CellSource, HaversineMatrix, Matrix, MatrixSource};

/// How aggressively to derive instead of buy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeriveMode {
    /// Buy the full matrix (baseline; no derivation). Useful for measurement.
    Off,
    /// Safe default: buy the skeleton exactly, derive only the cells the solver
    /// ignores.
    Skeleton,
}

/// Broker tuning. Defaults are the safe `Skeleton` policy.
#[derive(Debug, Clone)]
pub struct BrokerPolicy {
    /// Must be ≥ the solver's `granular_k` so every cell LS reads is bought.
    pub granular_k: usize,
    /// Extra neighbours bought on top of `granular_k` to absorb Haversine→true
    /// rank churn (the one quality knob).
    pub k_buy_margin: usize,
    pub derive: DeriveMode,
    /// Buy the depot's full row+column (index 0); it's on every route.
    pub buy_depot_full: bool,
    /// Number of farthest-point landmarks whose rows/cols are bought and used to
    /// derive the long-range cells.
    pub max_landmarks: usize,
}

impl Default for BrokerPolicy {
    fn default() -> Self {
        Self {
            granular_k: 20, // matches SolverConfig default
            k_buy_margin: 10,
            derive: DeriveMode::Skeleton,
            buy_depot_full: true,
            max_landmarks: 32,
        }
    }
}

/// What the last `build()` did — for measurement and the cost-vs-quality gate.
#[derive(Debug, Clone, Copy, Default)]
pub struct BrokerStats {
    pub n: usize,
    pub cells_total: usize,  // n*n
    pub cells_bought: usize, // requested from the provider
    pub cells_derived: usize,
    pub landmarks: usize,
    /// True when the bought sub-matrix passed the triangle-inequality check and
    /// min-plus derivation was used; false ⇒ Haversine fill only.
    pub metric_ok: bool,
}

impl BrokerStats {
    pub fn saved_fraction(&self) -> f64 {
        if self.cells_total == 0 {
            return 0.0;
        }
        1.0 - self.cells_bought as f64 / self.cells_total as f64
    }
}

/// Wraps a paid/limited [`CellSource`], buying few cells and deriving the rest.
pub struct BrokerMatrixSource<S: CellSource> {
    inner: S,
    policy: BrokerPolicy,
    stats: Mutex<BrokerStats>,
}

impl<S: CellSource> BrokerMatrixSource<S> {
    pub fn new(inner: S, policy: BrokerPolicy) -> Self {
        Self { inner, policy, stats: Mutex::new(BrokerStats::default()) }
    }

    /// Stats from the most recent `build()`.
    pub fn last_stats(&self) -> BrokerStats {
        *self.stats.lock().unwrap()
    }

    fn k_buy(&self) -> usize {
        self.policy.granular_k + self.policy.k_buy_margin
    }
}

impl<S: CellSource> MatrixSource for BrokerMatrixSource<S> {
    fn build(&self, coords: &[[f64; 2]]) -> Result<Matrix> {
        let n = coords.len();
        // Tiny problems, or Off mode: just buy everything.
        if self.policy.derive == DeriveMode::Off || n <= self.k_buy() + 2 {
            let m = self.inner.build(coords)?;
            *self.stats.lock().unwrap() = BrokerStats {
                n,
                cells_total: n * n,
                cells_bought: n * n,
                cells_derived: 0,
                landmarks: 0,
                metric_ok: true,
            };
            return Ok(m);
        }

        // 1. Free Haversine prior (used to rank candidates + as the fallback).
        let prior = HaversineMatrix::default().build(coords)?;

        // 2. Skeleton selection: K_buy nearest per node + depot + landmark rows/cols.
        let landmarks = pick_landmarks(&prior, self.policy.max_landmarks);
        let req = self.skeleton(&prior, &landmarks);

        // 3. Buy the skeleton.
        let resp = self.inner.fetch_cells(coords, &req)?;
        let cells_bought = req.pairs.len();

        // Assemble: start from the prior, overwrite bought cells with truth.
        let mut dur = prior.durations.clone();
        let mut dist = prior.distances.clone();
        let mut bought: HashSet<u64> = HashSet::with_capacity(cells_bought);
        for (k, &(i, j)) in req.pairs.iter().enumerate() {
            let idx = i as usize * n + j as usize;
            dur[idx] = resp.dur[k];
            if let (Some(out), Some(rd)) = (dist.as_mut(), resp.dist.as_ref()) {
                out[idx] = rd[k];
            }
            bought.insert(cell_key(i, j));
        }
        for i in 0..n {
            dur[i * n + i] = 0;
            if let Some(d) = dist.as_mut() {
                d[i * n + i] = 0;
            }
        }

        // 4. Derive the unbought cells via the landmark min-plus bridge, if the
        //    bought data looks metric. Otherwise keep the Haversine fill.
        let metric_ok = metric_check(&dur, n, &landmarks, &bought);
        let mut cells_derived = 0;
        if metric_ok {
            derive_min_plus(&mut dur, n, &landmarks, &bought, &mut cells_derived);
            if let Some(d) = dist.as_mut() {
                let mut _c = 0;
                derive_min_plus(d, n, &landmarks, &bought, &mut _c);
            }
        }

        *self.stats.lock().unwrap() = BrokerStats {
            n,
            cells_total: n * n,
            cells_bought,
            cells_derived,
            landmarks: landmarks.len(),
            metric_ok,
        };
        Ok(Matrix { n, durations: dur, distances: dist })
    }
}

impl<S: CellSource> BrokerMatrixSource<S> {
    /// Build the must-buy cell set: each node's K_buy nearest (by the Haversine
    /// prior), the depot row+col, and every landmark's row+col.
    fn skeleton(&self, prior: &Matrix, landmarks: &[u32]) -> CellRequest {
        let n = prior.n;
        let k = self.k_buy().min(n.saturating_sub(1));
        let mut set: HashSet<u64> = HashSet::new();
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let push = |i: u32, j: u32, set: &mut HashSet<u64>, pairs: &mut Vec<(u32, u32)>| {
            if i != j && set.insert(cell_key(i, j)) {
                pairs.push((i, j));
            }
        };

        // K_buy nearest per node (by prior duration).
        let mut buf: Vec<(i32, u32)> = Vec::with_capacity(n);
        for i in 0..n {
            buf.clear();
            for j in 0..n {
                if j != i {
                    buf.push((prior.durations[i * n + j], j as u32));
                }
            }
            if buf.len() > k {
                buf.select_nth_unstable_by_key(k - 1, |x| x.0);
                buf.truncate(k);
            }
            for &(_, j) in &buf {
                push(i as u32, j, &mut set, &mut pairs);
                push(j, i as u32, &mut set, &mut pairs); // both directions
            }
        }

        // Depot row + column (index 0).
        if self.policy.buy_depot_full {
            for j in 0..n as u32 {
                push(0, j, &mut set, &mut pairs);
                push(j, 0, &mut set, &mut pairs);
            }
        }

        // Landmark rows + columns (needed for min-plus derivation).
        for &l in landmarks {
            for j in 0..n as u32 {
                push(l, j, &mut set, &mut pairs);
                push(j, l, &mut set, &mut pairs);
            }
        }

        CellRequest { pairs }
    }
}

#[inline]
fn cell_key(i: u32, j: u32) -> u64 {
    ((i as u64) << 32) | j as u64
}

/// Farthest-point landmark sampling on the prior (greedy max-min).
fn pick_landmarks(prior: &Matrix, l: usize) -> Vec<u32> {
    let n = prior.n;
    let l = l.min(n);
    if l == 0 {
        return Vec::new();
    }
    let sym = |i: usize, j: usize| -> i64 {
        (prior.durations[i * n + j] as i64 + prior.durations[j * n + i] as i64) / 2
    };
    // start at the point with the largest row-max (an extreme)
    let mut i0 = 0usize;
    let mut best = i64::MIN;
    for i in 0..n {
        let mut rm = i64::MIN;
        for j in 0..n {
            rm = rm.max(sym(i, j));
        }
        if rm > best {
            best = rm;
            i0 = i;
        }
    }
    let mut chosen = vec![i0 as u32];
    let mut mind: Vec<i64> = (0..n).map(|i| sym(i, i0)).collect();
    while chosen.len() < l {
        let mut nxt = 0usize;
        let mut bv = i64::MIN;
        for i in 0..n {
            if mind[i] > bv {
                bv = mind[i];
                nxt = i;
            }
        }
        if chosen.contains(&(nxt as u32)) {
            break;
        }
        chosen.push(nxt as u32);
        for i in 0..n {
            let s = sym(i, nxt);
            if s < mind[i] {
                mind[i] = s;
            }
        }
    }
    chosen
}

/// Sanity check that the bought cells respect the triangle inequality through
/// landmarks: `d(i,j) <= d(i,l)+d(l,j)` for bought (i,j). Many large violations
/// ⇒ non-metric data where min-plus derivation would underestimate, so derive
/// only when violations are rare.
fn metric_check(dur: &[i32], n: usize, landmarks: &[u32], bought: &HashSet<u64>) -> bool {
    if landmarks.is_empty() {
        return false;
    }
    let mut checked = 0usize;
    let mut violations = 0usize;
    'outer: for &(i, j) in bought_pairs(bought).iter() {
        if i == j {
            continue;
        }
        let dij = dur[i as usize * n + j as usize] as i64;
        for &l in landmarks {
            if l == i || l == j {
                continue;
            }
            let via = dur[i as usize * n + l as usize] as i64
                + dur[l as usize * n + j as usize] as i64;
            checked += 1;
            // tolerance: rounding + 1% slack
            if dij > via + 2 + dij / 100 {
                violations += 1;
            }
            if checked >= 5000 {
                break 'outer;
            }
        }
    }
    checked == 0 || (violations as f64) / (checked as f64) < 0.02
}

fn bought_pairs(bought: &HashSet<u64>) -> Vec<(u32, u32)> {
    bought
        .iter()
        .take(2000)
        .map(|&k| ((k >> 32) as u32, (k & 0xffff_ffff) as u32))
        .collect()
}

/// Fill unbought cells with `min over landmarks of d(i,l)+d(l,j)` (an upper
/// bound on the true distance; directional, so asymmetric data is fine). Cells
/// with no usable landmark path keep their (Haversine) value.
fn derive_min_plus(
    dur: &mut [i32],
    n: usize,
    landmarks: &[u32],
    bought: &HashSet<u64>,
    derived: &mut usize,
) {
    if landmarks.is_empty() {
        return;
    }
    // Snapshot landmark rows/cols (all bought) so we don't read partially-updated cells.
    let lrows: Vec<Vec<i32>> = landmarks
        .iter()
        .map(|&l| dur[l as usize * n..l as usize * n + n].to_vec())
        .collect();
    let lcols: Vec<Vec<i32>> = landmarks
        .iter()
        .map(|&l| (0..n).map(|i| dur[i * n + l as usize]).collect())
        .collect();
    for i in 0..n {
        for j in 0..n {
            if i == j || bought.contains(&cell_key(i as u32, j as u32)) {
                continue;
            }
            let mut base = i64::MAX;
            for a in 0..landmarks.len() {
                let v = lcols[a][i] as i64 + lrows[a][j] as i64; // d(i,l)+d(l,j)
                if v < base {
                    base = v;
                }
            }
            if base < i64::MAX {
                dur[i * n + j] = base.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                *derived += 1;
            }
        }
    }
}
