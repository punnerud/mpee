//! Problem embeddings for similarity search across cached solves.
//!
//! Each problem is reduced to a small (≤20) vector of features that capture
//! its size, distance distribution, time-window stramhet, capacity tightness,
//! and geographic shape. Two problems whose embeddings are close in L2 are
//! "similar" — the assumption is that their optimal solutions share enough
//! structure that hyperparameters tuned for one transfer to the other.
//!
//! Standardization happens at search time: each feature is z-scored against
//! the corpus statistics so dimensions with different scales contribute
//! proportionally. Features dominated by problem-size (log_n_jobs) thus
//! don't drown out finer signals (TW stramhet, demand variance).
//!
//! This is the analog of "image embedding for visual search" in CV / RAG
//! pipelines: the embedding is what we index, the cached output is what we
//! retrieve.

use serde::{Deserialize, Serialize};

use crate::matrix::Matrix;
use crate::problem::Problem;

/// Compact feature vector for one problem instance. Field order is stable —
/// changing it invalidates all existing cache embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemEmbedding {
    // --- Problem size (log-scaled so 100 vs 1000 jobs differ by ~1) ---
    pub log_n_jobs: f32,
    pub log_n_vehicles: f32,
    pub log_n_shipments: f32,

    // --- Distance distribution ---
    /// Mean of all positive matrix durations.
    pub avg_distance: f32,
    /// Standard deviation of distances.
    pub std_distance: f32,
    /// Max distance / avg distance — measures spread of outliers.
    pub max_over_avg: f32,
    /// Skewness (tredje moment) — positive = long tail of large distances.
    pub distance_skewness: f32,

    // --- Time-window characteristics ---
    /// Fraction of jobs that have at least one explicit time window.
    pub frac_jobs_with_tw: f32,
    /// Mean (tw_width / vehicle_horizon) across jobs with TWs. 1.0 = no
    /// constraint, 0.0 = single instant.
    pub avg_tw_width_ratio: f32,
    /// Standard deviation of the ratio above — high std = mixed strict +
    /// loose TWs, low std = uniform tightness.
    pub std_tw_width_ratio: f32,

    // --- Capacity ---
    /// Total demand (summed first dim) / total fleet capacity.
    pub capacity_utilization: f32,
    /// Coefficient of variation of per-job demand (std/mean).
    pub demand_cv: f32,

    // --- Skills ---
    /// Fraction of jobs that have any skill requirement.
    pub frac_skill_constrained: f32,

    // --- Matrix shape ---
    /// max(d_ij) / max(d_ji) sum. 1.0 = symmetric, >1 = asymmetric.
    pub asymmetry: f32,

    // --- Geometry (only when coords are present; else 0) ---
    pub geographic_spread: f32,
    pub clustering_coefficient: f32,

    // --- Extended features (added later; default 0 so old caches still load) ---
    /// Ratio of largest-to-smallest 2D-PCA eigenvalue on coordinates.
    /// 1.0 = isotropic point cloud, large = elongated (highway-corridor shape).
    #[serde(default)]
    pub anisotropy_ratio: f32,

    /// MST total weight on a 200-sample, normalized by max pairwise distance.
    /// Captures "spread cost" — clumpy points have low MST weight, sparse
    /// uniform points have high.
    #[serde(default)]
    pub mst_total_weight_norm: f32,

    /// Kurtosis (4th moment) of nearest-neighbor distances. High = bimodal
    /// (mix of dense + sparse), low = uniform.
    #[serde(default)]
    pub nn_kurtosis: f32,

    /// Mean depot-to-job distance / max pairwise distance. Hub-and-spoke
    /// problems score low, distributed problems score higher.
    #[serde(default)]
    pub depot_distance_norm: f32,

    /// Fraction of jobs whose TW is "tight" (< 30% of vehicle horizon). The
    /// existing avg_tw_width_ratio doesn't separate "all loose" from "mix
    /// of tight + loose" — this complements it.
    #[serde(default)]
    pub frac_tight_tw: f32,

    /// Number of distinct skill IDs across all jobs. Captures fleet-skill
    /// heterogeneity that frac_skill_constrained alone misses.
    #[serde(default)]
    pub skill_diversity: f32,
}

impl ProblemEmbedding {
    /// Convert to a fixed-length f32 slice for L2-distance computation.
    /// Order MUST match `FEATURE_NAMES`. Old caches loaded via serde defaults
    /// will have 0.0 in extended slots.
    pub fn as_array(&self) -> [f32; 22] {
        [
            self.log_n_jobs,
            self.log_n_vehicles,
            self.log_n_shipments,
            self.avg_distance,
            self.std_distance,
            self.max_over_avg,
            self.distance_skewness,
            self.frac_jobs_with_tw,
            self.avg_tw_width_ratio,
            self.std_tw_width_ratio,
            self.capacity_utilization,
            self.demand_cv,
            self.frac_skill_constrained,
            self.asymmetry,
            self.geographic_spread,
            self.clustering_coefficient,
            self.anisotropy_ratio,
            self.mst_total_weight_norm,
            self.nn_kurtosis,
            self.depot_distance_norm,
            self.frac_tight_tw,
            self.skill_diversity,
        ]
    }

    pub fn dim() -> usize { 22 }
}

pub const FEATURE_NAMES: [&str; 22] = [
    "log_n_jobs",
    "log_n_vehicles",
    "log_n_shipments",
    "avg_distance",
    "std_distance",
    "max_over_avg",
    "distance_skewness",
    "frac_jobs_with_tw",
    "avg_tw_width_ratio",
    "std_tw_width_ratio",
    "capacity_utilization",
    "demand_cv",
    "frac_skill_constrained",
    "asymmetry",
    "geographic_spread",
    "clustering_coefficient",
    "anisotropy_ratio",
    "mst_total_weight_norm",
    "nn_kurtosis",
    "depot_distance_norm",
    "frac_tight_tw",
    "skill_diversity",
];

/// Extract embedding from a (problem, matrix) pair.
///
/// Most features come from the matrix and the jobs vector. The embedding is
/// computed once and stored; it does not depend on solver config.
pub fn extract(problem: &Problem, matrix: &Matrix) -> ProblemEmbedding {
    let n_jobs = problem.jobs.len();
    let n_vehicles = problem.vehicles.len();
    let n_shipments = problem.shipments.len();

    // --- Distance moments. Sample positive entries from the matrix. ---
    let mut dists: Vec<f64> = Vec::new();
    let n = matrix.n;
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            let d = matrix.duration(i, j) as f64;
            if d > 0.0 { dists.push(d); }
        }
    }
    let (avg_d, std_d, max_d, skew_d) = moments(&dists);

    // --- Asymmetry: ratio of |d_ij - d_ji| to (d_ij + d_ji) summed. ---
    let mut asym_num = 0.0_f64;
    let mut asym_den = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let a = matrix.duration(i, j) as f64;
            let b = matrix.duration(j, i) as f64;
            asym_num += (a - b).abs();
            asym_den += a + b;
        }
    }
    let asymmetry = if asym_den > 0.0 { (asym_num / asym_den) as f32 } else { 0.0 };

    // --- Time-window stats ---
    let mut tw_widths: Vec<f64> = Vec::new();
    let mut n_with_tw = 0usize;
    let horizon = problem
        .vehicles
        .iter()
        .filter_map(|v| v.time_window.map(|tw| (tw.end - tw.start) as f64))
        .fold(0.0_f64, f64::max)
        .max(1.0);
    for j in &problem.jobs {
        if !j.time_windows.is_empty() {
            n_with_tw += 1;
            let total: f64 = j.time_windows.iter()
                .map(|tw| (tw.end - tw.start) as f64)
                .sum();
            tw_widths.push(total / horizon);
        }
    }
    let frac_jobs_with_tw = if n_jobs > 0 {
        n_with_tw as f32 / n_jobs as f32
    } else { 0.0 };
    let (avg_tw, std_tw, _, _) = moments(&tw_widths);

    // --- Capacity utilization ---
    let total_demand: i64 = problem.jobs.iter()
        .map(|j| j.delivery.iter().sum::<i64>().max(j.pickup.iter().sum::<i64>()))
        .sum();
    let total_cap: i64 = problem.vehicles.iter()
        .map(|v| v.capacity.iter().sum::<i64>())
        .sum();
    let capacity_utilization = if total_cap > 0 {
        total_demand as f32 / total_cap as f32
    } else { 0.0 };

    // --- Demand variance ---
    let demands: Vec<f64> = problem.jobs.iter()
        .map(|j| j.delivery.iter().sum::<i64>().max(j.pickup.iter().sum::<i64>()) as f64)
        .collect();
    let (mean_dem, std_dem, _, _) = moments(&demands);
    let demand_cv = if mean_dem > 1e-9 { (std_dem / mean_dem) as f32 } else { 0.0 };

    // --- Skills ---
    let n_skill_jobs = problem.jobs.iter().filter(|j| !j.skills.is_empty()).count();
    let frac_skill_constrained = if n_jobs > 0 {
        n_skill_jobs as f32 / n_jobs as f32
    } else { 0.0 };

    // --- Geometry — only if coords present ---
    let coords: Vec<[f64; 2]> = problem.jobs.iter()
        .filter_map(|j| j.location.coord)
        .collect();
    let (geo_spread, cluster_coef) = if coords.len() >= 3 {
        geometric_features(&coords)
    } else { (0.0, 0.0) };

    // --- Extended features ---
    let anisotropy = if coords.len() >= 3 { pca_anisotropy(&coords) } else { 0.0 };

    // MST + nn_kurtosis: sample to keep cost bounded for large N.
    let max_d = max_d.max(1.0);
    let (mst_norm, nn_kurt) = if coords.len() >= 3 {
        mst_and_nn_features(&coords, max_d)
    } else { (0.0, 0.0) };

    // Depot-distance: mean d_0u over all (vehicle, job) pairs, normalized.
    let mut depot_sum = 0.0_f64;
    let mut depot_count = 0usize;
    for v in &problem.vehicles {
        let Some(start) = v.start.as_ref().and_then(|l| l.index) else { continue; };
        for j in &problem.jobs {
            let Some(loc) = j.location.index else { continue; };
            depot_sum += matrix.duration(start, loc) as f64;
            depot_count += 1;
        }
    }
    let depot_distance_norm = if depot_count > 0 && max_d > 1e-9 {
        ((depot_sum / depot_count as f64) / max_d) as f32
    } else { 0.0 };

    // frac_tight_tw: tight = ratio < 0.3
    let frac_tight_tw = if !tw_widths.is_empty() {
        let n_tight = tw_widths.iter().filter(|&&w| w < 0.3).count();
        n_tight as f32 / tw_widths.len() as f32
    } else { 0.0 };

    // Skill diversity: count unique skill IDs across all jobs + vehicles.
    let mut skills_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for j in &problem.jobs { skills_set.extend(j.skills.iter().copied()); }
    for v in &problem.vehicles { skills_set.extend(v.skills.iter().copied()); }
    let skill_diversity = skills_set.len() as f32;

    ProblemEmbedding {
        log_n_jobs: ((n_jobs.max(1)) as f32).ln(),
        log_n_vehicles: ((n_vehicles.max(1)) as f32).ln(),
        log_n_shipments: ((n_shipments + 1) as f32).ln(),
        avg_distance: avg_d as f32,
        std_distance: std_d as f32,
        max_over_avg: if avg_d > 1e-9 { (max_d / avg_d) as f32 } else { 0.0 },
        distance_skewness: skew_d as f32,
        frac_jobs_with_tw,
        avg_tw_width_ratio: avg_tw as f32,
        std_tw_width_ratio: std_tw as f32,
        capacity_utilization,
        demand_cv,
        frac_skill_constrained,
        asymmetry,
        geographic_spread: geo_spread,
        clustering_coefficient: cluster_coef,
        anisotropy_ratio: anisotropy,
        mst_total_weight_norm: mst_norm,
        nn_kurtosis: nn_kurt,
        depot_distance_norm,
        frac_tight_tw,
        skill_diversity,
    }
}

/// 2D PCA anisotropy: ratio of largest to smallest eigenvalue of the
/// covariance matrix. 1.0 = perfectly isotropic, large = elongated.
fn pca_anisotropy(coords: &[[f64; 2]]) -> f32 {
    let n = coords.len() as f64;
    let (mx, my) = coords.iter().fold((0.0_f64, 0.0_f64),
        |(sx, sy), [x, y]| (sx + x, sy + y));
    let (mx, my) = (mx / n, my / n);
    let (mut cxx, mut cyy, mut cxy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for [x, y] in coords {
        let dx = x - mx; let dy = y - my;
        cxx += dx * dx; cyy += dy * dy; cxy += dx * dy;
    }
    cxx /= n; cyy /= n; cxy /= n;
    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = ((trace * trace) / 4.0 - det).max(0.0).sqrt();
    let lam_max = trace / 2.0 + disc;
    let lam_min = trace / 2.0 - disc;
    if lam_min > 1e-12 { (lam_max / lam_min) as f32 } else { 0.0 }
}

/// MST total weight (Prim's, on a 200-sample) normalized by max-distance,
/// and kurtosis of nearest-neighbor distances.
fn mst_and_nn_features(coords: &[[f64; 2]], max_d: f64) -> (f32, f32) {
    let n = coords.len();
    let sample_size = n.min(200);
    let step = (n / sample_size).max(1);
    let sample: Vec<[f64; 2]> = coords.iter().step_by(step).take(sample_size).copied().collect();
    let m = sample.len();
    if m < 2 { return (0.0, 0.0); }

    // Prim's MST
    let mut in_mst = vec![false; m];
    let mut min_edge = vec![f64::INFINITY; m];
    in_mst[0] = true;
    for j in 1..m {
        min_edge[j] = euclid(sample[0], sample[j]);
    }
    let mut total = 0.0_f64;
    let mut nn_dists: Vec<f64> = Vec::with_capacity(m);
    nn_dists.push(min_edge[1..].iter().copied().fold(f64::INFINITY, f64::min));
    for _ in 1..m {
        let (mut best, mut best_w) = (-1i32, f64::INFINITY);
        for j in 0..m {
            if !in_mst[j] && min_edge[j] < best_w {
                best_w = min_edge[j];
                best = j as i32;
            }
        }
        if best < 0 { break; }
        in_mst[best as usize] = true;
        total += best_w;
        nn_dists.push(best_w);
        for j in 0..m {
            if !in_mst[j] {
                let d = euclid(sample[best as usize], sample[j]);
                if d < min_edge[j] { min_edge[j] = d; }
            }
        }
    }
    let mst_norm = (total / max_d) as f32;

    // Kurtosis of nn_dists (Pearson moment, excess form: -3 baseline).
    let nn = nn_dists.len() as f64;
    let mean = nn_dists.iter().sum::<f64>() / nn;
    let var = nn_dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / nn;
    let kurt = if var > 1e-12 {
        let std = var.sqrt();
        nn_dists.iter().map(|d| ((d - mean) / std).powi(4)).sum::<f64>() / nn - 3.0
    } else { 0.0 };
    (mst_norm, kurt as f32)
}

#[inline]
fn euclid(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Mean, std-dev, max, skewness (Pearson moment coefficient).
fn moments(xs: &[f64]) -> (f64, f64, f64, f64) {
    if xs.is_empty() { return (0.0, 0.0, 0.0, 0.0); }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let skew = if std > 1e-9 {
        xs.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n
    } else { 0.0 };
    (mean, std, max, skew)
}

/// Two cheap geometric descriptors:
/// - `spread`: bounding-box diagonal in degrees-of-arc (we don't bother with
///   haversine since this is just a relative shape feature).
/// - `clustering`: ratio of nearest-neighbor distance to mean pairwise
///   distance. Low = points are clumpy, high = points are uniformly spread.
fn geometric_features(coords: &[[f64; 2]]) -> (f32, f32) {
    let n = coords.len();
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for [x, y] in coords {
        if *x < min_x { min_x = *x; } if *x > max_x { max_x = *x; }
        if *y < min_y { min_y = *y; } if *y > max_y { max_y = *y; }
    }
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let spread = (dx * dx + dy * dy).sqrt() as f32;

    // Mean pairwise distance — sample to keep this O(n) for big problems.
    let sample = n.min(200);
    let step = (n / sample).max(1);
    let mut sum_pair = 0.0_f64;
    let mut count_pair = 0usize;
    let mut min_dist_per_point = vec![f64::INFINITY; sample];
    for (idx_a, a) in coords.iter().step_by(step).take(sample).enumerate() {
        for (idx_b, b) in coords.iter().step_by(step).take(sample).enumerate() {
            if idx_a == idx_b { continue; }
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
            sum_pair += d;
            count_pair += 1;
            if d < min_dist_per_point[idx_a] {
                min_dist_per_point[idx_a] = d;
            }
        }
    }
    let mean_pair = if count_pair > 0 { sum_pair / count_pair as f64 } else { 0.0 };
    let mean_nn = min_dist_per_point.iter()
        .copied()
        .filter(|d| d.is_finite())
        .sum::<f64>() / sample.max(1) as f64;
    let cluster = if mean_pair > 1e-9 {
        (mean_nn / mean_pair) as f32
    } else { 0.0 };
    (spread, cluster)
}

/// Compute L2 distance between two embeddings after per-feature standardization
/// using a precomputed corpus mean+std. If `stats` is None, raw L2 is used.
pub fn distance(a: &ProblemEmbedding, b: &ProblemEmbedding, stats: Option<&CorpusStats>) -> f32 {
    let av = a.as_array();
    let bv = b.as_array();
    let mut sum = 0.0_f32;
    for i in 0..av.len() {
        let d = av[i] - bv[i];
        let scaled = if let Some(s) = stats {
            if s.std[i] > 1e-9 { d / s.std[i] } else { d }
        } else { d };
        sum += scaled * scaled;
    }
    sum.sqrt()
}

/// Per-feature corpus statistics for z-score standardization.
#[derive(Debug, Clone)]
pub struct CorpusStats {
    pub mean: [f32; 22],
    pub std: [f32; 22],
}

impl CorpusStats {
    /// Build stats from a sample of cached embeddings.
    pub fn from_corpus(corpus: &[ProblemEmbedding]) -> Self {
        let n = corpus.len().max(1) as f32;
        let mut mean = [0.0_f32; 22];
        for emb in corpus {
            let v = emb.as_array();
            for i in 0..22 { mean[i] += v[i]; }
        }
        for i in 0..22 { mean[i] /= n; }

        let mut var = [0.0_f32; 22];
        for emb in corpus {
            let v = emb.as_array();
            for i in 0..22 {
                let d = v[i] - mean[i];
                var[i] += d * d;
            }
        }
        let mut std = [0.0_f32; 22];
        for i in 0..22 { std[i] = (var[i] / n).sqrt(); }
        Self { mean, std }
    }
}
