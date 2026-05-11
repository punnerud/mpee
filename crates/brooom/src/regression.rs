//! Linear regression embedding → SolverConfig parameter.
//!
//! Trains one regressor per hyperparameter (granular_k, ils_iters,
//! ils_kick_size, max_passes, multi_start) using ordinary least squares
//! over the cached corpus. Each row contributes (x = standardized
//! embedding, y = config-value) weighted by 1/cost — entries with lower
//! cost are presumed to have hit better hyperparameters and pull the fit
//! toward their region.
//!
//! Math: solve normal equations  (Xᵀ W X) β = Xᵀ W y
//! where W is diagonal of weights. Solved via Cholesky (for d=22 + 1
//! intercept = 23 params; matrix is 23×23 — trivial).
//!
//! When inverting fails (e.g. degenerate corpus), we fall back to median —
//! same baseline as the previous transfer mechanism. The user thus never
//! gets a *worse* config from regression than from median alone.

use crate::cache::{CacheMeta, SerializedConfig};
use crate::embedding::{CorpusStats, ProblemEmbedding};

const D: usize = 22;
const P: usize = D + 1; // +1 for intercept

/// Trained regressor: 5 coefficient vectors (one per hyperparameter).
#[derive(Debug, Clone)]
pub struct ConfigRegressor {
    pub coef_max_passes: [f64; P],
    pub coef_granular_k: [f64; P],
    pub coef_multi_start: [f64; P],
    pub coef_ils_iters: [f64; P],
    pub coef_ils_kick: [f64; P],
    pub stats: CorpusStats,
}

impl ConfigRegressor {
    /// Train one regressor per hyperparameter from a corpus of cache entries.
    /// Returns None when training fails (insufficient samples, degenerate
    /// covariance) — caller should fall back to median.
    pub fn train(entries: &[CacheMeta]) -> Option<Self> {
        // Need more samples than parameters; otherwise least-squares is
        // hopelessly under-determined and returns garbage.
        if entries.len() < P + 5 { return None; }

        let corpus: Vec<ProblemEmbedding> =
            entries.iter().map(|e| e.embedding.clone()).collect();
        let stats = CorpusStats::from_corpus(&corpus);

        // Build standardized design matrix X (n × P) and weight vector w.
        let n = entries.len();
        let mut x = vec![0.0_f64; n * P];
        let mut w = vec![0.0_f64; n];
        for (i, e) in entries.iter().enumerate() {
            let v = e.embedding.as_array();
            // Intercept column.
            x[i * P] = 1.0;
            for j in 0..D {
                let std_j = stats.std[j].max(1e-9);
                x[i * P + j + 1] = ((v[j] - stats.mean[j]) / std_j) as f64;
            }
            w[i] = if e.cost > 1e-9 { 1.0 / e.cost } else { 1.0 };
        }

        // Targets per hyperparameter.
        let y_max_passes: Vec<f64> = entries.iter().map(|e| e.config.max_passes as f64).collect();
        let y_granular_k: Vec<f64> = entries.iter().map(|e| e.config.granular_k as f64).collect();
        let y_multi_start: Vec<f64> = entries.iter().map(|e| e.config.multi_start as f64).collect();
        let y_ils_iters: Vec<f64> = entries.iter().map(|e| e.config.ils_iters as f64).collect();
        let y_ils_kick: Vec<f64> = entries.iter().map(|e| e.config.ils_kick_size).collect();

        // Solve normal equations once for the matrix; reuse across targets.
        let xtwx = build_xtwx(&x, &w, n);
        let chol = cholesky(&xtwx)?;

        let coef_max_passes = solve_chol(&chol, &xtwy(&x, &w, &y_max_passes, n));
        let coef_granular_k = solve_chol(&chol, &xtwy(&x, &w, &y_granular_k, n));
        let coef_multi_start = solve_chol(&chol, &xtwy(&x, &w, &y_multi_start, n));
        let coef_ils_iters = solve_chol(&chol, &xtwy(&x, &w, &y_ils_iters, n));
        let coef_ils_kick = solve_chol(&chol, &xtwy(&x, &w, &y_ils_kick, n));

        Some(Self {
            coef_max_passes,
            coef_granular_k,
            coef_multi_start,
            coef_ils_iters,
            coef_ils_kick,
            stats,
        })
    }

    /// Predict a config for a query embedding.
    pub fn predict(&self, query: &ProblemEmbedding) -> SerializedConfig {
        let v = query.as_array();
        let mut x = [0.0_f64; P];
        x[0] = 1.0;
        for j in 0..D {
            let std_j = self.stats.std[j].max(1e-9);
            x[j + 1] = ((v[j] - self.stats.mean[j]) / std_j) as f64;
        }
        let predict = |c: &[f64; P]| -> f64 {
            (0..P).map(|i| c[i] * x[i]).sum()
        };

        SerializedConfig {
            max_passes: predict(&self.coef_max_passes).round().max(10.0).min(500.0) as usize,
            granular_k: predict(&self.coef_granular_k).round().max(5.0).min(80.0) as usize,
            multi_start: predict(&self.coef_multi_start).round().max(1.0).min(32.0) as usize,
            ils_iters: predict(&self.coef_ils_iters).round().max(0.0).min(500.0) as usize,
            ils_kick_size: predict(&self.coef_ils_kick).max(0.05).min(0.95),
            time_limit_s: None,
        }
    }
}

// --- Linear-algebra primitives (small dense, no external deps) ---

/// Compute Xᵀ W X. Result is P×P, row-major.
fn build_xtwx(x: &[f64], w: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; P * P];
    for i in 0..n {
        let wi = w[i];
        for a in 0..P {
            let xa = x[i * P + a];
            for b in 0..P {
                out[a * P + b] += wi * xa * x[i * P + b];
            }
        }
    }
    out
}

/// Compute Xᵀ W y, length P.
fn xtwy(x: &[f64], w: &[f64], y: &[f64], n: usize) -> [f64; P] {
    let mut out = [0.0_f64; P];
    for i in 0..n {
        let wy = w[i] * y[i];
        for a in 0..P {
            out[a] += wy * x[i * P + a];
        }
    }
    out
}

/// Cholesky decomposition (lower-triangular L such that A = L Lᵀ). Returns
/// None if A is not positive-definite. Small ridge added on the diagonal
/// to combat near-singular cases.
fn cholesky(a: &[f64]) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; P * P];
    let ridge = 1e-6;
    for i in 0..P {
        for j in 0..=i {
            let mut s = a[i * P + j];
            if i == j { s += ridge; }
            for k in 0..j {
                s -= l[i * P + k] * l[j * P + k];
            }
            if i == j {
                if s <= 0.0 { return None; }
                l[i * P + j] = s.sqrt();
            } else {
                l[i * P + j] = s / l[j * P + j];
            }
        }
    }
    Some(l)
}

/// Solve L Lᵀ x = b given Cholesky factor L.
fn solve_chol(l: &[f64], b: &[f64; P]) -> [f64; P] {
    let mut y = [0.0_f64; P];
    // Forward: L y = b
    for i in 0..P {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * P + k] * y[k];
        }
        y[i] = s / l[i * P + i];
    }
    // Backward: Lᵀ x = y
    let mut x = [0.0_f64; P];
    for i in (0..P).rev() {
        let mut s = y[i];
        for k in (i + 1)..P {
            s -= l[k * P + i] * x[k];
        }
        x[i] = s / l[i * P + i];
    }
    x
}
