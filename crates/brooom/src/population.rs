//! Post-solve population polish: spawn N parallel ILS trajectories from
//! a single base solution and return the best.
//!
//! Why this exists separately from `multi_start`:
//!   - `multi_start` runs N full solves from scratch (insertion → LS → ILS).
//!     Wall time scales with N because every variant rebuilds from raw
//!     insertion. For N=1000 with `multi_start=8` the solve already takes
//!     ~46 s; bumping to 64 would push past 6 min.
//!   - Population polish takes the already-converged base solution as the
//!     starting point. Each trajectory only does the cheap part —
//!     destroy-and-repair (`kick`) followed by LS to re-converge. This
//!     means we can afford 64-128 trajectories without exploding wall
//!     time.
//!
//! In effect this is the CPU-side cousin of GPU Phase 8: many trajectories
//! exploring perturbations of one good solution, best-of-K wins.

use std::time::Instant;

use rand::SeedableRng;
use rayon::prelude::*;

use crate::granular::Granular;
use crate::local_search::local_search;
use crate::matrix::Matrix;
use crate::problem::Problem;
use crate::solution::Solution;
use crate::solver::kick;

/// Configuration for `polish_with_population`.
#[derive(Debug, Clone)]
pub struct PopulationConfig {
    /// Number of parallel trajectories. Each does its own ILS loop.
    pub n_trajectories: usize,
    /// ILS iterations per trajectory. Each iteration does one
    /// `kick + local_search` round and keeps the result if cheaper than
    /// that trajectory's best so far.
    pub ils_iters_per_trajectory: usize,
    /// Fraction of tasks to remove per kick (0..1).
    pub kick_frac: f64,
    /// LS pass cap inside each trajectory. Pass the same value the main
    /// solver used (typically `SolverConfig::max_local_search_passes`).
    pub max_local_search_passes: usize,
    /// Granular K (None = no granularity). Pass the same as the main solve.
    pub granular_k: Option<usize>,
    /// Optional wallclock deadline. Trajectories that hit the deadline
    /// return their current best; the reduce keeps the best across all.
    pub deadline: Option<Instant>,
    /// Print one line of progress at end of polish.
    pub verbose: bool,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            n_trajectories: 64,
            ils_iters_per_trajectory: 5,
            kick_frac: 0.3,
            max_local_search_passes: 30,
            granular_k: Some(40),
            deadline: None,
            verbose: false,
        }
    }
}

/// Stats reported back from `polish_with_population`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PopulationStats {
    pub n_trajectories: usize,
    pub initial_cost: f64,
    pub best_cost: f64,
    pub trajectories_improved: usize,
    pub wallclock_ms: f64,
}

/// Run `cfg.n_trajectories` ILS trajectories in parallel, each starting
/// from `base`. Returns the best solution found across all trajectories
/// (including `base` itself — the result is never worse than the input).
pub fn polish_with_population(
    problem: &Problem,
    matrix: &Matrix,
    base: &Solution,
    cfg: &PopulationConfig,
) -> (Solution, PopulationStats) {
    let t_start = Instant::now();
    let initial_cost = base.summary.cost;

    let n = cfg.n_trajectories.max(1);
    let granular = cfg.granular_k.map(|k| Granular::build(matrix, k));

    let trajectories: Vec<(Solution, bool)> = (0..n as u64)
        .into_par_iter()
        .map(|seed| {
            let mut local_best = base.clone();
            let mut local_best_cost = local_best.summary.cost;
            let mut rng =
                rand_chacha::ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xCAFE_F00D));
            let mut improved = false;

            for _iter in 0..cfg.ils_iters_per_trajectory {
                if let Some(d) = cfg.deadline {
                    if Instant::now() >= d {
                        break;
                    }
                }
                let mut perturbed = local_best.clone();
                kick(&mut perturbed, cfg.kick_frac, &mut rng, problem, matrix);
                local_search(
                    problem,
                    matrix,
                    &mut perturbed,
                    cfg.max_local_search_passes,
                    granular.as_ref(),
                );
                if perturbed.summary.cost + 1e-9 < local_best_cost {
                    local_best_cost = perturbed.summary.cost;
                    local_best = perturbed;
                    improved = true;
                }
            }
            (local_best, improved)
        })
        .collect();

    let mut best = base.clone();
    let mut best_cost = best.summary.cost;
    let mut improved_count = 0;
    for (s, imp) in trajectories {
        if imp {
            improved_count += 1;
        }
        if s.summary.cost + 1e-9 < best_cost {
            best_cost = s.summary.cost;
            best = s;
        }
    }

    let stats = PopulationStats {
        n_trajectories: n,
        initial_cost,
        best_cost,
        trajectories_improved: improved_count,
        wallclock_ms: t_start.elapsed().as_secs_f64() * 1000.0,
    };
    if cfg.verbose {
        let delta = initial_cost - best_cost;
        let pct = if initial_cost > 0.0 { 100.0 * delta / initial_cost } else { 0.0 };
        eprintln!(
            "population-polish: {}/{} trajectories improved, cost {:.2} → {:.2} (Δ={:.2}, -{:.4}%) in {:.0} ms",
            improved_count, n, initial_cost, best_cost, delta, pct, stats.wallclock_ms
        );
    }
    (best, stats)
}
