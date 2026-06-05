//! Cross-route (solution-level) objective+constraint hook.
//!
//! Mirrors [`crate::constraint`] but operates on the WHOLE candidate solution
//! inside `Solution::recompute_summary`. Each registered closure returns an
//! additive penalty (`Cost`); a hard violation is expressed as a very large but
//! finite penalty ([`HARD`]) so the search avoids it while still keeping a
//! gradient (one violation out-ranks two). This is what max-vehicles,
//! client-groups, and fairness ride on.
//!
//! Default solves pay one relaxed atomic load per `recompute_summary` (which is
//! far rarer than the per-route `evaluate_route`), so the cost is negligible.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use crate::problem::{Cost, Problem};
use crate::solution::{Route, TaskRef};

/// Penalty magnitude for a hard solution-level violation — large enough to
/// dominate any realistic route cost and the per-unassigned prize, but finite.
pub const HARD: Cost = 1e12;

/// What a global constraint sees: the whole candidate solution.
pub struct SolutionView<'a> {
    pub problem: &'a Problem,
    pub routes: &'a [Route],
    pub unassigned: &'a [TaskRef],
}

impl SolutionView<'_> {
    /// Number of non-empty routes (≈ vehicles used).
    pub fn vehicles_used(&self) -> usize {
        self.routes.iter().filter(|r| !r.steps.is_empty()).count()
    }
    /// Total route slots (including empty ones).
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
    /// Vehicle id backing route `i`.
    pub fn vehicle_id(&self, i: usize) -> u64 {
        self.problem.vehicles[self.routes[i].vehicle_idx].id
    }
    /// Ordered job ids on route `i`.
    pub fn job_ids(&self, i: usize) -> Vec<u64> {
        self.routes[i].steps.iter().map(|s| s.description(self.problem).id).collect()
    }
    /// Route `i` duration in seconds (end - start).
    pub fn duration(&self, i: usize) -> i64 {
        let m = &self.routes[i].metrics;
        m.end_time - m.start_time
    }
    pub fn distance(&self, i: usize) -> i64 {
        self.routes[i].metrics.distance
    }
    pub fn cost(&self, i: usize) -> Cost {
        self.routes[i].metrics.cost
    }
    /// Summed single-job delivery load (dimension 0) on route `i`.
    pub fn load(&self, i: usize) -> i64 {
        self.routes[i]
            .steps
            .iter()
            .filter(|s| matches!(s, TaskRef::Job(_)))
            .map(|s| s.description(self.problem).delivery.first().copied().unwrap_or(0))
            .sum()
    }
}

pub type GlobalConstraintFn = dyn Fn(&SolutionView) -> Cost + Send + Sync;

static HAS_GLOBAL: AtomicBool = AtomicBool::new(false);
static REGISTRY: RwLock<Vec<Arc<GlobalConstraintFn>>> = RwLock::new(Vec::new());

/// Replace the registered global constraints (empty vec clears).
pub fn set_global_constraints(list: Vec<Arc<GlobalConstraintFn>>) {
    let mut g = REGISTRY.write().unwrap();
    HAS_GLOBAL.store(!list.is_empty(), Ordering::SeqCst);
    *g = list;
}

/// Remove all global constraints.
pub fn clear_global_constraints() {
    set_global_constraints(Vec::new());
}

/// Whether any global constraint is registered (cheap, lock-free).
#[inline]
pub fn has_global() -> bool {
    HAS_GLOBAL.load(Ordering::Relaxed)
}

/// Total solution-level penalty for `view` across all registered constraints.
pub fn apply(view: &SolutionView) -> Cost {
    let g = REGISTRY.read().unwrap();
    g.iter().map(|c| c(view)).sum()
}

/// RAII guard: installs global constraints and clears them on drop.
pub struct GlobalConstraintGuard {
    _private: (),
}

impl GlobalConstraintGuard {
    pub fn install(list: Vec<Arc<GlobalConstraintFn>>) -> Self {
        set_global_constraints(list);
        GlobalConstraintGuard { _private: () }
    }
}

impl Drop for GlobalConstraintGuard {
    fn drop(&mut self) {
        clear_global_constraints();
    }
}

// ------------------------------------------------------------------------
// Built-in global constraints (Phase 5).
// ------------------------------------------------------------------------

/// Which per-route quantity to balance for fairness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FairnessMetric {
    Duration,
    Load,
}

impl Default for FairnessMetric {
    fn default() -> Self {
        FairnessMetric::Duration
    }
}

/// Penalize solutions that use more than `cap` vehicles.
pub fn max_vehicles(cap: usize) -> Arc<GlobalConstraintFn> {
    Arc::new(move |v: &SolutionView| {
        let used = v.vehicles_used();
        if used > cap {
            HARD * (used - cap) as Cost
        } else {
            0.0
        }
    })
}

/// Penalize any client-group whose number of served members falls outside
/// `[min, max]` — a "k-of-N" choose constraint. The penalty is `HARD` per
/// member over/under the bound, so the search keeps a gradient (one too many
/// out-ranks two too many) while still treating the bound as effectively hard.
///
/// `min == max == 1` reproduces the classic "exactly one per group" rule.
pub fn k_of_n_per_group(min: u32, max: u32) -> Arc<GlobalConstraintFn> {
    let lo = min as i64;
    let hi = max as i64;
    Arc::new(move |v: &SolutionView| {
        use std::collections::HashMap;
        let mut served: HashMap<u32, i64> = HashMap::new();
        for r in v.routes {
            for s in &r.steps {
                if let Some(g) = s.description(v.problem).group {
                    *served.entry(g).or_default() += 1;
                }
            }
        }
        // Every group declared on any job must end within [lo, hi] served.
        let mut groups: HashMap<u32, ()> = HashMap::new();
        for j in &v.problem.jobs {
            if let Some(g) = j.group {
                groups.insert(g, ());
            }
        }
        let mut pen = 0.0;
        for g in groups.keys() {
            let c = served.get(g).copied().unwrap_or(0);
            let over = (c - hi).max(0);
            let under = (lo - c).max(0);
            pen += HARD * (over + under) as Cost;
        }
        pen
    })
}

/// Penalize any client-group that doesn't have exactly one served member.
/// Thin wrapper over [`k_of_n_per_group`] with `min == max == 1`.
pub fn exactly_one_per_group() -> Arc<GlobalConstraintFn> {
    k_of_n_per_group(1, 1)
}

/// Penalize the spread (max - min) of a per-route metric across used routes,
/// scaled by `weight`. Soft by construction — trades against travel cost.
pub fn fairness(weight: Cost, metric: FairnessMetric) -> Arc<GlobalConstraintFn> {
    Arc::new(move |v: &SolutionView| {
        let vals: Vec<i64> = (0..v.route_count())
            .filter(|&i| !v.routes[i].steps.is_empty())
            .map(|i| match metric {
                FairnessMetric::Duration => v.duration(i),
                FairnessMetric::Load => v.load(i),
            })
            .collect();
        if vals.len() < 2 {
            return 0.0;
        }
        let min = *vals.iter().min().unwrap();
        let max = *vals.iter().max().unwrap();
        weight * (max - min) as Cost
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{Job, Location, Problem};
    use crate::solution::{Route, RouteMetrics, TaskRef};

    /// A grouped job at problem index `idx` with group `g`.
    fn grouped_job(id: u64, g: u32) -> Job {
        Job {
            id,
            location: Location { coord: None, index: Some(0) },
            kind: Default::default(),
            service: 0,
            setup: 0,
            release: 0,
            delivery: vec![1],
            pickup: vec![],
            skills: vec![],
            allowed_vehicles: None,
            priority: 0,
            time_windows: vec![],
            prize: crate::problem::DEFAULT_PRIZE,
            group: Some(g),
            description: None,
        }
    }

    /// Build a problem of `n` jobs all in group 1, and a single route serving
    /// the first `served` of them, then evaluate `c`'s penalty.
    fn penalty_for(c: &Arc<GlobalConstraintFn>, n: usize, served: usize) -> Cost {
        let mut p = Problem::default();
        p.jobs = (0..n).map(|i| grouped_job(i as u64 + 1, 1)).collect();
        let route = Route {
            vehicle_idx: 0,
            steps: (0..served).map(TaskRef::Job).collect(),
            metrics: RouteMetrics::default(),
        };
        let routes = vec![route];
        let view = SolutionView { problem: &p, routes: &routes, unassigned: &[] };
        c(&view)
    }

    #[test]
    fn exactly_one_is_one_one() {
        let c = exactly_one_per_group();
        // 0 served → 1 under → HARD; 1 → ok; 3 → 2 over → 2*HARD.
        assert_eq!(penalty_for(&c, 3, 0), HARD);
        assert_eq!(penalty_for(&c, 3, 1), 0.0);
        assert_eq!(penalty_for(&c, 3, 3), 2.0 * HARD);
    }

    #[test]
    fn k_of_n_allows_a_range() {
        // "between 2 and 3 of the group must be served".
        let c = k_of_n_per_group(2, 3);
        assert_eq!(penalty_for(&c, 5, 0), 2.0 * HARD, "0 served is 2 under the min");
        assert_eq!(penalty_for(&c, 5, 1), HARD, "1 served is 1 under the min");
        assert_eq!(penalty_for(&c, 5, 2), 0.0, "2 is inside [2,3]");
        assert_eq!(penalty_for(&c, 5, 3), 0.0, "3 is inside [2,3]");
        assert_eq!(penalty_for(&c, 5, 5), 2.0 * HARD, "5 served is 2 over the max");
    }
}
