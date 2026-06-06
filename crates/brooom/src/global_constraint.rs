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
    /// Makespan: the longest route duration (max over routes of end-start), i.e.
    /// when the last vehicle gets home. 0 when no route is used. This is the
    /// classic min-max objective: minimising it balances the slowest route down.
    pub fn makespan(&self) -> i64 {
        self.routes
            .iter()
            .filter(|r| !r.steps.is_empty())
            .map(|r| r.metrics.end_time - r.metrics.start_time)
            .max()
            .unwrap_or(0)
    }
    /// Total distance summed across all routes (metres in the matrix's unit).
    pub fn total_distance(&self) -> i64 {
        self.routes.iter().map(|r| r.metrics.distance).sum()
    }
    /// Number of unassigned *single jobs* (shipment halves excluded — they are
    /// not independently droppable). This is the count a `UnassignedCount`
    /// lexicographic level minimises.
    pub fn unassigned_count(&self) -> usize {
        self.unassigned
            .iter()
            .filter(|t| matches!(t, TaskRef::Job(_)))
            .count()
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

/// Charge a flat penalty per vehicle/route used. Drives phase 1 of the
/// lexicographic (two-phase) solver: with a `penalty` far larger than any
/// realistic travel-cost difference, the metaheuristic prefers fewer routes —
/// so `vehicles_used` on the result is the search's best-found minimum vehicle
/// count V*. Unlike [`max_vehicles`], this never makes a job infeasible (it has
/// no cap); it only biases the objective toward consolidation.
///
/// Soft and additive, so it composes with the other globals. It is installed
/// only during phase 1 and removed before phase 2 re-solves for cost.
pub fn vehicle_count_penalty(penalty: Cost) -> Arc<GlobalConstraintFn> {
    Arc::new(move |v: &SolutionView| penalty * v.vehicles_used() as Cost)
}

// ------------------------------------------------------------------------
// N-level lexicographic objective globals (Phase 1 engine extension).
//
// Each lexicographic level needs two flavours of global:
//   * a *bias penalty* installed while that level is the active objective, so
//     the metaheuristic drives the measure down (analogue of
//     `vehicle_count_penalty`); and
//   * a *hard cap* installed for ALL subsequent levels, pinning the achieved
//     value A_i so a later level can never regress it (analogue of
//     `max_vehicles`).
//
// CAVEAT (HARD-magnitude assumption): the cap globals charge a `HARD`-scaled
// penalty per unit over the cap. `HARD` (1e12) is assumed to out-rank any
// realistic route cost / prize the lower-priority levels minimise, exactly as
// `max_vehicles` and the group constraints already assume. If a single unit of
// the capped measure could legitimately exceed `HARD` in objective terms the
// pin would be only soft — true for all practical instances but stated here.
// ------------------------------------------------------------------------

/// Charge a flat `penalty` per unassigned single job. Drives a lexicographic
/// `UnassignedCount` level to serve as many jobs as feasible. Soft + additive,
/// so it composes with the other globals. Note the base objective already
/// charges each unassigned job its `prize`; this adds an extra uniform pressure
/// on top so the *count* (not the prize-weighted value) is what is minimised.
pub fn unassigned_count_penalty(penalty: Cost) -> Arc<GlobalConstraintFn> {
    Arc::new(move |v: &SolutionView| penalty * v.unassigned_count() as Cost)
}

/// HARD cap on the number of unassigned single jobs: charge `HARD` per job over
/// `cap`. Pins a `UnassignedCount` level's achieved value for later levels.
pub fn unassigned_count_cap(cap: usize) -> Arc<GlobalConstraintFn> {
    Arc::new(move |v: &SolutionView| {
        let u = v.unassigned_count();
        if u > cap {
            HARD * (u - cap) as Cost
        } else {
            0.0
        }
    })
}

/// HARD cap on the summed route cost: charge `HARD`-scaled penalty when the
/// total route cost exceeds `cap`. Pins a `Cost` level's achieved value for
/// later levels. The overage is scaled by `HARD` (a tiny multiplier keeps the
/// penalty finite yet dominating) so even a sub-unit cost regression is
/// out-ranked. `cap` is the level's measured cost plus a small epsilon slack,
/// chosen by the driver, to avoid pinning so tight that floating-point noise
/// makes the previous solution itself look infeasible.
pub fn cost_cap(cap: Cost) -> Arc<GlobalConstraintFn> {
    Arc::new(move |v: &SolutionView| {
        let total: Cost = v.routes.iter().map(|r| r.metrics.cost).sum();
        if total > cap {
            HARD * (total - cap)
        } else {
            0.0
        }
    })
}

/// Bias penalty for a `Makespan` level: charge `penalty` per second of the
/// longest route's duration, driving the search to balance the slowest route
/// down. Soft + additive.
pub fn makespan_penalty(penalty: Cost) -> Arc<GlobalConstraintFn> {
    Arc::new(move |v: &SolutionView| penalty * v.makespan() as Cost)
}

/// HARD cap on makespan (longest route duration, seconds): charge `HARD` per
/// second over `cap`. Pins a `Makespan` level's achieved value for later levels.
pub fn makespan_cap(cap: i64) -> Arc<GlobalConstraintFn> {
    Arc::new(move |v: &SolutionView| {
        let m = v.makespan();
        if m > cap {
            HARD * (m - cap) as Cost
        } else {
            0.0
        }
    })
}

/// Bias penalty for a `Distance` level: charge `penalty` per metre of total
/// distance summed across routes. Soft + additive.
pub fn distance_penalty(penalty: Cost) -> Arc<GlobalConstraintFn> {
    Arc::new(move |v: &SolutionView| penalty * v.total_distance() as Cost)
}

/// HARD cap on total distance (metres): charge `HARD` per metre over `cap`.
/// Pins a `Distance` level's achieved value for later levels.
pub fn distance_cap(cap: i64) -> Arc<GlobalConstraintFn> {
    Arc::new(move |v: &SolutionView| {
        let d = v.total_distance();
        if d > cap {
            HARD * (d - cap) as Cost
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

/// Per-objective-component weights applied as a global cost multiplier.
///
/// Weighted scalarization, NOT lexicographic. The base per-route `cost` already
/// equals `cost_travel + cost_span + cost_custom`; these weights re-scale each
/// component's contribution to the *global* objective. The constraint adds the
/// delta `Σ_routes (w_travel−1)·cost_travel + (w_span−1)·cost_span +
/// (w_custom−1)·cost_custom`, so the effective minimised objective becomes
/// `Σ (w_travel·cost_travel + w_span·cost_span + w_custom·cost_custom)` plus the
/// usual unassigned prizes and other global penalties. All weights default to
/// 1.0, which adds exactly 0.0 and leaves today's objective untouched.
///
/// NOTE: a true lexicographic solver (phase 1 minimise vehicle count, phase 2
/// minimise cost) is a separate two-phase search and out of scope here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectiveWeights {
    pub travel: Cost,
    pub span: Cost,
    pub custom: Cost,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self { travel: 1.0, span: 1.0, custom: 1.0 }
    }
}

impl ObjectiveWeights {
    /// True when every weight is exactly 1.0 (the multiplier is a no-op).
    pub fn is_identity(&self) -> bool {
        self.travel == 1.0 && self.span == 1.0 && self.custom == 1.0
    }
}

/// Re-weight the global objective's cost components by [`ObjectiveWeights`].
/// Adds only the delta from the unit-weight baseline, so an identity weight set
/// contributes 0.0 and reproduces today's objective exactly.
pub fn objective_weights(w: ObjectiveWeights) -> Arc<GlobalConstraintFn> {
    Arc::new(move |v: &SolutionView| {
        let mut delta = 0.0;
        for r in v.routes {
            let m = &r.metrics;
            delta += (w.travel - 1.0) * m.cost_travel
                + (w.span - 1.0) * m.cost_span
                + (w.custom - 1.0) * m.cost_custom;
        }
        delta
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
            disjunction_penalty: None,
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
