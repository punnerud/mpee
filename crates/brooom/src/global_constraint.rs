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

/// Penalize any client-group that doesn't have exactly one served member.
pub fn exactly_one_per_group() -> Arc<GlobalConstraintFn> {
    Arc::new(|v: &SolutionView| {
        use std::collections::HashMap;
        let mut served: HashMap<u32, i64> = HashMap::new();
        for r in v.routes {
            for s in &r.steps {
                if let Some(g) = s.description(v.problem).group {
                    *served.entry(g).or_default() += 1;
                }
            }
        }
        // Every group declared on any job must end with exactly one served.
        let mut groups: HashMap<u32, ()> = HashMap::new();
        for j in &v.problem.jobs {
            if let Some(g) = j.group {
                groups.insert(g, ());
            }
        }
        let mut pen = 0.0;
        for g in groups.keys() {
            let c = served.get(g).copied().unwrap_or(0);
            if c != 1 {
                pen += HARD * (c - 1).unsigned_abs() as Cost;
            }
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
