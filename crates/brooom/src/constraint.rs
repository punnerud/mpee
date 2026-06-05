//! User-supplied custom constraints, written as code.
//!
//! This is the "arbitrary constraint in code" escape hatch (the thing Timefold
//! and OR-Tools are known for): register one or more closures that the solver
//! calls on every *completed candidate route*, returning either a hard
//! rejection or a soft penalty added to that route's cost. Because
//! `evaluate_route` is the authority every accepted route passes through, a
//! custom constraint genuinely shapes the search — infeasible routes are never
//! committed, and penalised routes are out-competed by cheaper ones.
//!
//! ## Scope & cost
//! * Constraints are **per route**: a closure sees one vehicle, its ordered
//!   stops, and the route metrics. Cross-route / global constraints (e.g. "at
//!   most N vehicles") are out of scope for this hook.
//! * When no constraint is registered the hot path pays a single relaxed
//!   atomic load — effectively free. With constraints registered the engine
//!   stays on the CPU evaluator (the GPU megakernel cannot run arbitrary
//!   closures), so the solver automatically skips GPU polishing.
//! * Register **before** solving. Registering or clearing bumps the route-eval
//!   cache epoch so stale cached verdicts are never reused.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use crate::problem::{Problem, Vehicle};
use crate::solution::{RouteMetrics, TaskRef};

/// The verdict a custom constraint returns for one route.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// The route satisfies this constraint.
    Feasible,
    /// The route violates a hard constraint and must never be used.
    Infeasible,
    /// The route is allowed but carries this extra cost (a soft constraint).
    Penalty(f64),
}

/// What a custom constraint sees: one completed route in full.
pub struct RouteView<'a> {
    pub problem: &'a Problem,
    pub vehicle: &'a Vehicle,
    pub steps: &'a [TaskRef],
    pub metrics: &'a RouteMetrics,
    /// Per-route cumuls for any registered custom dimensions (P5). Empty unless
    /// a dimension is registered AND this view was built by `evaluate_route`
    /// (the full route walk that has the arrival times needed to accumulate).
    /// The pyspell evaluator reads `route.<dim>` / `route.<dim>[k]` from here.
    pub dim_cumuls: &'a crate::dimension::DimensionCumuls,
}

impl RouteView<'_> {
    /// Convenience: the `id`s of the jobs/shipment halves on this route, in
    /// visiting order.
    pub fn stop_ids(&self) -> Vec<u64> {
        self.steps.iter().map(|s| s.description(self.problem).id).collect()
    }
}

/// A shared empty `DimensionCumuls` for `RouteView`s built where no custom
/// dimension state is available (probe paths, unit tests, callers that don't run
/// `evaluate_route`). Avoids forcing every construction site to allocate.
pub fn empty_dim_cumuls() -> &'static crate::dimension::DimensionCumuls {
    static EMPTY: std::sync::OnceLock<crate::dimension::DimensionCumuls> = std::sync::OnceLock::new();
    EMPTY.get_or_init(crate::dimension::DimensionCumuls::default)
}

/// A custom constraint: any `Send + Sync` closure from a route to a verdict.
pub type CustomConstraintFn = dyn Fn(&RouteView) -> Verdict + Send + Sync;

// Fast path flag read on every `evaluate_route`; the RwLock is only touched
// when this is true.
static HAS_CUSTOM: AtomicBool = AtomicBool::new(false);
static REGISTRY: RwLock<Vec<Arc<CustomConstraintFn>>> = RwLock::new(Vec::new());

/// Replace the registered custom constraints. Pass an empty vec to clear.
/// Invalidates the route-eval cache so previously cached verdicts aren't reused.
pub fn set_constraints(list: Vec<Arc<CustomConstraintFn>>) {
    let mut g = REGISTRY.write().unwrap();
    HAS_CUSTOM.store(!list.is_empty(), Ordering::SeqCst);
    *g = list;
    drop(g);
    crate::solution::eval_cache_invalidate();
}

/// Remove all custom constraints (and any probe bounds they registered).
pub fn clear_constraints() {
    set_constraints(Vec::new());
    set_probe_bounds(Vec::new());
}

/// A whole-route metric the fast insertion probe (`eval.rs`) can check cheaply.
#[derive(Debug, Clone, Copy)]
pub enum ProbeMetric {
    TravelTime,
    Distance,
    Duration,
}

/// A hard upper bound on a probe-visible metric, mirrored from a DSL constraint
/// so the O(1) insertion probe can prune candidates before full evaluation.
#[derive(Debug, Clone, Copy)]
pub struct ProbeBound {
    pub metric: ProbeMetric,
    pub max: f64,
}

static HAS_PROBE: AtomicBool = AtomicBool::new(false);
static PROBE_BOUNDS: RwLock<Vec<ProbeBound>> = RwLock::new(Vec::new());

/// Register the probe bounds derived from the active constraints. Cleared by
/// [`clear_constraints`].
pub fn set_probe_bounds(list: Vec<ProbeBound>) {
    let mut g = PROBE_BOUNDS.write().unwrap();
    HAS_PROBE.store(!list.is_empty(), Ordering::SeqCst);
    *g = list;
}

/// Whether any probe bound is registered (cheap, lock-free).
#[inline]
pub fn has_probe_bounds() -> bool {
    HAS_PROBE.load(Ordering::Relaxed)
}

/// True when a route with these whole-route totals already breaks a registered
/// hard probe bound — lets `eval.rs::precompute` reject early.
pub fn probe_violates(travel_time: i64, distance: i64, duration: i64) -> bool {
    if !has_probe_bounds() {
        return false;
    }
    let g = PROBE_BOUNDS.read().unwrap();
    for b in g.iter() {
        let v = match b.metric {
            ProbeMetric::TravelTime => travel_time as f64,
            ProbeMetric::Distance => distance as f64,
            ProbeMetric::Duration => duration as f64,
        };
        if v > b.max {
            return true;
        }
    }
    false
}

/// Whether any custom constraint is currently registered (cheap, lock-free).
#[inline]
pub fn has_constraints() -> bool {
    HAS_CUSTOM.load(Ordering::Relaxed)
}

/// Apply every registered constraint to a finished route. Returns the total
/// soft penalty to add to the route cost, or `Err` if any constraint rejects it.
pub fn apply(view: &RouteView) -> Result<f64, &'static str> {
    let g = REGISTRY.read().unwrap();
    let mut penalty = 0.0;
    for c in g.iter() {
        match c(view) {
            Verdict::Feasible => {}
            Verdict::Infeasible => return Err("custom constraint violated"),
            Verdict::Penalty(p) => penalty += p,
        }
    }
    Ok(penalty)
}

/// RAII guard: installs constraints and clears them on drop, so a solve can be
/// scoped without leaking global state into the next one.
pub struct ConstraintGuard {
    _private: (),
}

impl ConstraintGuard {
    pub fn install(list: Vec<Arc<CustomConstraintFn>>) -> Self {
        set_constraints(list);
        ConstraintGuard { _private: () }
    }
}

impl Drop for ConstraintGuard {
    fn drop(&mut self) {
        clear_constraints();
    }
}
