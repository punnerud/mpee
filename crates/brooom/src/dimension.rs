//! User-defined custom accumulator dimensions (the OR-Tools `RoutingDimension`
//! gap).
//!
//! The solver already tracks multi-dimensional *capacity* internally
//! (`eval.rs` / `solution.rs`). What this module adds is the ability for a user
//! to register their **own** named quantity that accrues along a route via a
//! per-arc transit callback — fuel that drains per kilometre, a resource that
//! accumulates per stop, etc. — and then read it from a pyspell DSL constraint
//! as `route.<dim>` (whole-route aggregate) or `route.<dim>[k]` (the cumul at
//! stop `k`).
//!
//! ## Model (mirror of OR-Tools `RoutingDimension`, deliberately a first cut)
//! A dimension is a forward accumulation along one route:
//!   * `cumul[0] = start` (the value at the start depot),
//!   * for each arc from position `k` to `k+1`,
//!     `cumul[k+1] = cumul[k] + transit(from_loc, to_loc, cumul[k], arrival_at_to)`.
//! `cumul` therefore has one entry per route position (start depot, each stop,
//! end depot), exactly like the route walk in `solution::evaluate_route`.
//!
//! ## HONEST CAVEATS (do not pretend these are closed)
//!   * The transit callback is a **deterministic** function of
//!     `(from, to, cumul_before, arrival)` only. It is NOT a true vehicle-scoped
//!     history across trips: it cannot remember per-vehicle state from earlier
//!     trips, only the cumul threaded through the current route.
//!   * Arbitrary cumul bounds (`min`/`max`) are checked at **full route
//!     evaluation** (`evaluate_route`), NOT in the O(1) insertion probe in
//!     `eval.rs`. A bounded dimension is honoured (infeasible routes are never
//!     committed), but it does not prune candidates in the fast probe the way a
//!     travel/distance/duration bound does. We are explicit about this rather
//!     than pretending the probe understands arbitrary user callbacks.
//!   * No soft cumul bounds (no slack / span-cost on a custom dimension) and no
//!     cross-dimension coupling (e.g. `fuel = distance × factor` reading another
//!     dimension) in this first cut. The callback sees only its own cumul.
//!
//! ## Cost / registration
//!   * When no dimension is registered the hot path pays a single relaxed atomic
//!     load (`has_dimensions()`) — effectively free, and behaviour/cost is
//!     byte-identical to a build that never knew about this module.
//!   * Register **before** solving. Registering or clearing bumps the route-eval
//!     cache epoch so stale cached metrics are never reused.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use crate::problem::Time;

/// A per-arc transit callback: given the arc `(from_loc, to_loc)` as matrix
/// indices, the dimension's cumul *before* traversing the arc, and the arrival
/// time at `to_loc`, return the delta to add to the cumul. Deterministic and
/// side-effect free (see the module caveats).
pub type TransitFn = dyn Fn(usize, usize, i64, Time) -> i64 + Send + Sync;

/// One registered custom dimension.
#[derive(Clone)]
pub struct CustomDimension {
    /// The name a DSL constraint reads it by (`route.<name>`).
    pub name: String,
    /// Per-arc transit callback (see [`TransitFn`]).
    pub transit: Arc<TransitFn>,
    /// Value of the cumul at the start depot (position 0). Default 0.
    pub start: i64,
    /// Optional hard lower bound on every cumul value. Checked at full route
    /// evaluation (NOT in the O(1) probe).
    pub min: Option<i64>,
    /// Optional hard upper bound on every cumul value. Checked at full route
    /// evaluation (NOT in the O(1) probe).
    pub max: Option<i64>,
}

impl CustomDimension {
    /// A dimension with no bounds and a start value of 0.
    pub fn new(name: impl Into<String>, transit: Arc<TransitFn>) -> Self {
        CustomDimension { name: name.into(), transit, start: 0, min: None, max: None }
    }
    pub fn with_start(mut self, start: i64) -> Self {
        self.start = start;
        self
    }
    pub fn with_min(mut self, min: i64) -> Self {
        self.min = Some(min);
        self
    }
    pub fn with_max(mut self, max: i64) -> Self {
        self.max = Some(max);
        self
    }
}

// Fast-path flag read on every route walk; the RwLock is only touched when this
// is true. Mirrors `crate::constraint::HAS_CUSTOM`.
static HAS_DIM: AtomicBool = AtomicBool::new(false);
static REGISTRY: RwLock<Vec<CustomDimension>> = RwLock::new(Vec::new());

/// Replace the registered custom dimensions. Pass an empty vec to clear.
/// Invalidates the route-eval cache so previously cached metrics aren't reused.
pub fn set_dimensions(list: Vec<CustomDimension>) {
    let mut g = REGISTRY.write().unwrap();
    HAS_DIM.store(!list.is_empty(), Ordering::SeqCst);
    *g = list;
    drop(g);
    crate::solution::eval_cache_invalidate();
}

/// Remove all custom dimensions.
pub fn clear_dimensions() {
    set_dimensions(Vec::new());
}

/// Whether any custom dimension is currently registered (cheap, lock-free).
#[inline]
pub fn has_dimensions() -> bool {
    HAS_DIM.load(Ordering::Relaxed)
}

/// Number of registered dimensions (0 when none).
pub fn dimension_count() -> usize {
    if !has_dimensions() {
        return 0;
    }
    REGISTRY.read().unwrap().len()
}

/// The registered dimension names, in index order. Used by the DSL lowering to
/// resolve `route.<name>` to a `Field::CustomDimension(index)` at compile time.
pub fn dimension_names() -> Vec<String> {
    if !has_dimensions() {
        return Vec::new();
    }
    REGISTRY.read().unwrap().iter().map(|d| d.name.clone()).collect()
}

/// One arc of a route to feed the accumulator: the `from`/`to` matrix indices
/// and the arrival time at `to`. The route's leading entry is the start depot
/// (no incoming arc); thereafter one entry per traversed arc.
#[derive(Clone, Copy, Debug)]
pub struct Arc2 {
    pub from: usize,
    pub to: usize,
    pub arrival: Time,
}

/// Per-route cumuls for every registered dimension, plus the cheap prefix max
/// kept alongside (mirroring `max_load_prefix` in `eval.rs`). `cumul[d]` has one
/// entry per route position (length = arcs.len() + 1).
#[derive(Clone, Debug, Default)]
pub struct DimensionCumuls {
    /// `cumul[d][pos]` = the dimension-`d` cumul at route position `pos`.
    pub cumul: Vec<Vec<i64>>,
    /// `max_prefix[d][pos]` = max cumul of dimension `d` over positions `0..=pos`.
    pub max_prefix: Vec<Vec<i64>>,
    /// True iff a registered min/max bound is violated anywhere on this route.
    /// Honoured at full route evaluation (see module caveats).
    pub bound_violated: bool,
}

impl DimensionCumuls {
    pub fn is_empty(&self) -> bool {
        self.cumul.is_empty()
    }
    /// The cumul of dimension `d` at route position `pos` (0 if out of range).
    pub fn at(&self, d: usize, pos: usize) -> i64 {
        self.cumul.get(d).and_then(|c| c.get(pos)).copied().unwrap_or(0)
    }
    /// Whole-route aggregate of dimension `d`: the maximum cumul (the natural
    /// "peak" reading — e.g. peak resource held, or, for a draining quantity,
    /// the largest value, which is the start). 0 if the dimension is empty.
    pub fn aggregate_max(&self, d: usize) -> i64 {
        self.cumul.get(d).and_then(|c| c.iter().copied().max()).unwrap_or(0)
    }
    /// The cumul vector of dimension `d` as a borrowed slice (empty if absent).
    pub fn cumuls_of(&self, d: usize) -> &[i64] {
        self.cumul.get(d).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

/// Accumulate every registered dimension along a route described by its ordered
/// arcs. Returns the per-dimension cumul vectors (one entry per position) and a
/// flag for any bound violation. Only call when [`has_dimensions`] is true; with
/// no dimensions registered this allocates nothing for the caller (the caller
/// guards on the fast-path flag).
pub fn accumulate(arcs: &[Arc2]) -> DimensionCumuls {
    let g = REGISTRY.read().unwrap();
    let positions = arcs.len() + 1;
    let mut out = DimensionCumuls {
        cumul: Vec::with_capacity(g.len()),
        max_prefix: Vec::with_capacity(g.len()),
        bound_violated: false,
    };
    for dim in g.iter() {
        let mut cumul = Vec::with_capacity(positions);
        let mut prefix = Vec::with_capacity(positions);
        let mut v = dim.start;
        let mut running = v;
        let check = |val: i64| -> bool {
            (dim.min.map(|m| val < m).unwrap_or(false))
                || (dim.max.map(|m| val > m).unwrap_or(false))
        };
        if check(v) {
            out.bound_violated = true;
        }
        cumul.push(v);
        prefix.push(running);
        for a in arcs {
            // Transit threads the *current* cumul and the arrival at `to`.
            let delta = (dim.transit)(a.from, a.to, v, a.arrival);
            v += delta;
            if check(v) {
                out.bound_violated = true;
            }
            if v > running {
                running = v;
            }
            cumul.push(v);
            prefix.push(running);
        }
        out.cumul.push(cumul);
        out.max_prefix.push(prefix);
    }
    out
}

/// RAII guard: installs dimensions and clears them on drop, so a solve can be
/// scoped without leaking global state into the next one. Mirrors
/// [`crate::constraint::ConstraintGuard`].
pub struct DimensionGuard {
    _private: (),
}

impl DimensionGuard {
    pub fn install(list: Vec<CustomDimension>) -> Self {
        set_dimensions(list);
        DimensionGuard { _private: () }
    }
}

impl Drop for DimensionGuard {
    fn drop(&mut self) {
        clear_dimensions();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The registry is process-global, so these tests (which install/clear it)
    // must not run concurrently. Serialize them behind a shared lock, tolerating
    // a poisoned lock from a previously-panicking test.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn guard() -> std::sync::MutexGuard<'static, ()> {
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn fuel_drains_per_arc() {
        let _lock = guard();
        // A "fuel" dimension that starts at 100 and burns 10 units per arc,
        // independent of the actual nodes. Bound: never below 0.
        let dim = CustomDimension::new(
            "fuel",
            Arc::new(|_from, _to, _cumul, _arrival| -10),
        )
        .with_start(100)
        .with_min(0);
        let _g = DimensionGuard::install(vec![dim]);
        assert!(has_dimensions());

        // Three arcs → four positions: 100, 90, 80, 70. No bound violation.
        let arcs = [
            Arc2 { from: 0, to: 1, arrival: 100 },
            Arc2 { from: 1, to: 2, arrival: 200 },
            Arc2 { from: 2, to: 0, arrival: 300 },
        ];
        let c = accumulate(&arcs);
        assert_eq!(c.cumuls_of(0), &[100, 90, 80, 70]);
        assert!(!c.bound_violated);
        // aggregate (max) is the starting full tank.
        assert_eq!(c.aggregate_max(0), 100);
        assert_eq!(c.at(0, 2), 80);
    }

    #[test]
    fn bound_violation_flagged() {
        let _lock = guard();
        // Burns 40/arc from a 100 start with a min of 0 → after 3 arcs the cumul
        // would be -20, violating the lower bound.
        let dim = CustomDimension::new("fuel", Arc::new(|_, _, _, _| -40))
            .with_start(100)
            .with_min(0);
        let _g = DimensionGuard::install(vec![dim]);
        let arcs = [
            Arc2 { from: 0, to: 1, arrival: 1 },
            Arc2 { from: 1, to: 2, arrival: 2 },
            Arc2 { from: 2, to: 3, arrival: 3 },
        ];
        let c = accumulate(&arcs);
        assert_eq!(c.cumuls_of(0), &[100, 60, 20, -20]);
        assert!(c.bound_violated);
    }

    #[test]
    fn transit_reads_arrival_and_cumul() {
        let _lock = guard();
        // A dimension whose delta depends on the arrival time and current cumul
        // proves the callback genuinely threads state, not just a constant.
        let dim = CustomDimension::new(
            "accrue",
            // +1 only while the cumul is still under 3, and only at even arrivals.
            Arc::new(|_from, _to, cumul, arrival| {
                if cumul < 3 && arrival % 2 == 0 { 1 } else { 0 }
            }),
        );
        let _g = DimensionGuard::install(vec![dim]);
        let arcs = [
            Arc2 { from: 0, to: 1, arrival: 2 }, // even, cumul 0<3 → +1 → 1
            Arc2 { from: 1, to: 2, arrival: 3 }, // odd → +0 → 1
            Arc2 { from: 2, to: 3, arrival: 4 }, // even, 1<3 → +1 → 2
            Arc2 { from: 3, to: 4, arrival: 6 }, // even, 2<3 → +1 → 3
            Arc2 { from: 4, to: 5, arrival: 8 }, // even, 3<3 false → +0 → 3
        ];
        let c = accumulate(&arcs);
        assert_eq!(c.cumuls_of(0), &[0, 1, 1, 2, 3, 3]);
        assert_eq!(c.aggregate_max(0), 3);
    }

    #[test]
    fn no_dimensions_is_lockfree_empty() {
        let _lock = guard();
        clear_dimensions();
        assert!(!has_dimensions());
        assert_eq!(dimension_count(), 0);
        assert!(dimension_names().is_empty());
    }
}
