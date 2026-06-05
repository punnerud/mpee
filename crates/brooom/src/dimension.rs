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
//!   * Cumul bounds (`min`/`max`) are always checked at **full route
//!     evaluation** (`evaluate_route`), so a bounded dimension is honoured
//!     (infeasible routes are never committed). Spike `res` adds a *proactive*
//!     prune for the probe-expressible subset: a dimension declared
//!     [`CustomDimension::monotone`] with a `max` bound has that bound mirrored
//!     into the O(1) insertion probe in `eval.rs` (via
//!     [`probe_breaches_monotone_max`]), so a breaching insertion is rejected
//!     early, exactly like a travel/distance/duration bound. The residual
//!     caveat: **non-monotone or unbounded** dimensions, and the `min` bound,
//!     still fall back to full-eval-only — we do not pretend the probe
//!     understands arbitrary user callbacks. This narrows, but does not erase,
//!     the original P5 caveat.
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
    /// evaluation; ALSO mirrored into the O(1) insertion probe when [`monotone`]
    /// is set (see [`CustomDimension::monotone`]).
    pub max: Option<i64>,
    /// Declares this dimension is a **monotone non-decreasing prefix-accumulated
    /// resource**: every transit delta is `>= 0`, so the cumul never falls along
    /// a route and its running peak is simply the latest value. The caller
    /// asserts this property (it is NOT verified for arbitrary callbacks at
    /// registration — a lying flag is caught defensively at full evaluation,
    /// which remains the authority).
    ///
    /// When `monotone` is `true` AND `max` is `Some`, the fast insertion probe in
    /// `eval.rs` can prune a candidate that would breach the resource max BEFORE
    /// the full `evaluate_route`, exactly like a travel/distance/duration bound.
    /// The reasoning: inserting any task contributes a non-negative delta to every
    /// downstream cumul, so if the route's peak resource (its final cumul) plus
    /// the inserted task's own transit delta already exceeds `max`, no ordering
    /// can rescue it — the probe rejects early. Non-monotone or unbounded
    /// dimensions ignore this flag and fall back to full-eval (caveat preserved).
    pub monotone: bool,
}

impl CustomDimension {
    /// A dimension with no bounds and a start value of 0.
    pub fn new(name: impl Into<String>, transit: Arc<TransitFn>) -> Self {
        CustomDimension {
            name: name.into(),
            transit,
            start: 0,
            min: None,
            max: None,
            monotone: false,
        }
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
    /// Declare this dimension monotone non-decreasing (see
    /// [`CustomDimension::monotone`]). Combined with [`with_max`](Self::with_max)
    /// this makes the dimension's max bound prune in the O(1) insertion probe.
    pub fn monotone(mut self) -> Self {
        self.monotone = true;
        self
    }
}

// Fast-path flag read on every route walk; the RwLock is only touched when this
// is true. Mirrors `crate::constraint::HAS_CUSTOM`.
static HAS_DIM: AtomicBool = AtomicBool::new(false);
// Set iff at least one registered dimension is `monotone` with a `max` bound,
// i.e. its feasibility is probe-mirrorable. Lets the O(1) insertion probe in
// `eval.rs` skip all dimension work in the (common) case where no dimension is
// probe-expressible. Mirrors `crate::constraint::HAS_PROBE`.
static HAS_PROBE_DIM: AtomicBool = AtomicBool::new(false);
static REGISTRY: RwLock<Vec<CustomDimension>> = RwLock::new(Vec::new());

/// Replace the registered custom dimensions. Pass an empty vec to clear.
/// Invalidates the route-eval cache so previously cached metrics aren't reused.
pub fn set_dimensions(list: Vec<CustomDimension>) {
    let mut g = REGISTRY.write().unwrap();
    HAS_DIM.store(!list.is_empty(), Ordering::SeqCst);
    HAS_PROBE_DIM.store(
        list.iter().any(|d| d.monotone && d.max.is_some()),
        Ordering::SeqCst,
    );
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

/// Whether any registered dimension is probe-mirrorable (monotone + max bound).
/// Cheap, lock-free — read once at the top of the insertion-probe forward pass.
#[inline]
pub fn has_probe_dimensions() -> bool {
    HAS_PROBE_DIM.load(Ordering::Relaxed)
}

/// Test-only instrumentation: counts how many times a candidate route was
/// rejected *in the O(1) insertion probe* by a monotone dimension's max bound
/// (i.e. pruned before the full `evaluate_route`). Lets a test prove the
/// proactive prune actually fired rather than relying on the full-eval fallback.
#[cfg(test)]
pub static PROBE_PRUNE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Mirror of the monotone-dimension max bound into the O(1) insertion probe.
///
/// Given a route described by its ordered arcs (same `Arc2` sequence the full
/// evaluator threads), accumulate ONLY the dimensions declared `monotone` with a
/// `max` bound and return `true` if any of them breaches its max anywhere on the
/// route. Because such a dimension never decreases, its running peak is its final
/// cumul; checking each position is exactly the max-bound check, and a route that
/// breaches with the current stops can never be rescued by inserting more (every
/// insertion adds a non-negative delta). So `precompute` may reject early.
///
/// This is PRUNE-ONLY: it never reports a breach the full evaluator would not
/// also report (the full evaluator re-checks via `accumulate` and remains the
/// authority). Non-monotone / unbounded dimensions are skipped entirely and keep
/// their full-eval-only behaviour. Returns `false` immediately when no
/// probe-mirrorable dimension is registered.
pub fn probe_breaches_monotone_max(arcs: &[Arc2]) -> bool {
    if !has_probe_dimensions() {
        return false;
    }
    let g = REGISTRY.read().unwrap();
    for dim in g.iter() {
        let max = match (dim.monotone, dim.max) {
            (true, Some(m)) => m,
            _ => continue,
        };
        let mut v = dim.start;
        if v > max {
            return true;
        }
        for a in arcs {
            // Same threading as `accumulate`. The caller asserted monotonicity;
            // if a (buggy) callback returns a negative delta, the worst case is a
            // missed prune, never a false reject (we only ever return on `> max`).
            v += (dim.transit)(a.from, a.to, v, a.arrival);
            if v > max {
                return true;
            }
        }
    }
    false
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

    #[test]
    fn monotone_max_flags_only_when_probe_expressible() {
        let _lock = guard();
        // Monotone + max → probe-mirrorable.
        let mono = CustomDimension::new("load", Arc::new(|_, _, _, _| 10))
            .with_max(25)
            .monotone();
        let _g = DimensionGuard::install(vec![mono]);
        assert!(has_probe_dimensions(), "monotone+max is probe-expressible");

        // Monotone but NO max → not probe-mirrorable (nothing to prune against).
        let mono_nobound = CustomDimension::new("load", Arc::new(|_, _, _, _| 10)).monotone();
        set_dimensions(vec![mono_nobound]);
        assert!(!has_probe_dimensions(), "monotone without a max is not mirrorable");

        // Max but NOT declared monotone → stays on the full-eval fallback.
        let bounded_nonmono =
            CustomDimension::new("load", Arc::new(|_, _, _, _| 10)).with_max(25);
        set_dimensions(vec![bounded_nonmono]);
        assert!(!has_probe_dimensions(), "non-monotone keeps full-eval-only behaviour");
        clear_dimensions();
        assert!(!has_probe_dimensions());
    }

    #[test]
    fn probe_breaches_matches_accumulate_for_monotone_resource() {
        let _lock = guard();
        // A "load" resource that accrues +10 per arc, capped at 25.
        let dim = CustomDimension::new("load", Arc::new(|_, _, _, _| 10))
            .with_max(25)
            .monotone();
        let _g = DimensionGuard::install(vec![dim]);

        // Two arcs → cumuls [0, 10, 20]: peak 20 ≤ 25, no breach.
        let ok = [
            Arc2 { from: 0, to: 1, arrival: 1 },
            Arc2 { from: 1, to: 0, arrival: 2 },
        ];
        assert!(!probe_breaches_monotone_max(&ok));
        // The full-eval accumulator agrees: not bound_violated.
        assert!(!accumulate(&ok).bound_violated);

        // Three arcs → cumuls [0, 10, 20, 30]: 30 > 25 → breach.
        let bad = [
            Arc2 { from: 0, to: 1, arrival: 1 },
            Arc2 { from: 1, to: 2, arrival: 2 },
            Arc2 { from: 2, to: 0, arrival: 3 },
        ];
        assert!(probe_breaches_monotone_max(&bad), "probe detects the resource breach");
        // And the authoritative full-eval path reports the SAME breach.
        assert!(accumulate(&bad).bound_violated, "full-eval agrees with the probe");
    }

    #[test]
    fn non_monotone_dimension_is_not_probe_pruned() {
        let _lock = guard();
        // A genuinely non-monotone resource (drains) with a max bound, NOT
        // declared monotone. The probe mirror must skip it entirely (returns
        // false), leaving enforcement to full eval — the preserved caveat.
        let dim = CustomDimension::new("fuel", Arc::new(|_, _, _, _| -10))
            .with_max(1000); // huge max, never breached anyway
        let _g = DimensionGuard::install(vec![dim]);
        assert!(!has_probe_dimensions());
        let arcs = [Arc2 { from: 0, to: 1, arrival: 1 }];
        assert!(!probe_breaches_monotone_max(&arcs), "non-monotone is never probe-pruned");
    }
}
