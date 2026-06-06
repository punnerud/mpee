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
//!   * The transit callback is a **deterministic** function of an
//!     [`ArcCtx`] (`from`, `to`, `cumul_before`, `arrival`, `distance`,
//!     `duration`) only. It is NOT a true vehicle-scoped history across trips: it
//!     cannot remember per-vehicle state from earlier trips, only the cumul
//!     threaded through the current route. The `distance`/`duration` fields make a
//!     `fuel = distance × factor` rule expressible (cross-dim coupling on the
//!     physical arc), but the callback still sees only its *own* cumul, not other
//!     registered dimensions' cumuls.
//!   * Cumul bounds (`min`/`max`) are always checked at **full route
//!     evaluation** (`evaluate_route`), so a bounded dimension is honoured
//!     (infeasible routes are never committed). Phase-1 adds a *proactive* prune
//!     for the probe-expressible subset, in BOTH directions:
//!       - a [`Monotonicity::NonDecreasing`] dimension with a `max` bound has that
//!         bound mirrored into the O(1) insertion probe (via
//!         [`probe_breaches_monotone_max`]);
//!       - the dual: a [`Monotonicity::NonIncreasing`] (draining) dimension with a
//!         `min` bound mirrors that floor into the probe (via
//!         [`probe_breaches_monotone_min`]).
//!     So a breaching insertion is rejected early, exactly like a
//!     travel/distance/duration bound. The residual caveat: dimensions with
//!     [`Monotonicity::None`], or the *non-matching* bound direction (a
//!     non-decreasing dim's `min`, a draining dim's `max`), still fall back to
//!     full-eval-only — we do not pretend the probe understands arbitrary user
//!     callbacks. This narrows, but does not erase, the original P5 caveat.
//!   * Soft cumul bounds (`soft_min`/`soft_max` + `soft_weight`) add a penalty to
//!     the route cost when a cumul falls outside the soft band but stays within the
//!     hard band — mirroring OR-Tools `SetCumulVarSoft{Upper,Lower}Bound`. A
//!     cross-dimension coupling on the *physical arc* (distance/duration) is now
//!     expressible (see above), but coupling on another *registered dimension's*
//!     cumul is not.
//!
//! ## Cost / registration
//!   * When no dimension is registered the hot path pays a single relaxed atomic
//!     load (`has_dimensions()`) — effectively free, and behaviour/cost is
//!     byte-identical to a build that never knew about this module.
//!   * Register **before** solving. Registering or clearing bumps the route-eval
//!     cache epoch so stale cached metrics are never reused.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use crate::problem::{Cost, Time};

/// The context one arc presents to a transit callback. Threaded by the forward
/// pass of both the full evaluator (`solution::evaluate_route`) and the O(1)
/// probe (`eval::precompute`), so a callback sees identical inputs on both paths.
///
/// `distance`/`duration` are the *physical* arc cost (matrix distance and the
/// speed-scaled travel duration of this leg), which is what makes a cross-dim
/// coupling such as `fuel_burn = distance × factor` expressible without the
/// callback re-querying the matrix.
#[derive(Clone, Copy, Debug)]
pub struct ArcCtx {
    /// Matrix index of the arc's origin.
    pub from: usize,
    /// Matrix index of the arc's destination.
    pub to: usize,
    /// The dimension's cumul value *before* traversing this arc.
    pub cumul_before: i64,
    /// Arrival time at `to` (after travel, before setup/service).
    pub arrival: Time,
    /// Physical distance of this arc (matrix distance from `from` to `to`).
    pub distance: i64,
    /// Speed-scaled travel duration of this arc (the same value the evaluator
    /// added to the route clock for this leg).
    pub duration: Time,
}

/// A per-arc transit callback: given the arc context (see [`ArcCtx`]), return the
/// delta to add to the cumul. Deterministic and side-effect free (see the module
/// caveats).
pub type TransitFn = dyn Fn(&ArcCtx) -> i64 + Send + Sync;

/// How a dimension's cumul evolves along a route, declared by the caller so the
/// O(1) insertion probe can mirror the matching hard bound.
///
/// The property is the caller's assertion; it is NOT verified for arbitrary
/// callbacks at registration. A lying flag is caught defensively at full
/// evaluation (which re-checks every cumul and remains the authority) — the worst
/// case is a missed prune, never a false reject.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Monotonicity {
    /// No monotonicity asserted: the cumul may rise and fall. Not probe-mirrorable
    /// (the probe cannot bound a non-monotone walk in O(1)); bounds fall back to
    /// full-eval-only. This is the default and reproduces P5 behaviour.
    None,
    /// The cumul never decreases (every transit delta `>= 0`). Its running peak is
    /// its latest value, so a `max` bound is mirrorable into the probe.
    NonDecreasing,
    /// The cumul never increases (every transit delta `<= 0`) — a *draining*
    /// resource such as fuel. Its running trough is its latest value, so a `min`
    /// bound (the floor it may not fall through) is mirrorable into the probe.
    NonIncreasing,
}

impl Default for Monotonicity {
    fn default() -> Self {
        Monotonicity::None
    }
}

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
    /// evaluation; ALSO mirrored into the O(1) insertion probe when
    /// [`monotonicity`](Self::monotonicity) is [`Monotonicity::NonIncreasing`]
    /// (a draining resource may not fall below its floor).
    pub min: Option<i64>,
    /// Optional hard upper bound on every cumul value. Checked at full route
    /// evaluation; ALSO mirrored into the O(1) insertion probe when
    /// [`monotonicity`](Self::monotonicity) is [`Monotonicity::NonDecreasing`].
    pub max: Option<i64>,
    /// Optional soft upper bound: a cumul above this (but within the hard `max`)
    /// is allowed but adds `soft_weight × (cumul − soft_max)` to the route cost,
    /// per breaching position. Mirrors OR-Tools `SetCumulVarSoftUpperBound`.
    pub soft_max: Option<i64>,
    /// Optional soft lower bound: a cumul below this (but within the hard `min`)
    /// adds `soft_weight × (soft_min − cumul)` per breaching position. Mirrors
    /// OR-Tools `SetCumulVarSoftLowerBound`.
    pub soft_min: Option<i64>,
    /// Cost per unit of soft-bound violation (see `soft_max`/`soft_min`). 0 by
    /// default, so an unset soft band contributes no penalty.
    pub soft_weight: Cost,
    /// Declared monotonicity (see [`Monotonicity`]). Drives whether — and in which
    /// direction — the dimension's hard bound is mirrored into the O(1) probe.
    /// Defaults to [`Monotonicity::None`] (full-eval-only, P5 behaviour).
    pub monotonicity: Monotonicity,
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
            soft_max: None,
            soft_min: None,
            soft_weight: 0.0,
            monotonicity: Monotonicity::None,
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
    /// Set a soft upper bound and its per-unit penalty weight (see `soft_max`).
    pub fn with_soft_max(mut self, soft_max: i64, weight: Cost) -> Self {
        self.soft_max = Some(soft_max);
        self.soft_weight = weight;
        self
    }
    /// Set a soft lower bound and its per-unit penalty weight (see `soft_min`).
    pub fn with_soft_min(mut self, soft_min: i64, weight: Cost) -> Self {
        self.soft_min = Some(soft_min);
        self.soft_weight = weight;
        self
    }
    /// Declare this dimension monotone non-decreasing
    /// ([`Monotonicity::NonDecreasing`]). Combined with [`with_max`](Self::with_max)
    /// this makes the dimension's max bound prune in the O(1) insertion probe.
    /// Kept for back-compat with the P5 builder.
    pub fn monotone(mut self) -> Self {
        self.monotonicity = Monotonicity::NonDecreasing;
        self
    }
    /// Declare this dimension monotone non-increasing — a *draining* resource
    /// ([`Monotonicity::NonIncreasing`]). Combined with [`with_min`](Self::with_min)
    /// this makes the dimension's min bound (its floor) prune in the O(1) probe.
    pub fn draining(mut self) -> Self {
        self.monotonicity = Monotonicity::NonIncreasing;
        self
    }
}

// Fast-path flag read on every route walk; the RwLock is only touched when this
// is true. Mirrors `crate::constraint::HAS_CUSTOM`.
static HAS_DIM: AtomicBool = AtomicBool::new(false);
// Set iff at least one registered dimension is probe-mirrorable, i.e. either
// `NonDecreasing` with a `max` bound (prune on the peak) or `NonIncreasing` with
// a `min` bound (prune on the trough/floor). Lets the O(1) insertion probe in
// `eval.rs` skip all dimension work in the (common) case where no dimension is
// probe-expressible. Mirrors `crate::constraint::HAS_PROBE`.
static HAS_PROBE_DIM: AtomicBool = AtomicBool::new(false);
static REGISTRY: RwLock<Vec<CustomDimension>> = RwLock::new(Vec::new());

impl CustomDimension {
    /// Whether this dimension's hard bound can be mirrored into the O(1) probe:
    /// non-decreasing with a `max` (prune the peak), or non-increasing with a
    /// `min` (prune the floor). [`Monotonicity::None`] is never probe-mirrorable.
    #[inline]
    fn probe_mirrorable(&self) -> bool {
        matches!(
            (self.monotonicity, self.max, self.min),
            (Monotonicity::NonDecreasing, Some(_), _) | (Monotonicity::NonIncreasing, _, Some(_))
        )
    }
}

/// Replace the registered custom dimensions. Pass an empty vec to clear.
/// Invalidates the route-eval cache so previously cached metrics aren't reused.
pub fn set_dimensions(list: Vec<CustomDimension>) {
    let mut g = REGISTRY.write().unwrap();
    HAS_DIM.store(!list.is_empty(), Ordering::SeqCst);
    HAS_PROBE_DIM.store(
        list.iter().any(|d| d.probe_mirrorable()),
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

/// One arc of a route to feed the accumulator: the `from`/`to` matrix indices,
/// the arrival time at `to`, and the physical arc cost (distance/duration). The
/// route's leading entry is the start depot (no incoming arc); thereafter one
/// entry per traversed arc. `distance`/`duration` are threaded into [`ArcCtx`]
/// so a callback can express a coupling like `fuel = distance × factor`.
#[derive(Clone, Copy, Debug)]
pub struct Arc2 {
    pub from: usize,
    pub to: usize,
    pub arrival: Time,
    pub distance: i64,
    pub duration: Time,
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
    /// `min_prefix[d][pos]` = min cumul of dimension `d` over positions `0..=pos`.
    /// The dual of `max_prefix`, kept for draining (`NonIncreasing`) dimensions
    /// whose natural aggregate is the running trough.
    pub min_prefix: Vec<Vec<i64>>,
    /// True iff a registered HARD min/max bound is violated anywhere on this
    /// route. Honoured at full route evaluation (see module caveats).
    pub bound_violated: bool,
    /// Total soft-bound penalty accrued across all dimensions and positions on
    /// this route (0 when no soft band is configured or none is breached). Added
    /// to the route's `cost_custom` by the full evaluator; never a hard reject.
    pub soft_penalty: Cost,
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
    /// Whole-route trough of dimension `d`: the minimum cumul (the natural reading
    /// for a draining quantity — e.g. the lowest fuel level reached). 0 if empty.
    pub fn aggregate_min(&self, d: usize) -> i64 {
        self.cumul.get(d).and_then(|c| c.iter().copied().min()).unwrap_or(0)
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
        min_prefix: Vec::with_capacity(g.len()),
        bound_violated: false,
        soft_penalty: 0.0,
    };
    for dim in g.iter() {
        let mut cumul = Vec::with_capacity(positions);
        let mut max_pre = Vec::with_capacity(positions);
        let mut min_pre = Vec::with_capacity(positions);
        let mut v = dim.start;
        let mut running_max = v;
        let mut running_min = v;
        let check = |val: i64| -> bool {
            (dim.min.map(|m| val < m).unwrap_or(false))
                || (dim.max.map(|m| val > m).unwrap_or(false))
        };
        // Soft penalty for one cumul value: charged when it falls outside the
        // soft band (but, by construction of the caller, within the hard band).
        let soft = |val: i64| -> Cost {
            let mut pen = 0.0;
            if let Some(sm) = dim.soft_max {
                if val > sm {
                    pen += dim.soft_weight * (val - sm) as Cost;
                }
            }
            if let Some(sm) = dim.soft_min {
                if val < sm {
                    pen += dim.soft_weight * (sm - val) as Cost;
                }
            }
            pen
        };
        if check(v) {
            out.bound_violated = true;
        }
        out.soft_penalty += soft(v);
        cumul.push(v);
        max_pre.push(running_max);
        min_pre.push(running_min);
        for a in arcs {
            // Transit threads the *current* cumul and the full arc context.
            let ctx = ArcCtx {
                from: a.from,
                to: a.to,
                cumul_before: v,
                arrival: a.arrival,
                distance: a.distance,
                duration: a.duration,
            };
            let delta = (dim.transit)(&ctx);
            v += delta;
            if check(v) {
                out.bound_violated = true;
            }
            out.soft_penalty += soft(v);
            if v > running_max {
                running_max = v;
            }
            if v < running_min {
                running_min = v;
            }
            cumul.push(v);
            max_pre.push(running_max);
            min_pre.push(running_min);
        }
        out.cumul.push(cumul);
        out.max_prefix.push(max_pre);
        out.min_prefix.push(min_pre);
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
/// rejected *in the O(1) insertion probe* by a monotone dimension's bound (max
/// for non-decreasing, min for draining) — i.e. pruned before the full
/// `evaluate_route`. Lets a test prove the proactive prune actually fired rather
/// than relying on the full-eval fallback.
#[cfg(test)]
pub static PROBE_PRUNE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Thread one arc through a dimension's transit callback (helper shared by both
/// probe directions). Builds the same [`ArcCtx`] `accumulate` builds.
#[inline]
fn probe_delta(dim: &CustomDimension, a: &Arc2, cumul_before: i64) -> i64 {
    let ctx = ArcCtx {
        from: a.from,
        to: a.to,
        cumul_before,
        arrival: a.arrival,
        distance: a.distance,
        duration: a.duration,
    };
    (dim.transit)(&ctx)
}

/// Mirror of the non-decreasing-dimension `max` bound into the O(1) insertion
/// probe.
///
/// Given a route described by its ordered arcs (same `Arc2` sequence the full
/// evaluator threads), accumulate ONLY the dimensions declared
/// [`Monotonicity::NonDecreasing`] with a `max` bound and return `true` if any of
/// them breaches its max anywhere on the route. Because such a dimension never
/// decreases, its running peak is its final cumul; checking each position is
/// exactly the max-bound check, and a route that breaches with the current stops
/// can never be rescued by inserting more (every insertion adds a non-negative
/// delta). So `precompute` may reject early.
///
/// This is PRUNE-ONLY: it never reports a breach the full evaluator would not
/// also report (the full evaluator re-checks via `accumulate` and remains the
/// authority). Non-mirrorable dimensions are skipped entirely and keep their
/// full-eval-only behaviour. Returns `false` immediately when no probe-mirrorable
/// dimension is registered.
pub fn probe_breaches_monotone_max(arcs: &[Arc2]) -> bool {
    if !has_probe_dimensions() {
        return false;
    }
    let g = REGISTRY.read().unwrap();
    for dim in g.iter() {
        let max = match (dim.monotonicity, dim.max) {
            (Monotonicity::NonDecreasing, Some(m)) => m,
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
            v += probe_delta(dim, a, v);
            if v > max {
                return true;
            }
        }
    }
    false
}

/// The DUAL of [`probe_breaches_monotone_max`] for a *draining* resource: mirror
/// of the [`Monotonicity::NonIncreasing`] dimension's `min` bound (its floor)
/// into the O(1) probe.
///
/// A draining dimension's cumul never rises, so its running trough is its final
/// cumul; if any position already falls below `min`, no later insertion can lift
/// it back (every insertion adds a non-positive delta). So `precompute` may
/// reject early. PRUNE-ONLY, with the same authority/caveat split as the max
/// direction: a non-draining or `min`-less dimension is skipped and keeps its
/// full-eval-only behaviour.
pub fn probe_breaches_monotone_min(arcs: &[Arc2]) -> bool {
    if !has_probe_dimensions() {
        return false;
    }
    let g = REGISTRY.read().unwrap();
    for dim in g.iter() {
        let min = match (dim.monotonicity, dim.min) {
            (Monotonicity::NonIncreasing, Some(m)) => m,
            _ => continue,
        };
        let mut v = dim.start;
        if v < min {
            return true;
        }
        for a in arcs {
            // The caller asserted the cumul never rises; a (buggy) positive delta
            // can only cause a missed prune, never a false reject (we only ever
            // return on `< min`).
            v += probe_delta(dim, a, v);
            if v < min {
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
            Arc::new(|_ctx: &ArcCtx| -10),
        )
        .with_start(100)
        .with_min(0);
        let _g = DimensionGuard::install(vec![dim]);
        assert!(has_dimensions());

        // Three arcs → four positions: 100, 90, 80, 70. No bound violation.
        let arcs = [
            Arc2 { from: 0, to: 1, arrival: 100, distance: 0, duration: 0 },
            Arc2 { from: 1, to: 2, arrival: 200, distance: 0, duration: 0 },
            Arc2 { from: 2, to: 0, arrival: 300, distance: 0, duration: 0 },
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
        let dim = CustomDimension::new("fuel", Arc::new(|_: &ArcCtx| -40))
            .with_start(100)
            .with_min(0);
        let _g = DimensionGuard::install(vec![dim]);
        let arcs = [
            Arc2 { from: 0, to: 1, arrival: 1, distance: 0, duration: 0 },
            Arc2 { from: 1, to: 2, arrival: 2, distance: 0, duration: 0 },
            Arc2 { from: 2, to: 3, arrival: 3, distance: 0, duration: 0 },
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
            Arc::new(|c: &ArcCtx| {
                if c.cumul_before < 3 && c.arrival % 2 == 0 { 1 } else { 0 }
            }),
        );
        let _g = DimensionGuard::install(vec![dim]);
        let arcs = [
            Arc2 { from: 0, to: 1, arrival: 2, distance: 0, duration: 0 }, // even, cumul 0<3 → +1 → 1
            Arc2 { from: 1, to: 2, arrival: 3, distance: 0, duration: 0 }, // odd → +0 → 1
            Arc2 { from: 2, to: 3, arrival: 4, distance: 0, duration: 0 }, // even, 1<3 → +1 → 2
            Arc2 { from: 3, to: 4, arrival: 6, distance: 0, duration: 0 }, // even, 2<3 → +1 → 3
            Arc2 { from: 4, to: 5, arrival: 8, distance: 0, duration: 0 }, // even, 3<3 false → +0 → 3
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
        let mono = CustomDimension::new("load", Arc::new(|_: &ArcCtx| 10))
            .with_max(25)
            .monotone();
        let _g = DimensionGuard::install(vec![mono]);
        assert!(has_probe_dimensions(), "monotone+max is probe-expressible");

        // Monotone but NO max → not probe-mirrorable (nothing to prune against).
        let mono_nobound = CustomDimension::new("load", Arc::new(|_: &ArcCtx| 10)).monotone();
        set_dimensions(vec![mono_nobound]);
        assert!(!has_probe_dimensions(), "monotone without a max is not mirrorable");

        // Max but NOT declared monotone → stays on the full-eval fallback.
        let bounded_nonmono =
            CustomDimension::new("load", Arc::new(|_: &ArcCtx| 10)).with_max(25);
        set_dimensions(vec![bounded_nonmono]);
        assert!(!has_probe_dimensions(), "non-monotone keeps full-eval-only behaviour");

        // Draining + min → probe-mirrorable in the dual direction.
        let drain = CustomDimension::new("fuel", Arc::new(|_: &ArcCtx| -10))
            .with_start(100)
            .with_min(0)
            .draining();
        set_dimensions(vec![drain]);
        assert!(has_probe_dimensions(), "draining+min is probe-expressible (the dual)");

        // Draining but NO min → not mirrorable (no floor to prune against).
        let drain_nobound = CustomDimension::new("fuel", Arc::new(|_: &ArcCtx| -10)).draining();
        set_dimensions(vec![drain_nobound]);
        assert!(!has_probe_dimensions(), "draining without a min is not mirrorable");

        clear_dimensions();
        assert!(!has_probe_dimensions());
    }

    #[test]
    fn probe_breaches_matches_accumulate_for_monotone_resource() {
        let _lock = guard();
        // A "load" resource that accrues +10 per arc, capped at 25.
        let dim = CustomDimension::new("load", Arc::new(|_: &ArcCtx| 10))
            .with_max(25)
            .monotone();
        let _g = DimensionGuard::install(vec![dim]);

        // Two arcs → cumuls [0, 10, 20]: peak 20 ≤ 25, no breach.
        let ok = [
            Arc2 { from: 0, to: 1, arrival: 1, distance: 0, duration: 0 },
            Arc2 { from: 1, to: 0, arrival: 2, distance: 0, duration: 0 },
        ];
        assert!(!probe_breaches_monotone_max(&ok));
        // The full-eval accumulator agrees: not bound_violated.
        assert!(!accumulate(&ok).bound_violated);

        // Three arcs → cumuls [0, 10, 20, 30]: 30 > 25 → breach.
        let bad = [
            Arc2 { from: 0, to: 1, arrival: 1, distance: 0, duration: 0 },
            Arc2 { from: 1, to: 2, arrival: 2, distance: 0, duration: 0 },
            Arc2 { from: 2, to: 0, arrival: 3, distance: 0, duration: 0 },
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
        let dim = CustomDimension::new("fuel", Arc::new(|_: &ArcCtx| -10))
            .with_max(1000); // huge max, never breached anyway
        let _g = DimensionGuard::install(vec![dim]);
        assert!(!has_probe_dimensions());
        let arcs = [Arc2 { from: 0, to: 1, arrival: 1, distance: 0, duration: 0 }];
        assert!(!probe_breaches_monotone_max(&arcs), "non-monotone is never probe-pruned");
    }

    #[test]
    fn draining_min_probe_matches_accumulate() {
        let _lock = guard();
        // A draining fuel resource: starts at 100, burns 40/arc, floor 0.
        let dim = CustomDimension::new("fuel", Arc::new(|_: &ArcCtx| -40))
            .with_start(100)
            .with_min(0)
            .draining();
        let _g = DimensionGuard::install(vec![dim]);
        assert!(has_probe_dimensions(), "draining+min is probe-mirrorable");

        // Two arcs → cumuls [100, 60, 20]: trough 20 ≥ 0, no breach.
        let ok = [
            Arc2 { from: 0, to: 1, arrival: 1, distance: 0, duration: 0 },
            Arc2 { from: 1, to: 0, arrival: 2, distance: 0, duration: 0 },
        ];
        assert!(!probe_breaches_monotone_min(&ok), "trough 20 ≥ 0 is not pruned");
        assert!(!accumulate(&ok).bound_violated, "full-eval agrees: feasible");
        assert_eq!(accumulate(&ok).aggregate_min(0), 20, "trough reads 20");

        // Three arcs → cumuls [100, 60, 20, -20]: -20 < 0 → breach the floor.
        let bad = [
            Arc2 { from: 0, to: 1, arrival: 1, distance: 0, duration: 0 },
            Arc2 { from: 1, to: 2, arrival: 2, distance: 0, duration: 0 },
            Arc2 { from: 2, to: 0, arrival: 3, distance: 0, duration: 0 },
        ];
        assert!(probe_breaches_monotone_min(&bad), "probe detects the drained floor");
        assert!(accumulate(&bad).bound_violated, "full-eval agrees with the probe");
    }

    #[test]
    fn soft_bound_penalty_is_additive_not_a_reject() {
        let _lock = guard();
        // A load resource with a HARD max of 100 but a SOFT max of 15, weight 2.
        // Cumuls [0, 10, 20] → only position 2 (20) breaches soft by 5 → 2*5 = 10.
        let dim = CustomDimension::new("load", Arc::new(|_: &ArcCtx| 10))
            .with_max(100)
            .with_soft_max(15, 2.0)
            .monotone();
        let _g = DimensionGuard::install(vec![dim]);
        let arcs = [
            Arc2 { from: 0, to: 1, arrival: 1, distance: 0, duration: 0 },
            Arc2 { from: 1, to: 2, arrival: 2, distance: 0, duration: 0 },
        ];
        let c = accumulate(&arcs);
        assert!(!c.bound_violated, "still within the hard max → not a reject");
        assert_eq!(c.soft_penalty, 10.0, "2 × (20 − 15) over the one breaching position");
    }

    #[test]
    fn coupling_reads_arc_distance() {
        let _lock = guard();
        // Fuel burn proportional to arc distance: burn = distance / 10.
        let dim = CustomDimension::new(
            "fuel",
            Arc::new(|c: &ArcCtx| -(c.distance / 10)),
        )
        .with_start(100);
        let _g = DimensionGuard::install(vec![dim]);
        let arcs = [
            Arc2 { from: 0, to: 1, arrival: 1, distance: 200, duration: 30 }, // -20 → 80
            Arc2 { from: 1, to: 2, arrival: 2, distance: 500, duration: 70 }, // -50 → 30
        ];
        let c = accumulate(&arcs);
        assert_eq!(c.cumuls_of(0), &[100, 80, 30], "burn tracks per-arc distance");
    }
}
