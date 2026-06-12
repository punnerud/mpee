//! Per-route time-window slack + load-prefix structure for O(1) move
//! feasibility checks in the fast LS paths (Kindervater–Savelsbergh /
//! Vidal-style preprocessing).
//!
//! The fast operators already rank candidates by exact O(1) arc-cost deltas;
//! what made education expensive at n=400 was confirming FEASIBILITY of each
//! candidate with a full `evaluate_route` walk (O(route) per probe, dozens of
//! probes per task). With a per-route forward earliest-departure array (`ect`),
//! a backward latest-arrival array (`lat`) and load prefix/suffix maxima, a
//! candidate move is judged in O(moved stops) instead — and the arrays are
//! rebuilt O(route) only when a route is actually MODIFIED, not per candidate.
//!
//! ## Soundness contract (option B)
//!
//! The slack verdict is used as a PREFILTER: a candidate the slack check
//! rejects is skipped without evaluation; the surviving best candidate is
//! still confirmed by `evaluate_route` (whose metrics the Move needs anyway).
//! The LS trajectory is therefore identical to the eval-everything path as
//! long as the check has **no false negatives** — it must never reject a
//! candidate `evaluate_route` would accept. Under [`slack_eligible`] the
//! TW/load math below is exact (two-sided), and anything the math does not
//! model (skills, vehicle allowlists, unreachable-leg sentinels under
//! unbounded windows) errs on the PASS side and is caught by the confirm.
//!
//! Rounding parity is critical: every duration here is
//! `((matrix.duration(a, b) as f64) * speed).round() as i64`, identical to
//! `evaluate_route`.

use crate::matrix::Matrix;
use crate::problem::{Problem, TimeWindow, Vehicle};
use crate::solution::{Solution, TaskRef};

/// Sentinel for "no feasible arrival time exists" in `lat`. Far enough from
/// i64::MIN that subtracting a travel time can never underflow.
const LAT_NEG: i64 = i64::MIN / 4;

/// Kill-switch for A/B comparison (mirror of BROOOM_NO_FAST_LS). Read per
/// call — it is checked once per local_search invocation, and tests toggle
/// it in-process.
fn slack_ls_enabled() -> bool {
    std::env::var("BROOOM_NO_SLACK_LS").is_err()
}

/// Is the slack math EXACT for this problem? Conditions on top of the fast
/// arc-cost gate (`fast_cost_eligible`, checked by the caller since the fast
/// paths already require it):
///   - jobs only (no shipments → no pickup-before-delivery pairing, and the
///     load model `L_k = Σ future deliveries + Σ past pickups` holds);
///   - no first-class precedence pairs;
///   - at most one time window per job (multi-TW would need per-window
///     scanning in the backward recursion — a later arrival can be feasible
///     in a later window, so single-window `lat` math would falsely reject);
///   - no setup or release times (sequence-dependent terms the arrays don't
///     model);
///   - no driver breaks and no max_tasks / max_travel_time / max_distance
///     (route-level constraints outside the TW/load model).
///
/// Everything else (skills, allowed_vehicles, backhaul ordering, unreachable
/// legs) is deliberately NOT gated: ignoring them can only produce false
/// positives, which the confirm evaluation catches.
pub fn slack_eligible(problem: &Problem) -> bool {
    if !slack_ls_enabled() {
        return false;
    }
    if !problem.shipments.is_empty() || !problem.precedence.is_empty() {
        return false;
    }
    if !problem.jobs.iter().all(|j| {
        j.time_windows.len() <= 1 && j.setup == 0 && j.release == 0
    }) {
        return false;
    }
    problem.vehicles.iter().all(|v| {
        v.breaks.is_empty()
            && v.max_tasks.is_none()
            && v.max_travel_time.is_none()
            && v.max_distance.is_none()
    })
}

/// The single (or universal) time window of a job-step under the gate.
#[inline]
fn step_tw(problem: &Problem, t: TaskRef) -> TimeWindow {
    let job = t.description(problem);
    job.time_windows.first().copied().unwrap_or(TimeWindow::FOREVER)
}

/// Outcome of propagating a departure time through a chain of moved stops.
pub enum Chain {
    /// Some window in the chain is provably missed — the candidate is
    /// infeasible and may be skipped without evaluation.
    Infeasible,
    /// Departure time after the last chained stop, and its location.
    Dep { t: i64, loc: usize },
    /// A stop had no matrix location — the math cannot judge; treat as pass.
    Unknown,
}

/// Precomputed slack arrays for one route (one specific step sequence on one
/// specific vehicle). All load arrays are flattened `[index * dim + d]`.
pub struct RouteSlack {
    n: usize,
    dim: usize,
    speed: f64,
    vw: TimeWindow,
    start_idx: Option<usize>,
    end_idx: Option<usize>,
    /// Matrix location per stop.
    locs: Vec<usize>,
    /// `ect[k]` = departure clock after serving stop k (mirrors the `t` of
    /// `evaluate_route` right after `t += service`).
    ect: Vec<i64>,
    /// `lat[k]` = latest feasible ARRIVAL at stop k (the clock value on
    /// arrival, before waiting) such that stop k, the whole suffix and the
    /// final depot leg all meet their windows. `LAT_NEG` = impossible.
    lat: Vec<i64>,
    /// `init[d]` = load when leaving the depot = Σ deliveries.
    init: Vec<i64>,
    /// Prefix-exclusive delivery/pickup sums: `del_pre[k*dim+d]` = Σ of the
    /// first k stops' deliveries (so `del_pre[0] = 0`, `del_pre[n] = total`).
    del_pre: Vec<i64>,
    pick_pre: Vec<i64>,
    /// `pre_maxs[j*dim+d]` = max over the load checkpoints BEFORE gap j:
    /// init, L_0 .. L_{j-1}. (`pre_maxs[0] = init`.)
    pre_maxs: Vec<i64>,
    /// `suf_maxs[j*dim+d]` = max(L_j .. L_{n-1}); `suf_maxs[n] = i64::MIN`.
    suf_maxs: Vec<i64>,
}

impl RouteSlack {
    /// Build the arrays for `steps` on `veh`. Returns `None` when something
    /// prevents exact math (missing matrix index, a currently-infeasible
    /// route, an unexpected non-job step) — callers then skip slack checks
    /// for this route and confirm with `evaluate_route` as before.
    pub fn build(
        problem: &Problem,
        matrix: &Matrix,
        veh: &Vehicle,
        steps: &[TaskRef],
    ) -> Option<RouteSlack> {
        let n = steps.len();
        if n == 0 {
            return None;
        }
        let dim = problem.capacity_dim().max(veh.capacity.len()).max(1);
        let speed = veh.speed_factor.max(0.01);
        let vw = veh.time_window();
        let start_idx = veh
            .start
            .as_ref()
            .and_then(|l| l.index)
            .or_else(|| veh.end.as_ref().and_then(|l| l.index));
        let end_idx = veh.end.as_ref().and_then(|l| l.index).or(start_idx);

        let dur = |a: usize, b: usize| -> i64 {
            ((matrix.duration(a, b) as f64) * speed).round() as i64
        };

        let mut locs = Vec::with_capacity(n);
        for &s in steps {
            if !matches!(s, TaskRef::Job(_)) {
                return None; // reloads/shipments are gated out, but be safe
            }
            locs.push(s.description(problem).location.index?);
        }

        // Forward: earliest departure after each stop.
        let mut ect = Vec::with_capacity(n);
        let mut t = vw.start;
        let mut prev = start_idx;
        for (k, &s) in steps.iter().enumerate() {
            let job = s.description(problem);
            if let Some(p) = prev {
                t += dur(p, locs[k]);
            }
            let w = step_tw(problem, s);
            if t < w.start {
                t = w.start;
            }
            if t > w.end {
                return None; // route currently infeasible (shouldn't happen in hard mode)
            }
            t += job.service;
            ect.push(t);
            prev = Some(locs[k]);
        }

        // Backward: latest feasible arrival per stop. Base case is the final
        // depot leg (`evaluate_route` checks the vehicle window only at route
        // end): arrival at the end depot must be ≤ vw.end.
        let mut lat = vec![0i64; n];
        let mut next_lat = vw.end; // latest arrival at the "position after" point
        let mut next_loc: Option<usize> = end_idx; // None ⇒ no arc to it
        for k in (0..n).rev() {
            let job = steps[k].description(problem);
            let w = step_tw(problem, steps[k]);
            let d = match next_loc {
                Some(nl) => dur(locs[k], nl),
                None => 0,
            };
            let dep_limit = next_lat.saturating_sub(d).saturating_sub(job.service);
            lat[k] = if w.start > dep_limit {
                LAT_NEG
            } else {
                w.end.min(dep_limit)
            };
            next_lat = lat[k];
            next_loc = Some(locs[k]);
        }

        // Loads. Under the gate every step is a single job: deliveries are
        // subtracted, pickups added; initial load is the sum of deliveries.
        let mut init = vec![0i64; dim];
        let mut del_pre = vec![0i64; (n + 1) * dim];
        let mut pick_pre = vec![0i64; (n + 1) * dim];
        for (k, &s) in steps.iter().enumerate() {
            let job = s.description(problem);
            for d in 0..dim {
                let dl = job.delivery.get(d).copied().unwrap_or(0);
                let pk = job.pickup.get(d).copied().unwrap_or(0);
                del_pre[(k + 1) * dim + d] = del_pre[k * dim + d] + dl;
                pick_pre[(k + 1) * dim + d] = pick_pre[k * dim + d] + pk;
                init[d] += dl;
            }
        }
        // L_k = init − del_pre[k+1] + pick_pre[k+1]
        let load_at = |k: usize, d: usize| -> i64 {
            init[d] - del_pre[(k + 1) * dim + d] + pick_pre[(k + 1) * dim + d]
        };
        let mut pre_maxs = vec![0i64; (n + 1) * dim];
        for d in 0..dim {
            pre_maxs[d] = init[d];
        }
        for j in 1..=n {
            for d in 0..dim {
                pre_maxs[j * dim + d] = pre_maxs[(j - 1) * dim + d].max(load_at(j - 1, d));
            }
        }
        let mut suf_maxs = vec![i64::MIN; (n + 1) * dim];
        for j in (0..n).rev() {
            for d in 0..dim {
                suf_maxs[j * dim + d] = suf_maxs[(j + 1) * dim + d].max(load_at(j, d));
            }
        }

        Some(RouteSlack {
            n,
            dim,
            speed,
            vw,
            start_idx,
            end_idx,
            locs,
            ect,
            lat,
            init,
            del_pre,
            pick_pre,
            pre_maxs,
            suf_maxs,
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn dur(&self, matrix: &Matrix, a: usize, b: usize) -> i64 {
        ((matrix.duration(a, b) as f64) * self.speed).round() as i64
    }

    /// Departure time and location of the predecessor of gap `j` (insertion
    /// slot before original stop `j`): the previous stop's `ect`, or the
    /// vehicle's day start at the start depot for `j == 0`.
    #[inline]
    pub fn depart_before(&self, j: usize) -> (i64, Option<usize>) {
        if j == 0 {
            (self.vw.start, self.start_idx)
        } else {
            (self.ect[j - 1], Some(self.locs[j - 1]))
        }
    }

    /// Earliest departure after original stop `k`.
    #[inline]
    pub fn ect(&self, k: usize) -> i64 {
        self.ect[k]
    }

    /// Latest feasible arrival at original stop `j`; `j == n` means the
    /// end-depot position (`vw.end`).
    #[inline]
    pub fn lat(&self, j: usize) -> i64 {
        if j < self.n {
            self.lat[j]
        } else {
            self.vw.end
        }
    }

    /// Can a vehicle departing `from` at time `t` still serve original stop
    /// `j` and the whole suffix (or, for `j == n`, reach the end depot)?
    #[inline]
    pub fn admits_arrival(&self, matrix: &Matrix, j: usize, t: i64, from: Option<usize>) -> bool {
        if j < self.n {
            let arr = match from {
                Some(f) => t + self.dur(matrix, f, self.locs[j]),
                None => t,
            };
            arr <= self.lat[j]
        } else {
            let arr = match (from, self.end_idx) {
                (Some(f), Some(e)) => t + self.dur(matrix, f, e),
                _ => t,
            };
            arr <= self.vw.end
        }
    }

    /// Propagate a departure time through a chain of moved stops. `Infeasible`
    /// is exact under the gate; `Unknown` (missing matrix index) means the
    /// caller must fall back to evaluation.
    pub fn chain_dep(
        &self,
        problem: &Problem,
        matrix: &Matrix,
        mut t: i64,
        mut from: Option<usize>,
        tasks: &[TaskRef],
    ) -> Chain {
        let mut loc = 0usize;
        if tasks.is_empty() {
            return Chain::Unknown;
        }
        for &task in tasks {
            let job = task.description(problem);
            let Some(here) = job.location.index else {
                return Chain::Unknown;
            };
            if let Some(f) = from {
                t += self.dur(matrix, f, here);
            }
            let w = step_tw(problem, task);
            if t < w.start {
                t = w.start;
            }
            if t > w.end {
                return Chain::Infeasible;
            }
            t += job.service;
            from = Some(here);
            loc = here;
        }
        Chain::Dep { t, loc }
    }

    /// TW feasibility of removing stops `[i, k)` from this route: the bridge
    /// arc from the predecessor of `i` straight to stop `k` (or the end
    /// depot) must still meet every downstream window. Exact under the gate;
    /// removal is always load-feasible for plain jobs. Removing the WHOLE
    /// route is always fine (the route is dropped, never evaluated).
    pub fn removal_ok(&self, matrix: &Matrix, i: usize, k: usize) -> bool {
        if i == 0 && k >= self.n {
            return true;
        }
        let (t, from) = self.depart_before(i);
        self.admits_arrival(matrix, k, t, from)
    }

    /// Load before gap `j` (i.e. after stop `j-1`; the depot-departure load
    /// for `j == 0`), per dimension `d`.
    #[inline]
    pub fn load_before(&self, j: usize, d: usize) -> i64 {
        self.init[d] - self.del_pre[j * self.dim + d] + self.pick_pre[j * self.dim + d]
    }

    /// Σ deliveries of the first `k` stops, per dimension.
    #[inline]
    pub fn del_pre(&self, k: usize, d: usize) -> i64 {
        self.del_pre[k * self.dim + d]
    }

    /// Σ pickups of the first `k` stops, per dimension.
    #[inline]
    pub fn pick_pre(&self, k: usize, d: usize) -> i64 {
        self.pick_pre[k * self.dim + d]
    }

    #[inline]
    pub fn init(&self, d: usize) -> i64 {
        self.init[d]
    }

    /// Would every load checkpoint BEFORE gap `j` (depot departure and stops
    /// 0..j) stay within `veh`'s capacity after shifting by `shift[d]`?
    /// Mirrors `evaluate_route`: only dimensions the vehicle declares are
    /// checked.
    pub fn pre_shift_ok(&self, veh: &Vehicle, j: usize, shift: &[i64]) -> bool {
        let nd = self.dim.min(veh.capacity.len());
        for d in 0..nd {
            if self.pre_maxs[j * self.dim + d] + shift[d] > veh.capacity[d] {
                return false;
            }
        }
        true
    }

    /// Would every load checkpoint FROM original stop `j` on (stops j..n-1)
    /// stay within `veh`'s capacity after shifting by `shift[d]`?
    pub fn suf_shift_ok(&self, veh: &Vehicle, j: usize, shift: &[i64]) -> bool {
        if j >= self.n {
            return true;
        }
        let nd = self.dim.min(veh.capacity.len());
        for d in 0..nd {
            let m = self.suf_maxs[j * self.dim + d];
            if m != i64::MIN && m + shift[d] > veh.capacity[d] {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    #[inline]
    pub fn loc(&self, k: usize) -> usize {
        self.locs[k]
    }

    /// Feasibility of replacing the stops `[i, k)` of this route with
    /// `new_seg` (possibly empty): time windows of the new stops, the bridge
    /// into the unchanged suffix, and every load checkpoint. One method
    /// covers removal (`new_seg = []`), insertion at gap `j` (`i == k == j`),
    /// single exchange (`k == i + 1`), cross-exchange (`k == i + 2`) and
    /// best-insertion probes.
    ///
    /// Returns `false` only when the candidate is PROVEN infeasible under
    /// the gate; anything the math cannot judge returns `true` and is left
    /// to the confirm evaluation.
    pub fn replace_seg_ok(
        &self,
        problem: &Problem,
        matrix: &Matrix,
        veh: &Vehicle,
        i: usize,
        k: usize,
        new_seg: &[TaskRef],
    ) -> bool {
        // Removing the whole route: it is dropped, never evaluated.
        if i == 0 && k >= self.n && new_seg.is_empty() {
            return true;
        }

        // ── time windows ────────────────────────────────────────────────
        let (t0, from0) = self.depart_before(i);
        let (t_dep, from) = if new_seg.is_empty() {
            (t0, from0)
        } else {
            match self.chain_dep(problem, matrix, t0, from0, new_seg) {
                Chain::Infeasible => return false,
                Chain::Unknown => return true,
                Chain::Dep { t, loc } => (t, Some(loc)),
            }
        };
        if !self.admits_arrival(matrix, k, t_dep, from) {
            return false;
        }

        // ── loads ───────────────────────────────────────────────────────
        let dim = self.dim;
        let nd = dim.min(veh.capacity.len());
        if nd == 0 {
            return true;
        }
        let mut del_new = [0i64; 8];
        let mut pick_new = [0i64; 8];
        if dim > 8 {
            return true; // exotic dimensionality — leave to the evaluator
        }
        for &s in new_seg {
            let job = s.description(problem);
            for d in 0..dim {
                del_new[d] += job.delivery.get(d).copied().unwrap_or(0);
                pick_new[d] += job.pickup.get(d).copied().unwrap_or(0);
            }
        }
        // Prefix checkpoints (depot departure and stops 0..i) shift by the
        // change in total downstream deliveries.
        for d in 0..nd {
            let shift_pre = del_new[d] - (self.del_pre(k, d) - self.del_pre(i, d));
            if self.pre_maxs[i * dim + d] + shift_pre > veh.capacity[d] {
                return false;
            }
        }
        // Checkpoints inside the new segment.
        if !new_seg.is_empty() {
            let mut l = [0i64; 8];
            for d in 0..nd {
                let shift_pre = del_new[d] - (self.del_pre(k, d) - self.del_pre(i, d));
                l[d] = self.load_before(i, d) + shift_pre;
            }
            for &s in new_seg {
                let job = s.description(problem);
                for d in 0..nd {
                    l[d] -= job.delivery.get(d).copied().unwrap_or(0);
                    l[d] += job.pickup.get(d).copied().unwrap_or(0);
                    if l[d] > veh.capacity[d] {
                        return false;
                    }
                }
            }
        }
        // Suffix checkpoints (stops k..n-1) shift by the change in carried
        // pickups.
        for d in 0..nd {
            let shift_suf = pick_new[d] - (self.pick_pre(k, d) - self.pick_pre(i, d));
            if k < self.n {
                let m = self.suf_maxs[k * dim + d];
                if m != i64::MIN && m + shift_suf > veh.capacity[d] {
                    return false;
                }
            }
        }
        true
    }
}

/// One cache slot per route, index-aligned with `sol.routes`.
enum Slot {
    /// Not built (or invalidated by an applied move).
    Stale,
    /// Built and current.
    Ready(Box<RouteSlack>),
    /// Build failed (missing index / currently infeasible) — don't retry
    /// until the route changes.
    Unusable,
}

/// Lazily-built per-route slack arrays for the solution a local-search core
/// loop is working on. MUST be kept index-aligned with `sol.routes`: call
/// [`SlackCache::on_apply`] with the move's route updates right before
/// `apply_move`.
pub struct SlackCache {
    slots: Vec<Slot>,
}

impl SlackCache {
    pub fn new(n_routes: usize) -> SlackCache {
        SlackCache { slots: (0..n_routes).map(|_| Slot::Stale).collect() }
    }

    /// Build (if needed) the slack arrays for route `r`.
    pub fn ensure(&mut self, r: usize, problem: &Problem, matrix: &Matrix, sol: &Solution) {
        debug_assert_eq!(self.slots.len(), sol.routes.len());
        if let Slot::Stale = self.slots[r] {
            let route = &sol.routes[r];
            let veh = &problem.vehicles[route.vehicle_idx];
            self.slots[r] = match RouteSlack::build(problem, matrix, veh, &route.steps) {
                Some(s) => Slot::Ready(Box::new(s)),
                None => Slot::Unusable,
            };
        }
    }

    /// The slack arrays for route `r`, if buildable. Call [`ensure`] first.
    pub fn get(&self, r: usize) -> Option<&RouteSlack> {
        match &self.slots[r] {
            Slot::Ready(s) => Some(s),
            _ => None,
        }
    }

    /// Mirror an applied move: updated routes go stale, removed routes drop
    /// their slot (same descending-index order as `apply_move`).
    pub fn on_apply<T>(&mut self, route_updates: &[(usize, Option<T>)]) {
        let mut idxs: Vec<(usize, bool)> =
            route_updates.iter().map(|(r, p)| (*r, p.is_none())).collect();
        idxs.sort_by(|a, b| b.0.cmp(&a.0));
        for (r, removed) in idxs {
            if removed {
                self.slots.remove(r);
            } else {
                self.slots[r] = Slot::Stale;
            }
        }
    }
}
