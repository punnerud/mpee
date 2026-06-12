//! Per-route time-warp segment statistics for O(1) SOFT move deltas
//! (Vidal 2013 / PyVRP `TimeWindowSegment` concatenation).
//!
//! The hard-mode fast LS judges candidates with `crate::slack` (exact
//! feasibility verdicts). Under soft penalties there is no infeasibility to
//! prefilter — every candidate "passes" — so the work is computing the
//! PENALTY DELTA of a move cheaply. Carry-forward lateness (the public
//! soft-TW semantics) does not compose into constant-size segment state, but
//! warp semantics (clamp a late arrival back to the window end, charge the
//! clamped amount) does: a segment is summarised by four numbers
//! `(dur, warp, ear, lat)` and two segments concatenate in O(1). With forward
//! prefix segments and backward suffix segments per route, the violations of
//! a route with `[i, k)` replaced by an arbitrary new segment are exact in
//! O(|new segment|).
//!
//! These stats are only meaningful when `evaluate_route` runs with
//! `SoftMode::Warp` (see `crate::solution::set_soft_penalties_mode`): there
//! the confirm evaluation agrees exactly with the deltas computed here, so
//! the fast operators' rank-then-confirm-first contract carries over with no
//! mis-ranking. Only the HGS infeasible track arms warp mode; the public
//! carry-forward soft mode never uses this module.
//!
//! Rounding parity is critical and identical to `evaluate_route` and
//! `crate::slack`: every duration is
//! `((matrix.duration(a, b) as f64) * speed).round() as i64`.

use crate::matrix::Matrix;
use crate::problem::{Problem, TimeWindow, Vehicle};
use crate::solution::{SoftWeights, Solution, TaskRef};

/// Window bound used in the segment math for "unbounded" (FOREVER) windows
/// and the end-depot node. Far enough from i64::MAX that sums of a few terms
/// can never overflow (all merges also use saturating ops as a second guard).
const H: i64 = i64::MAX / 8;

/// Kill-switch for A/B comparison (mirror of BROOOM_NO_SLACK_LS). Read per
/// call — checked once per local_search invocation; tests toggle it
/// in-process.
fn warp_ls_enabled() -> bool {
    std::env::var("BROOOM_NO_WARP_LS").is_err()
}

/// Is the warp math EXACT for this problem? The structural TW/load envelope
/// is shared with `slack_eligible` (jobs only, ≤1 window, no setup/release/
/// breaks/max_*); on top of it the load-excess math needs a SINGLE capacity
/// dimension: `evaluate_route` tracks the peak of the per-checkpoint SUM of
/// per-dimension overloads, and a max-of-sums does not decompose into
/// per-dimension prefix maxima. (One dimension covers the GH/Solomon
/// benchmark families; multi-dim soft instances stay on the slow path.)
pub fn warp_eligible(problem: &Problem) -> bool {
    if !warp_ls_enabled() {
        return false;
    }
    if problem.capacity_dim() > 1 {
        return false;
    }
    crate::slack::tw_load_envelope(problem)
}

/// The single (or universal) time window of a job-step under the gate.
#[inline]
fn step_tw(problem: &Problem, t: TaskRef) -> TimeWindow {
    let job = t.description(problem);
    job.time_windows.first().copied().unwrap_or(TimeWindow::FOREVER)
}

/// Time-warp segment statistics: a chain of stops summarised by its total
/// `dur`ation on the clamped timeline (travel + service + waiting), the
/// accumulated `warp` (sum of clamped lateness), and the earliest/latest
/// start-of-service times `ear`/`lat` at its first stop that minimise
/// waiting/warp respectively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tws {
    pub dur: i64,
    pub warp: i64,
    pub ear: i64,
    pub lat: i64,
}

impl Tws {
    /// Segment of one job stop: its window and service time.
    #[inline]
    pub fn node(w: TimeWindow, service: i64) -> Tws {
        Tws { dur: service, warp: 0, ear: w.start.clamp(-H, H), lat: w.end.clamp(-H, H) }
    }

    /// Concatenate `a ⟶(edge)⟶ b` (PyVRP `TimeWindowSegment::merge`): `edge`
    /// is the rounded travel time from `a`'s last location to `b`'s first.
    #[inline]
    pub fn merge(a: Tws, edge: i64, b: Tws) -> Tws {
        let delta = a.dur - a.warp + edge;
        let wait = b.ear.saturating_sub(delta).saturating_sub(a.lat).max(0);
        let extra = a.ear.saturating_add(delta).saturating_sub(b.lat).max(0);
        Tws {
            dur: a.dur + b.dur + edge + wait,
            warp: a.warp + b.warp + extra,
            ear: a.ear.max(b.ear.saturating_sub(delta)) - wait,
            lat: a.lat.min(b.lat.saturating_sub(delta)) + extra,
        }
    }
}

/// Route violation totals, matching `evaluate_route`'s soft accumulators in
/// `SoftMode::Warp`: `tw` = summed clamped lateness, `load` = peak capacity
/// excess over the checkpoints, `dur` = route-end overrun past the vehicle
/// window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Viol {
    pub tw: i64,
    pub load: i64,
    pub dur: i64,
}

impl Viol {
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.tw == 0 && self.load == 0 && self.dur == 0
    }

    /// Weighted penalty of these violations (the term `evaluate_route` folds
    /// into `cost_custom`/`cost` in warp mode).
    #[inline]
    pub fn penalty(&self, sw: SoftWeights) -> f64 {
        sw.tw * self.tw as f64 + sw.load * self.load as f64 + sw.dur * self.dur as f64
    }
}

/// Precomputed warp/load arrays for one route (one specific step sequence on
/// one specific vehicle). Unlike `RouteSlack`, this ALWAYS builds on
/// hard-infeasible routes — that is the whole point: the soft LS works on
/// penalised, possibly-violating routes. Build fails only structurally
/// (empty route, non-job step, missing matrix index).
pub struct RouteWarp {
    n: usize,
    speed: f64,
    vw: TimeWindow,
    start_idx: Option<usize>,
    end_idx: Option<usize>,
    /// Matrix location per stop.
    locs: Vec<usize>,
    /// `pre[k]` = depot-start node ⊕ stops `0..k` (so `pre[0]` is the pinned
    /// depot-start segment alone; `pre[n]` covers the whole stop chain).
    pre: Vec<Tws>,
    /// `suf[k]` = stops `k..n` as a pure job chain (no depot ends);
    /// `suf[n]` is unused (the `k == n` case is handled directly).
    suf: Vec<Tws>,
    /// Single-dimension load model (dim ≤ 1 is enforced by `warp_eligible`):
    /// `init` = Σ deliveries; prefix-exclusive sums `del_pre[k]`/`pick_pre[k]`
    /// = Σ of the first k stops' deliveries/pickups.
    init: i64,
    cap: i64,
    del_pre: Vec<i64>,
    pick_pre: Vec<i64>,
    /// `pre_max[j]` = max over load checkpoints BEFORE gap j (init, L_0 ..
    /// L_{j-1}); `suf_max[j]` = max(L_j .. L_{n-1}), `suf_max[n]` = i64::MIN.
    pre_max: Vec<i64>,
    suf_max: Vec<i64>,
    /// Violations of the route as-is (the delta baseline).
    base: Viol,
}

impl RouteWarp {
    /// Empty scratch instance for hot call sites that rebuild against many
    /// transient step lists — Vec capacities survive across `rebuild`s.
    pub fn scratch() -> RouteWarp {
        RouteWarp {
            n: 0,
            speed: 1.0,
            vw: TimeWindow::FOREVER,
            start_idx: None,
            end_idx: None,
            locs: Vec::new(),
            pre: Vec::new(),
            suf: Vec::new(),
            init: 0,
            cap: 0,
            del_pre: Vec::new(),
            pick_pre: Vec::new(),
            pre_max: Vec::new(),
            suf_max: Vec::new(),
            base: Viol::default(),
        }
    }

    /// Build the arrays for `steps` on `veh`. `None` only on structural
    /// failure (empty, non-job step, missing matrix index) — never because
    /// the route violates windows or capacity.
    pub fn build(
        problem: &Problem,
        matrix: &Matrix,
        veh: &Vehicle,
        steps: &[TaskRef],
    ) -> Option<RouteWarp> {
        let mut w = RouteWarp::scratch();
        if w.rebuild(problem, matrix, veh, steps) {
            Some(w)
        } else {
            None
        }
    }

    /// Refill in place (clear + extend, allocations kept). `false` under the
    /// same conditions `build` returns `None`; the instance is then in an
    /// unspecified state until a `rebuild` succeeds.
    pub fn rebuild(
        &mut self,
        problem: &Problem,
        matrix: &Matrix,
        veh: &Vehicle,
        steps: &[TaskRef],
    ) -> bool {
        let n = steps.len();
        if n == 0 {
            return false;
        }
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

        let locs = &mut self.locs;
        locs.clear();
        locs.reserve(n);
        for &s in steps {
            if !matches!(s, TaskRef::Job(_)) {
                return false;
            }
            match s.description(problem).location.index {
                Some(l) => locs.push(l),
                None => return false,
            }
        }

        // Forward prefixes. The depot-start segment pins the route start at
        // vw.start (evaluate_route departs at vw.start, it never waits at the
        // start depot) — with ear == lat the pin is invariant under merges,
        // so the end clock of any prefix is `vw.start + dur − warp`.
        let pre = &mut self.pre;
        pre.clear();
        pre.reserve(n + 1);
        let start_seg = Tws { dur: 0, warp: 0, ear: vw.start, lat: vw.start };
        pre.push(start_seg);
        let mut prev_loc = start_idx;
        for (k, &s) in steps.iter().enumerate() {
            let node = Tws::node(step_tw(problem, s), s.description(problem).service);
            let edge = match prev_loc {
                Some(p) => dur(p, locs[k]),
                None => 0,
            };
            pre.push(Tws::merge(pre[k], edge, node));
            prev_loc = Some(locs[k]);
        }

        // Backward suffixes: pure job chains (no depot ends), so they compose
        // after any replacement segment.
        let suf = &mut self.suf;
        suf.clear();
        suf.resize(n + 1, Tws { dur: 0, warp: 0, ear: -H, lat: H });
        for k in (0..n).rev() {
            let node = Tws::node(step_tw(problem, steps[k]), steps[k].description(problem).service);
            suf[k] = if k + 1 < n {
                Tws::merge(node, dur(locs[k], locs[k + 1]), suf[k + 1])
            } else {
                node
            };
        }

        // Single-dimension loads (jobs only: deliveries subtract, pickups add,
        // initial load = Σ deliveries) — same model as RouteSlack at dim 1,
        // but tracked as VALUES so the peak EXCESS of a candidate is exact.
        let cap = veh.capacity.first().copied().unwrap_or(i64::MAX / 4);
        let del_pre = &mut self.del_pre;
        del_pre.clear();
        del_pre.resize(n + 1, 0i64);
        let pick_pre = &mut self.pick_pre;
        pick_pre.clear();
        pick_pre.resize(n + 1, 0i64);
        let mut init = 0i64;
        for (k, &s) in steps.iter().enumerate() {
            let job = s.description(problem);
            let dl = job.delivery.first().copied().unwrap_or(0);
            let pk = job.pickup.first().copied().unwrap_or(0);
            del_pre[k + 1] = del_pre[k] + dl;
            pick_pre[k + 1] = pick_pre[k] + pk;
            init += dl;
        }
        // L_k = init − del_pre[k+1] + pick_pre[k+1]
        let load_at = |k: usize| -> i64 { init - del_pre[k + 1] + pick_pre[k + 1] };
        let pre_max = &mut self.pre_max;
        pre_max.clear();
        pre_max.resize(n + 1, 0i64);
        pre_max[0] = init;
        for j in 1..=n {
            pre_max[j] = pre_max[j - 1].max(load_at(j - 1));
        }
        let suf_max = &mut self.suf_max;
        suf_max.clear();
        suf_max.resize(n + 1, i64::MIN);
        for j in (0..n).rev() {
            suf_max[j] = suf_max[j + 1].max(load_at(j));
        }

        self.n = n;
        self.speed = speed;
        self.vw = vw;
        self.start_idx = start_idx;
        self.end_idx = end_idx;
        self.init = init;
        self.cap = cap;
        let full = self.pre[n];
        let excess = (self.pre_max[n] - cap).max(0);
        self.base = self.tail_viol(matrix, full, prev_loc, excess);
        true
    }

    /// Violations of the route as it stands (the delta baseline).
    #[inline]
    pub fn viol(&self) -> Viol {
        self.base
    }

    #[inline]
    fn dur(&self, matrix: &Matrix, a: usize, b: usize) -> i64 {
        ((matrix.duration(a, b) as f64) * self.speed).round() as i64
    }

    /// Rounded travel time between two matrix locations at this route's
    /// speed — for callers composing their own segment stats (2-opt's
    /// incremental reversal scan).
    #[inline]
    pub fn edge_dur(&self, matrix: &Matrix, a: usize, b: usize) -> i64 {
        self.dur(matrix, a, b)
    }

    /// Close a full-route segment (depot-pinned prefix covering every stop)
    /// with the final depot leg and package the violations. `excess` is the
    /// already-computed peak load excess.
    fn tail_viol(&self, matrix: &Matrix, full: Tws, last_loc: Option<usize>, excess: i64) -> Viol {
        let end_edge = match (last_loc, self.end_idx) {
            (Some(p), Some(e)) => self.dur(matrix, p, e),
            _ => 0,
        };
        // The end depot has no window of its own (the vehicle window is
        // charged as dur excess below, not as warp), so the final leg only
        // extends the duration. With the start pinned, the end clock is
        // vw.start + dur − warp.
        let end_clock = self.vw.start + (full.dur - full.warp) + end_edge;
        Viol {
            tw: full.warp,
            load: excess,
            dur: (end_clock - self.vw.end).max(0),
        }
    }

    /// Violations of the route with `[i, k)` replaced by `new_seg`
    /// (`i == k` ⇒ pure insertion at gap i; empty `new_seg` ⇒ pure removal).
    /// Exact in O(|new_seg|). Returns `None` when a new stop is structurally
    /// unjudgeable (non-job, missing matrix index) — callers fall back to the
    /// slow path for that candidate.
    pub fn replace_seg_viol(
        &self,
        problem: &Problem,
        matrix: &Matrix,
        i: usize,
        k: usize,
        new_seg: &[TaskRef],
    ) -> Option<Viol> {
        debug_assert!(i <= k && k <= self.n);
        // ── Time: prefix ⊕ new nodes ⊕ suffix ⊕ final leg ──
        let mut cur = self.pre[i];
        let mut cur_loc = if i == 0 { self.start_idx } else { Some(self.locs[i - 1]) };
        for &s in new_seg {
            if !matches!(s, TaskRef::Job(_)) {
                return None;
            }
            let job = s.description(problem);
            let loc = job.location.index?;
            let edge = match cur_loc {
                Some(p) => self.dur(matrix, p, loc),
                None => 0,
            };
            cur = Tws::merge(cur, edge, Tws::node(step_tw(problem, s), job.service));
            cur_loc = Some(loc);
        }
        let last_loc = if k < self.n {
            let edge = match cur_loc {
                Some(p) => self.dur(matrix, p, self.locs[k]),
                None => 0,
            };
            cur = Tws::merge(cur, edge, self.suf[k]);
            Some(self.locs[self.n - 1])
        } else {
            cur_loc
        };

        // ── Load: uniform shifts on the prefix/suffix checkpoint maxima,
        // exact walk through the new segment (dim ≤ 1 by eligibility). ──
        let mut del_new = 0i64;
        let mut pick_new = 0i64;
        for &s in new_seg {
            let job = s.description(problem);
            del_new += job.delivery.first().copied().unwrap_or(0);
            pick_new += job.pickup.first().copied().unwrap_or(0);
        }
        let shift_pre = del_new - (self.del_pre[k] - self.del_pre[i]);
        let mut peak = self.pre_max[i] + shift_pre; // includes the new init
        if !new_seg.is_empty() {
            // Load entering the gap: L_{i-1} (or init) plus the prefix shift.
            let before = self.init - self.del_pre[i] + self.pick_pre[i];
            let mut l = before + shift_pre;
            for &s in new_seg {
                let job = s.description(problem);
                l -= job.delivery.first().copied().unwrap_or(0);
                l += job.pickup.first().copied().unwrap_or(0);
                peak = peak.max(l);
            }
        }
        if k < self.n {
            let shift_suf = pick_new - (self.pick_pre[k] - self.pick_pre[i]);
            let m = self.suf_max[k];
            if m != i64::MIN {
                peak = peak.max(m + shift_suf);
            }
        }
        let excess = (peak - self.cap).max(0);

        Some(self.tail_viol(matrix, cur, last_loc, excess))
    }

    /// Violations of the route with stops `[a, b]` REVERSED, where the caller
    /// maintains the reversed segment's stats incrementally (O(1) per
    /// extension in a 2-opt scan): `rev` is the Tws of `locs[b], …, locs[a]`
    /// and `rev_max_pref` the max over the reversed sequence's non-empty
    /// prefix sums of per-stop `pickup − delivery`. The load shifts are zero
    /// (same multiset), so only the in-segment peak changes.
    pub fn reversal_viol(
        &self,
        matrix: &Matrix,
        a: usize,
        b: usize,
        rev: Tws,
        rev_max_pref: i64,
    ) -> Viol {
        debug_assert!(a <= b && b < self.n);
        let mut cur = self.pre[a];
        let prev_loc = if a == 0 { self.start_idx } else { Some(self.locs[a - 1]) };
        let edge_in = match prev_loc {
            Some(p) => self.dur(matrix, p, self.locs[b]),
            None => 0,
        };
        cur = Tws::merge(cur, edge_in, rev);
        let last_loc = if b + 1 < self.n {
            cur = Tws::merge(cur, self.dur(matrix, self.locs[a], self.locs[b + 1]), self.suf[b + 1]);
            Some(self.locs[self.n - 1])
        } else {
            Some(self.locs[a])
        };
        let before = self.init - self.del_pre[a] + self.pick_pre[a];
        let mut peak = self.pre_max[a].max(before + rev_max_pref);
        if b + 1 < self.n {
            let m = self.suf_max[b + 1];
            if m != i64::MIN {
                peak = peak.max(m);
            }
        }
        self.tail_viol(matrix, cur, last_loc, (peak - self.cap).max(0))
    }

    /// `Tws::node` plus the per-stop load delta `pickup − delivery` for the
    /// caller-maintained reversal stats.
    pub fn stop_stats(&self, problem: &Problem, t: TaskRef) -> (Tws, i64) {
        let job = t.description(problem);
        let tw = job.time_windows.first().copied().unwrap_or(TimeWindow::FOREVER);
        let c = job.pickup.first().copied().unwrap_or(0) - job.delivery.first().copied().unwrap_or(0);
        (Tws::node(tw, job.service), c)
    }

    /// Penalty delta of replacing `[i, k)` with `new_seg`, relative to the
    /// route's current violations: `sw · (viol(candidate) − viol(base))`.
    /// `None` ⇒ structurally unjudgeable (fall back to the slow path).
    #[inline]
    pub fn penalty_delta(
        &self,
        problem: &Problem,
        matrix: &Matrix,
        sw: SoftWeights,
        i: usize,
        k: usize,
        new_seg: &[TaskRef],
    ) -> Option<f64> {
        let v = self.replace_seg_viol(problem, matrix, i, k, new_seg)?;
        Some(v.penalty(sw) - self.base.penalty(sw))
    }
}

/// One cache slot per route, index-aligned with `sol.routes` (the
/// `crate::slack::SlackCache` pattern).
enum Slot {
    Stale,
    Ready(Box<RouteWarp>),
    /// Structural build failure — don't retry until the route changes.
    Unusable,
}

/// Lazily-built per-route warp arrays for the solution a local-search core
/// loop is working on. MUST be kept index-aligned with `sol.routes`: call
/// [`WarpCache::on_apply`] with the move's route updates right before
/// `apply_move`.
pub struct WarpCache {
    slots: Vec<Slot>,
}

impl WarpCache {
    pub fn new(n_routes: usize) -> WarpCache {
        WarpCache { slots: (0..n_routes).map(|_| Slot::Stale).collect() }
    }

    /// Build (if needed) the warp arrays for route `r`.
    pub fn ensure(&mut self, r: usize, problem: &Problem, matrix: &Matrix, sol: &Solution) {
        debug_assert_eq!(self.slots.len(), sol.routes.len());
        if let Slot::Stale = self.slots[r] {
            let route = &sol.routes[r];
            let veh = &problem.vehicles[route.vehicle_idx];
            self.slots[r] = match RouteWarp::build(problem, matrix, veh, &route.steps) {
                Some(w) => Slot::Ready(Box::new(w)),
                None => Slot::Unusable,
            };
        }
    }

    /// The warp arrays for route `r`, if buildable. Call [`ensure`] first.
    pub fn get(&self, r: usize) -> Option<&RouteWarp> {
        match &self.slots[r] {
            Slot::Ready(w) => Some(w),
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
