//! Local search with don't-look bits.
//!
//! Each pass walks every "alive" task once. For a given task we look at all
//! moves involving it (relocate from its current slot, 2-opt across an edge
//! it touches, exchange against tasks in other routes) and apply the best
//! improving move we find. If no improving move exists the task is marked
//! `settled` and skipped on subsequent passes — until another move puts it
//! back into play. Convergence happens when every task is settled.
//!
//! This is dramatically cheaper than a full O(N²V) best-improvement sweep
//! because most tasks settle quickly and stop generating candidates.

use std::collections::HashSet;

use crate::granular::Granular;
use crate::matrix::Matrix;
use crate::problem::{Cost, Problem};
use crate::solution::{evaluate_route, Route, RouteMetrics, Solution, TaskRef};

const SEGMENT_LENS: [usize; 3] = [1, 2, 3];


// ── Fast O(1) cost-delta path ────────────────────────────────────────────────
//
// For the common objective (`span_cost == 0`, no custom dimensions, no soft
// penalties) a route's cost is
//   fixed + travel_time·(per_hour/3600·time_weight) + distance·distance_weight
//         + service_time·1e-6
// which depends ONLY on the arcs traversed — never on the time-window timing or
// waiting. So a move's COST delta is an O(1) edge calculation, and only its
// FEASIBILITY needs the full `evaluate_route` walk. We therefore enumerate
// candidates by exact O(1) cost-delta, then confirm feasibility lazily on the
// best-first — the first feasible candidate IS the best feasible move. This cuts
// `evaluate_route` calls from hundreds per task to a handful (validated for
// equivalence against the full evaluator in tests/incremental_ls.rs).

/// Per-vehicle cost coefficients for the arc-sum cost model.
#[derive(Clone, Copy)]
struct CostCoef {
    ct: f64, // cost per second of travel
    cd: f64, // cost per metre of distance
    speed: f64,
    fixed: Cost,
}

#[inline]
fn cost_coef(v: &crate::problem::Vehicle) -> CostCoef {
    CostCoef {
        ct: (v.per_hour / 3600.0).max(0.0) * v.time_weight,
        cd: v.distance_weight,
        speed: v.speed_factor.max(0.01),
        fixed: v.fixed,
    }
}

/// Cost of one arc a→b under a vehicle's coefficients (matches `evaluate_route`).
#[inline]
fn arc_cost(matrix: &Matrix, c: CostCoef, a: usize, b: usize) -> f64 {
    let dur = ((matrix.duration(a, b) as f64) * c.speed).round();
    dur * c.ct + (matrix.distance(a, b) as f64) * c.cd
}

/// Is the fast cost-delta model exact for this solve right now? It is whenever
/// the objective is the plain arc-based one: no span cost, no custom dimensions,
/// no soft-penalty mode, and the route isn't multi-trip (reloads insert depot
/// legs the simple arc walk doesn't model).
/// Runtime switch for the fast LS path. `cluster_decompose` flips this OFF around
/// its sub-solves so large-N (decomposed) results stay byte-identical to the
/// pre-fast-LS engine — the fast path gives no end-to-end benefit there (the cold
/// LS is a tiny fraction of the per-cluster budget) but would perturb tie-breaks.
/// Small-N flat solves (incl. HGS education) keep it on. Production runs one
/// top-level solve at a time, so the global flag is race-free there; in parallel
/// tests a race only toggles the (cost-identical) path, never correctness.
pub static FAST_LS_GLOBAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// Set the global fast-LS switch, returning the previous value (so callers can
/// restore it). Used by cluster_decompose to wrap its sub-solves.
pub fn set_fast_ls_global(v: bool) -> bool {
    FAST_LS_GLOBAL.swap(v, std::sync::atomic::Ordering::Relaxed)
}

/// Master switch for the fast LS path (off via BROOOM_NO_FAST_LS, for A/B).
fn fast_ls_enabled() -> bool {
    FAST_LS_GLOBAL.load(std::sync::atomic::Ordering::Relaxed)
        && std::env::var("BROOOM_NO_FAST_LS").is_err()
}

fn fast_cost_eligible(problem: &Problem) -> bool {
    if crate::dimension::has_dimensions() {
        return false;
    }
    // Soft penalties fold violation terms into the cost, which the arc-sum
    // model doesn't see — EXCEPT in warp mode, where crate::warp supplies the
    // penalty deltas exactly and the confirm evaluator agrees (SoftMode::Warp).
    if crate::solution::soft_is_active() && !crate::solution::warp_soft_active() {
        return false;
    }
    if problem.any_multi_trip() {
        return false;
    }
    if problem.vehicles.iter().any(|v| v.span_cost.max(0.0) != 0.0) {
        return false;
    }
    // Homogeneous COST coefficients: the cross-route fast operators cost a moved
    // segment's internal arcs once (they cancel between source and destination)
    // — exact only when every vehicle shares per_hour/time_weight/distance_weight/
    // speed_factor/fixed. Heterogeneous fleets fall back to full evaluation.
    homogeneous_cost(problem)
}

/// All vehicles share the cost coefficients that drive the arc-sum model.
fn homogeneous_cost(problem: &Problem) -> bool {
    let mut it = problem.vehicles.iter();
    let Some(v0) = it.next() else { return true };
    let c0 = cost_coef(v0);
    it.all(|v| {
        let c = cost_coef(v);
        c.ct == c0.ct && c.cd == c0.cd && c.speed == c0.speed && c.fixed == c0.fixed
    })
}

/// Matrix index of a step's location.
#[inline]
fn step_loc(problem: &Problem, t: TaskRef) -> Option<usize> {
    t.description(problem).location.index
}

/// Per-route candidate judge for the fast operators.
///
/// `Slack` (hard mode): O(1) FEASIBILITY verdicts — candidates the arrays
/// reject are provably infeasible and skipped; rank == exact arc delta.
///
/// `Warp` (SoftMode::Warp only): O(seg) PENALTY deltas — there is nothing to
/// reject (soft accepts any TW/load violation), instead the move's penalty
/// change is folded into the candidate delta at enumeration time so the rank
/// equals the true penalised delta and repair moves (penalty down, arc cost
/// up) become visible. The winner is still confirmed by `evaluate_route`,
/// whose warp-mode accumulators agree exactly with these deltas.
///
/// Operators not yet warp-converted treat `Warp` as "no judge": they rank by
/// arc delta only (blind to penalties) — sound, because every confirm
/// computes the real delta from penalised metrics and `consider` filters,
/// but they can miss repair moves and may break early on a false first
/// winner. Convert them when measurements say they matter.
pub(crate) enum Judge {
    Slack(crate::slack::SlackCache),
    Warp(crate::warp::WarpCache, crate::solution::SoftWeights),
}

impl Judge {
    /// Arm the right judge for this problem/mode, or `None` when neither
    /// math is exact. Mirrors the old slack_cache arming.
    fn arm(problem: &Problem, n_routes: usize, granular_on: bool) -> Option<Judge> {
        if !(granular_on && fast_ls_enabled() && fast_cost_eligible(problem)) {
            return None;
        }
        if crate::solution::warp_soft_active() {
            crate::warp::warp_eligible(problem).then(|| {
                Judge::Warp(crate::warp::WarpCache::new(n_routes), crate::solution::soft_weights())
            })
        } else {
            crate::slack::slack_eligible(problem)
                .then(|| Judge::Slack(crate::slack::SlackCache::new(n_routes)))
        }
    }

    /// Mirror an applied move into whichever cache is armed.
    fn on_apply(&mut self, route_updates: &[(usize, Option<(Vec<TaskRef>, RouteMetrics)>)]) {
        match self {
            Judge::Slack(c) => c.on_apply(route_updates),
            Judge::Warp(c, _) => c.on_apply(route_updates),
        }
    }
}

// ── env-gated per-operator statistics (BROOOM_LS_STATS=1) ────────────────────
// One slot per operator in the probe order of the core loop. `probes` counts
// invocations, `nanos` wall time inside the operator (including its confirm
// evaluations), `confirm_evals` the evaluate_route calls made by the fast
// confirm loops. All increments are guarded by `ls_stats_on()` so the default
// path pays nothing beyond a cached-bool read.
const OP_NAMES: [&str; 6] = ["reloc", "2opt", "exch", "2opt*", "xchg2", "swap*"];
const OP_RELOC: usize = 0;
const OP_TWO_OPT: usize = 1;
const OP_EXCH: usize = 2;
const OP_TWO_OPT_STAR: usize = 3;
const OP_CROSS: usize = 4;
const OP_SWAP_STAR: usize = 5;
struct OpStats {
    probes: std::sync::atomic::AtomicU64,
    nanos: std::sync::atomic::AtomicU64,
    confirm_evals: std::sync::atomic::AtomicU64,
    slack_skips: std::sync::atomic::AtomicU64,
}
#[allow(clippy::declare_interior_mutable_const)]
const OP_STATS_INIT: OpStats = OpStats {
    probes: std::sync::atomic::AtomicU64::new(0),
    nanos: std::sync::atomic::AtomicU64::new(0),
    confirm_evals: std::sync::atomic::AtomicU64::new(0),
    slack_skips: std::sync::atomic::AtomicU64::new(0),
};
static OP_STATS: [OpStats; 6] = [OP_STATS_INIT; 6];

#[inline]
fn op_confirm_eval(op: usize) {
    if crate::solution::ls_stats_on() {
        OP_STATS[op].confirm_evals.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[inline]
fn op_slack_skip(op: usize) {
    if crate::solution::ls_stats_on() {
        OP_STATS[op].slack_skips.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Run one operator probe, timing it when stats are on.
#[inline]
fn op_timed(op: usize, stats: bool, f: impl FnOnce()) {
    if stats {
        let t0 = std::time::Instant::now();
        f();
        OP_STATS[op].probes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        OP_STATS[op]
            .nanos
            .fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    } else {
        f();
    }
}

/// Render and reset the global LS statistics (operators + evaluate_route).
/// Returns an empty string when BROOOM_LS_STATS is off.
pub fn ls_stats_snapshot_and_reset() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    if !crate::solution::ls_stats_on() {
        return String::new();
    }
    let calls = crate::solution::EVAL_CALLS.swap(0, Relaxed);
    let hits = crate::solution::EVAL_CACHE_HITS.swap(0, Relaxed);
    let mut out = format!("LS-STATS: eval_calls={} cache_hits={}", calls, hits);
    for (k, name) in OP_NAMES.iter().enumerate() {
        let p = OP_STATS[k].probes.swap(0, Relaxed);
        let ns = OP_STATS[k].nanos.swap(0, Relaxed);
        let ce = OP_STATS[k].confirm_evals.swap(0, Relaxed);
        let sk = OP_STATS[k].slack_skips.swap(0, Relaxed);
        out.push_str(&format!(
            " | {} probes={} t={:.2}s evals={} skips={}",
            name,
            p,
            ns as f64 / 1e9,
            ce,
            sk
        ));
    }
    out
}

// Thread-local epoch-stamped membership buffer for granular neighbours. Replaces
// a per-call `HashSet` (alloc + K hashes) in the hot LS operators with O(1)
// array writes and zero allocation after warmup. Each operator stamps the
// neighbours of its anchor once, then tests membership in its candidate loop.
thread_local! {
    static NMARK: std::cell::RefCell<(u32, Vec<u32>)> = const { std::cell::RefCell::new((0, Vec::new())) };
}
/// Stamp the granular neighbours of `loc` into the thread-local buffer.
fn nmark_set(g: &Granular, loc: usize, n_locs: usize) {
    NMARK.with(|cell| {
        let mut b = cell.borrow_mut();
        if b.1.len() < n_locs {
            b.1.resize(n_locs, 0);
        }
        b.0 = b.0.wrapping_add(1);
        if b.0 == 0 {
            for m in b.1.iter_mut() { *m = 0; }
            b.0 = 1;
        }
        let epoch = b.0;
        for nb in g.neighbors(loc) {
            if nb < b.1.len() { b.1[nb] = epoch; }
        }
    });
}
/// Test membership against the most recent `nmark_set`.
#[inline]
fn nmark_has(l: usize) -> bool {
    NMARK.with(|cell| {
        let b = cell.borrow();
        b.1.get(l).copied() == Some(b.0)
    })
}

// Thread-local wall-clock deadline for local search. Construction-time LS runs
// to convergence, which at n≥400 takes whole seconds per variant — unchecked,
// the multi-start construction alone blows past the user's time limit (measured
// 1.1–2× overshoot at G&H 400 @ -l 10) and starves the HGS phase that follows.
// A deadline here caps every pass loop; an interrupted solution is merely
// unconverged, never invalid (moves apply atomically). `None` (the default, and
// always the case without a time limit) changes nothing.
thread_local! {
    static LS_DEADLINE: std::cell::Cell<Option<std::time::Instant>> =
        const { std::cell::Cell::new(None) };
}
/// Arm (or clear) the wall-clock deadline LS honours on this thread. Callers
/// that run on pooled threads (rayon) must set it on entry — workers are reused
/// and may carry a previous phase's value.
pub fn set_ls_deadline(d: Option<std::time::Instant>) {
    LS_DEADLINE.with(|c| c.set(d));
}
/// True once the armed deadline (if any) has passed. Public so other
/// construction-time loops (greedy insertion) can honour the same wall-clock.
#[inline]
pub fn ls_deadline_hit() -> bool {
    LS_DEADLINE.with(|c| c.get().is_some_and(|d| std::time::Instant::now() >= d))
}

/// The depot index a vehicle starts/ends at (for arc endpoints).
#[inline]
fn depot_start(v: &crate::problem::Vehicle) -> Option<usize> {
    v.start.as_ref().and_then(|l| l.index).or_else(|| v.end.as_ref().and_then(|l| l.index))
}
#[inline]
fn depot_end(v: &crate::problem::Vehicle) -> Option<usize> {
    v.end.as_ref().and_then(|l| l.index).or_else(|| v.start.as_ref().and_then(|l| l.index))
}

/// One concrete move ready to apply. `route_updates` carries the new step
/// list and metrics for each route that changes; `None` payload means the
/// route ended up empty and should be removed.
#[derive(Clone)]
struct Move {
    delta: Cost,
    route_updates: Vec<(usize, Option<(Vec<TaskRef>, RouteMetrics)>)>,
    /// Tasks whose involvement should clear their `settled` flag.
    touched: Vec<TaskRef>,
}

/// Run local search until every task is settled or `max_passes` is hit.
///
/// `granular_k` caps how many nearest-neighbor target positions we consider
/// per move. Pass `None` to disable granularity (vanilla LS — slower).
pub fn local_search(
    problem: &Problem,
    matrix: &Matrix,
    sol: &mut Solution,
    max_passes: usize,
    granular: Option<&Granular>,
) {
    local_search_with_settled(problem, matrix, sol, max_passes, granular, HashSet::new());
}

/// Local search that starts with the don't-look bits pre-populated. Used by the
/// ILS loop to re-converge after a small perturbation: the working solution was
/// already LS-converged, so every task in an UNTOUCHED route still carries a
/// valid "no improving move" verdict under the don't-look-bit semantics
/// (verdicts are only invalidated when the task's own route changes — exactly
/// the rule the move-application below uses). Seeding those verdicts skips
/// re-probing the whole solution and makes one ILS iteration cost
/// O(perturbed region), not O(N) — the trick PyVRP's ILS uses to run orders of
/// magnitude more iterations in the same budget.
pub fn local_search_seeded(
    problem: &Problem,
    matrix: &Matrix,
    sol: &mut Solution,
    max_passes: usize,
    granular: Option<&Granular>,
    unsettled: &HashSet<TaskRef>,
) {
    let settled: HashSet<TaskRef> = sol
        .routes
        .iter()
        .flat_map(|r| r.steps.iter().copied())
        .filter(|t| !unsettled.contains(t))
        .collect();
    local_search_with_settled(problem, matrix, sol, max_passes, granular, settled);
}

fn local_search_with_settled(
    problem: &Problem,
    matrix: &Matrix,
    sol: &mut Solution,
    max_passes: usize,
    granular: Option<&Granular>,
    mut settled: HashSet<TaskRef>,
) {
    // O(1) candidate judge for the fast operators (see slack.rs / warp.rs).
    // Only armed when both the fast cost model and the per-mode math are
    // exact; the cache stays index-aligned with sol.routes via on_apply below.
    let mut judge = Judge::arm(problem, sol.routes.len(), granular.is_some());

    'outer: for _ in 0..max_passes {
        if ls_deadline_hit() { break 'outer; }
        let task_seq: Vec<TaskRef> = sol.routes.iter().flat_map(|r| r.steps.iter().copied()).collect();

        let mut any_change = false;

        for (ti, task) in task_seq.into_iter().enumerate() {
            // A single pass at n≥400 costs hundreds of ms — too coarse for a
            // 10 s budget — so also poll the deadline inside the pass.
            if ti % 64 == 0 && ls_deadline_hit() { break 'outer; }
            if settled.contains(&task) { continue; }

            let Some((r1, i)) = locate(sol, task) else { continue; };

            let mut best: Option<Move> = None;

            let stats = crate::solution::ls_stats_on();
            let use_slack = matches!(&judge, Some(Judge::Slack(_)));
            let warp_mode = matches!(&judge, Some(Judge::Warp(..)));
            let sc = &mut judge;
            op_timed(OP_RELOC, stats, || try_relocate_task(problem, matrix, sol, r1, i, granular, sc.as_mut(), &mut best));
            op_timed(OP_TWO_OPT, stats, || try_two_opt_through(problem, matrix, sol, r1, i, sc.as_mut(), &mut best));
            op_timed(OP_EXCH, stats, || try_exchange_with(problem, matrix, sol, r1, i, granular, sc.as_mut(), &mut best));
            op_timed(OP_TWO_OPT_STAR, stats, || try_two_opt_star(problem, matrix, sol, r1, i, granular, sc.as_mut(), &mut best));
            op_timed(OP_CROSS, stats, || try_cross_exchange_with(problem, matrix, sol, r1, i, granular, sc.as_mut(), &mut best));
            // swap* has no warp conversion yet: its slack arrays are
            // meaningless on violating routes and its eval fallback is the
            // per-pair full-evaluation bomb — skip it under warp mode.
            if !warp_mode {
                op_timed(OP_SWAP_STAR, stats, || try_swap_star(problem, matrix, sol, r1, i, granular, use_slack, &mut best));
            }

            match best {
                Some(mv) if mv.delta < -1e-9 => {
                    // Clear settled for every task in the affected routes —
                    // their context just changed, so previous "no improvement"
                    // verdicts may be stale. The cost is a few extra probes
                    // per move, well worth the better convergence.
                    for (route_idx, payload) in &mv.route_updates {
                        if let Some((steps, _)) = payload {
                            for t in steps { settled.remove(t); }
                        } else if let Some(r) = sol.routes.get(*route_idx) {
                            for t in &r.steps { settled.remove(t); }
                        }
                    }
                    for t in &mv.touched { settled.remove(t); }
                    if let Some(j) = judge.as_mut() {
                        j.on_apply(&mv.route_updates);
                    }
                    apply_move(sol, mv);
                    any_change = true;
                }
                _ => {
                    settled.insert(task);
                }
            }
        }

        if !any_change { break 'outer; }
    }

    sol.routes.retain(|r| !r.steps.is_empty());
    sol.recompute_summary(problem);
}

/// Full LS pass without don't-look bits. Every task is reconsidered every
/// pass — slower (O(N) probes per pass) but rediscovers moves that
/// don't-look has prematurely settled. Run as a finishing pass after
/// `local_search` has converged. Vroom equivalent: their main loop is
/// effectively this (best-improvement, no don't-look bits).
pub fn local_search_full(
    problem: &Problem,
    matrix: &Matrix,
    sol: &mut Solution,
    max_passes: usize,
    granular: Option<&Granular>,
) {
    let mut judge = Judge::arm(problem, sol.routes.len(), granular.is_some());

    'outer: for _ in 0..max_passes {
        if ls_deadline_hit() { break 'outer; }
        let task_seq: Vec<TaskRef> = sol.routes.iter().flat_map(|r| r.steps.iter().copied()).collect();
        let mut any_change = false;

        for (ti, task) in task_seq.into_iter().enumerate() {
            if ti % 64 == 0 && ls_deadline_hit() { break 'outer; }
            let Some((r1, i)) = locate(sol, task) else { continue; };

            let mut best: Option<Move> = None;
            let stats = crate::solution::ls_stats_on();
            let use_slack = matches!(&judge, Some(Judge::Slack(_)));
            let warp_mode = matches!(&judge, Some(Judge::Warp(..)));
            let sc = &mut judge;
            op_timed(OP_RELOC, stats, || try_relocate_task(problem, matrix, sol, r1, i, granular, sc.as_mut(), &mut best));
            op_timed(OP_TWO_OPT, stats, || try_two_opt_through(problem, matrix, sol, r1, i, sc.as_mut(), &mut best));
            op_timed(OP_EXCH, stats, || try_exchange_with(problem, matrix, sol, r1, i, granular, sc.as_mut(), &mut best));
            op_timed(OP_TWO_OPT_STAR, stats, || try_two_opt_star(problem, matrix, sol, r1, i, granular, sc.as_mut(), &mut best));
            op_timed(OP_CROSS, stats, || try_cross_exchange_with(problem, matrix, sol, r1, i, granular, sc.as_mut(), &mut best));
            // See local_search_with_settled: swap* is skipped in warp mode.
            if !warp_mode {
                op_timed(OP_SWAP_STAR, stats, || try_swap_star(problem, matrix, sol, r1, i, granular, use_slack, &mut best));
            }

            if let Some(mv) = best {
                if mv.delta < -1e-9 {
                    if let Some(j) = judge.as_mut() {
                        j.on_apply(&mv.route_updates);
                    }
                    apply_move(sol, mv);
                    any_change = true;
                }
            }
        }

        if !any_change { break 'outer; }
    }

    sol.routes.retain(|r| !r.steps.is_empty());
    sol.recompute_summary(problem);
}

fn locate(sol: &Solution, task: TaskRef) -> Option<(usize, usize)> {
    for (r, route) in sol.routes.iter().enumerate() {
        if let Some(i) = route.steps.iter().position(|t| *t == task) {
            return Some((r, i));
        }
    }
    None
}

fn apply_move(sol: &mut Solution, mv: Move) {
    let mut updates = mv.route_updates;
    updates.sort_by(|a, b| b.0.cmp(&a.0));
    for (idx, payload) in updates {
        match payload {
            Some((steps, metrics)) => {
                sol.routes[idx].steps = steps;
                sol.routes[idx].metrics = metrics;
            }
            None => {
                sol.routes.remove(idx);
            }
        }
    }
}

fn consider(best: &mut Option<Move>, candidate: Move) {
    if candidate.delta >= -1e-9 { return; }
    if best.as_ref().map_or(true, |b| candidate.delta < b.delta) {
        *best = Some(candidate);
    }
}

/// Relocate around (r1, i). With granularity, only the K nearest matrix
/// neighbors of the moved segment's first stop are valid insertion targets;
/// the bordering position before/after each such neighbor is what we try.
fn try_relocate_task(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    granular: Option<&Granular>,
    judge: Option<&mut Judge>,
    best: &mut Option<Move>,
) {
    // Fast O(1) cost-delta path: enumerate candidates by exact edge-delta cost,
    // then confirm feasibility lazily best-first (the first feasible candidate is
    // the best feasible move). Only when the arc-cost model is exact.
    if granular.is_some() && fast_ls_enabled() && fast_cost_eligible(problem) {
        try_relocate_task_fast(problem, matrix, sol, r1, i, granular.unwrap(), judge, best);
        return;
    }
    try_relocate_task_slow(problem, matrix, sol, r1, i, granular, best);
}

fn try_relocate_task_slow(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    granular: Option<&Granular>,
    best: &mut Option<Move>,
) {
    let route1 = &sol.routes[r1];
    let n1 = route1.steps.len();
    for &seg_len in &SEGMENT_LENS {
        if i + seg_len > n1 { continue; }
        let segment: Vec<TaskRef> = route1.steps[i..i + seg_len].to_vec();
        let mut route1_after: Vec<TaskRef> = route1.steps.clone();
        route1_after.drain(i..i + seg_len);

        let veh1 = &problem.vehicles[route1.vehicle_idx];
        let metrics1_after = if route1_after.is_empty() {
            RouteMetrics::default()
        } else {
            match evaluate_route(problem, matrix, veh1, &route1_after) {
                Ok(m) => m, Err(_) => continue,
            }
        };
        let cost1_delta = metrics1_after.cost - route1.metrics.cost;

        // Where in matrix-space does this segment live? Use the first stop
        // as the anchor for granularity lookups.
        let seg_loc = segment[0].description(problem).location.index;

        // Collect the set of route-positions (r2, j) we want to probe. With
        // granularity we limit j to slots that are adjacent to a top-K
        // matrix neighbor of `seg_loc`; without, we probe every slot.
        for r2 in 0..sol.routes.len() {
            let route2 = &sol.routes[r2];
            let veh2 = &problem.vehicles[route2.vehicle_idx];
            if !segment.iter().all(|t| veh2.has_skills(t.skills(problem))) { continue; }

            let base = if r1 == r2 { &route1_after } else { &route2.steps };
            let nb = base.len();

            // Build the set of insertion positions to try.
            let positions: Vec<usize> = match (granular, seg_loc) {
                (Some(g), Some(loc)) => {
                    let neighbor_locs: HashSet<usize> = g.neighbors(loc).collect();
                    let mut out: Vec<usize> = Vec::new();
                    for j in 0..=nb {
                        // A slot is "near" if it's adjacent to a step whose
                        // location is one of our K nearest neighbors.
                        let prev_loc = if j == 0 { None } else {
                            base[j - 1].description(problem).location.index
                        };
                        let next_loc = if j == nb { None } else {
                            base[j].description(problem).location.index
                        };
                        if prev_loc.map_or(false, |l| neighbor_locs.contains(&l))
                            || next_loc.map_or(false, |l| neighbor_locs.contains(&l))
                        {
                            out.push(j);
                        }
                    }
                    // Always try the very-end and very-start slots so a
                    // segment can leave a saturated neighborhood entirely.
                    if !out.contains(&0) { out.push(0); }
                    if !out.contains(&nb) { out.push(nb); }
                    out
                }
                _ => (0..=nb).collect(),
            };

            // Or-opt with reversal: also try inserting the segment reversed
            // (only meaningful for length ≥ 2). Often unlocks an
            // improvement when the natural orientation breaks downstream
            // TWs but the reversed one fits.
            let segment_rev: Option<Vec<TaskRef>> = if seg_len > 1 {
                let mut r = segment.clone();
                r.reverse();
                Some(r)
            } else { None };

            for j in positions {
                if r1 == r2 && j == i { continue; }
                for variant in [Some(segment.as_slice()), segment_rev.as_deref()] {
                    let Some(seg) = variant else { continue; };
                    let mut cand = Vec::with_capacity(base.len() + seg_len);
                    cand.extend_from_slice(&base[..j]);
                    cand.extend_from_slice(seg);
                    cand.extend_from_slice(&base[j..]);

                    let metrics2_after = match evaluate_route(problem, matrix, veh2, &cand) {
                        Ok(m) => m, Err(_) => continue,
                    };

                    if r1 == r2 {
                        let total_delta = metrics2_after.cost - route1.metrics.cost;
                        consider(best, Move {
                            delta: total_delta,
                            route_updates: vec![(r1, Some((cand, metrics2_after)))],
                            touched: segment.clone(),
                        });
                    } else {
                        let cost2_delta = metrics2_after.cost - route2.metrics.cost;
                        let total_delta = cost1_delta + cost2_delta;
                        let upd1 = if route1_after.is_empty() {
                            (r1, None)
                        } else {
                            (r1, Some((route1_after.clone(), metrics1_after)))
                        };
                        let upd2 = (r2, Some((cand, metrics2_after)));
                        consider(best, Move {
                            delta: total_delta,
                            route_updates: vec![upd1, upd2],
                            touched: segment.clone(),
                        });
                    }
                }
            }
        }
    }
}

/// Full arc-sum cost of a route under the simple objective: fixed + Σ arc costs
/// (incl. both depot legs) + Σ service·1e-6. Exact iff `fast_cost_eligible`.
fn route_arc_cost(problem: &Problem, matrix: &Matrix, veh: &crate::problem::Vehicle, steps: &[TaskRef]) -> f64 {
    if steps.is_empty() {
        return 0.0;
    }
    let c = cost_coef(veh);
    let mut sum = c.fixed;
    let mut prev = depot_start(veh);
    for &s in steps {
        let here = match step_loc(problem, s) {
            Some(h) => h,
            None => continue,
        };
        if let Some(p) = prev {
            sum += arc_cost(matrix, c, p, here);
        }
        prev = Some(here);
        sum += (s.description(problem).service as f64) * 1e-6;
    }
    if let (Some(p), Some(e)) = (prev, depot_end(veh)) {
        sum += arc_cost(matrix, c, p, e);
    }
    sum
}

/// Cheapest insertion cost-delta of `task` into `base`, IGNORING feasibility
/// (edge math only, O(len)). A lower bound on the true cheapest *feasible*
/// insertion, used to prune hopeless swap* pairs before any full evaluation.
fn pred_cheapest_insert(
    problem: &Problem, matrix: &Matrix, veh: &crate::problem::Vehicle,
    base: &[TaskRef], task: TaskRef,
) -> f64 {
    let Some(tloc) = step_loc(problem, task) else { return f64::INFINITY };
    let c = cost_coef(veh);
    let tserv = task.description(problem).service as f64 * 1e-6;
    let nb = base.len();
    let ds = depot_start(veh);
    let de = depot_end(veh);
    let mut best = f64::INFINITY;
    for pos in 0..=nb {
        let prev = if pos == 0 { ds } else { step_loc(problem, base[pos - 1]) };
        let next = if pos == nb { de } else { step_loc(problem, base[pos]) };
        let added = prev.map_or(0.0, |p| arc_cost(matrix, c, p, tloc))
            + next.map_or(0.0, |n| arc_cost(matrix, c, tloc, n));
        let removed = match (prev, next) {
            (Some(p), Some(n)) => arc_cost(matrix, c, p, n),
            _ => 0.0,
        };
        best = best.min(added - removed + tserv);
    }
    best
}

/// Cost-delta of REMOVING the task at `pos` from `steps` (edge math, O(1)):
/// cost(route_without) − cost(route). Exact in the simple envelope.
fn pred_remove_delta(
    problem: &Problem, matrix: &Matrix, veh: &crate::problem::Vehicle,
    steps: &[TaskRef], pos: usize,
) -> f64 {
    let c = cost_coef(veh);
    let n = steps.len();
    let tloc = match step_loc(problem, steps[pos]) { Some(l) => l, None => return 0.0 };
    let prev = if pos == 0 { depot_start(veh) } else { step_loc(problem, steps[pos - 1]) };
    let next = if pos + 1 >= n { depot_end(veh) } else { step_loc(problem, steps[pos + 1]) };
    let removed = prev.map_or(0.0, |p| arc_cost(matrix, c, p, tloc))
        + next.map_or(0.0, |nx| arc_cost(matrix, c, tloc, nx));
    let added = match (prev, next) {
        (Some(p), Some(nx)) => arc_cost(matrix, c, p, nx),
        _ => 0.0,
    };
    let tserv = steps[pos].description(problem).service as f64 * 1e-6;
    added - removed - tserv
}

/// Fast relocate: O(1) edge-delta cost enumeration + lazy feasibility confirm.
/// Equivalent to `try_relocate_task_slow` on `fast_cost_eligible` problems
/// (verified in tests/incremental_ls.rs), but evaluates the full route only on
/// the best-by-cost candidates instead of every one.
fn try_relocate_task_fast(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    granular: &Granular,
    judge: Option<&mut Judge>,
    best: &mut Option<Move>,
) {
    let route1 = &sol.routes[r1];
    let n1 = route1.steps.len();
    let veh1 = &problem.vehicles[route1.vehicle_idx];
    let c1 = cost_coef(veh1);

    // Judge for route1: slack gives an O(1) removal feasibility check (the
    // route1-after-removal evaluation is deferred to the confirm loop); warp
    // gives the removal's exact penalty term folded into delta_r1.
    let mut judge = judge;
    let mut have_s1 = false;
    let mut warp_sw: Option<crate::solution::SoftWeights> = None;
    match judge.as_deref_mut() {
        Some(Judge::Slack(cache)) => {
            cache.ensure(r1, problem, matrix, sol);
            have_s1 = cache.get(r1).is_some();
        }
        Some(Judge::Warp(cache, sw)) => {
            cache.ensure(r1, problem, matrix, sol);
            warp_sw = Some(*sw);
        }
        None => {}
    }

    struct Cand {
        delta: f64,
        seg_len: usize,
        r2: usize,
        j: usize,
        rev: bool,
    }
    let mut cands: Vec<Cand> = Vec::new();
    // Per seg_len: the route1-after-removal step list and its cost delta.
    let mut r1_after_cache: Vec<Option<(Vec<TaskRef>, f64)>> = vec![None; 4];
    // Arc-sum cost of route1, walked at most ONCE per probe (was: a full
    // route1_after walk per seg_len); each seg_len derives its removal delta
    // with O(1) arc math on top.
    let mut arc_base1: Option<f64> = None;
    // Insertion-position scratch, reused across the (seg_len, r2) loops.
    let mut positions: Vec<usize> = Vec::new();
    // Per seg_len, the metrics of route1-after-removal: None = not evaluated
    // yet (deferred until a candidate of this seg_len survives the slack
    // check), Some(None) = removal infeasible, Some(Some(m)) = metrics.
    let mut m1_eval: [Option<Option<RouteMetrics>>; 4] = [None, None, None, None];
    // Warp mode: transient warp arrays for the intra-route base (route1 minus
    // the segment), built lazily per seg_len AT ENUMERATION time — intra
    // candidates need the combined route's ABSOLUTE penalty in their rank.
    let mut intra_warp: [Option<Option<Box<crate::warp::RouteWarp>>>; 4] =
        [None, None, None, None];
    // Warp mode: reversed segment per seg_len (the enumeration otherwise
    // never materialises it).
    let mut seg_rev_v: Vec<TaskRef> = Vec::new();

    for &seg_len in &SEGMENT_LENS {
        if i + seg_len > n1 {
            continue;
        }
        let segment: &[TaskRef] = &route1.steps[i..i + seg_len];
        if segment.iter().any(|t| step_loc(problem, *t).is_none()) {
            continue;
        }
        let seg0 = step_loc(problem, segment[0]).unwrap();
        let seg_last = step_loc(problem, segment[seg_len - 1]).unwrap();
        let seg_service: f64 =
            segment.iter().map(|t| t.description(problem).service as f64 * 1e-6).sum();
        // Internal arc cost of the segment, forward and reversed.
        let mut internal_fwd = 0.0;
        let mut internal_rev = 0.0;
        for w in 0..seg_len.saturating_sub(1) {
            let a = step_loc(problem, segment[w]).unwrap();
            let b = step_loc(problem, segment[w + 1]).unwrap();
            internal_fwd += arc_cost(matrix, c1, a, b);
            internal_rev += arc_cost(matrix, c1, b, a);
        }

        // route1 after removing the segment (used for r1==r2 base + the Move).
        let mut route1_after: Vec<TaskRef> = Vec::with_capacity(n1 - seg_len);
        route1_after.extend_from_slice(&route1.steps[..i]);
        route1_after.extend_from_slice(&route1.steps[i + seg_len..]);
        if have_s1 {
            // O(1) removal check; route1-after metrics are evaluated lazily in
            // the confirm loop, only when a candidate of this seg_len survives.
            let Some(Judge::Slack(cache)) = judge.as_deref() else { unreachable!() };
            let s1 = cache.get(r1).unwrap();
            if !s1.replace_seg_ok(problem, matrix, veh1, i, i + seg_len, &[]) {
                op_slack_skip(OP_RELOC);
                continue; // removal provably infeasible — skip seg_len
            }
        } else if warp_sw.is_none() {
            let m1_after = if route1_after.is_empty() {
                Some(RouteMetrics::default())
            } else {
                op_confirm_eval(OP_RELOC);
                evaluate_route(problem, matrix, veh1, &route1_after).ok()
            };
            if m1_after.is_none() {
                m1_eval[seg_len] = Some(None);
                continue; // removal infeasible (e.g. split a shipment) — skip seg_len
            }
            m1_eval[seg_len] = Some(m1_after);
        }
        // Warp mode: removal never fails — instead its penalty term enters the
        // rank. ABSOLUTE penalty of route1-after (route1.metrics.cost below
        // already carries the CURRENT penalty, so subtracting it and adding
        // the candidate's absolute penalty yields the true penalised delta).
        let pen_r1_after: f64 = match (warp_sw, judge.as_deref()) {
            (Some(sw), Some(Judge::Warp(cache, _))) if !route1_after.is_empty() => cache
                .get(r1)
                .and_then(|w1| w1.replace_seg_viol(problem, matrix, i, i + seg_len, &[]))
                .map_or(0.0, |v| v.penalty(sw)),
            _ => 0.0,
        };
        let delta_r1 = if route1_after.is_empty() {
            0.0 - route1.metrics.cost
        } else {
            let base = *arc_base1
                .get_or_insert_with(|| route_arc_cost(problem, matrix, veh1, &route1.steps));
            let prev = if i == 0 { depot_start(veh1) } else { step_loc(problem, route1.steps[i - 1]) };
            let next = if i + seg_len == n1 {
                depot_end(veh1)
            } else {
                step_loc(problem, route1.steps[i + seg_len])
            };
            let removed = prev.map_or(0.0, |p| arc_cost(matrix, c1, p, seg0))
                + internal_fwd
                + next.map_or(0.0, |nx| arc_cost(matrix, c1, seg_last, nx));
            let added = match (prev, next) {
                (Some(p), Some(nx)) => arc_cost(matrix, c1, p, nx),
                _ => 0.0,
            };
            base + added - removed - seg_service - route1.metrics.cost
        };
        r1_after_cache[seg_len] = Some((route1_after, delta_r1));
        let route1_after: &[TaskRef] = &r1_after_cache[seg_len].as_ref().unwrap().0;
        if warp_sw.is_some() {
            seg_rev_v.clear();
            seg_rev_v.extend(segment.iter().rev().copied());
        }

        nmark_set(granular, seg0, matrix.n);

        for r2 in 0..sol.routes.len() {
            let route2 = &sol.routes[r2];
            let veh2 = &problem.vehicles[route2.vehicle_idx];
            if !segment.iter().all(|t| veh2.has_skills(t.skills(problem))) {
                continue;
            }
            let c2 = cost_coef(veh2);
            let base: &[TaskRef] = if r1 == r2 { &route1_after } else { &route2.steps };
            let nb = base.len();
            let ds2 = depot_start(veh2);
            let de2 = depot_end(veh2);

            // Warp mode: penalty source for this destination — the combined
            // intra base (absolute penalty) or route2 (relative delta).
            let intra_w: Option<&crate::warp::RouteWarp> = if warp_sw.is_some() && r1 == r2 {
                intra_warp[seg_len]
                    .get_or_insert_with(|| {
                        crate::warp::RouteWarp::build(problem, matrix, veh1, route1_after)
                            .map(Box::new)
                    })
                    .as_deref()
            } else {
                None
            };
            if r1 != r2 {
                if let Some(Judge::Warp(cache, _)) = judge.as_deref_mut() {
                    cache.ensure(r2, problem, matrix, sol);
                }
            }
            let warp2: Option<&crate::warp::RouteWarp> = match judge.as_deref() {
                Some(Judge::Warp(cache, _)) if r1 != r2 => cache.get(r2),
                _ => None,
            };

            // Candidate positions: granular (adjacent to a top-K neighbour of
            // seg0) plus the two extremes — mirrors the slow path.
            positions.clear();
            for j in 0..=nb {
                let prev_loc = if j == 0 { None } else { step_loc(problem, base[j - 1]) };
                let next_loc = if j == nb { None } else { step_loc(problem, base[j]) };
                if prev_loc.map_or(false, nmark_has) || next_loc.map_or(false, nmark_has) {
                    positions.push(j);
                }
            }
            if !positions.contains(&0) {
                positions.push(0);
            }
            if !positions.contains(&nb) {
                positions.push(nb);
            }

            for &j in &positions {
                if r1 == r2 && j == i {
                    // (no-op slot in the ORIGINAL route; harmless here since base
                    // is route1_after, but keep parity with the slow guard)
                }
                let pjprev = if j == 0 { ds2 } else { step_loc(problem, base[j - 1]) };
                let pjnext = if j == nb { de2 } else { step_loc(problem, base[j]) };
                // Removed arc at the insertion gap.
                let removed_gap = match (pjprev, pjnext) {
                    (Some(p), Some(n)) => arc_cost(matrix, c2, p, n),
                    _ => 0.0,
                };
                for rev in [false, true] {
                    if rev && seg_len < 2 {
                        continue;
                    }
                    let (ins0, ins_last, internal) = if rev {
                        (seg_last, seg0, internal_rev)
                    } else {
                        (seg0, seg_last, internal_fwd)
                    };
                    let added = match pjprev {
                        Some(p) => arc_cost(matrix, c2, p, ins0),
                        None => 0.0,
                    } + internal
                        + match pjnext {
                            Some(n) => arc_cost(matrix, c2, ins_last, n),
                            None => 0.0,
                        };
                    let delta_r2 = added - removed_gap + seg_service;
                    let mut delta = delta_r1 + delta_r2;
                    if let Some(sw) = warp_sw {
                        let seg_or: &[TaskRef] = if rev { &seg_rev_v } else { segment };
                        if r1 == r2 {
                            // Absolute penalty of the combined candidate route
                            // (delta_r1 already subtracted the penalised cost).
                            if let Some(v) = intra_w
                                .and_then(|w| w.replace_seg_viol(problem, matrix, j, j, seg_or))
                            {
                                delta += v.penalty(sw);
                            }
                        } else {
                            delta += pen_r1_after;
                            if let Some(d) = warp2.and_then(|w| {
                                w.penalty_delta(problem, matrix, sw, j, j, seg_or)
                            }) {
                                delta += d;
                            }
                        }
                    }
                    if delta < -1e-9 {
                        cands.push(Cand { delta, seg_len, r2, j, rev });
                    }
                }
            }
        }
    }

    if cands.is_empty() {
        return;
    }
    cands.sort_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap());

    // Confirm feasibility best-first; first feasible candidate is the best move.
    // Transient slack arrays for the intra-route base (route1 minus the
    // segment), built lazily per seg_len.
    let mut intra_slack: [Option<Option<Box<crate::slack::RouteSlack>>>; 4] =
        [None, None, None, None];
    for cand in &cands {
        if best.as_ref().map_or(false, |b| cand.delta >= b.delta - 1e-12) {
            break; // can't beat the incumbent from here on (sorted)
        }
        // seg_len found removal-infeasible by a deferred evaluation?
        if matches!(&m1_eval[cand.seg_len], Some(None)) {
            continue;
        }
        let (route1_after, _) = r1_after_cache[cand.seg_len].as_ref().unwrap();
        let segment: Vec<TaskRef> = route1.steps[i..i + cand.seg_len].to_vec();
        let mut seg_oriented = segment.clone();
        if cand.rev {
            seg_oriented.reverse();
        }
        let route2 = &sol.routes[cand.r2];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        // O(seg) slack prefilter on the destination route (hard mode only —
        // warp mode has no infeasibility to prefilter; its penalty already
        // entered the rank and the confirm below is exact).
        if r1 == cand.r2 {
            if have_s1 {
                let s = intra_slack[cand.seg_len].get_or_insert_with(|| {
                    crate::slack::RouteSlack::build(problem, matrix, veh1, route1_after)
                        .map(Box::new)
                });
                if let Some(s) = s {
                    if !s.replace_seg_ok(problem, matrix, veh1, cand.j, cand.j, &seg_oriented) {
                        op_slack_skip(OP_RELOC);
                        continue;
                    }
                }
            }
        } else if let Some(Judge::Slack(cache)) = judge.as_deref_mut() {
            cache.ensure(cand.r2, problem, matrix, sol);
            if let Some(s2) = cache.get(cand.r2) {
                if !s2.replace_seg_ok(problem, matrix, veh2, cand.j, cand.j, &seg_oriented) {
                    op_slack_skip(OP_RELOC);
                    continue;
                }
            }
        }
        // Deferred route1-after evaluation (its metrics go into the Move).
        if r1 != cand.r2 && m1_eval[cand.seg_len].is_none() {
            let m = if route1_after.is_empty() {
                Some(RouteMetrics::default())
            } else {
                op_confirm_eval(OP_RELOC);
                evaluate_route(problem, matrix, veh1, route1_after).ok()
            };
            let dead = m.is_none();
            m1_eval[cand.seg_len] = Some(m);
            if dead {
                continue;
            }
        }
        let base: &[TaskRef] = if r1 == cand.r2 { route1_after } else { &route2.steps };
        let mut cand2 = Vec::with_capacity(base.len() + cand.seg_len);
        cand2.extend_from_slice(&base[..cand.j]);
        cand2.extend_from_slice(&seg_oriented);
        cand2.extend_from_slice(&base[cand.j..]);
        op_confirm_eval(OP_RELOC);
        let m2 = match evaluate_route(problem, matrix, veh2, &cand2) {
            Ok(m) => m,
            Err(_) => continue,
        };
        // Build the move with authoritative metrics.
        let touched = segment;
        if r1 == cand.r2 {
            let real_delta = m2.cost - route1.metrics.cost;
            consider(best, Move {
                delta: real_delta,
                route_updates: vec![(r1, Some((cand2, m2)))],
                touched,
            });
        } else {
            let m1 = m1_eval[cand.seg_len].as_ref().unwrap().clone().unwrap();
            let real_delta = (m1.cost - route1.metrics.cost) + (m2.cost - route2.metrics.cost);
            let upd1 = if route1_after.is_empty() {
                (r1, None)
            } else {
                (r1, Some((route1_after.clone(), m1)))
            };
            consider(best, Move {
                delta: real_delta,
                route_updates: vec![upd1, (cand.r2, Some((cand2, m2)))],
                touched,
            });
        }
        break; // first feasible (best-by-cost) accepted
    }
}

/// 2-opt restricted to reversals that include position `i` in route `r1`.
fn try_two_opt_through(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    judge: Option<&mut Judge>,
    best: &mut Option<Move>,
) {
    let mut judge = judge;
    let route = &sol.routes[r1];
    let n = route.steps.len();
    if n < 3 { return; }
    let veh = &problem.vehicles[route.vehicle_idx];

    // Fast path: O(1) edge-delta cost per (a,b) reversal (internal arc sums
    // maintained incrementally), then confirm feasibility lazily best-first.
    if fast_ls_enabled() && fast_cost_eligible(problem) {
        let locs: Vec<Option<usize>> = route.steps.iter().map(|&s| step_loc(problem, s)).collect();
        if locs.iter().all(|l| l.is_some()) {
            let c = cost_coef(veh);
            let loc = |k: usize| locs[k].unwrap();
            let ds = depot_start(veh);
            let de = depot_end(veh);
            struct C2 { delta: f64, a: usize, b: usize }
            let mut cands: Vec<C2> = Vec::new();
            // Warp mode: judge reversals by their exact penalty delta too —
            // reversals are THE repair move for time-window order violations.
            if let Some(Judge::Warp(cache, _)) = judge.as_deref_mut() {
                cache.ensure(r1, problem, matrix, sol);
            }
            let (warp1, warp_sw) = match judge.as_deref() {
                Some(Judge::Warp(cache, sw)) => (cache.get(r1), Some(*sw)),
                _ => (None, None),
            };
            let base_pen = match (warp1, warp_sw) {
                (Some(w), Some(sw)) => w.viol().penalty(sw),
                _ => 0.0,
            };
            for a in 0..n - 1 {
                if a > i { break; } // need a ≤ i
                let prev = if a == 0 { ds } else { Some(loc(a - 1)) };
                let mut fwd = 0.0; // forward internal arc sum of [a..b]
                let mut rev = 0.0; // reversed internal arc sum
                // Reversed-segment stats for warp: Tws of locs[b..=a] reversed
                // and the max prefix sum of per-stop pickup−delivery, both
                // extended O(1) per b (front extension).
                let mut rev_stats = warp1.map(|w| w.stop_stats(problem, route.steps[a]));
                for b in a + 1..n {
                    // extend the window to include edge (b-1,b)
                    fwd += arc_cost(matrix, c, loc(b - 1), loc(b));
                    rev += arc_cost(matrix, c, loc(b), loc(b - 1));
                    if let (Some(w), Some((rtws, rmax))) = (warp1, rev_stats.as_mut()) {
                        let (node_b, c_b) = w.stop_stats(problem, route.steps[b]);
                        let edge = w.edge_dur(matrix, loc(b), loc(b - 1));
                        *rtws = crate::warp::Tws::merge(node_b, edge, *rtws);
                        *rmax = c_b + (*rmax).max(0);
                    }
                    if b < i { continue; } // need b ≥ i
                    let next = if b + 1 >= n { de } else { Some(loc(b + 1)) };
                    let old_e = prev.map_or(0.0, |p| arc_cost(matrix, c, p, loc(a)))
                        + next.map_or(0.0, |nx| arc_cost(matrix, c, loc(b), nx));
                    let new_e = prev.map_or(0.0, |p| arc_cost(matrix, c, p, loc(b)))
                        + next.map_or(0.0, |nx| arc_cost(matrix, c, loc(a), nx));
                    let mut delta = (new_e + rev) - (old_e + fwd);
                    if let (Some(w), Some(sw), Some((rtws, rmax))) = (warp1, warp_sw, rev_stats.as_ref()) {
                        delta += w.reversal_viol(matrix, a, b, *rtws, *rmax).penalty(sw) - base_pen;
                    }
                    if delta < -1e-9 {
                        cands.push(C2 { delta, a, b });
                    }
                }
            }
            if cands.is_empty() {
                return;
            }
            cands.sort_by(|x, y| x.delta.partial_cmp(&y.delta).unwrap());
            let mut rev_seg: Vec<TaskRef> = Vec::new();
            for cd in &cands {
                if best.as_ref().map_or(false, |bm| cd.delta >= bm.delta - 1e-12) {
                    break;
                }
                // O(segment) slack prefilter: the reversal replaces [a, b]
                // with its mirror — chain the reversed stops, land on lat[b+1].
                if let Some(Judge::Slack(cache)) = judge.as_deref_mut() {
                    cache.ensure(r1, problem, matrix, sol);
                    if let Some(s) = cache.get(r1) {
                        rev_seg.clear();
                        rev_seg.extend(route.steps[cd.a..=cd.b].iter().rev().copied());
                        if !s.replace_seg_ok(problem, matrix, veh, cd.a, cd.b + 1, &rev_seg) {
                            op_slack_skip(OP_TWO_OPT);
                            continue;
                        }
                    }
                }
                let mut cand = route.steps.clone();
                cand[cd.a..=cd.b].reverse();
                op_confirm_eval(OP_TWO_OPT);
                let m = match evaluate_route(problem, matrix, veh, &cand) {
                    Ok(m) => m, Err(_) => continue,
                };
                let delta = m.cost - route.metrics.cost;
                let touched = cand[cd.a..=cd.b].to_vec();
                consider(best, Move { delta, route_updates: vec![(r1, Some((cand, m)))], touched });
                break;
            }
            return;
        }
    }

    for a in 0..n - 1 {
        for b in a + 1..n {
            if i < a || i > b { continue; }
            let mut cand = route.steps.clone();
            cand[a..=b].reverse();
            let m = match evaluate_route(problem, matrix, veh, &cand) {
                Ok(m) => m, Err(_) => continue,
            };
            let delta = m.cost - route.metrics.cost;
            consider(best, Move {
                delta,
                route_updates: vec![(r1, Some((cand.clone(), m)))],
                touched: cand[a..=b].to_vec(),
            });
        }
    }
}

/// Exchange operator: swap the task at (r1, i) with each task in another
/// route. With granularity, only swap with tasks whose location is among
/// the K nearest neighbors of (r1, i)'s location.
fn try_exchange_with(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    granular: Option<&Granular>,
    judge: Option<&mut Judge>,
    best: &mut Option<Move>,
) {
    let mut judge = judge;
    let n_routes = sol.routes.len();
    let a = sol.routes[r1].steps[i];
    let a_loc = a.description(problem).location.index;

    // Fast path: O(1) edge-delta per swap, lazy feasibility confirm best-first.
    let fast = granular.is_some() && fast_ls_enabled() && fast_cost_eligible(problem);
    if fast {
        let Some(aloc) = a_loc else { return };
        nmark_set(granular.unwrap(), aloc, matrix.n);
        let route1 = &sol.routes[r1];
        let veh1 = &problem.vehicles[route1.vehicle_idx];
        let c1 = cost_coef(veh1);
        let n1 = route1.steps.len();
        let a_serv = a.description(problem).service as f64 * 1e-6;
        let pi = if i == 0 { depot_start(veh1) } else { step_loc(problem, route1.steps[i - 1]) };
        let ni = if i + 1 >= n1 { depot_end(veh1) } else { step_loc(problem, route1.steps[i + 1]) };
        let old_a = pi.map_or(0.0, |p| arc_cost(matrix, c1, p, aloc))
            + ni.map_or(0.0, |n| arc_cost(matrix, c1, aloc, n));
        struct EC { delta: f64, r2: usize, j: usize }
        let mut cands: Vec<EC> = Vec::new();
        if let Some(Judge::Warp(cache, _)) = judge.as_deref_mut() {
            cache.ensure(r1, problem, matrix, sol);
        }
        for r2 in 0..n_routes {
            if r2 == r1 { continue; }
            let route2 = &sol.routes[r2];
            let veh2 = &problem.vehicles[route2.vehicle_idx];
            if !veh2.has_skills(a.skills(problem)) { continue; }
            let c2 = cost_coef(veh2);
            let n2 = route2.steps.len();
            // Warp mode: both sides' penalty deltas are RELATIVE (the confirm
            // subtracts both routes' penalised metrics).
            if let Some(Judge::Warp(cache, _)) = judge.as_deref_mut() {
                cache.ensure(r2, problem, matrix, sol);
            }
            let (warp1, warp2, wsw) = match judge.as_deref() {
                Some(Judge::Warp(cache, sw)) => (cache.get(r1), cache.get(r2), Some(*sw)),
                _ => (None, None, None),
            };
            for j in 0..n2 {
                let b = route2.steps[j];
                let Some(bloc) = step_loc(problem, b) else { continue };
                if !nmark_has(bloc) { continue; }
                if !veh1.has_skills(b.skills(problem)) { continue; }
                let b_serv = b.description(problem).service as f64 * 1e-6;
                // r1: a→b at i.
                let new_a = pi.map_or(0.0, |p| arc_cost(matrix, c1, p, bloc))
                    + ni.map_or(0.0, |n| arc_cost(matrix, c1, bloc, n));
                let d1 = (new_a - old_a) + (b_serv - a_serv);
                // r2: b→a at j.
                let pj = if j == 0 { depot_start(veh2) } else { step_loc(problem, route2.steps[j - 1]) };
                let nj = if j + 1 >= n2 { depot_end(veh2) } else { step_loc(problem, route2.steps[j + 1]) };
                let old_b = pj.map_or(0.0, |p| arc_cost(matrix, c2, p, bloc))
                    + nj.map_or(0.0, |n| arc_cost(matrix, c2, bloc, n));
                let new_b = pj.map_or(0.0, |p| arc_cost(matrix, c2, p, aloc))
                    + nj.map_or(0.0, |n| arc_cost(matrix, c2, aloc, n));
                let d2 = (new_b - old_b) + (a_serv - b_serv);
                let mut delta = d1 + d2;
                if let (Some(w1), Some(w2), Some(sw)) = (warp1, warp2, wsw) {
                    if let (Some(p1), Some(p2)) = (
                        w1.penalty_delta(problem, matrix, sw, i, i + 1, &[b]),
                        w2.penalty_delta(problem, matrix, sw, j, j + 1, &[a]),
                    ) {
                        delta += p1 + p2;
                    }
                }
                if delta < -1e-9 {
                    cands.push(EC { delta, r2, j });
                }
            }
        }
        if cands.is_empty() { return; }
        cands.sort_by(|x, y| x.delta.partial_cmp(&y.delta).unwrap());
        for cd in &cands {
            if best.as_ref().map_or(false, |bm| cd.delta >= bm.delta - 1e-12) { break; }
            let route2 = &sol.routes[cd.r2];
            let veh2 = &problem.vehicles[route2.vehicle_idx];
            let b = route2.steps[cd.j];
            // O(1) slack prefilter: replace a with b at i in r1, b with a at j in r2.
            if let Some(Judge::Slack(cache)) = judge.as_deref_mut() {
                cache.ensure(r1, problem, matrix, sol);
                cache.ensure(cd.r2, problem, matrix, sol);
                let ok = match (cache.get(r1), cache.get(cd.r2)) {
                    (Some(s1), Some(s2)) => {
                        s1.replace_seg_ok(problem, matrix, veh1, i, i + 1, &[b])
                            && s2.replace_seg_ok(problem, matrix, veh2, cd.j, cd.j + 1, &[a])
                    }
                    _ => true,
                };
                if !ok {
                    op_slack_skip(OP_EXCH);
                    continue;
                }
            }
            let mut cand1 = route1.steps.clone();
            let mut cand2 = route2.steps.clone();
            cand1[i] = b;
            cand2[cd.j] = a;
            op_confirm_eval(OP_EXCH);
            let m1 = match evaluate_route(problem, matrix, veh1, &cand1) { Ok(m) => m, Err(_) => continue };
            op_confirm_eval(OP_EXCH);
            let m2 = match evaluate_route(problem, matrix, veh2, &cand2) { Ok(m) => m, Err(_) => continue };
            let delta = (m1.cost - route1.metrics.cost) + (m2.cost - route2.metrics.cost);
            consider(best, Move {
                delta,
                route_updates: vec![(r1, Some((cand1, m1))), (cd.r2, Some((cand2, m2)))],
                touched: vec![a, b],
            });
            break;
        }
        return;
    }

    // Slow path (non-eligible / no granular): build the membership set once.
    let neighbor_set: Option<HashSet<usize>> = match (granular, a_loc) {
        (Some(g), Some(loc)) => Some(g.neighbors(loc).collect()),
        _ => None,
    };
    for r2 in 0..n_routes {
        if r2 == r1 { continue; }
        let route1 = &sol.routes[r1];
        let route2 = &sol.routes[r2];
        let veh1 = &problem.vehicles[route1.vehicle_idx];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        if !veh2.has_skills(a.skills(problem)) { continue; }

        let n2 = route2.steps.len();
        for j in 0..n2 {
            let b = route2.steps[j];
            if let Some(set) = &neighbor_set {
                let b_loc = b.description(problem).location.index;
                if b_loc.map_or(true, |l| !set.contains(&l)) { continue; }
            }
            if !veh1.has_skills(b.skills(problem)) { continue; }
            let mut cand1 = route1.steps.clone();
            let mut cand2 = route2.steps.clone();
            cand1[i] = b;
            cand2[j] = a;
            let m1 = match evaluate_route(problem, matrix, veh1, &cand1) {
                Ok(m) => m, Err(_) => continue,
            };
            let m2 = match evaluate_route(problem, matrix, veh2, &cand2) {
                Ok(m) => m, Err(_) => continue,
            };
            let delta = (m1.cost - route1.metrics.cost) + (m2.cost - route2.metrics.cost);
            consider(best, Move {
                delta,
                route_updates: vec![
                    (r1, Some((cand1, m1))),
                    (r2, Some((cand2, m2))),
                ],
                touched: vec![a, b],
            });
        }
    }
}

/// 2-opt*: cut both routes at one point each, swap their tails. Specifically
/// new_r1 = r1[..=i] + r2[j..], new_r2 = r2[..j] + r1[i+1..]. Granularity
/// limits the j range to cut-points adjacent to a top-K matrix neighbor of
/// the (r1, i) location.
fn try_two_opt_star(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    granular: Option<&Granular>,
    judge: Option<&mut Judge>,
    best: &mut Option<Move>,
) {
    // Large-N guard (matrix.n ≥ 500): keep the slow path so the top-level
    // polish on big instances stays byte-identical to the shipped engine —
    // the fast variant is cost-identical but tie-breaks differently, which
    // measured +0.13% on one of three N=1000 seeds. Small-N (where the HGS
    // education lives) takes the fast path.
    // NOTE: not warp-converted — under warp mode it runs BLIND (arc-only
    // rank, exact penalised confirm): sound, but misses penalty-repair tail
    // swaps. Convert if education measurements say it matters.
    if granular.is_some() && fast_ls_enabled() && fast_cost_eligible(problem) && matrix.n < 500 {
        try_two_opt_star_fast(problem, matrix, sol, r1, i, granular.unwrap(), judge, best);
        return;
    }
    try_two_opt_star_slow(problem, matrix, sol, r1, i, granular, best);
}

/// Fast 2-opt*: the tail swap changes exactly four boundary arcs — everything
/// else (tail internals, the tails' return-to-depot legs, service totals,
/// fixed costs) moves between the two routes unchanged. Exact only when both
/// vehicles share start AND end depots (otherwise the tail's depot leg does
/// not cancel); pairs with differing depots fall back to full evaluation
/// inline. Candidates are ranked by the O(1) delta, then confirmed lazily
/// best-first by the authoritative evaluator (TW/capacity feasibility of a
/// tail swap is what usually kills it).
fn try_two_opt_star_fast(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    granular: &Granular,
    judge: Option<&mut Judge>,
    best: &mut Option<Move>,
) {
    let mut judge = judge;
    let n_routes = sol.routes.len();
    let route1 = &sol.routes[r1];
    let veh1 = &problem.vehicles[route1.vehicle_idx];
    let n1 = route1.steps.len();
    let c = cost_coef(veh1); // homogeneous (fast_cost_eligible)
    let Some(a_i) = step_loc(problem, route1.steps[i]) else {
        return;
    };
    nmark_set(granular, a_i, matrix.n);
    let next1 = if i + 1 < n1 {
        step_loc(problem, route1.steps[i + 1])
    } else {
        depot_end(veh1)
    };

    struct Cand {
        delta: f64,
        r2: usize,
        j: usize,
    }
    let mut cands: Vec<Cand> = Vec::new();

    for r2 in 0..n_routes {
        if r2 == r1 {
            continue;
        }
        let route2 = &sol.routes[r2];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        // Same-depot gate: tails carry their depot legs with them, which only
        // cancels when both vehicles use the same depots. Otherwise evaluate
        // this pair the slow way (rare: multi-depot heterogeneous fleets).
        if depot_start(veh1) != depot_start(veh2) || depot_end(veh1) != depot_end(veh2) {
            two_opt_star_pair_slow(problem, matrix, sol, r1, i, r2, true, best);
            continue;
        }
        let n2 = route2.steps.len();

        for j in 0..n2 {
            // Granular parity with the slow path: only cut points whose r2[j]
            // is a top-K neighbour of the anchor (j == n2 has no r2[j] and is
            // skipped there too).
            let Some(b_j) = step_loc(problem, route2.steps[j]) else {
                continue;
            };
            if !nmark_has(b_j) {
                continue;
            }
            let prev2 = if j > 0 {
                step_loc(problem, route2.steps[j - 1])
            } else {
                depot_start(veh2)
            };
            // Δ = new boundary arcs − old boundary arcs.
            let old = match next1 {
                Some(n) => arc_cost(matrix, c, a_i, n),
                None => 0.0,
            } + match prev2 {
                Some(p) => arc_cost(matrix, c, p, b_j),
                None => 0.0,
            };
            let new = arc_cost(matrix, c, a_i, b_j)
                + match (prev2, next1) {
                    (Some(p), Some(n)) => arc_cost(matrix, c, p, n),
                    _ => 0.0,
                };
            let delta = new - old;
            if delta < -1e-9 {
                cands.push(Cand { delta, r2, j });
            }
        }
    }

    if cands.is_empty() {
        return;
    }
    cands.sort_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap());

    for cand in &cands {
        if best.as_ref().map_or(false, |b| cand.delta >= b.delta - 1e-12) {
            break;
        }
        let route2 = &sol.routes[cand.r2];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        let n2 = route2.steps.len();
        let j = cand.j;
        let r1_tail: &[TaskRef] = &route1.steps[i + 1..];
        let r2_tail: &[TaskRef] = &route2.steps[j..];
        if !r1_tail.iter().all(|t| veh2.has_skills(t.skills(problem))) {
            continue;
        }
        if !r2_tail.iter().all(|t| veh1.has_skills(t.skills(problem))) {
            continue;
        }
        // O(1) slack prefilter for the tail swap. Each route's `lat`/load
        // arrays remain valid for the new host only when both vehicles share
        // the same day window (depots/speed already equal on this path).
        if let Some(Judge::Slack(cache)) = judge.as_deref_mut() {
            if veh1.time_window() == veh2.time_window() {
                cache.ensure(r1, problem, matrix, sol);
                cache.ensure(cand.r2, problem, matrix, sol);
                if let (Some(s1), Some(s2)) = (cache.get(r1), cache.get(cand.r2)) {
                    let dim = s1.dim();
                    if dim == s2.dim() && dim <= 8 {
                        let mut ok = true;
                        // new r1 = r1[..=i] + r2[j..]  (r2 tail is non-empty)
                        ok &= s2.admits_arrival(matrix, j, s1.ect(i), Some(s1.loc(i)));
                        // new r2 = r2[..j] + r1[i+1..]
                        if i + 1 < n1 {
                            let (t2, from2) = s2.depart_before(j);
                            ok &= s1.admits_arrival(matrix, i + 1, t2, from2);
                        } else if j > 0 {
                            // truncated r2 must still reach its end depot
                            let (t2, from2) = s2.depart_before(j);
                            ok &= s2.admits_arrival(matrix, n2, t2, from2);
                        }
                        if ok {
                            // Load checkpoints: prefixes shift by the change in
                            // downstream deliveries, tails by the change in
                            // carried pickups.
                            let mut shift1 = [0i64; 8];
                            let mut shift2 = [0i64; 8];
                            let mut nshift1 = [0i64; 8];
                            let mut nshift2 = [0i64; 8];
                            for d in 0..dim {
                                let del_tail1 = s1.del_pre(n1, d) - s1.del_pre(i + 1, d);
                                let del_tail2 = s2.del_pre(n2, d) - s2.del_pre(j, d);
                                shift1[d] = del_tail2 - del_tail1;
                                nshift1[d] = -shift1[d];
                                shift2[d] = s1.pick_pre(i + 1, d) - s2.pick_pre(j, d);
                                nshift2[d] = -shift2[d];
                            }
                            ok &= s1.pre_shift_ok(veh1, i + 1, &shift1[..dim]);
                            ok &= s2.suf_shift_ok(veh1, j, &shift2[..dim]);
                            if i + 1 < n1 || j > 0 {
                                ok &= s2.pre_shift_ok(veh2, j, &nshift1[..dim]);
                                ok &= s1.suf_shift_ok(veh2, i + 1, &nshift2[..dim]);
                            }
                        }
                        if !ok {
                            op_slack_skip(OP_TWO_OPT_STAR);
                            continue;
                        }
                    }
                }
            }
        }
        let mut cand1: Vec<TaskRef> = Vec::with_capacity(i + 1 + (n2 - j));
        cand1.extend_from_slice(&route1.steps[..=i]);
        cand1.extend_from_slice(r2_tail);
        let mut cand2: Vec<TaskRef> = Vec::with_capacity(j + (n1 - i - 1));
        cand2.extend_from_slice(&route2.steps[..j]);
        cand2.extend_from_slice(r1_tail);
        op_confirm_eval(OP_TWO_OPT_STAR);
        let m1 = match evaluate_route(problem, matrix, veh1, &cand1) {
            Ok(m) => m,
            Err(_) => continue,
        };
        op_confirm_eval(OP_TWO_OPT_STAR);
        let m2 = match evaluate_route(problem, matrix, veh2, &cand2) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let delta = (m1.cost - route1.metrics.cost) + (m2.cost - route2.metrics.cost);
        let mut touched: Vec<TaskRef> = Vec::new();
        touched.push(route1.steps[i]);
        if j < n2 {
            touched.push(route2.steps[j]);
        }
        consider(best, Move {
            delta,
            route_updates: vec![(r1, Some((cand1, m1))), (cand.r2, Some((cand2, m2)))],
            touched,
        });
        break; // first feasible = best feasible (sorted by exact cost)
    }
}

/// Slow-path 2-opt* for a single (r1, r2) pair — used by the fast path as a
/// fallback when the pair's depots differ. `use_nmark` says the caller already
/// stamped the anchor's neighbours into the thread-local buffer.
fn two_opt_star_pair_slow(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    r2: usize,
    use_nmark: bool,
    best: &mut Option<Move>,
) {
    let route1 = &sol.routes[r1];
    let veh1 = &problem.vehicles[route1.vehicle_idx];
    let n1 = route1.steps.len();
    let route2 = &sol.routes[r2];
    let veh2 = &problem.vehicles[route2.vehicle_idx];
    let n2 = route2.steps.len();
    for j in 0..=n2 {
        if use_nmark {
            let r2_j_loc = if j < n2 { step_loc(problem, route2.steps[j]) } else { None };
            if !r2_j_loc.map_or(false, nmark_has) {
                continue;
            }
        }
        let r1_tail: &[TaskRef] = &route1.steps[i + 1..];
        let r2_tail: &[TaskRef] = &route2.steps[j..];
        if !r1_tail.iter().all(|t| veh2.has_skills(t.skills(problem))) {
            continue;
        }
        if !r2_tail.iter().all(|t| veh1.has_skills(t.skills(problem))) {
            continue;
        }
        let mut cand1: Vec<TaskRef> = Vec::with_capacity(i + 1 + (n2 - j));
        cand1.extend_from_slice(&route1.steps[..=i]);
        cand1.extend_from_slice(r2_tail);
        let mut cand2: Vec<TaskRef> = Vec::with_capacity(j + (n1 - i - 1));
        cand2.extend_from_slice(&route2.steps[..j]);
        cand2.extend_from_slice(r1_tail);
        let m1 = match evaluate_route(problem, matrix, veh1, &cand1) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let m2 = match evaluate_route(problem, matrix, veh2, &cand2) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let delta = (m1.cost - route1.metrics.cost) + (m2.cost - route2.metrics.cost);
        let mut touched: Vec<TaskRef> = Vec::new();
        touched.push(route1.steps[i]);
        if j < n2 {
            touched.push(route2.steps[j]);
        }
        consider(best, Move {
            delta,
            route_updates: vec![(r1, Some((cand1, m1))), (r2, Some((cand2, m2)))],
            touched,
        });
    }
}

fn try_two_opt_star_slow(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    granular: Option<&Granular>,
    best: &mut Option<Move>,
) {
    let n_routes = sol.routes.len();
    let route1 = &sol.routes[r1];
    let veh1 = &problem.vehicles[route1.vehicle_idx];
    let n1 = route1.steps.len();
    let anchor = route1.steps[i].description(problem).location.index;
    let has_gran = match (granular, anchor) {
        (Some(g), Some(loc)) => { nmark_set(g, loc, matrix.n); true }
        _ => false,
    };

    for r2 in 0..n_routes {
        if r2 == r1 { continue; }
        let route2 = &sol.routes[r2];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        let n2 = route2.steps.len();

        // r1[..=i] gets r2[j..] tail; r2[..j] gets r1[i+1..] tail.
        // Iterate j over every cut point in r2 where the resulting swap
        // brings something near i's location into the swap zone.
        for j in 0..=n2 {
            // Granular: skip if neither (r1[i], r2[j]) edge nor the new
            // (r2[j-1], r1[i+1]) edge involves a near neighbor.
            if has_gran {
                let r2_j_loc = if j < n2 {
                    route2.steps[j].description(problem).location.index
                } else { None };
                let near = r2_j_loc.map_or(false, nmark_has);
                if !near { continue; }
            }

            let r1_tail: &[TaskRef] = &route1.steps[i + 1..];
            let r2_tail: &[TaskRef] = &route2.steps[j..];

            // Skill checks: r2's tail must fit r1's vehicle and vice versa.
            if !r1_tail.iter().all(|t| veh2.has_skills(t.skills(problem))) { continue; }
            if !r2_tail.iter().all(|t| veh1.has_skills(t.skills(problem))) { continue; }

            // Build candidates.
            let mut cand1: Vec<TaskRef> = Vec::with_capacity(i + 1 + (n2 - j));
            cand1.extend_from_slice(&route1.steps[..=i]);
            cand1.extend_from_slice(r2_tail);

            let mut cand2: Vec<TaskRef> = Vec::with_capacity(j + (n1 - i - 1));
            cand2.extend_from_slice(&route2.steps[..j]);
            cand2.extend_from_slice(r1_tail);

            let m1 = match evaluate_route(problem, matrix, veh1, &cand1) {
                Ok(m) => m, Err(_) => continue,
            };
            let m2 = match evaluate_route(problem, matrix, veh2, &cand2) {
                Ok(m) => m, Err(_) => continue,
            };
            let delta = (m1.cost - route1.metrics.cost)
                      + (m2.cost - route2.metrics.cost);

            // Touched: the boundary tasks that just got new neighbors.
            let mut touched: Vec<TaskRef> = Vec::new();
            touched.push(route1.steps[i]);
            if j < n2 { touched.push(route2.steps[j]); }
            consider(best, Move {
                delta,
                route_updates: vec![
                    (r1, Some((cand1, m1))),
                    (r2, Some((cand2, m2))),
                ],
                touched,
            });
        }
    }
}

thread_local! {
    /// swap*'s per-pair transient slack arrays (r2 \ {t2}): one scratch per
    /// thread, rebuilt in place — the millions of per-pair builds reuse the
    /// same Vec capacities instead of allocating 7 fresh Vecs each.
    static SWAP_S2_SCRATCH: std::cell::RefCell<crate::slack::RouteSlack> =
        std::cell::RefCell::new(crate::slack::RouteSlack::scratch());
    /// Insertion-position ranking buffer for `best_feasible_insertion_delta`.
    static INS_ORDER: std::cell::RefCell<Vec<(f64, usize)>> =
        std::cell::RefCell::new(Vec::new());
}

/// SwapStar (Vidal 2022, HGS-CVRP): swap two tasks t1 in r1, t2 in r2 but
/// place each task at its BEST position in the other route — not at the
/// old position of the swapped-out task. Unlike try_exchange_with, which
/// locks the swap to the same position. More expensive but finds moves
/// the other operators systematically miss.
///
/// Granular: only pairs (t1, t2) where t2 is among the K nearest to t1.
fn try_swap_star(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    granular: Option<&Granular>,
    use_slack: bool,
    best: &mut Option<Move>,
) {
    let n_routes = sol.routes.len();
    let route1 = &sol.routes[r1];
    let n1 = route1.steps.len();
    let veh1 = &problem.vehicles[route1.vehicle_idx];
    let t1 = route1.steps[i];
    let t1_loc = t1.description(problem).location.index;

    let has_gran = match (granular, t1_loc) {
        (Some(g), Some(loc)) => { nmark_set(g, loc, matrix.n); true }
        _ => false,
    };

    let mut r1_minus: Vec<TaskRef> = Vec::with_capacity(n1.saturating_sub(1));
    r1_minus.extend_from_slice(&route1.steps[..i]);
    r1_minus.extend_from_slice(&route1.steps[i + 1..]);

    // Fast pre-filter (simple envelope): a valid lower bound on each pair's
    // cost-delta lets us skip hopeless pairs before any full route evaluation.
    let prefilter = fast_ls_enabled() && fast_cost_eligible(problem);
    let rem1 = if prefilter {
        pred_remove_delta(problem, matrix, veh1, &route1.steps, i)
    } else { 0.0 };

    // Transient slack arrays for the hypothetical bases: r1\{t1} is shared
    // across all pairs (built lazily on the first surviving pair), r2\{t2}
    // is built per pair. One O(route) build replaces up to O(route) full
    // evaluations inside best_insertion.
    let use_slack = use_slack && prefilter;
    let mut s1m: Option<Option<Box<crate::slack::RouteSlack>>> = None;

    // Slack-judged pair candidates: exact O(1) cost deltas + O(1) feasibility,
    // confirmed best-first after the scan (same shape as the other fast ops).
    struct SCand {
        delta: f64,
        r2: usize,
        j: usize,
        pos1: usize,
        pos2: usize,
    }
    let mut scands: Vec<SCand> = Vec::new();
    // Running best among the slack-judged pairs — keeps the lower-bound prune
    // as tight as the old inline-evaluation flow kept it via `best`.
    let mut best_sc = f64::INFINITY;

    // r2\{t2} scratch, reused across the whole pair scan (clear + extend).
    let mut r2_minus: Vec<TaskRef> = Vec::new();

    let t1serv = t1.description(problem).service as f64 * 1e-6;

    // Largest arc removable by inserting INTO r1_minus: the t2-insertion
    // delta is ≥ −max_removed_r1 (added arcs and service are ≥ 0), giving an
    // O(1) stage-1 prune before the O(n1) exact insertion scan. Stage 1 only
    // prunes pairs the exact lower bound would also prune ⇒ pair set (and
    // trajectory) unchanged.
    let max_removed_r1 = if prefilter {
        let c1 = cost_coef(veh1);
        let ds1 = depot_start(veh1);
        let de1 = depot_end(veh1);
        let nb = r1_minus.len();
        let mut mx = 0.0f64;
        for pos in 0..=nb {
            let prev = if pos == 0 { ds1 } else { step_loc(problem, r1_minus[pos - 1]) };
            let next = if pos == nb { de1 } else { step_loc(problem, r1_minus[pos]) };
            if let (Some(p), Some(n)) = (prev, next) {
                mx = mx.max(arc_cost(matrix, c1, p, n));
            }
        }
        mx
    } else { 0.0 };

    for r2 in 0..n_routes {
        if r2 == r1 { continue; }
        let route2 = &sol.routes[r2];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        let n2 = route2.steps.len();
        if n2 == 0 { continue; }
        if !veh2.has_skills(t1.skills(problem)) { continue; }
        let c2 = cost_coef(veh2);
        let ds2 = depot_start(veh2);
        let de2 = depot_end(veh2);
        // Lazy top-3 cheapest feasibility-ignoring insertion deltas of t1 in
        // the FULL route2 (position kept to exclude pair-adjacent slots).
        // r2_minus positions are exactly the full-route positions ∉ {j, j+1}
        // plus the gap at j, so min(first non-adjacent top-3 entry, gap delta)
        // is BIT-IDENTICAL to pred_cheapest_insert on r2_minus — computed
        // O(1) per pair instead of O(n2).
        let mut ins2_top3: Option<[(f64, usize); 3]> = None;

        for j in 0..n2 {
            let t2 = route2.steps[j];

            // Granular filter: skip pairs where t2 is not near t1.
            if has_gran {
                let l = t2.description(problem).location.index;
                if l.map_or(true, |l| !nmark_has(l)) { continue; }
            }
            if !veh1.has_skills(t2.skills(problem)) { continue; }

            let rem2 = if prefilter {
                pred_remove_delta(problem, matrix, veh2, &route2.steps, j)
            } else { 0.0 };

            // Lower-bound prune: (cheapest feasibility-ignoring inserts) +
            // (exact removals) ≤ true pair delta. Skip if it can't beat the
            // incumbent. Never prunes an improving move ⇒ equivalence preserved.
            if prefilter {
                let top3 = ins2_top3.get_or_insert_with(|| {
                    let mut t = [(f64::INFINITY, usize::MAX); 3];
                    if let Some(tloc) = t1_loc {
                        for pos in 0..=n2 {
                            let prev = if pos == 0 { ds2 } else { step_loc(problem, route2.steps[pos - 1]) };
                            let next = if pos == n2 { de2 } else { step_loc(problem, route2.steps[pos]) };
                            let added = prev.map_or(0.0, |p| arc_cost(matrix, c2, p, tloc))
                                + next.map_or(0.0, |n| arc_cost(matrix, c2, tloc, n));
                            let removed = match (prev, next) {
                                (Some(p), Some(n)) => arc_cost(matrix, c2, p, n),
                                _ => 0.0,
                            };
                            let d = added - removed + t1serv;
                            if d < t[2].0 {
                                t[2] = (d, pos);
                                if t[2].0 < t[1].0 { t.swap(1, 2); }
                                if t[1].0 < t[0].0 { t.swap(0, 1); }
                            }
                        }
                    }
                    t
                });
                let ins2_lb = {
                    // Gap delta: inserting t1 where t2 used to sit.
                    let prev = if j == 0 { ds2 } else { step_loc(problem, route2.steps[j - 1]) };
                    let next = if j + 1 == n2 { de2 } else { step_loc(problem, route2.steps[j + 1]) };
                    let mut lb = match t1_loc {
                        Some(tloc) => {
                            let added = prev.map_or(0.0, |p| arc_cost(matrix, c2, p, tloc))
                                + next.map_or(0.0, |n| arc_cost(matrix, c2, tloc, n));
                            let removed = match (prev, next) {
                                (Some(p), Some(n)) => arc_cost(matrix, c2, p, n),
                                _ => 0.0,
                            };
                            added - removed + t1serv
                        }
                        None => f64::INFINITY,
                    };
                    // First non-adjacent entry = exact min over all
                    // non-adjacent positions (at most 2 are excluded).
                    for &(d, p) in top3.iter() {
                        if p != j && p != j + 1 {
                            lb = lb.min(d);
                            break;
                        }
                    }
                    lb
                };
                let thresh = best.as_ref().map(|b| b.delta).unwrap_or(-1e-9).min(best_sc);
                // Stage 1: O(1) optimistic bound on the r1-side insert.
                if rem1 + rem2 + ins2_lb - max_removed_r1 >= thresh - 1e-12 {
                    continue;
                }
                let lb = rem1 + pred_cheapest_insert(problem, matrix, veh1, &r1_minus, t2)
                    + rem2 + ins2_lb;
                if lb >= thresh - 1e-12 {
                    continue;
                }
            }

            // r2\{t2} — only built for pairs that survive the prune.
            r2_minus.clear();
            r2_minus.extend_from_slice(&route2.steps[..j]);
            r2_minus.extend_from_slice(&route2.steps[j + 1..]);

            // Slack branch: judge both insertions with O(1) checks and exact
            // arc deltas — zero evaluations until the confirm phase below.
            // Outcomes: None ⇒ arrays unavailable, fall to the eval path;
            // Some(None) ⇒ no feasible insertion, the pair is dead;
            // Some(Some(..)) ⇒ judged.
            if use_slack {
                let s1m_ref = s1m
                    .get_or_insert_with(|| {
                        crate::slack::RouteSlack::build(problem, matrix, veh1, &r1_minus)
                            .map(Box::new)
                    })
                    .as_deref();
                if let Some(s1ref) = s1m_ref {
                    let judged: Option<Option<(usize, f64, usize, f64)>> =
                        SWAP_S2_SCRATCH.with(|cell| {
                            // r1-side first: its arrays are already built, and
                            // a dead r1-side saves the per-pair s2m rebuild.
                            let Some((pos1, d1)) = best_feasible_insertion_delta(
                                problem, matrix, veh1, &r1_minus, t2, s1ref,
                            ) else { return Some(None) };
                            let mut s2m = cell.borrow_mut();
                            if !s2m.rebuild(problem, matrix, veh2, &r2_minus) {
                                return None;
                            }
                            let Some((pos2, d2)) = best_feasible_insertion_delta(
                                problem, matrix, veh2, &r2_minus, t1, &s2m,
                            ) else { return Some(None) };
                            Some(Some((pos1, d1, pos2, d2)))
                        });
                    match judged {
                        Some(Some((pos1, d1, pos2, d2))) => {
                            let delta = rem1 + d1 + rem2 + d2;
                            if delta < -1e-9 {
                                best_sc = best_sc.min(delta);
                                scands.push(SCand { delta, r2, j, pos1, pos2 });
                            }
                            continue;
                        }
                        Some(None) => continue,
                        None => {} // arrays unavailable — eval path below
                    }
                }
            }

            // Eval path (no slack, or arrays unavailable for this pair).
            let best1 = best_insertion(problem, matrix, veh1, &r1_minus, t2, None);
            let Some((cand1, m1)) = best1 else { continue };

            // Best position for t1 in r2\{t2}
            let best2 = best_insertion(problem, matrix, veh2, &r2_minus, t1, None);
            let Some((cand2, m2)) = best2 else { continue };

            let delta = (m1.cost - route1.metrics.cost)
                      + (m2.cost - route2.metrics.cost);
            consider(best, Move {
                delta,
                route_updates: vec![
                    (r1, Some((cand1, m1))),
                    (r2, Some((cand2, m2))),
                ],
                touched: vec![t1, t2],
            });
        }
    }

    // Confirm the slack-judged pairs best-first; the first pair whose two
    // routes evaluate cleanly is the best feasible swap.
    if scands.is_empty() {
        return;
    }
    scands.sort_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap());
    for sc in &scands {
        if best.as_ref().map_or(false, |b| sc.delta >= b.delta - 1e-12) {
            break;
        }
        let route2 = &sol.routes[sc.r2];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        let n2 = route2.steps.len();
        let t2 = route2.steps[sc.j];
        let mut cand1: Vec<TaskRef> = Vec::with_capacity(n1);
        cand1.extend_from_slice(&r1_minus[..sc.pos1]);
        cand1.push(t2);
        cand1.extend_from_slice(&r1_minus[sc.pos1..]);
        let mut cand2: Vec<TaskRef> = Vec::with_capacity(n2);
        cand2.extend_from_slice(&route2.steps[..sc.j]);
        cand2.extend_from_slice(&route2.steps[sc.j + 1..]);
        cand2.insert(sc.pos2, t1);
        op_confirm_eval(OP_SWAP_STAR);
        let Ok(m1) = evaluate_route(problem, matrix, veh1, &cand1) else { continue };
        op_confirm_eval(OP_SWAP_STAR);
        let Ok(m2) = evaluate_route(problem, matrix, veh2, &cand2) else { continue };
        let delta = (m1.cost - route1.metrics.cost) + (m2.cost - route2.metrics.cost);
        consider(best, Move {
            delta,
            route_updates: vec![(r1, Some((cand1, m1))), (sc.r2, Some((cand2, m2)))],
            touched: vec![t1, t2],
        });
        break;
    }
}

/// SwapStar helper: the best insertion position of `task` into `base` judged
/// purely by O(1) slack checks — positions ranked by exact arc-cost delta,
/// first slack-feasible wins. Mirrors `best_insertion`'s choice (which the
/// confirm evaluation still validates) without any route evaluation.
fn best_feasible_insertion_delta(
    problem: &Problem,
    matrix: &Matrix,
    veh: &crate::problem::Vehicle,
    base: &[TaskRef],
    task: TaskRef,
    slack_base: &crate::slack::RouteSlack,
) -> Option<(usize, f64)> {
    let tloc = step_loc(problem, task)?;
    // O(dim) whole-route load filter: if the task provably fits at NO
    // position, skip the full position scan.
    if !slack_base.task_load_fits(veh, problem, task) {
        op_slack_skip(OP_SWAP_STAR);
        return None;
    }
    let c = cost_coef(veh);
    let nb = base.len();
    let tserv = task.description(problem).service as f64 * 1e-6;
    let ds = depot_start(veh);
    let de = depot_end(veh);
    INS_ORDER.with(|cell| {
        let mut order = cell.borrow_mut();
        order.clear();
        order.extend((0..=nb).map(|pos| {
            let prev = if pos == 0 { ds } else { step_loc(problem, base[pos - 1]) };
            let next = if pos == nb { de } else { step_loc(problem, base[pos]) };
            let added = prev.map_or(0.0, |p| arc_cost(matrix, c, p, tloc))
                + next.map_or(0.0, |n| arc_cost(matrix, c, tloc, n));
            let removed = match (prev, next) {
                (Some(p), Some(n)) => arc_cost(matrix, c, p, n),
                _ => 0.0,
            };
            (added - removed + tserv, pos)
        }));
        order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let seg = [task];
        for &(d, pos) in order.iter() {
            if slack_base.replace_seg_ok(problem, matrix, veh, pos, pos, &seg) {
                return Some((pos, d));
            }
            op_slack_skip(OP_SWAP_STAR);
        }
        None
    })
}

/// Helper for SwapStar: try each insertion position for `task` in `base`,
/// return the one with the lowest evaluated cost (or None if none feasible).
fn best_insertion(
    problem: &Problem,
    matrix: &Matrix,
    veh: &crate::problem::Vehicle,
    base: &[TaskRef],
    task: TaskRef,
    slack_base: Option<&crate::slack::RouteSlack>,
) -> Option<(Vec<TaskRef>, RouteMetrics)> {
    // Fast path: rank positions by exact O(1) insertion cost-delta, then confirm
    // feasibility best-first — the first feasible position is the cheapest one.
    if fast_ls_enabled() && fast_cost_eligible(problem) {
        if let Some(tloc) = step_loc(problem, task) {
            let c = cost_coef(veh);
            let nb = base.len();
            let tserv = task.description(problem).service as f64 * 1e-6;
            let ds = depot_start(veh);
            let de = depot_end(veh);
            let mut order: Vec<(f64, usize)> = (0..=nb)
                .map(|pos| {
                    let prev = if pos == 0 { ds } else { step_loc(problem, base[pos - 1]) };
                    let next = if pos == nb { de } else { step_loc(problem, base[pos]) };
                    let added = prev.map_or(0.0, |p| arc_cost(matrix, c, p, tloc))
                        + next.map_or(0.0, |n| arc_cost(matrix, c, tloc, n));
                    let removed = match (prev, next) {
                        (Some(p), Some(n)) => arc_cost(matrix, c, p, n),
                        _ => 0.0,
                    };
                    (added - removed + tserv, pos)
                })
                .collect();
            order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let seg = [task];
            for (_, pos) in order {
                // O(1) slack prefilter: positions that provably miss a time
                // window or a load checkpoint are skipped without evaluation.
                if let Some(s) = slack_base {
                    if !s.replace_seg_ok(problem, matrix, veh, pos, pos, &seg) {
                        op_slack_skip(OP_SWAP_STAR);
                        continue;
                    }
                }
                let mut cand: Vec<TaskRef> = Vec::with_capacity(nb + 1);
                cand.extend_from_slice(&base[..pos]);
                cand.push(task);
                cand.extend_from_slice(&base[pos..]);
                op_confirm_eval(OP_SWAP_STAR);
                if let Ok(m) = evaluate_route(problem, matrix, veh, &cand) {
                    return Some((cand, m));
                }
            }
            return None;
        }
    }
    let mut best: Option<(Vec<TaskRef>, RouteMetrics)> = None;
    for pos in 0..=base.len() {
        let mut cand: Vec<TaskRef> = Vec::with_capacity(base.len() + 1);
        cand.extend_from_slice(&base[..pos]);
        cand.push(task);
        cand.extend_from_slice(&base[pos..]);
        if let Ok(m) = evaluate_route(problem, matrix, veh, &cand) {
            if best.as_ref().map_or(true, |(_, bm)| m.cost < bm.cost) {
                best = Some((cand, m));
            }
        }
    }
    best
}

/// Cross-exchange: swap a length-2 segment from r1 (containing i) with a
/// length-2 segment from r2. (Segments of length 1 are covered by single
/// exchange; longer segments locked us into worse local optima in earlier
/// experiments. Length 2 untangles two-edge crossings between routes.)
fn try_cross_exchange_with(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    granular: Option<&Granular>,
    judge: Option<&mut Judge>,
    best: &mut Option<Move>,
) {
    // Large-N guard: see try_two_opt_star — slow path above matrix.n ≥ 500
    // keeps the N=1000 headline byte-identical.
    if granular.is_some() && fast_ls_enabled() && fast_cost_eligible(problem) && matrix.n < 500 {
        try_cross_exchange_fast(problem, matrix, sol, r1, i, granular.unwrap(), judge, best);
        return;
    }
    try_cross_exchange_slow(problem, matrix, sol, r1, i, granular, best);
}

/// Fast cross-exchange: swapping the two length-2 segments changes exactly six
/// arcs (the boundary arcs around each gap plus each segment's internal arc,
/// now charged on the other route) and trades the segments' service tie-break
/// terms. Both routes keep their prefix/suffix and depot legs, so unlike
/// 2-opt* no same-depot gate is needed — homogeneous cost coefficients (the
/// `fast_cost_eligible` gate) suffice. Rank by exact delta, confirm lazily.
fn try_cross_exchange_fast(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    granular: &Granular,
    judge: Option<&mut Judge>,
    best: &mut Option<Move>,
) {
    let mut judge = judge;
    let n_routes = sol.routes.len();
    let route1 = &sol.routes[r1];
    let n1 = route1.steps.len();
    if i + 2 > n1 {
        return;
    }
    let veh1 = &problem.vehicles[route1.vehicle_idx];
    let c = cost_coef(veh1); // homogeneous
    let (Some(s1a), Some(s1b)) = (
        step_loc(problem, route1.steps[i]),
        step_loc(problem, route1.steps[i + 1]),
    ) else {
        return;
    };
    nmark_set(granular, s1a, matrix.n);
    let prev1 = if i > 0 { step_loc(problem, route1.steps[i - 1]) } else { depot_start(veh1) };
    let nxt1 = if i + 2 < n1 { step_loc(problem, route1.steps[i + 2]) } else { depot_end(veh1) };
    let seg1_service: f64 = route1.steps[i..i + 2]
        .iter()
        .map(|t| t.description(problem).service as f64 * 1e-6)
        .sum();
    // Arcs r1 loses around its gap (boundary + internal).
    let r1_old = match prev1 {
        Some(p) => arc_cost(matrix, c, p, s1a),
        None => 0.0,
    } + arc_cost(matrix, c, s1a, s1b)
        + match nxt1 {
            Some(n) => arc_cost(matrix, c, s1b, n),
            None => 0.0,
        };

    struct Cand {
        delta: f64,
        r2: usize,
        j: usize,
    }
    let mut cands: Vec<Cand> = Vec::new();
    if let Some(Judge::Warp(cache, _)) = judge.as_deref_mut() {
        cache.ensure(r1, problem, matrix, sol);
    }

    for r2 in 0..n_routes {
        if r2 == r1 {
            continue;
        }
        let route2 = &sol.routes[r2];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        let n2 = route2.steps.len();
        if 2 > n2 {
            continue;
        }
        // Warp mode: both penalty deltas are RELATIVE (confirm subtracts both
        // routes' penalised metrics).
        if let Some(Judge::Warp(cache, _)) = judge.as_deref_mut() {
            cache.ensure(r2, problem, matrix, sol);
        }
        let (warp1, warp2, wsw) = match judge.as_deref() {
            Some(Judge::Warp(cache, sw)) => (cache.get(r1), cache.get(r2), Some(*sw)),
            _ => (None, None, None),
        };
        for j in 0..=n2 - 2 {
            let (Some(s2a), Some(s2b)) = (
                step_loc(problem, route2.steps[j]),
                step_loc(problem, route2.steps[j + 1]),
            ) else {
                continue;
            };
            // Granular parity: at least one of seg2's locations near the anchor.
            if !nmark_has(s2a) && !nmark_has(s2b) {
                continue;
            }
            let prev2 = if j > 0 { step_loc(problem, route2.steps[j - 1]) } else { depot_start(veh2) };
            let nxt2 = if j + 2 < n2 { step_loc(problem, route2.steps[j + 2]) } else { depot_end(veh2) };
            let seg2_service: f64 = route2.steps[j..j + 2]
                .iter()
                .map(|t| t.description(problem).service as f64 * 1e-6)
                .sum();
            let r2_old = match prev2 {
                Some(p) => arc_cost(matrix, c, p, s2a),
                None => 0.0,
            } + arc_cost(matrix, c, s2a, s2b)
                + match nxt2 {
                    Some(n) => arc_cost(matrix, c, s2b, n),
                    None => 0.0,
                };
            // seg2 spliced into r1's gap, seg1 into r2's gap.
            let r1_new = match prev1 {
                Some(p) => arc_cost(matrix, c, p, s2a),
                None => 0.0,
            } + arc_cost(matrix, c, s2a, s2b)
                + match nxt1 {
                    Some(n) => arc_cost(matrix, c, s2b, n),
                    None => 0.0,
                };
            let r2_new = match prev2 {
                Some(p) => arc_cost(matrix, c, p, s1a),
                None => 0.0,
            } + arc_cost(matrix, c, s1a, s1b)
                + match nxt2 {
                    Some(n) => arc_cost(matrix, c, s1b, n),
                    None => 0.0,
                };
            // Service tie-break swaps with the segments; everything else cancels.
            let mut delta = (r1_new - r1_old + seg2_service - seg1_service)
                + (r2_new - r2_old + seg1_service - seg2_service);
            if let (Some(w1), Some(w2), Some(sw)) = (warp1, warp2, wsw) {
                if let (Some(p1), Some(p2)) = (
                    w1.penalty_delta(problem, matrix, sw, i, i + 2, &route2.steps[j..j + 2]),
                    w2.penalty_delta(problem, matrix, sw, j, j + 2, &route1.steps[i..i + 2]),
                ) {
                    delta += p1 + p2;
                }
            }
            if delta < -1e-9 {
                cands.push(Cand { delta, r2, j });
            }
        }
    }

    if cands.is_empty() {
        return;
    }
    cands.sort_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap());

    let seg1: Vec<TaskRef> = route1.steps[i..i + 2].to_vec();
    for cand in &cands {
        if best.as_ref().map_or(false, |b| cand.delta >= b.delta - 1e-12) {
            break;
        }
        let route2 = &sol.routes[cand.r2];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        let n2 = route2.steps.len();
        let j = cand.j;
        let seg2: &[TaskRef] = &route2.steps[j..j + 2];
        if !seg1.iter().all(|t| veh2.has_skills(t.skills(problem))) {
            continue;
        }
        if !seg2.iter().all(|t| veh1.has_skills(t.skills(problem))) {
            continue;
        }
        // O(1) slack prefilter: seg2 replaces [i, i+2) in r1, seg1 replaces
        // [j, j+2) in r2.
        if let Some(Judge::Slack(cache)) = judge.as_deref_mut() {
            cache.ensure(r1, problem, matrix, sol);
            cache.ensure(cand.r2, problem, matrix, sol);
            let ok = match (cache.get(r1), cache.get(cand.r2)) {
                (Some(s1), Some(s2)) => {
                    s1.replace_seg_ok(problem, matrix, veh1, i, i + 2, seg2)
                        && s2.replace_seg_ok(problem, matrix, veh2, j, j + 2, &seg1)
                }
                _ => true,
            };
            if !ok {
                op_slack_skip(OP_CROSS);
                continue;
            }
        }
        let mut cand1: Vec<TaskRef> = Vec::with_capacity(n1);
        cand1.extend_from_slice(&route1.steps[..i]);
        cand1.extend_from_slice(seg2);
        cand1.extend_from_slice(&route1.steps[i + 2..]);
        let mut cand2: Vec<TaskRef> = Vec::with_capacity(n2);
        cand2.extend_from_slice(&route2.steps[..j]);
        cand2.extend_from_slice(&seg1);
        cand2.extend_from_slice(&route2.steps[j + 2..]);
        op_confirm_eval(OP_CROSS);
        let m1 = match evaluate_route(problem, matrix, veh1, &cand1) {
            Ok(m) => m,
            Err(_) => continue,
        };
        op_confirm_eval(OP_CROSS);
        let m2 = match evaluate_route(problem, matrix, veh2, &cand2) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let delta = (m1.cost - route1.metrics.cost) + (m2.cost - route2.metrics.cost);
        let mut touched = seg1.clone();
        touched.extend_from_slice(seg2);
        consider(best, Move {
            delta,
            route_updates: vec![(r1, Some((cand1, m1))), (cand.r2, Some((cand2, m2)))],
            touched,
        });
        break;
    }
}

fn try_cross_exchange_slow(
    problem: &Problem,
    matrix: &Matrix,
    sol: &Solution,
    r1: usize,
    i: usize,
    granular: Option<&Granular>,
    best: &mut Option<Move>,
) {
    let n_routes = sol.routes.len();
    let route1 = &sol.routes[r1];
    let n1 = route1.steps.len();
    if i + 2 > n1 { return; }
    let veh1 = &problem.vehicles[route1.vehicle_idx];

    let anchor = route1.steps[i].description(problem).location.index;
    let has_gran = match (granular, anchor) {
        (Some(g), Some(loc)) => { nmark_set(g, loc, matrix.n); true }
        _ => false,
    };

    let seg1: Vec<TaskRef> = route1.steps[i..i + 2].to_vec();

    for r2 in 0..n_routes {
        if r2 == r1 { continue; }
        let route2 = &sol.routes[r2];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        let n2 = route2.steps.len();
        if 2 > n2 { continue; }

        if !seg1.iter().all(|t| veh2.has_skills(t.skills(problem))) { continue; }

        for j in 0..=n2 - 2 {
            let seg2: &[TaskRef] = &route2.steps[j..j + 2];

            // Granularity: at least one of seg2's locations should be near the anchor.
            if has_gran {
                let any_near = seg2.iter().any(|t| {
                    t.description(problem).location.index.map_or(false, nmark_has)
                });
                if !any_near { continue; }
            }

            if !seg2.iter().all(|t| veh1.has_skills(t.skills(problem))) { continue; }

            let mut cand1: Vec<TaskRef> = Vec::with_capacity(n1);
            cand1.extend_from_slice(&route1.steps[..i]);
            cand1.extend_from_slice(seg2);
            cand1.extend_from_slice(&route1.steps[i + 2..]);

            let mut cand2: Vec<TaskRef> = Vec::with_capacity(n2);
            cand2.extend_from_slice(&route2.steps[..j]);
            cand2.extend_from_slice(&seg1);
            cand2.extend_from_slice(&route2.steps[j + 2..]);

            let m1 = match evaluate_route(problem, matrix, veh1, &cand1) {
                Ok(m) => m, Err(_) => continue,
            };
            let m2 = match evaluate_route(problem, matrix, veh2, &cand2) {
                Ok(m) => m, Err(_) => continue,
            };
            let delta = (m1.cost - route1.metrics.cost)
                      + (m2.cost - route2.metrics.cost);

            let mut touched = seg1.clone();
            touched.extend_from_slice(seg2);
            consider(best, Move {
                delta,
                route_updates: vec![
                    (r1, Some((cand1, m1))),
                    (r2, Some((cand2, m2))),
                ],
                touched,
            });
        }
    }
}

/// Tour-split pass: try to split long routes in two by assigning the second
/// half to an unused vehicle. Repeats until no improving split exists.
///
/// **Status: tested but unused.** When wired into the multi-start loop, this
/// produced a -0.7% gain on N=500 but +0.6% regression on N=1000 (post-LS,
/// post-split route structures landed re-LS on worse local optima than the
/// no-split path). A "revert if regressed" wrapper helped but didn't fully
/// eliminate the loss. Kept as dead code (`#[allow(dead_code)]`) for future
/// integration with a smarter accept/reject (eg. compare best-of-K with vs.
/// without and keep the winner).
///
/// Cost gain comes from: shorter individual routes finish their TWs with
/// less waiting / less violation. Cost penalty: extra fixed-vehicle cost
/// (Solomon instances have 0 fixed cost; real problems may not).
///
/// Cost: O(R · L · V_unused · L) per pass — bounded by pre-screening to
/// only consider routes ≥ `min_route_len`.
pub fn route_split_pass(
    problem: &Problem,
    matrix: &Matrix,
    sol: &mut Solution,
    min_route_len: usize,
) {
    loop {
        let used: HashSet<usize> = sol.routes.iter().map(|r| r.vehicle_idx).collect();
        let unused: Vec<usize> = (0..problem.vehicles.len())
            .filter(|v| !used.contains(v))
            .collect();
        if unused.is_empty() { return; }

        // (r1_idx, part1_steps, m1, v2_idx, part2_steps, m2, delta)
        let mut best: Option<(usize, Vec<TaskRef>, RouteMetrics, usize, Vec<TaskRef>, RouteMetrics, f64)> = None;

        for r1_idx in 0..sol.routes.len() {
            let route = &sol.routes[r1_idx];
            if route.steps.len() < min_route_len { continue; }
            let veh1 = &problem.vehicles[route.vehicle_idx];
            let cur_cost = route.metrics.cost;

            // split_at = number of stops kept in part1; both halves must be non-empty.
            for split_at in 1..route.steps.len() {
                let part1: Vec<TaskRef> = route.steps[..split_at].to_vec();
                let part2: Vec<TaskRef> = route.steps[split_at..].to_vec();

                let m1 = match evaluate_route(problem, matrix, veh1, &part1) {
                    Ok(m) => m, Err(_) => continue,
                };

                for &v2_idx in &unused {
                    let veh2 = &problem.vehicles[v2_idx];
                    if !part2.iter().all(|t| veh2.has_skills(t.skills(problem))) { continue; }
                    let m2 = match evaluate_route(problem, matrix, veh2, &part2) {
                        Ok(m) => m, Err(_) => continue,
                    };
                    let delta = (m1.cost + m2.cost) - cur_cost;
                    if delta < -1e-9 && best.as_ref().map_or(true, |b| delta < b.6) {
                        best = Some((r1_idx, part1.clone(), m1, v2_idx, part2.clone(), m2, delta));
                    }
                }
            }
        }

        if let Some((r1_idx, p1, m1, v2_idx, p2, m2, _)) = best {
            sol.routes[r1_idx].steps = p1;
            sol.routes[r1_idx].metrics = m1;
            sol.routes.push(Route {
                vehicle_idx: v2_idx,
                steps: p2,
                metrics: m2,
            });
        } else {
            break;
        }
    }
    sol.recompute_summary(problem);
}
