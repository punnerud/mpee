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
/// Master switch for the fast LS path (off via BROOOM_NO_FAST_LS, for A/B).
fn fast_ls_enabled() -> bool {
    std::env::var("BROOOM_NO_FAST_LS").is_err()
}

fn fast_cost_eligible(problem: &Problem) -> bool {
    if crate::dimension::has_dimensions() {
        return false;
    }
    if crate::solution::soft_is_active() {
        return false;
    }
    if problem.any_multi_trip() {
        return false;
    }
    problem.vehicles.iter().all(|v| v.span_cost.max(0.0) == 0.0)
}

/// Matrix index of a step's location.
#[inline]
fn step_loc(problem: &Problem, t: TaskRef) -> Option<usize> {
    t.description(problem).location.index
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
    let mut settled: HashSet<TaskRef> = HashSet::new();

    'outer: for _ in 0..max_passes {
        let task_seq: Vec<TaskRef> = sol.routes.iter().flat_map(|r| r.steps.iter().copied()).collect();

        let mut any_change = false;

        for task in task_seq {
            if settled.contains(&task) { continue; }

            let Some((r1, i)) = locate(sol, task) else { continue; };

            let mut best: Option<Move> = None;

            try_relocate_task(problem, matrix, sol, r1, i, granular, &mut best);
            try_two_opt_through(problem, matrix, sol, r1, i, &mut best);
            try_exchange_with(problem, matrix, sol, r1, i, granular, &mut best);
            try_two_opt_star(problem, matrix, sol, r1, i, granular, &mut best);
            try_cross_exchange_with(problem, matrix, sol, r1, i, granular, &mut best);
            try_swap_star(problem, matrix, sol, r1, i, granular, &mut best);

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
    'outer: for _ in 0..max_passes {
        let task_seq: Vec<TaskRef> = sol.routes.iter().flat_map(|r| r.steps.iter().copied()).collect();
        let mut any_change = false;

        for task in task_seq {
            let Some((r1, i)) = locate(sol, task) else { continue; };

            let mut best: Option<Move> = None;
            try_relocate_task(problem, matrix, sol, r1, i, granular, &mut best);
            try_two_opt_through(problem, matrix, sol, r1, i, &mut best);
            try_exchange_with(problem, matrix, sol, r1, i, granular, &mut best);
            try_two_opt_star(problem, matrix, sol, r1, i, granular, &mut best);
            try_cross_exchange_with(problem, matrix, sol, r1, i, granular, &mut best);
            try_swap_star(problem, matrix, sol, r1, i, granular, &mut best);

            if let Some(mv) = best {
                if mv.delta < -1e-9 {
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
    best: &mut Option<Move>,
) {
    // Fast O(1) cost-delta path: enumerate candidates by exact edge-delta cost,
    // then confirm feasibility lazily best-first (the first feasible candidate is
    // the best feasible move). Only when the arc-cost model is exact.
    if granular.is_some() && fast_ls_enabled() && fast_cost_eligible(problem) {
        try_relocate_task_fast(problem, matrix, sol, r1, i, granular.unwrap(), best);
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
    best: &mut Option<Move>,
) {
    let route1 = &sol.routes[r1];
    let n1 = route1.steps.len();
    let veh1 = &problem.vehicles[route1.vehicle_idx];
    let c1 = cost_coef(veh1);

    struct Cand {
        delta: f64,
        seg_len: usize,
        r2: usize,
        j: usize,
        rev: bool,
    }
    let mut cands: Vec<Cand> = Vec::new();
    // Per seg_len: the route1-after-removal step list, its metrics, feasibility.
    let mut r1_after_cache: Vec<Option<(Vec<TaskRef>, Option<RouteMetrics>, f64)>> = vec![None; 4];

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
        let m1_after = if route1_after.is_empty() {
            Some(RouteMetrics::default())
        } else {
            evaluate_route(problem, matrix, veh1, &route1_after).ok()
        };
        if m1_after.is_none() {
            continue; // removal infeasible (e.g. split a shipment) — skip seg_len
        }
        let cost_r1_after = route_arc_cost(problem, matrix, veh1, &route1_after);
        let delta_r1 = cost_r1_after - route1.metrics.cost;
        r1_after_cache[seg_len] = Some((route1_after.clone(), m1_after, delta_r1));

        let seg_neighbors: std::collections::HashSet<usize> = granular.neighbors(seg0).collect();

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

            // Candidate positions: granular (adjacent to a top-K neighbour of
            // seg0) plus the two extremes — mirrors the slow path.
            let mut positions: Vec<usize> = Vec::new();
            for j in 0..=nb {
                let prev_loc = if j == 0 { None } else { step_loc(problem, base[j - 1]) };
                let next_loc = if j == nb { None } else { step_loc(problem, base[j]) };
                if prev_loc.map_or(false, |l| seg_neighbors.contains(&l))
                    || next_loc.map_or(false, |l| seg_neighbors.contains(&l))
                {
                    positions.push(j);
                }
            }
            if !positions.contains(&0) {
                positions.push(0);
            }
            if !positions.contains(&nb) {
                positions.push(nb);
            }

            for j in positions {
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
                    let delta = delta_r1 + delta_r2;
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
    for cand in &cands {
        if best.as_ref().map_or(false, |b| cand.delta >= b.delta - 1e-12) {
            break; // can't beat the incumbent from here on (sorted)
        }
        let (route1_after, m1_after, _) = r1_after_cache[cand.seg_len].as_ref().unwrap();
        let segment: Vec<TaskRef> = route1.steps[i..i + cand.seg_len].to_vec();
        let mut seg_oriented = segment.clone();
        if cand.rev {
            seg_oriented.reverse();
        }
        let route2 = &sol.routes[cand.r2];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        let base: &[TaskRef] = if r1 == cand.r2 { route1_after } else { &route2.steps };
        let mut cand2 = Vec::with_capacity(base.len() + cand.seg_len);
        cand2.extend_from_slice(&base[..cand.j]);
        cand2.extend_from_slice(&seg_oriented);
        cand2.extend_from_slice(&base[cand.j..]);
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
            let m1 = m1_after.clone().unwrap();
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
    best: &mut Option<Move>,
) {
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
            for a in 0..n - 1 {
                if a > i { break; } // need a ≤ i
                let prev = if a == 0 { ds } else { Some(loc(a - 1)) };
                let mut fwd = 0.0; // forward internal arc sum of [a..b]
                let mut rev = 0.0; // reversed internal arc sum
                for b in a + 1..n {
                    // extend the window to include edge (b-1,b)
                    fwd += arc_cost(matrix, c, loc(b - 1), loc(b));
                    rev += arc_cost(matrix, c, loc(b), loc(b - 1));
                    if b < i { continue; } // need b ≥ i
                    let next = if b + 1 >= n { de } else { Some(loc(b + 1)) };
                    let old_e = prev.map_or(0.0, |p| arc_cost(matrix, c, p, loc(a)))
                        + next.map_or(0.0, |nx| arc_cost(matrix, c, loc(b), nx));
                    let new_e = prev.map_or(0.0, |p| arc_cost(matrix, c, p, loc(b)))
                        + next.map_or(0.0, |nx| arc_cost(matrix, c, loc(a), nx));
                    let delta = (new_e + rev) - (old_e + fwd);
                    if delta < -1e-9 {
                        cands.push(C2 { delta, a, b });
                    }
                }
            }
            if cands.is_empty() {
                return;
            }
            cands.sort_by(|x, y| x.delta.partial_cmp(&y.delta).unwrap());
            for cd in &cands {
                if best.as_ref().map_or(false, |bm| cd.delta >= bm.delta - 1e-12) {
                    break;
                }
                let mut cand = route.steps.clone();
                cand[cd.a..=cd.b].reverse();
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
    best: &mut Option<Move>,
) {
    let n_routes = sol.routes.len();
    let a = sol.routes[r1].steps[i];
    let a_loc = a.description(problem).location.index;
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
    best: &mut Option<Move>,
) {
    let n_routes = sol.routes.len();
    let route1 = &sol.routes[r1];
    let veh1 = &problem.vehicles[route1.vehicle_idx];
    let n1 = route1.steps.len();
    let anchor = route1.steps[i].description(problem).location.index;
    let neighbor_set: Option<HashSet<usize>> = match (granular, anchor) {
        (Some(g), Some(loc)) => Some(g.neighbors(loc).collect()),
        _ => None,
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
            if let Some(set) = &neighbor_set {
                let r2_j_loc = if j < n2 {
                    route2.steps[j].description(problem).location.index
                } else { None };
                let near = r2_j_loc.map_or(false, |l| set.contains(&l));
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
    best: &mut Option<Move>,
) {
    let n_routes = sol.routes.len();
    let route1 = &sol.routes[r1];
    let n1 = route1.steps.len();
    let veh1 = &problem.vehicles[route1.vehicle_idx];
    let t1 = route1.steps[i];
    let t1_loc = t1.description(problem).location.index;

    let neighbor_set: Option<HashSet<usize>> = match (granular, t1_loc) {
        (Some(g), Some(loc)) => Some(g.neighbors(loc).collect()),
        _ => None,
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

    for r2 in 0..n_routes {
        if r2 == r1 { continue; }
        let route2 = &sol.routes[r2];
        let veh2 = &problem.vehicles[route2.vehicle_idx];
        let n2 = route2.steps.len();
        if n2 == 0 { continue; }
        if !veh2.has_skills(t1.skills(problem)) { continue; }

        for j in 0..n2 {
            let t2 = route2.steps[j];

            // Granular filter: skip pairs where t2 is not near t1.
            if let Some(set) = &neighbor_set {
                let l = t2.description(problem).location.index;
                if l.map_or(true, |l| !set.contains(&l)) { continue; }
            }
            if !veh1.has_skills(t2.skills(problem)) { continue; }

            // Lower-bound prune: (cheapest feasibility-ignoring inserts) +
            // (exact removals) ≤ true pair delta. Skip if it can't beat the
            // incumbent. Never prunes an improving move ⇒ equivalence preserved.
            if prefilter {
                let mut r2_minus_tmp: Vec<TaskRef> = Vec::with_capacity(n2 - 1);
                r2_minus_tmp.extend_from_slice(&route2.steps[..j]);
                r2_minus_tmp.extend_from_slice(&route2.steps[j + 1..]);
                let rem2 = pred_remove_delta(problem, matrix, veh2, &route2.steps, j);
                let lb = rem1 + pred_cheapest_insert(problem, matrix, veh1, &r1_minus, t2)
                    + rem2 + pred_cheapest_insert(problem, matrix, veh2, &r2_minus_tmp, t1);
                let thresh = best.as_ref().map(|b| b.delta).unwrap_or(-1e-9);
                if lb >= thresh - 1e-12 {
                    continue;
                }
            }

            // Best position for t2 in r1\{t1}
            let best1 = best_insertion(problem, matrix, veh1, &r1_minus, t2);
            let Some((cand1, m1)) = best1 else { continue };

            // r2\{t2}
            let mut r2_minus: Vec<TaskRef> = Vec::with_capacity(n2 - 1);
            r2_minus.extend_from_slice(&route2.steps[..j]);
            r2_minus.extend_from_slice(&route2.steps[j + 1..]);

            // Best position for t1 in r2\{t2}
            let best2 = best_insertion(problem, matrix, veh2, &r2_minus, t1);
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
}

/// Helper for SwapStar: try each insertion position for `task` in `base`,
/// return the one with the lowest evaluated cost (or None if none feasible).
fn best_insertion(
    problem: &Problem,
    matrix: &Matrix,
    veh: &crate::problem::Vehicle,
    base: &[TaskRef],
    task: TaskRef,
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
            for (_, pos) in order {
                let mut cand: Vec<TaskRef> = Vec::with_capacity(nb + 1);
                cand.extend_from_slice(&base[..pos]);
                cand.push(task);
                cand.extend_from_slice(&base[pos..]);
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
    best: &mut Option<Move>,
) {
    let n_routes = sol.routes.len();
    let route1 = &sol.routes[r1];
    let n1 = route1.steps.len();
    if i + 2 > n1 { return; }
    let veh1 = &problem.vehicles[route1.vehicle_idx];

    let anchor = route1.steps[i].description(problem).location.index;
    let neighbor_set: Option<HashSet<usize>> = match (granular, anchor) {
        (Some(g), Some(loc)) => Some(g.neighbors(loc).collect()),
        _ => None,
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
            if let Some(set) = &neighbor_set {
                let any_near = seg2.iter().any(|t| {
                    t.description(problem).location.index.map_or(false, |l| set.contains(&l))
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
