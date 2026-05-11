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
    sol.recompute_summary();
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
    sol.recompute_summary();
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
    sol.recompute_summary();
}
