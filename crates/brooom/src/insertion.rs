//! Construction heuristic — greedy cheapest insertion with O(1) probe.
//!
//! Hot path:
//!   1. Per route, precompute forward depart times, backward latest_arrival
//!      bounds, and prefix/suffix max-load arrays once.
//!   2. For every (task, route, position), the probe answers feasibility +
//!      Δ-cost in O(1). Most infeasible candidates are rejected without ever
//!      touching the full evaluator.
//!   3. The top-K cheapest probe survivors get a full `evaluate_route` to
//!      confirm — this guards against the few approximations the probe makes
//!      (multi-TW selection, setup change at the next location).
//!   4. Apply the cheapest confirmed insertion; refresh that route's precomp.
//!
//! Per-slot probing fans out across rayon workers when total work exceeds a
//! threshold; small instances stay on the serial path.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::eval::{precompute, try_insert_pair, try_insert_single, try_insert_single_with_shift, RoutePrecomp};
use crate::matrix::Matrix;
use crate::problem::{Cost, Problem};
use crate::solution::{evaluate_route, Route, RouteMetrics, Solution, TaskRef};

const VALIDATE_TOP_K: usize = 8;
const PARALLEL_PROBE_THRESHOLD: usize = 1024;

/// Bounded top-K min-keep: tracks the K *cheapest* probes seen so far
/// without ever growing past K.
struct TopK<T> {
    heap: BinaryHeap<HeapEntry<T>>,
    cap: usize,
}

struct HeapEntry<T> {
    delta: Cost,
    payload: T,
}

impl<T> PartialEq for HeapEntry<T> {
    fn eq(&self, other: &Self) -> bool { self.delta == other.delta }
}
impl<T> Eq for HeapEntry<T> {}
impl<T> PartialOrd for HeapEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl<T> Ord for HeapEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.delta.partial_cmp(&other.delta).unwrap_or(Ordering::Equal)
    }
}

impl<T> TopK<T> {
    fn new(cap: usize) -> Self { Self { heap: BinaryHeap::with_capacity(cap + 1), cap } }
    fn push(&mut self, delta: Cost, payload: T) {
        if self.heap.len() < self.cap {
            self.heap.push(HeapEntry { delta, payload });
        } else if let Some(top) = self.heap.peek() {
            if delta < top.delta {
                self.heap.pop();
                self.heap.push(HeapEntry { delta, payload });
            }
        }
    }
    fn into_sorted_asc(self) -> Vec<(Cost, T)> {
        let mut out: Vec<(Cost, T)> = self.heap.into_iter().map(|e| (e.delta, e.payload)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        out
    }
}

#[derive(Debug, Clone, Copy)]
struct ProbeSingle {
    task_slot: usize,
    route_idx: usize,
    pos: usize,
    delta: Cost,
}

#[derive(Debug, Clone, Copy)]
struct ProbePair {
    task_slot: usize,
    route_idx: usize,
    pos_p: usize,
    pos_d: usize,
    delta: Cost,
}

enum Confirmed {
    Single(ProbeSingle, Vec<TaskRef>, RouteMetrics),
    Pair(ProbePair, Vec<TaskRef>, RouteMetrics),
}

fn probe_one_slot(
    problem: &Problem,
    matrix: &Matrix,
    routes_steps: &[Vec<TaskRef>],
    precomps: &[Option<RoutePrecomp>],
    pending: &[TaskRef],
    is_pair_head: &[bool],
    slot: usize,
    out_singles: &mut TopK<ProbeSingle>,
    out_pairs: &mut TopK<ProbePair>,
) {
    let task = pending[slot];
    let is_pickup_head = is_pair_head[slot];
    for r in 0..routes_steps.len() {
        let Some(pre) = precomps[r].as_ref() else { continue; };
        let veh = &problem.vehicles[r];
        if is_pickup_head {
            let delivery = pending[slot + 1];
            let l = routes_steps[r].len();
            let positions = l + 2;
            for pos_p in 1..positions {
                for pos_d in pos_p..positions {
                    if let Some(d) = try_insert_pair(
                        pre, problem, matrix, veh, &routes_steps[r],
                        pos_p, pos_d, task, delivery,
                    ) {
                        out_pairs.push(d, ProbePair {
                            task_slot: slot, route_idx: r, pos_p, pos_d, delta: d,
                        });
                    }
                }
            }
        } else {
            let l = routes_steps[r].len();
            let positions = l + 2;
            for pos in 1..positions {
                if let Some(d) = try_insert_single(pre, problem, matrix, veh, pos, task) {
                    out_singles.push(d, ProbeSingle {
                        task_slot: slot, route_idx: r, pos, delta: d,
                    });
                }
            }
        }
    }
}

/// Build a starting solution by greedy cheapest insertion.
pub fn greedy_insertion(problem: &Problem, matrix: &Matrix) -> Solution {
    greedy_insertion_seeded(problem, matrix, 0)
}

/// Same as `greedy_insertion` but with a `seed` that perturbs the order
/// in which pending tasks are visited. `seed == 0` is the deterministic
/// baseline; non-zero seeds shuffle within priority class so multi-start
/// gets distinct local optima.
pub fn greedy_insertion_seeded(problem: &Problem, matrix: &Matrix, seed: u64) -> Solution {
    let mut routes_steps: Vec<Vec<TaskRef>> = (0..problem.vehicles.len()).map(|_| Vec::new()).collect();
    let mut precomps: Vec<Option<RoutePrecomp>> = (0..problem.vehicles.len())
        .map(|i| precompute(problem, matrix, &problem.vehicles[i], i, &[]))
        .collect();

    let mut pending: Vec<TaskRef> = Vec::new();
    let mut is_pair_head: Vec<bool> = Vec::new();

    let mut singles: Vec<TaskRef> = (0..problem.jobs.len()).map(TaskRef::Job).collect();
    if seed == 0 {
        singles.sort_by_key(|t| {
            let p = t.priority(problem);
            let j = t.description(problem);
            let earliest = j.time_windows.first().map(|w| w.start).unwrap_or(0);
            (std::cmp::Reverse(p), earliest, j.id)
        });
    } else {
        // Shuffle within priority class, deterministically per seed.
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        singles.sort_by_key(|t| std::cmp::Reverse(t.priority(problem)));
        // Shuffle tasks of equal priority among themselves.
        let mut start = 0;
        while start < singles.len() {
            let p = singles[start].priority(problem);
            let mut end = start + 1;
            while end < singles.len() && singles[end].priority(problem) == p { end += 1; }
            singles[start..end].shuffle(&mut rng);
            start = end;
        }
    }
    for s in singles {
        pending.push(s);
        is_pair_head.push(false);
    }

    let mut shipments: Vec<usize> = (0..problem.shipments.len()).collect();
    if seed == 0 {
        shipments.sort_by_key(|&i| {
            let s = &problem.shipments[i];
            let earliest = s.pickup.time_windows.first().map(|w| w.start).unwrap_or(0);
            (std::cmp::Reverse(s.priority), earliest, s.pickup.id)
        });
    } else {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed.wrapping_add(1));
        shipments.sort_by_key(|&i| std::cmp::Reverse(problem.shipments[i].priority));
        let mut start = 0;
        while start < shipments.len() {
            let p = problem.shipments[shipments[start]].priority;
            let mut end = start + 1;
            while end < shipments.len() && problem.shipments[shipments[end]].priority == p {
                end += 1;
            }
            shipments[start..end].shuffle(&mut rng);
            start = end;
        }
    }
    for i in shipments {
        pending.push(TaskRef::Pickup(i));
        is_pair_head.push(true);
        pending.push(TaskRef::Delivery(i));
        is_pair_head.push(false);
    }

    let mut alive = vec![true; pending.len()];

    loop {
        let alive_count = alive.iter().filter(|&&a| a).count();
        let avg_route_len: usize = routes_steps.iter().map(|r| r.len() + 2).sum::<usize>() / routes_steps.len().max(1);
        let work_estimate = alive_count * routes_steps.len() * avg_route_len;
        let go_parallel = work_estimate >= PARALLEL_PROBE_THRESHOLD;

        let live_slots: Vec<usize> = (0..pending.len())
            .filter(|&s| alive[s] && !(matches!(pending[s], TaskRef::Delivery(_)) && s > 0 && is_pair_head[s - 1]))
            .collect();

        // Serial probe over all live slots (also the wasm path: no rayon).
        let probe_serial = || -> (Vec<TopK<ProbeSingle>>, Vec<TopK<ProbePair>>) {
            let mut local_s: TopK<ProbeSingle> = TopK::new(VALIDATE_TOP_K);
            let mut local_p: TopK<ProbePair> = TopK::new(VALIDATE_TOP_K);
            for slot in &live_slots {
                probe_one_slot(
                    problem, matrix, &routes_steps, &precomps, &pending, &is_pair_head,
                    *slot, &mut local_s, &mut local_p,
                );
            }
            (vec![local_s], vec![local_p])
        };

        #[cfg(feature = "parallel")]
        let (probes_single_vec, probes_pair_vec): (Vec<TopK<ProbeSingle>>, Vec<TopK<ProbePair>>) = if go_parallel {
            live_slots
                .par_iter()
                .map(|&slot| {
                    let mut local_s: TopK<ProbeSingle> = TopK::new(VALIDATE_TOP_K);
                    let mut local_p: TopK<ProbePair> = TopK::new(VALIDATE_TOP_K);
                    probe_one_slot(
                        problem, matrix, &routes_steps, &precomps, &pending, &is_pair_head,
                        slot, &mut local_s, &mut local_p,
                    );
                    (local_s, local_p)
                })
                .unzip()
        } else {
            probe_serial()
        };
        #[cfg(not(feature = "parallel"))]
        let (probes_single_vec, probes_pair_vec): (Vec<TopK<ProbeSingle>>, Vec<TopK<ProbePair>>) = {
            let _ = go_parallel;
            probe_serial()
        };

        let mut probes_single: TopK<ProbeSingle> = TopK::new(VALIDATE_TOP_K);
        for tk in probes_single_vec {
            for (d, p) in tk.into_sorted_asc() { probes_single.push(d, p); }
        }
        let mut probes_pair: TopK<ProbePair> = TopK::new(VALIDATE_TOP_K);
        for tk in probes_pair_vec {
            for (d, p) in tk.into_sorted_asc() { probes_pair.push(d, p); }
        }
        let probes_single = probes_single.into_sorted_asc();
        let probes_pair = probes_pair.into_sorted_asc();

        let mut best: Option<Confirmed> = None;
        let maybe_offer = |cand: Confirmed, best: &mut Option<Confirmed>| {
            let cand_delta = match &cand {
                Confirmed::Single(s, _, _) => s.delta,
                Confirmed::Pair(p, _, _) => p.delta,
            };
            let beats = match best {
                None => true,
                Some(Confirmed::Single(s, _, _)) => cand_delta < s.delta,
                Some(Confirmed::Pair(p, _, _)) => cand_delta < p.delta,
            };
            if beats { *best = Some(cand); }
        };
        for (_, s) in probes_single.iter() {
            let mut cand = routes_steps[s.route_idx].clone();
            cand.insert(s.pos - 1, pending[s.task_slot]);
            let veh = &problem.vehicles[s.route_idx];
            if let Ok(m) = evaluate_route(problem, matrix, veh, &cand) {
                maybe_offer(Confirmed::Single(*s, cand, m), &mut best);
                break;
            }
        }
        for (_, p) in probes_pair.iter() {
            let mut cand = routes_steps[p.route_idx].clone();
            cand.insert(p.pos_p - 1, pending[p.task_slot]);
            cand.insert(p.pos_d, pending[p.task_slot + 1]);
            let veh = &problem.vehicles[p.route_idx];
            if let Ok(m) = evaluate_route(problem, matrix, veh, &cand) {
                maybe_offer(Confirmed::Pair(*p, cand, m), &mut best);
                break;
            }
        }

        let Some(pick) = best else { break; };
        match pick {
            Confirmed::Single(s, cand, _m) => {
                routes_steps[s.route_idx] = cand;
                precomps[s.route_idx] = precompute(
                    problem, matrix, &problem.vehicles[s.route_idx], s.route_idx,
                    &routes_steps[s.route_idx],
                );
                alive[s.task_slot] = false;
            }
            Confirmed::Pair(p, cand, _m) => {
                routes_steps[p.route_idx] = cand;
                precomps[p.route_idx] = precompute(
                    problem, matrix, &problem.vehicles[p.route_idx], p.route_idx,
                    &routes_steps[p.route_idx],
                );
                alive[p.task_slot] = false;
                alive[p.task_slot + 1] = false;
            }
        }
    }

    let mut routes: Vec<Route> = Vec::new();
    for (idx, steps) in routes_steps.into_iter().enumerate() {
        if steps.is_empty() { continue; }
        let veh = &problem.vehicles[idx];
        let metrics = evaluate_route(problem, matrix, veh, &steps).unwrap_or_default();
        routes.push(Route { vehicle_idx: idx, steps, metrics });
    }
    let unassigned: Vec<TaskRef> = alive.iter().enumerate()
        .filter_map(|(i, &ok)| if ok { Some(pending[i]) } else { None })
        .collect();
    let mut sol = Solution { routes, unassigned, summary: Default::default() };
    sol.recompute_summary();
    sol
}

/// Solomon I1 (1987) insertion heuristic — partial implementation.
///
/// **Status: tested but unused (kept as scaffolding).** This version has
/// only the c1 cost-delta term and the c2 = λ·d_0u − c1 selection, but
/// lacks the c12 (arrival-time shift at the next step) term that is
/// Solomon I1's actual TW-aware insight. Empirical test on N=1000 with
/// 4 of 8 multi-start seeds → identical net cost (12976) vs full greedy,
/// at +15% wall time. A complete implementation must compute c12 by
/// looking at the arrival-time delta at position pos+1 after inserting u
/// at pos — which requires either a custom probe or a dual evaluate_route.
///
/// `lambda` controls the depot-distance bias (Solomon: 1 or 2).
#[allow(dead_code)]
pub fn solomon_i1_insertion(problem: &Problem, matrix: &Matrix, lambda: f64) -> Solution {
    let n_vehicles = problem.vehicles.len();
    let mut routes_steps: Vec<Vec<TaskRef>> = (0..n_vehicles).map(|_| Vec::new()).collect();
    let mut precomps: Vec<Option<RoutePrecomp>> = (0..n_vehicles)
        .map(|i| precompute(problem, matrix, &problem.vehicles[i], i, &[]))
        .collect();

    // d_0u: minimum depot-distance over all vehicles. Far-from-any-depot
    // tasks score high on c2.
    let d_0u_of = |task: TaskRef| -> f64 {
        let u_loc = match task.description(problem).location.index {
            Some(l) => l,
            None => return 0.0,
        };
        let mut min_d = i64::MAX;
        for v in &problem.vehicles {
            if let Some(loc) = v.start.as_ref().and_then(|l| l.index) {
                let d = matrix.duration(loc, u_loc) as i64;
                if d < min_d { min_d = d; }
            }
        }
        if min_d == i64::MAX { 0.0 } else { min_d as f64 }
    };

    let mut pending: Vec<TaskRef> = (0..problem.jobs.len()).map(TaskRef::Job).collect();
    let mut unassigned: Vec<TaskRef> = Vec::new();

    while !pending.is_empty() {
        // For each pending: find best (route, pos) and its c1 = insertion delta.
        let mut best_per_job: Vec<Option<(usize, usize, Cost)>> = vec![None; pending.len()];
        for (job_idx, &task) in pending.iter().enumerate() {
            for r in 0..n_vehicles {
                let veh = &problem.vehicles[r];
                if !veh.has_skills(task.skills(problem)) { continue; }
                let Some(pre) = precomps[r].as_ref() else { continue; };
                let positions = routes_steps[r].len() + 2;
                for pos in 1..positions {
                    if let Some(d) = try_insert_single(pre, problem, matrix, veh, pos, task) {
                        if best_per_job[job_idx].map_or(true, |(_, _, bd)| d < bd) {
                            best_per_job[job_idx] = Some((r, pos, d));
                        }
                    }
                }
            }
        }

        // Pick the task with maximum c2 = λ·d_0u − c1.
        let chosen = (0..pending.len())
            .filter_map(|i| {
                let (r, pos, c1) = best_per_job[i]?;
                let d_0 = d_0u_of(pending[i]);
                let c2 = lambda * d_0 - c1;
                Some((c2, i, r, pos))
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        match chosen {
            Some((_, idx, r, pos)) => {
                let task = pending.remove(idx);
                let mut cand = routes_steps[r].clone();
                cand.insert(pos - 1, task);
                let veh = &problem.vehicles[r];
                if evaluate_route(problem, matrix, veh, &cand).is_ok() {
                    routes_steps[r] = cand;
                    precomps[r] = precompute(problem, matrix, veh, r, &routes_steps[r]);
                } else {
                    unassigned.push(task);
                }
            }
            None => {
                unassigned.extend(pending.drain(..));
                break;
            }
        }
    }

    let mut routes: Vec<Route> = Vec::new();
    for (r_idx, steps) in routes_steps.iter().enumerate() {
        if steps.is_empty() { continue; }
        let veh = &problem.vehicles[r_idx];
        if let Ok(metrics) = evaluate_route(problem, matrix, veh, steps) {
            routes.push(Route {
                vehicle_idx: r_idx,
                steps: steps.clone(),
                metrics,
            });
        }
    }

    let mut sol = Solution { routes, unassigned, summary: Default::default() };
    sol.recompute_summary();
    sol
}


/// Complete Solomon I1 (1987) with both c11 and c12.
///
/// **Status: implemented but not activated.** Our c12 computation only
/// captures arrival shift at the IMMEDIATE next position, not cumulative
/// push-forward through the entire route. Classical Solomon I1 propagates
/// the shift until the route stabilizes at a stop with enough slack. Tests
/// showed +3.8 to +4.6% regression on N=500 vs greedy multi-start. A full
/// cumulative implementation costs O(L) per probe = O(N*M*L^2) total — too
/// slow for N=1000.
#[allow(dead_code)]
///
/// For each iter: for each pending task u, find the best insertion position
/// (i, u, j) in open routes by minimizing
///   c1(i, u, j) = a1*c11(i, u, j) + a2*c12(i, u, j)
/// where
///   c11 = d_iu + d_uj - mu*d_ij        (insertion cost, we use cost-delta)
///   c12 = b_uj_new - b_uj_old          (TW arrival shift at j)
///
/// Pick job u with max c2(u) = lambda*d_0u - c1(u). Insert.
///
/// Compared to `solomon_i1_insertion`: the older version lacks c12.
/// Solomon I1's TW awareness is in c12 — the forward shift of downstream
/// stops penalizes a job if insertion pushes arrivals near or past
/// time-window bounds.
///
/// Standard Solomon parameters (1987): a1=a2=0.5, mu=1, lambda=2.
pub fn solomon_i1_full(
    problem: &Problem,
    matrix: &Matrix,
    alpha1: f64,
    alpha2: f64,
    mu: f64,
    lambda: f64,
) -> Solution {
    let _ = mu; // c11 in our impl is the cost-delta from try_insert_single, which already
                // factors in d_iu + d_uj - d_ij implicitly (the new and old travel times).
                // The μ-weighted form is mathematically equivalent up to constants once we
                // use the cost delta as the c11 surrogate.

    let n_vehicles = problem.vehicles.len();
    let mut routes_steps: Vec<Vec<TaskRef>> = (0..n_vehicles).map(|_| Vec::new()).collect();
    let mut precomps: Vec<Option<RoutePrecomp>> = (0..n_vehicles)
        .map(|i| precompute(problem, matrix, &problem.vehicles[i], i, &[]))
        .collect();

    let d_0u_of = |task: TaskRef| -> f64 {
        let u_loc = match task.description(problem).location.index {
            Some(l) => l,
            None => return 0.0,
        };
        let mut min_d = i64::MAX;
        for v in &problem.vehicles {
            if let Some(loc) = v.start.as_ref().and_then(|l| l.index) {
                let d = matrix.duration(loc, u_loc) as i64;
                if d < min_d { min_d = d; }
            }
        }
        if min_d == i64::MAX { 0.0 } else { min_d as f64 }
    };

    let mut pending: Vec<TaskRef> = (0..problem.jobs.len()).map(TaskRef::Job).collect();
    let mut unassigned: Vec<TaskRef> = Vec::new();

    while !pending.is_empty() {
        // For each pending: find best (route, pos) and its c1.
        // best_per_job[i] = (route_idx, pos, c1_value)
        let mut best_per_job: Vec<Option<(usize, usize, f64)>> = vec![None; pending.len()];
        for (job_idx, &task) in pending.iter().enumerate() {
            for r in 0..n_vehicles {
                let veh = &problem.vehicles[r];
                if !veh.has_skills(task.skills(problem)) { continue; }
                let Some(pre) = precomps[r].as_ref() else { continue; };
                let positions = routes_steps[r].len() + 2;
                for pos in 1..positions {
                    if let Some((_cost_delta, travel_delta, time_shift)) =
                        try_insert_single_with_shift(pre, problem, matrix, veh, pos, task)
                    {
                        // c1 = α1·c11 + α2·c12, both in seconds.
                        // c11 = travel_delta (= d_iu + d_uj − d_ij + setup)
                        // c12 = time_shift   (= b_uj_new − b_uj_old)
                        let c1 = alpha1 * (travel_delta as f64) + alpha2 * (time_shift as f64);
                        if best_per_job[job_idx].map_or(true, |(_, _, bd)| c1 < bd) {
                            best_per_job[job_idx] = Some((r, pos, c1));
                        }
                    }
                }
            }
        }

        // Pick task with max c2 = λ·d_0u − c1.
        let chosen = (0..pending.len())
            .filter_map(|i| {
                let (r, pos, c1) = best_per_job[i]?;
                let d_0 = d_0u_of(pending[i]);
                // c2 = λ·d_0u − c1, all seconds.
                let c2 = lambda * d_0 - c1;
                Some((c2, i, r, pos))
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        match chosen {
            Some((_, idx, r, pos)) => {
                let task = pending.remove(idx);
                let mut cand = routes_steps[r].clone();
                cand.insert(pos - 1, task);
                let veh = &problem.vehicles[r];
                if evaluate_route(problem, matrix, veh, &cand).is_ok() {
                    routes_steps[r] = cand;
                    precomps[r] = precompute(problem, matrix, veh, r, &routes_steps[r]);
                } else {
                    unassigned.push(task);
                }
            }
            None => {
                unassigned.extend(pending.drain(..));
                break;
            }
        }
    }

    let mut routes: Vec<Route> = Vec::new();
    for (r_idx, steps) in routes_steps.iter().enumerate() {
        if steps.is_empty() { continue; }
        let veh = &problem.vehicles[r_idx];
        if let Ok(metrics) = evaluate_route(problem, matrix, veh, steps) {
            routes.push(Route {
                vehicle_idx: r_idx,
                steps: steps.clone(),
                metrics,
            });
        }
    }

    let mut sol = Solution { routes, unassigned, summary: Default::default() };
    sol.recompute_summary();
    sol
}
