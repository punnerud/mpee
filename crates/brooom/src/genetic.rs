//! Population-based Hybrid Genetic Search (Vidal/Prins) — CPU.
//!
//! This is the recombination lever the perturbation-based ILS cannot reach.
//! A warm-start hold test proved brooom's local search *holds* PyVRP's optima
//! at +0.00% but our search never *generates* a trajectory that reaches those
//! route structures. The missing mechanism is the classic HGS-CVRP triad:
//!
//!   giant tour (a permutation of all customers, no route delimiters)
//!     → **Split** (a Bellman DP that optimally partitions the giant tour into
//!        routes, deciding the route COUNT optimally for that ordering)
//!     → education (local search) + adaptive-penalty repair.
//!
//! Order crossover (OX) recombines two giant tours; Split turns the child
//! ordering into the cost-optimal multi-route solution. This is precisely the
//! global move ILS lacks — it reconstructs route partitions from scratch rather
//! than nudging an existing one.
//!
//! Scope (v1, gated): single-dimension capacity, job-only (no shipments),
//! no breaks / multi-trip, homogeneous fleet, time windows allowed. The caller
//! falls back to the existing search for anything outside this envelope, so this
//! is purely additive — best-of with the proven variants ⇒ non-regressing.

use crate::granular::Granular;
use crate::insertion::greedy_insertion_seeded;
use crate::local_search::{local_search, local_search_full};
use crate::matrix::Matrix;
use crate::problem::Problem;
use crate::solution::{evaluate_route, Route, Solution, TaskRef};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use web_time::Instant;

/// Whether this problem fits the v1 HGS envelope (see module docs). Outside it,
/// the caller keeps the proven search path.
pub fn hgs_applicable(problem: &Problem) -> bool {
    if problem.jobs.is_empty() || problem.vehicles.is_empty() {
        return false;
    }
    if !problem.shipments.is_empty() {
        return false;
    }
    if problem.any_multi_trip() {
        return false;
    }
    // Job-only: every task we Split is a plain Job. Reject backhaul-only mixes by
    // requiring deliveries (the Solomon-style envelope); pickups handled as load.
    if problem.vehicles.iter().any(|v| !v.breaks.is_empty()) {
        return false;
    }
    // Homogeneous fleet: identical capacity + time window so a single Split cost
    // model is exact (any vehicle can host any route at the same cost).
    let v0 = &problem.vehicles[0];
    let cap0 = &v0.capacity;
    let tw0 = v0.time_window();
    problem.vehicles.iter().all(|v| {
        v.capacity == *cap0
            && v.time_window().start == tw0.start
            && v.time_window().end == tw0.end
            && v.breaks.is_empty()
            && !v.is_multi_trip()
    })
}

/// Flatten a solution into a giant tour: the job indices visited, route after
/// route, with route boundaries dropped. Only `TaskRef::Job` steps are kept
/// (the v1 envelope has no others). Unassigned jobs are appended so the giant
/// tour is a complete permutation of all jobs.
pub fn solution_to_giant_tour(sol: &Solution, problem: &Problem) -> Vec<usize> {
    let mut tour = Vec::with_capacity(problem.jobs.len());
    let mut seen = vec![false; problem.jobs.len()];
    for r in &sol.routes {
        for s in &r.steps {
            if let TaskRef::Job(j) = s {
                if !seen[*j] {
                    seen[*j] = true;
                    tour.push(*j);
                }
            }
        }
    }
    // Any job not in a route (unassigned / dropped) — append in index order so
    // Split still has every customer to place.
    for (j, &s) in seen.iter().enumerate() {
        if !s {
            tour.push(j);
        }
    }
    tour
}

/// **Split** — Prins/Vidal optimal partition of a giant tour into routes.
///
/// Builds the auxiliary shortest-path DP: `dp[j]` = min cost to serve the first
/// `j` customers of the tour using whole routes. An arc `(i → j)` means one
/// vehicle serves `tour[i..j]`. Both capacity and tail-TW infeasibility are
/// monotonic under append, so we break the inner extension as soon as a
/// segment turns infeasible.
///
/// Arc relaxation has two engines:
///   - **incremental** (the default inside the HGS envelope): per anchor `i`,
///     extending the segment by one customer updates a forward state (time,
///     load prefix extrema, travel/distance/service totals) in O(1) and mirrors
///     `evaluate_route`'s verdict + cost exactly. This is the Split hot path —
///     the eval-based engine re-walked the whole segment per extension, making
///     Split O(L) per arc instead of O(1) and capping HGS generations.
///   - **eval** (fallback): full `evaluate_route` per extension, used whenever
///     a feature outside the mirrored envelope is active (custom dimensions,
///     soft mode, code constraints, precedence, breaks).
///
/// The backtracked routes are ALWAYS re-validated by `evaluate_route`, so the
/// evaluator remains the authority for whatever Solution leaves this function.
///
/// Returns `None` if no feasible partition exists within the fleet size, or if
/// the optimal partition needs more routes than vehicles available.
pub fn split(tour: &[usize], problem: &Problem, matrix: &Matrix) -> Option<Solution> {
    let use_fast = split_fast_eligible(problem)
        && std::env::var("BROOOM_NO_FAST_SPLIT").is_err();
    split_with_engine(tour, problem, matrix, use_fast)
}

/// Reference Split that always uses the full-evaluator arc engine. Exposed for
/// equivalence tests (`tests/genetic_split.rs`) — not part of the public API.
#[doc(hidden)]
pub fn split_reference(tour: &[usize], problem: &Problem, matrix: &Matrix) -> Option<Solution> {
    split_with_engine(tour, problem, matrix, false)
}

/// The incremental arc engine mirrors `evaluate_route` exactly only when none
/// of the features it doesn't model are active. (Breaks/multi-trip/shipments
/// are already excluded by `hgs_applicable`; these are the rest.)
fn split_fast_eligible(problem: &Problem) -> bool {
    !crate::dimension::has_dimensions()
        && !crate::solution::soft_is_active()
        && !crate::constraint::has_constraints()
        && problem.precedence.is_empty()
        && problem.vehicles.iter().all(|v| v.breaks.is_empty())
}

fn split_with_engine(
    tour: &[usize],
    problem: &Problem,
    matrix: &Matrix,
    fast: bool,
) -> Option<Solution> {
    let n = tour.len();
    if n == 0 {
        return Some(Solution::default());
    }
    let fleet = problem.vehicles.len();
    let veh = &problem.vehicles[0]; // homogeneous (gated by hgs_applicable)

    const INF: f64 = f64::INFINITY;
    let mut dp = vec![INF; n + 1];
    let mut pred = vec![0usize; n + 1];
    let mut nroutes = vec![usize::MAX; n + 1]; // routes used to reach j (for fleet cap)
    dp[0] = 0.0;
    nroutes[0] = 0;

    let mut relax = |dp: &mut Vec<f64>, pred: &mut Vec<usize>, nroutes: &mut Vec<usize>,
                     i: usize, j: usize, seg_cost: f64| {
        let cand = dp[i] + seg_cost;
        let cand_routes = nroutes[i] + 1;
        // Prefer lower cost; among equal cost prefer fewer routes.
        if cand < dp[j] - 1e-9 || (cand <= dp[j] + 1e-9 && cand_routes < nroutes[j]) {
            dp[j] = cand;
            pred[j] = i;
            nroutes[j] = cand_routes;
        }
    };

    if fast {
        let mut walk = SegWalk::new(problem, matrix, veh);
        for i in 0..n {
            if dp[i] == INF {
                continue;
            }
            walk.reset();
            for j in (i + 1)..=n {
                if !walk.append(tour[j - 1]) {
                    break; // monotonic: longer segment stays infeasible
                }
                match walk.close() {
                    Some(cost) => relax(&mut dp, &mut pred, &mut nroutes, i, j, cost),
                    None => break, // mirror of the eval engine: closing failure breaks too
                }
            }
        }
    } else {
        let mut seg: Vec<TaskRef> = Vec::with_capacity(32);
        for i in 0..n {
            if dp[i] == INF {
                continue;
            }
            seg.clear();
            for j in (i + 1)..=n {
                seg.push(TaskRef::Job(tour[j - 1]));
                match evaluate_route(problem, matrix, veh, &seg) {
                    Ok(m) => relax(&mut dp, &mut pred, &mut nroutes, i, j, m.cost),
                    Err(_) => break, // monotonic: longer segment stays infeasible
                }
            }
        }
    }

    if dp[n] == INF || nroutes[n] > fleet {
        return None;
    }

    // Backtrack the boundaries into routes, assigning distinct vehicles.
    let mut bounds = Vec::new();
    let mut j = n;
    while j > 0 {
        let i = pred[j];
        bounds.push((i, j));
        j = i;
    }
    bounds.reverse();

    let mut sol = Solution::default();
    for (vehicle_idx, (a, b)) in bounds.into_iter().enumerate() {
        let steps: Vec<TaskRef> = tour[a..b].iter().map(|&t| TaskRef::Job(t)).collect();
        let v = &problem.vehicles[vehicle_idx.min(fleet - 1)];
        match evaluate_route(problem, matrix, v, &steps) {
            Ok(metrics) => sol.routes.push(Route { vehicle_idx, steps, metrics }),
            Err(_) => return None, // shouldn't happen (segment was feasible on veh0)
        }
    }
    sol.recompute_summary(problem);
    Some(sol)
}

/// Incremental forward state for one Split segment (a candidate route serving
/// `tour[i..j]`). `append` extends the segment by one customer in O(1) per
/// capacity dimension; `close` adds the return-to-depot leg and the end-of-route
/// checks without mutating the walk, returning the route cost.
///
/// This mirrors `evaluate_route` for the envelope admitted by
/// `split_fast_eligible` + `hgs_applicable`: plain jobs, single trip, hard mode,
/// no breaks / custom dimensions / code constraints / precedence. The cost
/// formula and every arithmetic step (incl. the `f64` rounding of speed-scaled
/// arcs and the order of the cost summation) are copied from the evaluator so
/// the DP sees bit-identical segment costs.
///
/// Capacity over a growing segment: the evaluator loads the vehicle with the
/// FULL segment's deliveries up front, so appending customer `j` raises the
/// load at *every* earlier position by `delivery_j`. We therefore track, per
/// dimension, the running net flow after each served stop
/// (`net = pickups − deliveries` so far) and its prefix max/min; with
/// `d_total` = sum of all deliveries in the segment the evaluator's checks
/// become:
///   - start:      d_total ≤ cap
///   - mid (peak): d_total + net_max ≤ cap
///   - no negative: d_total + net_min ≥ 0
struct SegWalk<'a> {
    problem: &'a Problem,
    matrix: &'a Matrix,
    veh: &'a crate::problem::Vehicle,
    dim: usize,
    speed: f64,
    vw: crate::problem::TimeWindow,
    start_idx: Option<usize>,
    end_idx: Option<usize>,
    // walk state
    t: i64,
    prev: Option<usize>,
    travel: i64,
    distance: i64,
    service: i64,
    tasks: usize,
    seen_backhaul: bool,
    d_total: Vec<i64>,
    net: Vec<i64>,
    net_max: Vec<i64>,
    net_min: Vec<i64>,
}

/// Matrix legs at/above this are the routing engine's "no path" sentinel
/// (mirror of `solution::UNREACHABLE_LEG`, which is private).
const UNREACHABLE_LEG: i64 = 100_000_000;

impl<'a> SegWalk<'a> {
    fn new(problem: &'a Problem, matrix: &'a Matrix, veh: &'a crate::problem::Vehicle) -> Self {
        let dim = problem.capacity_dim().max(veh.capacity.len()).max(1);
        let start_idx = veh
            .start
            .as_ref()
            .and_then(|l| l.index)
            .or_else(|| veh.end.as_ref().and_then(|l| l.index));
        let end_idx = veh.end.as_ref().and_then(|l| l.index).or(start_idx);
        let vw = veh.time_window();
        Self {
            problem,
            matrix,
            veh,
            dim,
            speed: veh.speed_factor.max(0.01),
            vw,
            start_idx,
            end_idx,
            t: vw.start,
            prev: start_idx,
            travel: 0,
            distance: 0,
            service: 0,
            tasks: 0,
            seen_backhaul: false,
            d_total: vec![0; dim],
            net: vec![0; dim],
            net_max: vec![i64::MIN; dim],
            net_min: vec![i64::MAX; dim],
        }
    }

    fn reset(&mut self) {
        self.t = self.vw.start;
        self.prev = self.start_idx;
        self.travel = 0;
        self.distance = 0;
        self.service = 0;
        self.tasks = 0;
        self.seen_backhaul = false;
        for v in &mut self.d_total {
            *v = 0;
        }
        for v in &mut self.net {
            *v = 0;
        }
        for v in &mut self.net_max {
            *v = i64::MIN;
        }
        for v in &mut self.net_min {
            *v = i64::MAX;
        }
    }

    /// Extend the segment with job `job_idx`. Returns false when the extended
    /// segment is infeasible (the caller must stop extending this anchor, the
    /// same break the eval engine takes).
    fn append(&mut self, job_idx: usize) -> bool {
        let task = TaskRef::Job(job_idx);
        let job = task.description(self.problem);
        if !self.veh.has_skills(task.skills(self.problem)) {
            return false;
        }
        if !job.allows_vehicle(self.veh.id) {
            return false;
        }
        let Some(here) = job.location.index else {
            return false;
        };

        // Backhaul ordering (linehaul-before-backhaul on a route).
        let backhaul = !job.pickup.is_empty() && job.delivery.is_empty();
        if backhaul {
            self.seen_backhaul = true;
        } else if !job.delivery.is_empty() && self.seen_backhaul {
            return false;
        }

        // Travel arc.
        if let Some(p) = self.prev {
            let raw = self.matrix.duration(p, here);
            if raw as i64 >= UNREACHABLE_LEG {
                return false;
            }
            let dur = ((raw as f64) * self.speed).round() as i64;
            self.t += dur;
            self.travel += dur;
            self.distance += self.matrix.distance(p, here);
        }

        // Setup, release, time window — same order as the evaluator.
        let do_setup = match self.prev {
            Some(p) => p != here && job.setup > 0,
            None => job.setup > 0,
        };
        if do_setup {
            self.t += job.setup;
        }
        if self.t < job.release {
            self.t = job.release;
        }
        let Some(tw) = crate::solution::pick_time_window(&job.time_windows, self.t) else {
            return false;
        };
        if self.t < tw.start {
            self.t = tw.start;
        }
        if self.t > tw.end {
            return false;
        }
        self.t += job.service;
        self.service += job.service;

        // Capacity: fold this job into the totals, then re-check the whole
        // segment via the prefix extrema (O(dim)).
        for i in 0..self.dim {
            let d = job.delivery.get(i).copied().unwrap_or(0);
            let p = job.pickup.get(i).copied().unwrap_or(0);
            self.d_total[i] += d;
            self.net[i] += p - d;
            if self.net[i] > self.net_max[i] {
                self.net_max[i] = self.net[i];
            }
            if self.net[i] < self.net_min[i] {
                self.net_min[i] = self.net[i];
            }
        }
        for (i, &cap_i) in self.veh.capacity.iter().enumerate() {
            if i >= self.dim {
                break;
            }
            if self.d_total[i] > cap_i {
                return false; // capacity exceeded at route start
            }
            if self.d_total[i] + self.net_max[i] > cap_i {
                return false; // capacity exceeded mid-route
            }
        }
        for i in 0..self.dim {
            if self.d_total[i] + self.net_min[i] < 0 {
                return false; // negative load (over-delivery)
            }
        }

        self.tasks += 1;
        if let Some(max) = self.veh.max_tasks {
            if self.tasks > max {
                return false;
            }
        }
        self.prev = Some(here);
        true
    }

    /// Close the current segment as a complete route (return leg + end-of-route
    /// checks) WITHOUT mutating the walk. Returns the route cost, or `None`
    /// when closing is infeasible.
    fn close(&self) -> Option<f64> {
        let mut t = self.t;
        let mut travel = self.travel;
        let mut distance = self.distance;
        if let (Some(p), Some(e)) = (self.prev, self.end_idx) {
            let raw = self.matrix.duration(p, e);
            if raw as i64 >= UNREACHABLE_LEG {
                return None;
            }
            let dur = ((raw as f64) * self.speed).round() as i64;
            t += dur;
            travel += dur;
            distance += self.matrix.distance(p, e);
        }
        if t > self.vw.end {
            return None;
        }
        if let Some(max) = self.veh.max_travel_time {
            if travel > max {
                return None;
            }
        }
        if let Some(max) = self.veh.max_distance {
            if distance > max {
                return None;
            }
        }
        // Bit-identical to the evaluator's cost summation.
        let cost_travel = self.veh.fixed
            + (travel as f64) * (self.veh.per_hour / 3600.0).max(0.0) * self.veh.time_weight
            + (distance as f64) * self.veh.distance_weight
            + (self.service as f64) * 1e-6;
        let cost_span = ((t - self.vw.start) as f64) * self.veh.span_cost.max(0.0);
        Some(cost_travel + cost_span)
    }
}

/// Order crossover (OX) on two giant tours of the same job set. Copies a random
/// contiguous slice from `pa`, then fills the remaining positions with the jobs
/// of `pb` in their order, skipping those already taken. Produces a valid
/// permutation of all jobs.
pub fn order_crossover(pa: &[usize], pb: &[usize], rng: &mut ChaCha8Rng) -> Vec<usize> {
    let n = pa.len();
    if n <= 2 {
        return pa.to_vec();
    }
    let mut a = rng.gen_range(0..n);
    let mut b = rng.gen_range(0..n);
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    let mut child = vec![usize::MAX; n];
    let mut taken = vec![false; n.max(1)];
    // Map job id → taken needs a set sized to max job id; jobs are 0..n indices
    // into problem.jobs, and the tour is a permutation of those, so `n` slots
    // (indexed by job id) is exactly right.
    let mut in_child = vec![false; n];
    for k in a..=b {
        child[k] = pa[k];
        in_child[pa[k]] = true;
    }
    let mut fill = (b + 1) % n;
    for off in 0..n {
        let src = pb[(b + 1 + off) % n];
        if !in_child[src] {
            child[fill] = src;
            in_child[src] = true;
            fill = (fill + 1) % n;
        }
    }
    let _ = &mut taken;
    debug_assert!(child.iter().all(|&x| x != usize::MAX));
    child
}

// ── SREX crossover (Nagata 2010, the VRPTW crossover PyVRP uses) ────────────
//
// Order crossover + Split recombines *orderings* — it deliberately forgets the
// parents' route partitions, which is the right global move for CVRP but
// discards exactly the structure time windows shape. SREX instead exchanges
// whole ROUTES: the child keeps parent A's unselected routes intact and
// adopts parent B's most-overlapping routes intact, so two good partial
// partitions recombine without being re-derived. Running both operators side
// by side gives the population the CVRP-style reshuffle AND the VRPTW-style
// structure exchange.

/// Selective route exchange on two job-only solutions (the v1 HGS envelope).
/// Returns `None` when the child cannot host every job within the fleet.
fn srex_crossover(
    pa: &Solution,
    pb: &Solution,
    problem: &Problem,
    matrix: &Matrix,
    rng: &mut ChaCha8Rng,
) -> Option<Solution> {
    let na = pa.routes.len();
    let nb = pb.routes.len();
    if na == 0 || nb == 0 {
        return None;
    }
    let fleet = problem.vehicles.len();
    let frac = rng.gen_range(0.3..=0.6);
    let n_select = ((na as f64 * frac).round() as usize).clamp(1, na.min(nb));

    // 1. Random A-routes to give up.
    let mut a_idx: Vec<usize> = (0..na).collect();
    for k in 0..n_select {
        let j = rng.gen_range(k..na);
        a_idx.swap(k, j);
    }
    let a_selected = &a_idx[..n_select];
    let mut is_a_selected = vec![false; na];
    for &i in a_selected {
        is_a_selected[i] = true;
    }

    // 2. Job set of A's selected routes.
    let mut in_va = vec![false; problem.jobs.len()];
    for &i in a_selected {
        for s in &pa.routes[i].steps {
            if let TaskRef::Job(j) = s {
                in_va[*j] = true;
            }
        }
    }

    // 3. B-routes with the highest overlap with V_A.
    let mut overlap: Vec<(usize, usize)> = pb
        .routes
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let ov = r
                .steps
                .iter()
                .filter(|s| matches!(s, TaskRef::Job(j) if in_va[*j]))
                .count();
            (i, ov)
        })
        .collect();
    overlap.sort_by(|a, b| b.1.cmp(&a.1));
    let b_selected: Vec<usize> = overlap.into_iter().take(n_select).map(|(i, _)| i).collect();

    // 4. Child = A's unselected routes + B's selected routes, deduped (first
    // occurrence wins, so A's preserved structures keep their jobs).
    let mut seen = vec![false; problem.jobs.len()];
    let mut child_steps: Vec<Vec<TaskRef>> = Vec::with_capacity(na);
    for (i, r) in pa.routes.iter().enumerate() {
        if is_a_selected[i] {
            continue;
        }
        let steps: Vec<TaskRef> = r
            .steps
            .iter()
            .filter(|s| matches!(s, TaskRef::Job(j) if !std::mem::replace(&mut seen[*j], true)))
            .copied()
            .collect();
        if !steps.is_empty() {
            child_steps.push(steps);
        }
    }
    for &i in &b_selected {
        let steps: Vec<TaskRef> = pb.routes[i]
            .steps
            .iter()
            .filter(|s| matches!(s, TaskRef::Job(j) if !std::mem::replace(&mut seen[*j], true)))
            .copied()
            .collect();
        if !steps.is_empty() {
            child_steps.push(steps);
        }
    }
    if child_steps.len() > fleet {
        return None;
    }

    // 5. Re-evaluate every child route on sequentially assigned vehicles
    // (homogeneous fleet, gated by hgs_applicable). A deduped route can turn
    // infeasible only via... it can't: removing stops keeps a feasible route
    // feasible in this envelope (capacity shrinks, arrivals only get earlier).
    // Still, the evaluator stays the authority — bail rather than trust that.
    let mut sol = Solution::default();
    for (vehicle_idx, steps) in child_steps.into_iter().enumerate() {
        let veh = &problem.vehicles[vehicle_idx];
        let metrics = evaluate_route(problem, matrix, veh, &steps).ok()?;
        sol.routes.push(Route { vehicle_idx, steps, metrics });
    }

    // 6. Reinsert missing jobs (in A's selected but not B's selected),
    // largest demand first, each at its cheapest feasible position; open a
    // fresh route when nothing fits.
    let mut missing: Vec<usize> = (0..problem.jobs.len()).filter(|&j| !seen[j]).collect();
    missing.sort_by_key(|&j| -(problem.jobs[j].delivery.first().copied().unwrap_or(0)));
    for j in missing {
        let task = TaskRef::Job(j);
        let mut best: Option<(usize, usize, f64)> = None; // (route, pos, delta)
        for (ri, route) in sol.routes.iter().enumerate() {
            let veh = &problem.vehicles[route.vehicle_idx];
            let Some(pre) = crate::eval::precompute(problem, matrix, veh, route.vehicle_idx, &route.steps) else {
                continue;
            };
            for pos in 1..=route.steps.len() + 1 {
                if let Some(d) = crate::eval::try_insert_single(&pre, problem, matrix, veh, pos, task) {
                    if best.map_or(true, |(_, _, bd)| d < bd) {
                        best = Some((ri, pos - 1, d));
                    }
                }
            }
        }
        match best {
            Some((ri, pos, _)) => {
                let veh = &problem.vehicles[sol.routes[ri].vehicle_idx];
                let mut steps = sol.routes[ri].steps.clone();
                steps.insert(pos, task);
                // The O(1) probe is a pre-filter; the evaluator confirms.
                let Ok(metrics) = evaluate_route(problem, matrix, veh, &steps) else {
                    return None;
                };
                sol.routes[ri].steps = steps;
                sol.routes[ri].metrics = metrics;
            }
            None => {
                let vehicle_idx = sol.routes.len();
                if vehicle_idx >= fleet {
                    return None;
                }
                let veh = &problem.vehicles[vehicle_idx];
                let steps = vec![task];
                let metrics = evaluate_route(problem, matrix, veh, &steps).ok()?;
                sol.routes.push(Route { vehicle_idx, steps, metrics });
            }
        }
    }
    sol.recompute_summary(problem);
    Some(sol)
}

// ── Population HGS ──────────────────────────────────────────────────────────

/// One population member: its educated solution plus the giant tour and edge
/// fingerprint used for recombination and diversity.
struct Indiv {
    tour: Vec<usize>,
    sol: Solution,
    cost: f64,
    edges: HashSet<(u32, u32)>,
    /// Vidal biased fitness: cost_rank + DIV·div_rank over the current
    /// population (lower = better). Recomputed by `recompute_fitness`; parents
    /// are tournament-selected on THIS, not raw cost, so a diverse-but-decent
    /// individual still breeds — the mechanism that keeps the population from
    /// collapsing onto one basin.
    fitness: f64,
}

/// Directed consecutive-job edges of a solution (depot = sentinel `u32::MAX`).
/// Used for the broken-pairs diversity distance, à la Vidal HGS.
fn edge_set(sol: &Solution) -> HashSet<(u32, u32)> {
    let mut e = HashSet::new();
    const DEPOT: u32 = u32::MAX;
    for r in &sol.routes {
        let mut prev = DEPOT;
        for s in &r.steps {
            if let TaskRef::Job(j) = s {
                let cur = *j as u32;
                e.insert((prev, cur));
                prev = cur;
            }
        }
        e.insert((prev, DEPOT));
    }
    e
}

/// Broken-pairs distance in [0,1]: share of edges the two solutions disagree on.
fn edge_distance(a: &HashSet<(u32, u32)>, b: &HashSet<(u32, u32)>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let diff = a.symmetric_difference(b).count();
    diff as f64 / (a.len() + b.len()) as f64
}

fn make_indiv(sol: Solution, problem: &Problem) -> Indiv {
    let tour = solution_to_giant_tour(&sol, problem);
    let cost = sol.summary.cost;
    let edges = edge_set(&sol);
    Indiv { tour, sol, cost, edges, fitness: 0.0 }
}

const DIV: f64 = 0.4; // diversity weight on the rank sum

/// Recompute every member's biased fitness (cost_rank + DIV·div_rank) over the
/// CURRENT population. O(n²) broken-pairs distances — cheap (n ≤ µ+λ) next to
/// one education. Returns the index of the best-cost member.
fn recompute_fitness(pop: &mut [Indiv]) -> usize {
    let n = pop.len();
    if n == 0 {
        return 0;
    }
    let mut diversity = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            if i != j {
                acc += edge_distance(&pop[i].edges, &pop[j].edges);
            }
        }
        diversity[i] = if n > 1 { acc / (n - 1) as f64 } else { 0.0 };
    }
    // cost rank (0 = cheapest)
    let mut by_cost: Vec<usize> = (0..n).collect();
    by_cost.sort_by(|&a, &b| pop[a].cost.partial_cmp(&pop[b].cost).unwrap());
    let mut cost_rank = vec![0usize; n];
    for (r, &idx) in by_cost.iter().enumerate() {
        cost_rank[idx] = r;
    }
    // diversity rank (0 = most diverse)
    let mut by_div: Vec<usize> = (0..n).collect();
    by_div.sort_by(|&a, &b| diversity[b].partial_cmp(&diversity[a]).unwrap());
    for (r, &idx) in by_div.iter().enumerate() {
        pop[idx].fitness = cost_rank[idx] as f64 + DIV * r as f64;
    }
    by_cost[0]
}

/// Cull `pop` down to `mu` survivors by Vidal biased fitness. The best-cost
/// member is always retained.
fn cull(pop: &mut Vec<Indiv>, mu: usize) {
    if pop.len() <= mu {
        return;
    }
    let best_cost_idx = recompute_fitness(pop);
    let n = pop.len();
    let mut scored: Vec<(f64, usize)> = (0..n).map(|i| (pop[i].fitness, i)).collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut keep: Vec<usize> = scored.iter().take(mu).map(|&(_, i)| i).collect();
    if !keep.contains(&best_cost_idx) {
        keep.pop();
        keep.push(best_cost_idx);
    }
    keep.sort_unstable();
    // Rebuild pop keeping only `keep` (in place, preserving order).
    let mut kept: Vec<Indiv> = Vec::with_capacity(keep.len());
    let mut ki = 0usize;
    for (i, ind) in pop.drain(..).enumerate() {
        if ki < keep.len() && keep[ki] == i {
            kept.push(ind);
            ki += 1;
        }
    }
    *pop = kept;
}

/// Binary-tournament parent index by biased fitness (lower wins). Falls back
/// to raw cost when fitness is stale-equal (e.g. right after init).
fn tournament(pop: &[Indiv], rng: &mut ChaCha8Rng) -> usize {
    let a = rng.gen_range(0..pop.len());
    let b = rng.gen_range(0..pop.len());
    let (fa, fb) = (pop[a].fitness, pop[b].fitness);
    if fa != fb {
        return if fa < fb { a } else { b };
    }
    if pop[a].cost <= pop[b].cost {
        a
    } else {
        b
    }
}

/// Shared elite pool for light island migration: islands publish every new
/// best and occasionally adopt someone else's. Kept tiny and pull-rare so it
/// cross-pollinates basins without homogenising the islands (seeding every
/// island with the same solution measurably hurt: +1–3% on R2/RC2).
pub struct MigrationPool {
    entries: std::sync::Mutex<Vec<(f64, Solution)>>,
}

impl MigrationPool {
    pub fn new() -> Self {
        Self { entries: std::sync::Mutex::new(Vec::new()) }
    }
    fn publish(&self, sol: &Solution) {
        let mut e = self.entries.lock().unwrap();
        let cost = sol.summary.cost;
        if e.iter().any(|(c, _)| (c - cost).abs() < 1e-6) {
            return; // already have this basin's representative
        }
        e.push((cost, sol.clone()));
        e.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        e.truncate(8);
    }
    fn pull(&self, worse_than: f64, rng: &mut ChaCha8Rng) -> Option<Solution> {
        let e = self.entries.lock().unwrap();
        let better: Vec<&(f64, Solution)> =
            e.iter().filter(|(c, _)| *c < worse_than - 1e-6).collect();
        if better.is_empty() {
            return None;
        }
        Some(better[rng.gen_range(0..better.len())].1.clone())
    }
}

impl Default for MigrationPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Population Hybrid Genetic Search with giant-tour OX crossover + Split.
///
/// Returns the best hard-feasible solution found, or `None` if the problem is
/// outside the v1 envelope / Split never produced a feasible partition. Runs
/// until `deadline` (or a generation cap when no deadline is set). `seeds` may
/// supply warm individuals (e.g. the multi-start best) to fold into the initial
/// population — they only help (best-of by cost).
pub fn solve_genetic(
    problem: &Problem,
    matrix: &Matrix,
    granular: Option<&Granular>,
    max_passes: usize,
    seed: u64,
    deadline: Option<Instant>,
    seeds: &[Solution],
    migration: Option<&MigrationPool>,
) -> Option<Solution> {
    if !hgs_applicable(problem) {
        return None;
    }
    // Force HARD mode for this island: education and Split must stay
    // hard-feasible. rayon reuses worker threads, so a prior ILS phase may have
    // left soft penalties armed on this thread — clear them or `evaluate_route`
    // would accept penalised-infeasible routes into the population.
    crate::solution::set_soft_penalties(None);
    // Population sizing: Vidal's 25/40 now that incremental Split + the full
    // fast-LS operator set made education cheap (was 12/25 when cold education
    // was the budget). Overridable for A/B via BROOOM_HGS_MU / BROOOM_HGS_LAMBDA.
    let env_usize = |k: &str, d: usize| {
        std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
    };
    let mu = env_usize("BROOOM_HGS_MU", 25).max(2);
    let lambda = env_usize("BROOOM_HGS_LAMBDA", 40).max(1);
    let max_pop = mu + lambda;
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x6857_4753);

    // ── Initial population: diverse greedy seeds + any provided warm solutions,
    // each Split-normalised then educated. Init uses a CHEAP education (2 passes)
    // so it doesn't eat the whole budget — offspring get the full `max_passes`.
    // Deadline-aware so islands honour the shared wall-clock. ──────────────────
    let init_passes = max_passes.min(2);
    let past_deadline = |d: &Option<Instant>| d.map_or(false, |dd| Instant::now() >= dd);
    let mut pop: Vec<Indiv> = Vec::with_capacity(max_pop);
    let mut push_educated = |pop: &mut Vec<Indiv>, mut sol: Solution, passes: usize| {
        // Normalise through Split (optimal partition of this ordering) then educate.
        let tour = solution_to_giant_tour(&sol, problem);
        if let Some(s) = split(&tour, problem, matrix) {
            sol = s;
        }
        local_search(problem, matrix, &mut sol, passes, granular);
        if sol.unassigned.is_empty() {
            pop.push(make_indiv(sol, problem));
        }
    };
    for s in seeds {
        push_educated(&mut pop, s.clone(), init_passes);
    }
    let mut si = 0u64;
    while pop.len() < mu && !past_deadline(&deadline) {
        let s = greedy_insertion_seeded(problem, matrix, seed.wrapping_add(si).wrapping_add(1));
        push_educated(&mut pop, s, init_passes);
        si += 1;
        if si > mu as u64 * 4 {
            break; // safety: construction can't fill the population
        }
    }
    if pop.len() < 2 {
        return pop.into_iter().min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap()).map(|i| i.sol);
    }
    if pop.is_empty() {
        return None;
    }

    let mut best: Solution = pop
        .iter()
        .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap())
        .map(|i| i.sol.clone())
        .unwrap();
    let mut best_cost = best.summary.cost;

    let max_gens = if deadline.is_some() { usize::MAX } else { 2000 };
    let mut gens_since_improve = 0usize;
    let mut gen_count = 0usize;
    // Share of offspring produced by SREX (whole-route exchange) instead of
    // OX+Split (ordering recombination). Both operators feed the same
    // education + culling. Override via BROOOM_SREX (0.0 disables).
    let srex_p: f64 = std::env::var("BROOOM_SREX")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|p: &f64| (0.0..=1.0).contains(p))
        .unwrap_or(0.5);
    recompute_fitness(&mut pop);
    // How often (in generations) an island refreshes fitness ranks and checks
    // the migration pool. Fitness staleness between culls is acceptable; full
    // recomputes every generation measured ~equal and cost time.
    const MIGRATE_EVERY: usize = 150;
    for gen in 0..max_gens {
        gen_count = gen;
        if let Some(d) = deadline {
            if Instant::now() >= d {
                break;
            }
        }
        // Light island migration: occasionally adopt a foreign elite that is
        // strictly better than our current best. Split-normalise + educate
        // cheaply like any other entrant; culling's diversity term keeps it
        // from flooding the population.
        if let Some(pool) = migration {
            if gen > 0 && gen % MIGRATE_EVERY == 0 {
                if let Some(s) = pool.pull(best_cost, &mut rng) {
                    push_educated(&mut pop, s, init_passes);
                    recompute_fitness(&mut pop);
                }
            }
        }
        // Produce one offspring: crossover(parent_a, parent_b) → educate.
        let pa = tournament(&pop, &mut rng);
        let mut pb = tournament(&pop, &mut rng);
        if pb == pa {
            pb = (pb + 1) % pop.len();
        }
        let child_opt = if rng.gen_bool(srex_p) {
            // SREX keeps whole routes from both parents (no Split — the
            // partition IS the inherited structure).
            srex_crossover(&pop[pa].sol, &pop[pb].sol, problem, matrix, &mut rng)
        } else {
            let child_tour = order_crossover(&pop[pa].tour, &pop[pb].tour, &mut rng);
            split(&child_tour, problem, matrix)
        };
        let Some(mut child) = child_opt else {
            continue;
        };
        local_search(problem, matrix, &mut child, max_passes, granular);
        if !child.unassigned.is_empty() {
            continue;
        }
        let child_cost = child.summary.cost;
        if child_cost < best_cost - 1e-9 {
            // exhaustive-on-best polish
            let mut polished = child.clone();
            local_search_full(problem, matrix, &mut polished, max_passes, granular);
            if polished.unassigned.is_empty() && polished.summary.cost < child_cost - 1e-9 {
                child = polished;
            }
            best_cost = child.summary.cost;
            best = child.clone();
            gens_since_improve = 0;
            if let Some(pool) = migration {
                pool.publish(&best);
            }
        } else {
            gens_since_improve += 1;
        }
        // Provisional fitness = cost rank among the current population (full
        // diversity-aware ranks are refreshed at cull; recomputing the O(n²)
        // broken-pairs matrix per offspring would rival the education cost).
        let mut child_indiv = make_indiv(child, problem);
        child_indiv.fitness = pop.iter().filter(|p| p.cost < child_indiv.cost).count() as f64;
        pop.push(child_indiv);

        if pop.len() >= max_pop {
            cull(&mut pop, mu);
        }

        // Diversification restart: if stuck, refresh the worse half with fresh
        // greedy seeds (keeps the elite, injects new genetic material).
        if gens_since_improve >= 400 {
            cull(&mut pop, mu / 2);
            for _ in 0..(mu / 2) {
                let s = greedy_insertion_seeded(
                    problem,
                    matrix,
                    seed.wrapping_add(gen as u64).wrapping_add(0x1234),
                );
                push_educated(&mut pop, s, init_passes);
            }
            recompute_fitness(&mut pop);
            gens_since_improve = 0;
        }
    }

    if std::env::var("BROOOM_HGS_DEBUG").is_ok() {
        eprintln!(
            "HGS: gens={} pop={} best={:.0} routes={}",
            gen_count,
            pop.len(),
            best_cost,
            best.routes.iter().filter(|r| !r.steps.is_empty()).count()
        );
    }
    Some(best)
}

/// Run `n_islands` independent GA populations in parallel (island model) and
/// return the cheapest hard-feasible result. Each island is single-threaded;
/// rayon spreads them across cores. This is how HGS uses the machine the way the
/// multi-start ILS does. `seeds` are shared as warm individuals for every island.
#[cfg(feature = "parallel")]
pub fn solve_genetic_parallel(
    problem: &Problem,
    matrix: &Matrix,
    granular: Option<&Granular>,
    max_passes: usize,
    base_seed: u64,
    deadline: Option<Instant>,
    n_islands: usize,
    seeds: &[Solution],
) -> Option<Solution> {
    use rayon::prelude::*;
    if !hgs_applicable(problem) {
        return None;
    }
    let pool = MigrationPool::new();
    let migrate = std::env::var("BROOOM_NO_MIGRATION").is_err();
    (0..n_islands.max(1) as u64)
        .into_par_iter()
        .filter_map(|i| {
            // Only island 0 is anchored on the provided seed (e.g. the ILS best):
            // it can never return worse than that seed, giving a safety floor.
            // Islands 1.. explore from diverse greedy starts — seeding them all
            // with the same solution homogenises the population and hurts
            // exploration (measured: +1–3% on R2/RC2).
            let isl_seeds: &[Solution] = if i == 0 { seeds } else { &[] };
            solve_genetic(
                problem,
                matrix,
                granular,
                max_passes,
                base_seed.wrapping_add(i.wrapping_mul(0x9E37)),
                deadline,
                isl_seeds,
                if migrate { Some(&pool) } else { None },
            )
        })
        .min_by(|a, b| a.summary.cost.partial_cmp(&b.summary.cost).unwrap())
}
