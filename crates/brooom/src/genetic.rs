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
/// vehicle serves `tour[i..j]`; its cost/feasibility come from the real
/// `evaluate_route` (so the partition optimises brooom's true objective). Both
/// capacity and tail-TW infeasibility are monotonic under append, so we break
/// the inner extension as soon as a segment turns infeasible.
///
/// Returns `None` if no feasible partition exists within the fleet size, or if
/// the optimal partition needs more routes than vehicles available.
pub fn split(tour: &[usize], problem: &Problem, matrix: &Matrix) -> Option<Solution> {
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

    let mut seg: Vec<TaskRef> = Vec::with_capacity(32);
    for i in 0..n {
        if dp[i] == INF {
            continue;
        }
        seg.clear();
        for j in (i + 1)..=n {
            seg.push(TaskRef::Job(tour[j - 1]));
            match evaluate_route(problem, matrix, veh, &seg) {
                Ok(m) => {
                    let cand = dp[i] + m.cost;
                    let cand_routes = nroutes[i] + 1;
                    // Prefer lower cost; among equal cost prefer fewer routes.
                    if cand < dp[j] - 1e-9
                        || (cand <= dp[j] + 1e-9 && cand_routes < nroutes[j])
                    {
                        dp[j] = cand;
                        pred[j] = i;
                        nroutes[j] = cand_routes;
                    }
                }
                Err(_) => break, // monotonic: longer segment stays infeasible
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

// ── Population HGS ──────────────────────────────────────────────────────────

/// One population member: its educated solution plus the giant tour and edge
/// fingerprint used for recombination and diversity.
struct Indiv {
    tour: Vec<usize>,
    sol: Solution,
    cost: f64,
    edges: HashSet<(u32, u32)>,
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
    Indiv { tour, sol, cost, edges }
}

/// Cull `pop` down to `mu` survivors by Vidal biased fitness: rank by cost
/// (ascending) and by diversity contribution (mean broken-pairs distance to the
/// rest, descending), keep the lowest `cost_rank + DIV·div_rank`. The best-cost
/// member is always retained. Near-duplicate (clone) members are dropped first.
fn cull(pop: &mut Vec<Indiv>, mu: usize) {
    const DIV: f64 = 0.4; // diversity weight on the rank sum
    if pop.len() <= mu {
        return;
    }
    let n = pop.len();
    // Diversity contribution: mean distance to all others.
    let mut diversity = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            if i != j {
                acc += edge_distance(&pop[i].edges, &pop[j].edges);
            }
        }
        diversity[i] = acc / (n - 1) as f64;
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
    let mut div_rank = vec![0usize; n];
    for (r, &idx) in by_div.iter().enumerate() {
        div_rank[idx] = r;
    }
    let best_cost_idx = by_cost[0];
    let mut scored: Vec<(f64, usize)> = (0..n)
        .map(|i| (cost_rank[i] as f64 + DIV * div_rank[i] as f64, i))
        .collect();
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

/// Binary-tournament parent index by cost (lower wins).
fn tournament(pop: &[Indiv], rng: &mut ChaCha8Rng) -> usize {
    let a = rng.gen_range(0..pop.len());
    let b = rng.gen_range(0..pop.len());
    if pop[a].cost <= pop[b].cost {
        a
    } else {
        b
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
) -> Option<Solution> {
    if !hgs_applicable(problem) {
        return None;
    }
    // Force HARD mode for this island: education and Split must stay
    // hard-feasible. rayon reuses worker threads, so a prior ILS phase may have
    // left soft penalties armed on this thread — clear them or `evaluate_route`
    // would accept penalised-infeasible routes into the population.
    crate::solution::set_soft_penalties(None);
    let mu = 12usize; // target population (small: cold education is the budget)
    let lambda = 25usize; // offspring before each cull (μ+λ generational)
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
    for gen in 0..max_gens {
        gen_count = gen;
        if let Some(d) = deadline {
            if Instant::now() >= d {
                break;
            }
        }
        // Produce one offspring: OX(parent_a, parent_b) → Split → educate.
        let pa = tournament(&pop, &mut rng);
        let mut pb = tournament(&pop, &mut rng);
        if pb == pa {
            pb = (pb + 1) % pop.len();
        }
        let child_tour = order_crossover(&pop[pa].tour, &pop[pb].tour, &mut rng);
        let Some(mut child) = split(&child_tour, problem, matrix) else {
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
        } else {
            gens_since_improve += 1;
        }
        pop.push(make_indiv(child, problem));

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
            )
        })
        .min_by(|a, b| a.summary.cost.partial_cmp(&b.summary.cost).unwrap())
}
