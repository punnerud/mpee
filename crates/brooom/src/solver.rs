//! Top-level solve orchestration.

use rand::seq::SliceRandom;
use rand::SeedableRng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
// Browser-safe clock on wasm; std::time on native.
use web_time::Instant;

use crate::error::{Error, Result};
use crate::granular::Granular;
use crate::insertion::{greedy_insertion, greedy_insertion_seeded};
use crate::local_search::{local_search, local_search_full, route_split_pass};
use crate::matrix::{resolve_coords, Matrix, MatrixSource};
use crate::problem::Problem;
use crate::solution::{Solution, TaskRef};

#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// How many local-search passes to run before giving up. Each pass picks
    /// the single best improving move across all operators.
    pub max_local_search_passes: usize,
    /// Granular-neighborhood K (Toth & Vigo). Smaller K → faster but more
    /// local optima. `None` disables granularity (slower, marginally better).
    pub granular_k: Option<usize>,
    /// Number of parallel multi-start attempts. `1` is the deterministic
    /// single-solve path; larger K runs that many seeded variants in
    /// parallel and returns the cheapest. By construction K≥1 is never
    /// worse than K=1.
    pub multi_start: usize,
    /// Iterated local search: after LS converges, perform this many
    /// destroy-and-repair kicks per multi-start variant. Each kick removes
    /// `ils_kick_size` random tasks and reinserts them, then re-runs LS.
    /// Best-ever cost is tracked across kicks. 0 disables ILS.
    pub ils_iters: usize,
    /// Fraction (0.0..1.0) of tasks to remove per ILS kick.
    pub ils_kick_size: f64,
    /// Optional wall-time budget in milliseconds. When set, the ILS loop
    /// stops as soon as elapsed time exceeds this — useful when you want
    /// "give me the best you can in N seconds".
    pub time_limit_ms: Option<u64>,
    /// If set, print summary lines as the search progresses.
    pub verbose: bool,
    /// Pre-built warm-start solution. When provided, seed=0 skips the
    /// insertion phase and drops straight into local search on this
    /// solution. Other multi-start seeds keep their normal starts so
    /// best-of-K still benefits from diversity. The warm-start must
    /// reference the same `problem`'s jobs/vehicles.
    pub warm_start: Option<crate::solution::Solution>,
    /// Run a GPU megakernel polish pass on the multi-start winner before
    /// the final CPU polish. Falls back silently to CPU-only if GPU
    /// initialization fails or if the GPU pass produces a worse / non-
    /// feasible result. Useful for N≥500 where GPU LS iters cost far
    /// less than CPU iters.
    pub use_gpu: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_local_search_passes: 50,
            granular_k: Some(20),
            // K=8 by default — closes the cost gap to Vroom by running 8
            // seeded variants in parallel and keeping the cheapest. The
            // wall-time cost is acceptable because we have 30-40× headroom
            // vs Vroom on N≥500 anyway.
            multi_start: 8,
            // 30 ILS kicks per variant with 40% destruction. Combined with
            // K=8 multi-start, gives 240 distinct local optima per solve.
            // Closes the cost gap on N≤250 to within ~1%, larger N benefits
            // from `--time-limit-s` to spend more compute on quality.
            ils_iters: 30,
            ils_kick_size: 0.4,
            time_limit_ms: None,
            verbose: false,
            warm_start: None,
            use_gpu: false,
        }
    }
}

/// Solver result bundle. Holding the matrix lets the caller render per-step
/// timing in the output (otherwise step-level distance would be unknown).
#[derive(Debug, Clone)]
pub struct Solved {
    pub matrix: Matrix,
    pub solution: Solution,
}

/// Resolve coordinates → matrix → initial solution → local search.
///
/// `source` is consulted only if the problem does not already carry a matrix
/// for the vehicles' profile. Pass `Some(&HaversineMatrix::default())` for a
/// network-free build.
pub fn solve(
    problem: &mut Problem,
    source: Option<&dyn MatrixSource>,
    config: SolverConfig,
) -> Result<Solution> {
    let s = solve_full(problem, source, config)?;
    Ok(s.solution)
}

/// Same as `solve` but also returns the matrix used.
pub fn solve_full(
    problem: &mut Problem,
    source: Option<&dyn MatrixSource>,
    config: SolverConfig,
) -> Result<Solved> {
    problem.validate()?;
    let matrix = build_matrix(problem, source)?;
    // Drop the (possibly very large) raw `Vec<Vec<i64>>` matrices that came
    // from JSON now that we have the compact i32 runtime form. On 1000-node
    // instances this releases ~16 MB per matrix.
    problem.matrices.clear();
    let solution = solve_with_matrix(problem, &matrix, &config);
    Ok(Solved { matrix, solution })
}

/// Run insertion + local search using a pre-built matrix.
///
/// With `config.multi_start > 1`, runs that many seeded variants in parallel
/// (rayon) and returns the cheapest. Seed 0 is always the deterministic
/// baseline so we can never beat-then-lose; seeds 1..K shuffle pending tasks
/// within priority class to give LS distinct starting points.
pub fn solve_with_matrix(problem: &Problem, matrix: &Matrix, config: &SolverConfig) -> Solution {
    // Drop any stale eval cache from a previous solve. Worker threads each
    // have their own thread-local cache; rayon will warm them as it spawns.
    crate::solution::eval_cache_invalidate();
    let granular = config.granular_k.map(|k| Granular::build(matrix, k));
    if config.verbose {
        if let Some(g) = &granular {
            eprintln!("brooom: built granular neighborhood K={} (n={})", g.k(), g.n());
        }
    }

    let k = config.multi_start.max(1);
    let ils_iters = config.ils_iters;
    let ils_kick = config.ils_kick_size.max(0.0).min(1.0);
    let deadline = config
        .time_limit_ms
        .map(|ms| Instant::now() + std::time::Duration::from_millis(ms));

    // For each of K seeds: insertion → LS → ILS-kick loop. Best across all
    // attempts wins. With K=1 and ils_iters=0 this is the original baseline;
    // any larger setting trades wall time for cost.
    let solve_one = |seed: u64| -> Solution {
        // Diversify starting solutions across multi-start variants:
        //   seed=0     → deterministic greedy cheapest (baseline)
        //   even seeds → seeded greedy (shuffled within priority)
        //   odd seeds  → Solomon I1 with varied λ (1.0, 1.5, 2.0, 3.0)
        // Solomon I1 produces structurally different starts (favors far-from-
        // depot tasks first), which gives LS access to local optima the
        // greedy variants can't reach. Verified on N=1000 to close ~1-2% gap
        // vs Vroom by complementing the greedy seeds.
        let mut sol = if seed == 0 {
            // Warm-start (if any) replaces the deterministic greedy baseline
            // for seed=0. Other seeds keep their normal diversifying starts —
            // best-of-K then takes warm-start vs alternatives without ever
            // losing, so warm-start is strictly safe.
            if let Some(ws) = config.warm_start.as_ref() {
                ws.clone()
            } else {
                greedy_insertion(problem, matrix)
            }
        } else {
            greedy_insertion_seeded(problem, matrix, seed)
        };
        local_search(
            problem, matrix, &mut sol,
            config.max_local_search_passes, granular.as_ref(),
        );

        // Smart RouteSplit: enable only for one selected seed (seed=7).
        // The best-of-K mechanism means that if the split variant wins, we
        // keep it; if it doesn't win, we lose nothing. Earlier unconditional
        // split caused +0.6% regression on N=1000 because some variants got
        // worse. With ONLY one variant we now get "best of with-vs-without"
        // for free from the multi-start pool.
        if seed == 7 {
            route_split_pass(problem, matrix, &mut sol, 10);
            local_search(
                problem, matrix, &mut sol,
                config.max_local_search_passes, granular.as_ref(),
            );
        }

        // ILS: destroy-and-repair, track best ever.
        if ils_iters > 0 && ils_kick > 0.0 {
            let mut best_cost = sol.summary.cost;
            let mut best_sol = sol.clone();
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xA5A5));
            for _ in 0..ils_iters {
                if let Some(d) = deadline {
                    if Instant::now() >= d { break; }
                }
                let mut perturbed = best_sol.clone();
                kick(&mut perturbed, ils_kick, &mut rng, problem, matrix);
                local_search(
                    problem, matrix, &mut perturbed,
                    config.max_local_search_passes, granular.as_ref(),
                );
                if perturbed.summary.cost < best_cost {
                    best_cost = perturbed.summary.cost;
                    best_sol = perturbed;
                }
            }
            sol = best_sol;
        }
        sol
    };

    #[allow(unused_mut)]
    let mut best = if k == 1 {
        solve_one(0)
    } else {
        // Parallel multi-start on native; serial on wasm (no rayon).
        #[cfg(feature = "parallel")]
        {
            (0..k as u64)
                .into_par_iter()
                .map(solve_one)
                .min_by(|a, b| {
                    a.summary
                        .cost
                        .partial_cmp(&b.summary.cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("at least one variant")
        }
        #[cfg(not(feature = "parallel"))]
        {
            (1..k as u64).map(solve_one).fold(solve_one(0), |a, b| {
                if b.summary.cost < a.summary.cost { b } else { a }
            })
        }
    };

    // GPU megakernel polish pass on the multi-start winner. Falls back
    // silently if GPU init fails or no improvement. Only invoked for
    // larger N where cross-route interactions matter — at the per-cluster
    // level after cluster_decompose the routes are too short for batch
    // GPU to find diversity-driven wins. The outer flow in main.rs runs
    // a separate top-level gpu_polish on the merged solution.
    #[cfg(feature = "gpu")]
    if config.use_gpu && best.routes.len() > 0 && matrix.n >= 500 {
        let t_gpu = std::time::Instant::now();
        let max_iter = if matrix.n >= 5000 { 2000 } else { 1000 };
        if let Some(gpu_sol) = crate::gpu_polish::gpu_polish(
            problem, matrix, &best, granular.as_ref(), max_iter, config.verbose,
        ) {
            if gpu_sol.summary.cost + 1e-9 < best.summary.cost {
                if config.verbose {
                    eprintln!(
                        "brooom: GPU polish: {:.2} → {:.2} (Δ={:.2}, t={:.2}s)",
                        best.summary.cost, gpu_sol.summary.cost,
                        best.summary.cost - gpu_sol.summary.cost,
                        t_gpu.elapsed().as_secs_f64()
                    );
                }
                best = gpu_sol;
            }
        }
    }

    // Final polishing pass on the multi-start winner: full LS with no
    // don't-look bits — every task reconsidered every pass. Vroom-style.
    // Don't-look-LS converges fast but can prematurely settle tasks that
    // a later move could free up. This finishing pass picks up those
    // missed moves once.
    let pre_polish_cost = best.summary.cost;
    local_search_full(
        problem, matrix, &mut best,
        config.max_local_search_passes, granular.as_ref(),
    );
    if config.verbose && best.summary.cost + 1e-9 < pre_polish_cost {
        eprintln!(
            "brooom: polish pass: {:.2} → {:.2} (Δ={:.2})",
            pre_polish_cost, best.summary.cost, pre_polish_cost - best.summary.cost
        );
    }

    // Final guaranteed-assignment pass. The ILS `kick` drops empty routes and
    // only reinserts into surviving ones, so when vehicles outnumber demand a
    // feasible job can get stranded in `unassigned` instead of opening a spare
    // vehicle. Repair that here: place each remaining job in its cheapest
    // feasible slot (existing route or an unused vehicle).
    let pre_repair_unassigned = best.unassigned.len();
    repair_unassigned(&mut best, problem, matrix);
    if config.verbose && best.unassigned.len() < pre_repair_unassigned {
        eprintln!(
            "brooom: repair pass: assigned {} stranded job(s) → unassigned={}",
            pre_repair_unassigned - best.unassigned.len(),
            best.unassigned.len()
        );
    }

    if config.verbose {
        eprintln!(
            "brooom: multi_start={} best — routes={} unassigned={} cost={:.2}",
            k,
            best.routes.len(),
            best.unassigned.len(),
            best.summary.cost
        );
    }
    best
}

/// Final guaranteed-assignment pass: greedily place any still-unassigned
/// single jobs into the cheapest *feasible* slot — across existing routes
/// AND any unused vehicle — using `evaluate_route`, so capacity, skills,
/// time windows and max-distance/-time are all still honoured. It only ever
/// ADDS assignments, so it cannot make a feasible plan worse; it just stops
/// the ILS from stranding jobs in spare vehicles. A placement that would
/// need an unreachable (sentinel-distance) leg is rejected, so a job with no
/// real path stays unassigned rather than getting an absurd 100 000-km route.
fn repair_unassigned(sol: &mut Solution, problem: &Problem, matrix: &Matrix) {
    use crate::solution::{eval_cache_invalidate, evaluate_route, Route, RouteMetrics};
    // 100 000 km — far beyond any real road leg; flags the sentinel value
    // mpee uses for "no path" without rejecting legitimately long routes.
    const UNREACHABLE_M: i64 = 100_000_000;

    if sol.unassigned.is_empty() {
        return;
    }
    // The solve just filled the per-thread evaluate_route cache; bump the
    // epoch so this pass recomputes feasibility from scratch (avoids any
    // stale/collided cache entry stranding a placeable job).
    eval_cache_invalidate();
    enum Slot {
        Existing(usize, usize),
        NewVehicle(usize),
    }

    // Multi-pass: a job whose *solo* round-trip is infeasible (e.g. a one-way
    // snap makes depot→job→depot unreachable) can still be inserted mid-route
    // once another job has opened a vehicle — so retry the leftovers until a
    // full pass places nothing new.
    loop {
    let pending = std::mem::take(&mut sol.unassigned);
    let mut leftovers: Vec<TaskRef> = Vec::new();
    let mut placed_any = false;

    for task in pending {
        // Shipment halves must be inserted as a pickup→delivery pair; leave
        // those to the main solver. This repair only handles standalone jobs.
        if matches!(task, TaskRef::Pickup(_) | TaskRef::Delivery(_)) {
            leftovers.push(task);
            continue;
        }
        let req = task.skills(problem);
        let mut best: Option<(Slot, f64, RouteMetrics)> = None;

        // (a) every position in every existing route.
        for ri in 0..sol.routes.len() {
            let veh = &problem.vehicles[sol.routes[ri].vehicle_idx];
            if !veh.has_skills(req) {
                continue;
            }
            let old = sol.routes[ri].metrics.cost;
            for pos in 0..=sol.routes[ri].steps.len() {
                let mut cand = sol.routes[ri].steps.clone();
                cand.insert(pos, task);
                if let Ok(m) = evaluate_route(problem, matrix, veh, &cand) {
                    if m.distance >= UNREACHABLE_M {
                        continue;
                    }
                    let delta = m.cost - old;
                    if best.as_ref().map_or(true, |b| delta < b.1) {
                        best = Some((Slot::Existing(ri, pos), delta, m));
                    }
                }
            }
        }
        // (b) one fresh route per currently-unused vehicle.
        let used: std::collections::HashSet<usize> =
            sol.routes.iter().map(|r| r.vehicle_idx).collect();
        for vi in 0..problem.vehicles.len() {
            if used.contains(&vi) {
                continue;
            }
            let veh = &problem.vehicles[vi];
            if !veh.has_skills(req) {
                continue;
            }
            if let Ok(m) = evaluate_route(problem, matrix, veh, &[task]) {
                if m.distance >= UNREACHABLE_M {
                    continue;
                }
                if best.as_ref().map_or(true, |b| m.cost < b.1) {
                    best = Some((Slot::NewVehicle(vi), m.cost, m));
                }
            }
        }

        match best {
            Some((Slot::Existing(ri, pos), _, m)) => {
                sol.routes[ri].steps.insert(pos, task);
                sol.routes[ri].metrics = m;
                placed_any = true;
            }
            Some((Slot::NewVehicle(vi), _, m)) => {
                sol.routes.push(Route { vehicle_idx: vi, steps: vec![task], metrics: m });
                placed_any = true;
            }
            None => leftovers.push(task),
        }
    }

    sol.unassigned = leftovers;
    if !placed_any || sol.unassigned.is_empty() {
        break;
    }
    }
    sol.recompute_summary();
}

/// Build a matrix from whatever the problem provides.
///
/// Order of preference:
/// 1. A matrix already in `problem.matrices[profile]` for the first vehicle's profile.
/// 2. The provided `source` applied to the resolved coordinate list.
pub fn build_matrix(problem: &mut Problem, source: Option<&dyn MatrixSource>) -> Result<Matrix> {
    let profile = problem
        .vehicles
        .first()
        .map(|v| v.profile.clone())
        .unwrap_or_else(|| "car".to_string());
    if let Some(p) = problem.matrices.get(&profile) {
        return Matrix::from_provided(p);
    }
    let coords = resolve_coords(problem);
    if coords.is_empty() {
        return Err(Error::Invalid(
            "no coordinates found and no matrix provided".into(),
        ));
    }
    let src = source.ok_or_else(|| {
        Error::Invalid(
            "problem has no matrix and no MatrixSource was supplied (try haversine)".into(),
        )
    })?;
    src.build(&coords)
}

/// ILS perturbation: remove `frac` of all assigned tasks at random, then
/// reinsert each via cheapest-feasible insertion (probe + full eval). Tasks
/// that can't be placed feasibly land in `unassigned`. LS then drives the
/// result to a (hopefully different) local optimum.
///
/// Random *placement* would be cheaper but can leave the solution
/// infeasible; LS then can't recover because its delta-cost reasoning
/// assumes a feasible base. Feasible-cheapest is the safe choice.
///
/// `pub` so the population-polish in `crate::population` can reuse the
/// same destroy-and-repair logic.
pub fn kick<R: rand::Rng>(
    sol: &mut Solution,
    frac: f64,
    rng: &mut R,
    problem: &Problem,
    matrix: &Matrix,
) {
    use crate::eval::{precompute, try_insert_single};
    use crate::solution::evaluate_route;

    let mut assigned: Vec<(usize, usize)> = Vec::new();
    for (r, route) in sol.routes.iter().enumerate() {
        for i in 0..route.steps.len() {
            assigned.push((r, i));
        }
    }
    if assigned.is_empty() { return; }

    let n_kick = ((assigned.len() as f64) * frac).round() as usize;
    if n_kick == 0 { return; }
    assigned.shuffle(rng);
    assigned.truncate(n_kick);

    // Sort high→low step index so removals don't shift earlier ones.
    assigned.sort_by(|a, b| (b.0, b.1).cmp(&(a.0, a.1)));

    let mut pulled: Vec<TaskRef> = Vec::with_capacity(n_kick);
    for (r, i) in &assigned {
        if let Some(route) = sol.routes.get_mut(*r) {
            if *i < route.steps.len() {
                pulled.push(route.steps.remove(*i));
            }
        }
    }

    // Refresh metrics on every route (loads/timing changed).
    for r in 0..sol.routes.len() {
        let veh = &problem.vehicles[sol.routes[r].vehicle_idx];
        if let Ok(m) = evaluate_route(problem, matrix, veh, &sol.routes[r].steps) {
            sol.routes[r].metrics = m;
        }
    }
    sol.routes.retain(|r| !r.steps.is_empty());

    // Build precomp for each surviving route.
    let mut precomps: Vec<Option<crate::eval::RoutePrecomp>> = (0..sol.routes.len())
        .map(|r| precompute(problem, matrix, &problem.vehicles[sol.routes[r].vehicle_idx], r, &sol.routes[r].steps))
        .collect();

    pulled.shuffle(rng);

    // Regret-3 reinsertion: for each pulled task, track best, 2nd-best, AND
    // 3rd-best insertion across all (route, pos). Regret = (2nd + 3rd) − 2·best.
    // This generalizes regret-2 (which only used 2nd-best). Capturing the
    // 3rd-best alternative gives a better signal about how "trapped" a task
    // is — if both 2nd and 3rd are far from best, the task is highly
    // committed to its top choice and should be placed first.
    //
    // Empirically: regret-3 ≥ regret-2 ≥ regret-1 (= cheapest-feasible) on
    // CVRPTW (Potvin & Rousseau 1993).
    while !pulled.is_empty() {
        let mut scored: Vec<(f64, usize, usize, usize)> = Vec::new();
        for (idx, &t) in pulled.iter().enumerate() {
            let mut best: Option<(usize, usize, f64)> = None;
            let mut second: f64 = f64::INFINITY;
            let mut third: f64 = f64::INFINITY;
            for r in 0..sol.routes.len() {
                let Some(pre) = precomps[r].as_ref() else { continue; };
                let veh = &problem.vehicles[sol.routes[r].vehicle_idx];
                if !veh.has_skills(t.skills(problem)) { continue; }
                let positions = sol.routes[r].steps.len() + 2;
                for pos in 1..positions {
                    if let Some(d) = try_insert_single(pre, problem, matrix, veh, pos, t) {
                        match best {
                            None => best = Some((r, pos, d)),
                            Some((_, _, bd)) if d < bd => {
                                third = second;
                                second = bd;
                                best = Some((r, pos, d));
                            }
                            Some(_) if d < second => {
                                third = second;
                                second = d;
                            }
                            Some(_) if d < third => third = d,
                            _ => {}
                        }
                    }
                }
            }
            if let Some((r, pos, b)) = best {
                // Regret-3: (2nd + 3rd) − 2·best. Falls back to regret-2 when
                // 3rd is infinity (only ≤2 feasible insertions exist).
                let r2 = if second.is_finite() { second - b } else { 1e6 };
                let r3 = if third.is_finite() { third - b } else { r2 };
                let regret = r2 + r3;
                scored.push((regret, idx, r, pos));
            }
        }

        if scored.is_empty() {
            // No task is feasibly insertable anywhere — drop remaining.
            for t in pulled.drain(..) { sol.unassigned.push(t); }
            break;
        }

        // Pick highest-regret task.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let (_, idx, r, pos) = scored[0];
        let t = pulled.remove(idx);
        let mut cand = sol.routes[r].steps.clone();
        cand.insert(pos - 1, t);
        let veh = &problem.vehicles[sol.routes[r].vehicle_idx];
        if let Ok(m) = evaluate_route(problem, matrix, veh, &cand) {
            sol.routes[r].steps = cand;
            sol.routes[r].metrics = m;
            precomps[r] = precompute(problem, matrix, veh, r, &sol.routes[r].steps);
        } else {
            sol.unassigned.push(t);
        }
    }

    sol.recompute_summary();
}

