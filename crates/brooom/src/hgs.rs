//! Hybrid Genetic Search (HGS) with GPU-accelerated LS-education.
//!
//! Population-based metaheuristic that combines genetic recombination
//! (SREX-style route exchange crossover) with local-search "education"
//! of every offspring. The LS-education runs in batch on the GPU
//! megakernel — pop_size trajectories per dispatch.
//!
//! Loosely follows Vidal et al. (2014) "A Hybrid Genetic Algorithm for
//! Multidepot and Periodic Vehicle Routing Problems" and the PyVRP/
//! HGS-CVRP implementations. The novelty here is offloading the LS
//! education to a single GPU dispatch per generation.
//!
//! MVP scope:
//! - Job-only (no pickup/delivery shipments)
//! - Homogeneous fleet (single TW + capacity)
//! - Generational replacement (pop_size offspring per generation)
//! - Random parent selection (full tournament for V2)

use std::collections::{HashMap, HashSet};

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::error::Error;
use crate::granular::Granular;
use crate::gpu_population::GpuPopulation;
use crate::matrix::Matrix;
use crate::problem::Problem;
use crate::solution::{evaluate_route, Route, Solution, TaskRef};

#[derive(Debug, Clone)]
pub struct HgsConfig {
    pub pop_size: u32,
    /// Max generations. Ignored if `time_limit_ms` is set and elapses first.
    pub max_generations: u32,
    pub time_limit_ms: Option<u64>,
    /// Fraction of parent A's routes copied to offspring (0.3-0.5 typical).
    pub crossover_route_frac: f64,
    pub verbose: bool,
}

impl Default for HgsConfig {
    fn default() -> Self {
        Self {
            pop_size: 64,
            max_generations: 100,
            time_limit_ms: Some(30_000),
            crossover_route_frac: 0.4,
            verbose: false,
        }
    }
}

/// Run HGS-GPU and return the best solution found.
///
/// `initial` is the seed solution (typically from `greedy_insertion` or
/// `solve_with_matrix`). We diversify it via random restart + crossover
/// to build the initial population.
pub fn solve_hgs(
    problem: &Problem,
    matrix: &Matrix,
    granular: &Granular,
    initial: &Solution,
    config: &HgsConfig,
) -> Result<Solution, Error> {
    let t_start = std::time::Instant::now();
    let deadline = config.time_limit_ms.map(|ms| {
        t_start + std::time::Duration::from_millis(ms)
    });
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xC0FFEE);

    // 1. Build initial population via seeded insertion variants.
    let mut population = build_initial_population(
        problem, matrix, granular, initial, config, &mut rng,
    )?;
    if config.verbose {
        let best_cost = population.iter().map(|s| s.summary.cost).fold(f64::INFINITY, f64::min);
        eprintln!(
            "hgs: initial pop = {} individuals, best cost = {:.2} (t={:.2}s)",
            population.len(), best_cost, t_start.elapsed().as_secs_f64()
        );
    }

    // 2. Generation loop.
    let mut best_cost = population.iter().map(|s| s.summary.cost).fold(f64::INFINITY, f64::min);
    let mut generations = 0u32;
    let mut stale_gens = 0u32;
    let task_count: usize = problem.jobs.len();

    for gen in 0..config.max_generations {
        if let Some(d) = deadline {
            if std::time::Instant::now() >= d { break; }
        }
        generations = gen + 1;

        // 2a. Generate `pop_size` offspring via tournament + SREX-light crossover.
        let pop_size = config.pop_size as usize;
        let mut offspring: Vec<Solution> = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let pa_idx = tournament_select(&population, 2, &mut rng);
            let pb_idx = tournament_select(&population, 2, &mut rng);
            let pa = &population[pa_idx];
            let pb = &population[pb_idx];
            let child = srex_crossover(pa, pb, problem, matrix, config.crossover_route_frac, &mut rng);
            offspring.push(child);
        }
        // Filter offspring with task leaks before educate.
        offspring.retain(|s| {
            let assigned: usize = s.routes.iter().map(|r| r.steps.len()).sum();
            assigned == task_count
        });
        if offspring.is_empty() {
            stale_gens += 1;
            if stale_gens > 5 { break; }
            continue;
        }

        // 2b. LS-educate offspring on GPU.
        let educated = match educate_batch(&offspring, problem, matrix, granular) {
            Ok(e) => e,
            Err(e) => {
                if config.verbose { eprintln!("hgs: educate_batch failed: {e}"); }
                break;
            }
        };

        // 2c. Diversity-aware replacement: among (population ∪ educated),
        // keep the best pop_size considering both cost AND structural
        // diversity. The cheapest individual is always kept; the rest
        // are scored by a combined "fitness" = cost_rank + diversity_rank.
        let mut union: Vec<Solution> = population.iter().cloned().chain(educated).collect();
        union = select_diverse(&union, pop_size);
        // Sort by cost so population[0] is always the cheapest.
        union.sort_by(|a, b| a.summary.cost.partial_cmp(&b.summary.cost).unwrap_or(std::cmp::Ordering::Equal));
        population = union;

        let new_best = population[0].summary.cost;
        if new_best + 1e-9 < best_cost {
            if config.verbose {
                eprintln!(
                    "hgs: gen {gen}: best {:.2} → {:.2} (Δ={:.2}, t={:.2}s)",
                    best_cost, new_best, best_cost - new_best,
                    t_start.elapsed().as_secs_f64()
                );
            }
            best_cost = new_best;
            stale_gens = 0;
        } else {
            stale_gens += 1;
            // Restart trigger: if diversity is dead, regenerate bottom half.
            if stale_gens >= 8 {
                if config.verbose { eprintln!("hgs: gen {gen}: restart triggered (stale_gens={stale_gens})"); }
                // Keep top 25% (lowest cost), regenerate rest from random seeds + educate.
                // population is already sorted by cost (above), so truncate keeps the best.
                let keep = (pop_size / 4).max(1);
                population.truncate(keep);
                while population.len() < pop_size {
                    let seed = rng.gen::<u64>();
                    let mut sol = crate::insertion::greedy_insertion_seeded(problem, matrix, seed);
                    sol.recompute_summary(problem);
                    population.push(sol);
                }
                // Educate the new ones.
                if let Ok(educated) = educate_batch(&population[keep..], problem, matrix, granular) {
                    for (i, e) in educated.into_iter().enumerate() {
                        population[keep + i] = e;
                    }
                }
                population.sort_by(|a, b| a.summary.cost.partial_cmp(&b.summary.cost).unwrap_or(std::cmp::Ordering::Equal));
                stale_gens = 0;
            }
        }
    }

    if config.verbose {
        eprintln!(
            "hgs: done, {generations} generations, best cost = {:.2}, t={:.2}s",
            population[0].summary.cost, t_start.elapsed().as_secs_f64()
        );
    }
    Ok(population[0].clone())
}

// ---- Population init ----

fn build_initial_population(
    problem: &Problem,
    matrix: &Matrix,
    granular: &Granular,
    initial: &Solution,
    config: &HgsConfig,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> Result<Vec<Solution>, Error> {
    let pop_size = config.pop_size as usize;
    let mut pop: Vec<Solution> = Vec::with_capacity(pop_size);
    pop.push(initial.clone());
    while pop.len() < pop_size {
        let seed: u64 = rng.gen();
        let mut sol = crate::insertion::greedy_insertion_seeded(problem, matrix, seed);
        sol.recompute_summary(problem);
        pop.push(sol);
    }
    // Educate everyone with one LS pass on GPU.
    let educated = educate_batch(&pop, problem, matrix, granular)?;
    Ok(educated)
}

// ---- Tournament selection ----

// ---- Diversity-aware selection ----
//
// Bag-of-edges fingerprint: for each individual, hash its directed edge
// set (depot → first stop, first → second, ..., last → depot). Pairwise
// "broken-pair distance" = |edges(A) ∆ edges(B)| / 2, divisor by 2 since
// a swap appears once in each direction.
//
// Selection criterion (Vidal): rank by COST and DIVERSITY-CONTRIBUTION
// separately, take sum of ranks. Diversity contribution = average
// distance to other individuals in population.

fn edge_set(sol: &Solution) -> HashSet<(u32, u32)> {
    let mut edges = HashSet::new();
    for r in &sol.routes {
        let mut prev: u32 = 0;  // depot symbolic; we don't have direct depot id here
        // Use a sentinel (u32::MAX) for depot since TaskRef doesn't expose location index
        // directly without &Problem. For the bag-of-edges fingerprint we use the
        // TaskRef as the "vertex" instead — encoded as (kind, idx) packed into u32.
        let _ = prev;
        let mut last: u64 = 0;  // dummy depot encoding = 0
        for &t in &r.steps {
            let v = match t {
                TaskRef::Job(i) => (i as u64) << 2 | 1,
                TaskRef::Pickup(i) => (i as u64) << 2 | 2,
                TaskRef::Delivery(i) => (i as u64) << 2 | 3,
            };
            edges.insert((last as u32, v as u32));
            last = v;
        }
        edges.insert((last as u32, 0));  // back to depot
    }
    edges
}

fn select_diverse(pool: &[Solution], target_size: usize) -> Vec<Solution> {
    if pool.len() <= target_size {
        return pool.to_vec();
    }
    // Compute edge-set fingerprints once.
    let fingerprints: Vec<HashSet<(u32, u32)>> = pool.iter().map(edge_set).collect();
    let n = pool.len();

    // For each i, average broken-pair distance to all others.
    let mut diversity: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            if i == j { continue; }
            let d = fingerprints[i].symmetric_difference(&fingerprints[j]).count() as f64;
            sum += d / 2.0;
        }
        diversity[i] = sum / (n - 1) as f64;
    }

    // Rank by cost (ascending) and diversity (descending). Sum of ranks
    // = combined fitness; lower is better.
    let mut cost_order: Vec<usize> = (0..n).collect();
    cost_order.sort_by(|&a, &b| pool[a].summary.cost.partial_cmp(&pool[b].summary.cost).unwrap());
    let mut cost_rank: Vec<usize> = vec![0; n];
    for (rank, &idx) in cost_order.iter().enumerate() { cost_rank[idx] = rank; }

    let mut div_order: Vec<usize> = (0..n).collect();
    div_order.sort_by(|&a, &b| diversity[b].partial_cmp(&diversity[a]).unwrap());
    let mut div_rank: Vec<usize> = vec![0; n];
    for (rank, &idx) in div_order.iter().enumerate() { div_rank[idx] = rank; }

    // Combined: 70% cost, 30% diversity (typical HGS weighting).
    let mut combined: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, 0.7 * cost_rank[i] as f64 + 0.3 * div_rank[i] as f64))
        .collect();
    combined.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    combined.truncate(target_size);
    // Ensure the absolute best-cost individual is always kept.
    let best_cost_idx = cost_order[0];
    if !combined.iter().any(|(i, _)| *i == best_cost_idx) {
        combined.pop();
        combined.push((best_cost_idx, 0.0));
    }
    combined.into_iter().map(|(i, _)| pool[i].clone()).collect()
}

fn tournament_select(
    pop: &[Solution],
    k: usize,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> usize {
    let mut best_idx = rng.gen_range(0..pop.len());
    let mut best_cost = pop[best_idx].summary.cost;
    for _ in 1..k {
        let i = rng.gen_range(0..pop.len());
        if pop[i].summary.cost < best_cost {
            best_idx = i;
            best_cost = pop[i].summary.cost;
        }
    }
    best_idx
}

// ---- Proper SREX (Selective Route Exchange) crossover ----
//
// Per Nagata 2010 / Vidal et al.'s HGS-CVRP:
//   1. Pick n_select random routes from parent A (the "selected" set).
//   2. Compute the set V_A of customers visited by those routes.
//   3. Find n_select routes in B whose customers overlap V_A most.
//   4. Build offspring:
//        - All UNSELECTED routes from A (preserved as-is)
//        - All SELECTED routes from B (preserved as-is)
//   5. Remove duplicates (a task in B's added route may also be in A's
//      kept route): keep first occurrence, strip from later.
//   6. Reinsert missing tasks (in A's selected but not in B's added)
//      via cheapest-feasible-position.
//
// This preserves complete route STRUCTURES from both parents, unlike
// "take subset of A + reinsert B's tasks" which loses B's structure.
fn srex_crossover(
    pa: &Solution,
    pb: &Solution,
    problem: &Problem,
    matrix: &Matrix,
    route_frac: f64,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> Solution {
    let n_routes_a = pa.routes.len();
    let n_routes_b = pb.routes.len();
    if n_routes_a == 0 || n_routes_b == 0 {
        return pa.clone();
    }
    let n_select = ((n_routes_a as f64 * route_frac).round() as usize)
        .max(1)
        .min(n_routes_a)
        .min(n_routes_b);

    // 1. Pick n_select random A-indices.
    let mut a_idx_pool: Vec<usize> = (0..n_routes_a).collect();
    a_idx_pool.shuffle(rng);
    let a_selected: HashSet<usize> = a_idx_pool.into_iter().take(n_select).collect();

    // 2. Compute customer set of A's selected routes.
    let mut v_a: HashSet<TaskRef> = HashSet::new();
    for (i, r) in pa.routes.iter().enumerate() {
        if a_selected.contains(&i) {
            for &t in &r.steps { v_a.insert(t); }
        }
    }

    // 3. Score each B route by overlap with V_A; pick top n_select.
    let mut b_overlap: Vec<(usize, usize)> = pb.routes.iter().enumerate()
        .map(|(i, r)| {
            let ov = r.steps.iter().filter(|t| v_a.contains(t)).count();
            (i, ov)
        })
        .collect();
    b_overlap.sort_by(|a, b| b.1.cmp(&a.1));  // descending
    let b_selected: HashSet<usize> = b_overlap.into_iter()
        .take(n_select)
        .map(|(i, _)| i)
        .collect();

    // 4. Build offspring: A's unselected routes + B's selected routes.
    let mut child_routes: Vec<Route> = Vec::with_capacity(n_routes_a);
    for (i, r) in pa.routes.iter().enumerate() {
        if !a_selected.contains(&i) { child_routes.push(r.clone()); }
    }
    for (i, r) in pb.routes.iter().enumerate() {
        if b_selected.contains(&i) { child_routes.push(r.clone()); }
    }

    // 5. De-duplicate. A task might appear in both A's kept and B's added.
    let mut seen: HashSet<TaskRef> = HashSet::new();
    for route in &mut child_routes {
        route.steps.retain(|t| seen.insert(*t));
    }
    // Recompute metrics for any route whose contents changed (a deduped
    // route's metrics no longer match its current steps).
    for route in &mut child_routes {
        let vehicle = &problem.vehicles[route.vehicle_idx];
        if let Ok(m) = evaluate_route(problem, matrix, vehicle, &route.steps) {
            route.metrics = m;
        }
    }
    // Drop empty routes (could happen if all their steps were duplicates).
    child_routes.retain(|r| !r.steps.is_empty());

    // 6. Find missing tasks (were in V_A but not in current child).
    let child_tasks: HashSet<TaskRef> = child_routes.iter()
        .flat_map(|r| r.steps.iter().copied())
        .collect();
    let missing: Vec<TaskRef> = v_a.iter()
        .filter(|t| !child_tasks.contains(t))
        .copied()
        .collect();

    let mut child = Solution {
        routes: child_routes,
        unassigned: Vec::new(),
        summary: Default::default(),
    };

    // 7. Reinsert missing tasks at cheapest feasible position. Order them
    // by demand descending so big tasks get prime real estate first.
    let mut missing = missing;
    missing.sort_by_key(|t| {
        let job = t.description(problem);
        -(job.delivery.first().copied().unwrap_or(0) as i64)
    });
    for t in missing {
        let inserted = try_insert_cheapest_feasible(&mut child, t, problem, matrix);
        if !inserted {
            // Try opening a new route.
            let vehicle_idx = choose_unused_vehicle(&child, problem);
            if vehicle_idx < problem.vehicles.len() {
                let vehicle = &problem.vehicles[vehicle_idx];
                let steps = vec![t];
                match evaluate_route(problem, matrix, vehicle, &steps) {
                    Ok(metrics) => child.routes.push(Route { vehicle_idx, steps, metrics }),
                    Err(_) => child.unassigned.push(t),
                }
            } else {
                child.unassigned.push(t);
            }
        }
    }
    child.recompute_summary(problem);
    child
}

fn try_insert_cheapest_feasible(
    sol: &mut Solution,
    task: TaskRef,
    problem: &Problem,
    matrix: &Matrix,
) -> bool {
    let mut best: Option<(usize, usize, f64)> = None;  // (route_idx, pos, new_cost)
    let mut scratch: Vec<TaskRef> = Vec::with_capacity(16);
    for (r_idx, route) in sol.routes.iter().enumerate() {
        let vehicle = &problem.vehicles[route.vehicle_idx];
        for pos in 0..=route.steps.len() {
            scratch.clear();
            scratch.extend_from_slice(&route.steps[..pos]);
            scratch.push(task);
            scratch.extend_from_slice(&route.steps[pos..]);
            match evaluate_route(problem, matrix, vehicle, &scratch) {
                Ok(m) => {
                    let candidate_cost = m.cost;
                    let delta_cost = candidate_cost - route.metrics.cost;
                    match best {
                        None => best = Some((r_idx, pos, delta_cost)),
                        Some((_, _, bd)) if delta_cost < bd => best = Some((r_idx, pos, delta_cost)),
                        _ => {}
                    }
                }
                Err(_) => continue,
            }
        }
    }
    match best {
        Some((r_idx, pos, _)) => {
            let vehicle_idx = sol.routes[r_idx].vehicle_idx;
            let vehicle = &problem.vehicles[vehicle_idx];
            sol.routes[r_idx].steps.insert(pos, task);
            sol.routes[r_idx].metrics = evaluate_route(problem, matrix, vehicle, &sol.routes[r_idx].steps)
                .expect("insertion that passed eval shouldn't fail re-eval");
            true
        }
        None => false,
    }
}

fn choose_unused_vehicle(sol: &Solution, problem: &Problem) -> usize {
    let used: HashSet<usize> = sol.routes.iter().map(|r| r.vehicle_idx).collect();
    for v in 0..problem.vehicles.len() {
        if !used.contains(&v) { return v; }
    }
    // No unused vehicle; reuse last (this is a "soft fail" — fleet limit
    // breached but we keep the task assigned).
    problem.vehicles.len().saturating_sub(1).max(0)
}

// ---- Batch GPU LS-education ----

/// Upload `individuals` to the megakernel as pop_size trajectories and
/// run one LS pass each (no kick). Returns educated solutions in the
/// same order, with re-evaluated metrics from CPU evaluate_route.
fn educate_batch(
    individuals: &[Solution],
    problem: &Problem,
    matrix: &Matrix,
    granular: &Granular,
) -> Result<Vec<Solution>, Error> {
    if individuals.is_empty() {
        return Ok(Vec::new());
    }
    let n_loc = matrix.n;
    let veh0 = &problem.vehicles[0];
    let veh_cap = veh0.capacity.get(0).copied().unwrap_or(0) as i32;
    let veh_tw = veh0.time_window();
    let depot = veh0.start.as_ref().and_then(|l| l.index).unwrap_or(0) as u32;

    // Per-location problem data.
    let mut loc_service = vec![0i32; n_loc];
    let mut loc_demand = vec![0i32; n_loc];
    let mut loc_tw_s = vec![veh_tw.start as i32; n_loc];
    let mut loc_tw_e = vec![veh_tw.end as i32; n_loc];
    for j in &problem.jobs {
        let li = j.location.index.ok_or_else(|| Error::Other("hgs: job missing location index".into()))?;
        loc_service[li] = j.service as i32;
        loc_demand[li] = j.delivery.get(0).copied().unwrap_or(0) as i32;
        if let Some(tw) = j.time_windows.first() {
            loc_tw_s[li] = tw.start as i32;
            loc_tw_e[li] = tw.end as i32;
        }
    }

    // Convert individuals to GPU tours. All individuals must have the
    // same n_routes (max across them) to fit fixed-slot layout.
    let max_routes = individuals.iter().map(|s| s.routes.len()).max().unwrap_or(0);
    if max_routes == 0 { return Ok(individuals.to_vec()); }
    let max_route_len_actual = individuals.iter()
        .flat_map(|s| s.routes.iter())
        .map(|r| r.steps.len() + 2)
        .max().unwrap_or(0);
    let max_route_len = (max_route_len_actual as u32) + 16;
    let tour_capacity = (max_routes as u32) * max_route_len;

    let pop_size = individuals.len() as u32;
    let gpu = GpuPopulation::new(
        &matrix.durations,
        n_loc as u32,
        pop_size,
        max_routes as u32,
        tour_capacity,
    )?;

    // Build tour data for each individual. Pad routes with empty
    // depot-only routes so each has max_routes entries.
    let mut all_tours: Vec<Vec<Vec<u32>>> = Vec::with_capacity(individuals.len());
    for ind in individuals {
        let mut tours: Vec<Vec<u32>> = Vec::with_capacity(max_routes);
        for r in &ind.routes {
            let mut tour = vec![depot];
            for s in &r.steps {
                let li = s.description(problem).location.index.ok_or_else(|| {
                    Error::Other("hgs: step missing location index".into())
                })? as u32;
                tour.push(li);
            }
            tour.push(depot);
            tours.push(tour);
        }
        while tours.len() < max_routes {
            tours.push(vec![depot, depot]);
        }
        all_tours.push(tours);
    }
    gpu.upload(&all_tours)?;
    gpu.upload_problem_data(&loc_service, &loc_demand, &loc_tw_s, &loc_tw_e)?;
    let veh_cap_vec = vec![veh_cap; pop_size as usize];
    let veh_tw_s_vec = vec![veh_tw.start as i32; pop_size as usize];
    let veh_tw_e_vec = vec![veh_tw.end as i32; pop_size as usize];
    gpu.upload_vehicle_data(&veh_cap_vec, &veh_tw_s_vec, &veh_tw_e_vec)?;

    // Granular K-NN. GPU caps K at 64 (MAX_GRANULAR_K); truncate if needed.
    let k_orig = granular.k();
    let k = k_orig.min(64);
    let mut near_flat: Vec<u32> = Vec::with_capacity(n_loc * k);
    for i in 0..n_loc {
        let mut found = 0;
        for nb in granular.neighbors(i) {
            if found >= k { break; }
            near_flat.push(nb as u32);
            found += 1;
        }
        while found < k {
            near_flat.push(i as u32);
            found += 1;
        }
    }
    gpu.upload_granular(&near_flat, k as u32)?;

    // Run batch megakernel. HGS_EDUCATE_KICK env var lets us add a kick
    // during educate (0 = pure LS only, 3 = light perturbation). Light
    // kick gives offspring a chance to escape SREX-induced local opts.
    let kick: u32 = std::env::var("HGS_EDUCATE_KICK")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let kick_seed: u32 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32).unwrap_or(0xCAFEBABE);
    let max_iter = if n_loc >= 1000 { 1500 } else { 800 };
    gpu.run_megakernel_2opt_batch_with_kick(max_iter, kick, kick_seed)?;
    let all_final = gpu.read_back_all()?;

    // Build location → TaskRef lookup.
    let mut loc_to_task: HashMap<usize, TaskRef> = HashMap::new();
    for (idx, j) in problem.jobs.iter().enumerate() {
        if let Some(li) = j.location.index {
            loc_to_task.insert(li, TaskRef::Job(idx));
        }
    }

    // Reconstruct each Solution from its returned tours. Tours have
    // max_routes slots per trajectory; the megakernel may have moved
    // tasks between slots (e.g. relocate inter-route between a "real"
    // slot and a previously-padded slot). So we iterate over ALL slots
    // and treat any slot with interior customers (length > 2) as a real
    // route. Empty slots ([depot, depot]) are dropped.
    let mut educated: Vec<Solution> = Vec::with_capacity(individuals.len());
    for (t_idx, ind) in individuals.iter().enumerate() {
        let tours = &all_final[t_idx];
        let mut new_routes: Vec<Route> = Vec::with_capacity(tours.len());
        let mut ok = true;
        for (r_idx, tour) in tours.iter().enumerate() {
            // Skip empty (depot-only) routes.
            if tour.len() < 3 { continue; }
            // Vehicle assignment: use ind.routes[r_idx] if available,
            // otherwise the r_idx-th vehicle in the fleet. For homogeneous
            // fleets this is sufficient.
            let veh_idx = if r_idx < ind.routes.len() {
                ind.routes[r_idx].vehicle_idx
            } else {
                r_idx.min(problem.vehicles.len() - 1)
            };
            let vehicle = &problem.vehicles[veh_idx];
            let mut steps: Vec<TaskRef> = Vec::with_capacity(tour.len() - 2);
            for &li in tour.iter().skip(1).take(tour.len() - 2) {
                match loc_to_task.get(&(li as usize)) {
                    Some(&tr) => steps.push(tr),
                    None => { ok = false; break; }
                }
            }
            if !ok { break; }
            match evaluate_route(problem, matrix, vehicle, &steps) {
                Ok(m) => new_routes.push(Route { vehicle_idx: veh_idx, steps, metrics: m }),
                Err(_) => { ok = false; break; }
            }
        }
        if ok {
            let mut s = Solution {
                routes: new_routes,
                unassigned: ind.unassigned.clone(),
                summary: Default::default(),
            };
            s.recompute_summary(problem);
            // Verify all tasks present (no leak).
            let n_assigned: usize = s.routes.iter().map(|r| r.steps.len()).sum();
            let n_orig_assigned: usize = ind.routes.iter().map(|r| r.steps.len()).sum();
            if n_assigned == n_orig_assigned {
                // Accept the educated solution unconditionally if feasible —
                // even slight increases are tolerated since selection downstream
                // ranks by cost. Diversity is preserved this way.
                educated.push(s);
            } else {
                educated.push(ind.clone());
            }
        } else {
            educated.push(ind.clone());
        }
    }
    Ok(educated)
}
