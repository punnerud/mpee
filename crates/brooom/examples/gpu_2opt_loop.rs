//! Phase-4 milestone: GPU-only 2-opt local search loop on a single
//! trajectory.
//!
//! The loop is *CPU-orchestrated* (variant A in the plan) but no CPU code
//! ever inspects, modifies, or evaluates the tour data — every iteration:
//!
//!   1. GPU: `precompute_all`     (TW + capacity + per-route metrics)
//!   2. GPU: `find_best_2opt_all` (per-route argmin, distance only)
//!   3. CPU: read 12 bytes per route → pick globally most-improving move
//!   4. GPU: `apply_2opt`         (reverse segment in place)
//!   5. GPU: `precompute_all` re-validates feasibility
//!   6. CPU: if new feasibility flag is 0, undo by re-applying the same
//!      move (2-opt is its own inverse) — distance-improving but TW-
//!      infeasible move; mark route as no-progress
//!   7. Loop until no improving feasible move remains.
//!
//! The persistent `tour_buf` on GPU is the single source of truth for the
//! tour throughout. CPU only reads small per-route summary records (best
//! moves, feasibility flags) — never the tour itself, until the very end.
//!
//! Run with:
//!   cargo run --release --example gpu_2opt_loop
//!   cargo run --release --example gpu_2opt_loop -- benchmarks/instances/r1_0200.json

use std::collections::HashSet;
use std::time::Instant;

use brooom::gpu_population::GpuPopulation;
use brooom::problem::Problem;
use brooom::solution::evaluate_route;
use brooom::solver::{solve_full, SolverConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "benchmarks/instances/r1_0100.json".into());

    println!("Loading {} ...", path);
    let f = std::fs::File::open(&path)?;
    let mut problem: Problem = brooom::io::parse_input_reader(std::io::BufReader::new(f))?;

    // Initial CPU solve — gives us a starting solution. We deliberately
    // skip CPU local search so the GPU loop has something to optimize.
    let cfg = SolverConfig {
        max_local_search_passes: 0,
        granular_k: Some(20),
        multi_start: 1,
        ils_iters: 0,
        ils_kick_size: 0.4,
        time_limit_ms: None,
        verbose: false,
        warm_start: None,
            ..Default::default()
    };
    let t0 = Instant::now();
    let solved = solve_full(&mut problem, None, cfg)?;
    println!(
        "Initial CPU solve: cost={:.1} routes={} ({:?})\n",
        solved.solution.summary.cost,
        solved.solution.routes.len(),
        t0.elapsed()
    );

    let n_loc = solved.matrix.n;

    // Per-location problem data.
    let veh = &problem.vehicles[0];
    let veh_cap = veh.capacity[0] as i32;
    let veh_tw = veh.time_window();
    let veh_tw_s = veh_tw.start as i32;
    let veh_tw_e = veh_tw.end as i32;

    let mut loc_service = vec![0i32; n_loc];
    let mut loc_demand = vec![0i32; n_loc];
    let mut loc_tw_s = vec![veh_tw_s; n_loc];
    let mut loc_tw_e = vec![veh_tw_e; n_loc];
    for j in &problem.jobs {
        let li = j.location.index.ok_or("job missing index")?;
        loc_service[li] = j.service as i32;
        loc_demand[li] = j.delivery.get(0).copied().unwrap_or(0) as i32;
        if let Some(tw) = j.time_windows.first() {
            loc_tw_s[li] = tw.start as i32;
            loc_tw_e[li] = tw.end as i32;
        }
    }

    // Build initial tours.
    let depot = veh.start.as_ref().and_then(|l| l.index).unwrap_or(0) as u32;
    let mut tours: Vec<Vec<u32>> = Vec::new();
    for r in &solved.solution.routes {
        let mut tour = vec![depot];
        for &task in &r.steps {
            let li = task.description(&problem).location.index
                .ok_or("step missing index")? as u32;
            tour.push(li);
        }
        tour.push(depot);
        tours.push(tour);
    }
    let n_routes = tours.len() as u32;

    let max_route_len = tours.iter().map(|r| r.len() as u32).max().unwrap_or(0) + 8;
    let tour_capacity = n_routes * max_route_len;

    let gpu = GpuPopulation::new(
        &solved.matrix.durations,
        n_loc as u32,
        1,
        n_routes,
        tour_capacity,
    )?;
    gpu.upload(&[tours.clone()])?;
    gpu.upload_problem_data(&loc_service, &loc_demand, &loc_tw_s, &loc_tw_e)?;
    gpu.upload_vehicle_data(&[veh_cap], &[veh_tw_s], &[veh_tw_e])?;

    gpu.precompute_all()?;
    let initial_metrics = gpu.read_all_route_metrics()?;
    let init_total_travel: i64 = initial_metrics[0].iter().map(|m| m.travel_time as i64).sum();
    let init_feas: bool = initial_metrics[0].iter().all(|m| m.feasible);
    println!(
        "Uploaded {n_routes} routes to GPU. Initial total travel = {} ({}).",
        init_total_travel,
        if init_feas { "all feasible" } else { "WARNING: some routes infeasible" }
    );

    // ---- The Phase-4 loop ----
    let loop_start = Instant::now();
    let mut iter = 0;
    let mut applied = 0;
    let mut rolled_back = 0;
    // (route, i, j) tuples we've tried and that broke feasibility — taboo.
    let mut tabu: HashSet<(u32, u32, u32)> = HashSet::new();

    loop {
        iter += 1;

        // 1. find best 2-opt per route
        gpu.find_best_2opt_all()?;
        let bests = gpu.read_best_2opt_all()?;

        // 2. globally most-improving (skipping tabu).
        let mut global_best: Option<(u32, u32, u32, i32)> = None; // (route, i, j, delta)
        for r in 0..n_routes as usize {
            let b = bests[r];
            if b.delta >= 0 || b.delta == i32::MAX { continue; }
            if tabu.contains(&(r as u32, b.i, b.j)) { continue; }
            match global_best {
                None => global_best = Some((r as u32, b.i, b.j, b.delta)),
                Some((_, _, _, gd)) if b.delta < gd => {
                    global_best = Some((r as u32, b.i, b.j, b.delta));
                }
                _ => {}
            }
        }

        let Some((r_idx, i, j, delta)) = global_best else {
            println!("Iter {iter}: no improving non-tabu move found — converged.");
            break;
        };

        // 3. apply on GPU.
        gpu.apply_2opt(0, r_idx, i, j)?;

        // 4. re-precompute and check feasibility.
        gpu.precompute_all()?;
        let metrics = gpu.read_all_route_metrics()?;
        if !metrics[0][r_idx as usize].feasible {
            // Roll back — apply same 2-opt to undo (it's its own inverse).
            gpu.apply_2opt(0, r_idx, i, j)?;
            gpu.precompute_all()?;
            tabu.insert((r_idx, i, j));
            rolled_back += 1;
            if rolled_back <= 3 {
                println!("Iter {iter}: route {r_idx} (i={i}, j={j}) Δ={delta} broke TW; rolled back.");
            }
            continue;
        }

        applied += 1;
        if applied <= 5 || applied % 10 == 0 {
            println!("Iter {iter}: applied 2-opt route={r_idx} i={i} j={j} Δ={delta}");
        }
        // Safety cutoff in case something pathological happens.
        if iter > 5000 { break; }
    }

    let loop_elapsed = loop_start.elapsed();
    let final_metrics = gpu.read_all_route_metrics()?;
    let final_travel: i64 = final_metrics[0].iter().map(|m| m.travel_time as i64).sum();
    let all_feas: bool = final_metrics[0].iter().all(|m| m.feasible);

    println!("\n=== Loop summary ===");
    println!("Iterations:       {iter}");
    println!("Applied moves:    {applied}");
    println!("Rolled back (TW): {rolled_back}");
    println!("Final feasible:   {all_feas}");
    println!("Initial travel:   {init_total_travel}");
    println!("Final travel:     {final_travel}");
    println!("Improvement:      {} ({:.2}%)",
             init_total_travel - final_travel,
             100.0 * (init_total_travel - final_travel) as f64 / init_total_travel as f64);
    println!("Wall time:        {:?}", loop_elapsed);
    println!("Per iteration:    {:?}", loop_elapsed / iter.max(1) as u32);

    // Cross-validate: read final tours back, run CPU evaluate_route, compare.
    let final_tours = gpu.read_back(0)?;
    let mut cpu_total: i64 = 0;
    let mut cpu_all_feas = true;
    for (r_idx, route) in solved.solution.routes.iter().enumerate() {
        let vehicle = &problem.vehicles[route.vehicle_idx];
        // Reconstruct steps from the GPU tour by mapping location → TaskRef::Job.
        // Build lookup once.
        let location_to_job: std::collections::HashMap<usize, brooom::solution::TaskRef> =
            problem.jobs.iter().enumerate().filter_map(|(idx, j)| {
                j.location.index.map(|li| (li, brooom::solution::TaskRef::Job(idx)))
            }).collect();

        let mut steps = Vec::new();
        // Skip first and last (depots).
        for &li in final_tours[r_idx].iter().skip(1).take(final_tours[r_idx].len() - 2) {
            if let Some(&tr) = location_to_job.get(&(li as usize)) {
                steps.push(tr);
            }
        }
        match evaluate_route(&problem, &solved.matrix, vehicle, &steps) {
            Ok(m) => cpu_total += m.travel_time,
            Err(e) => {
                println!("  ✗ CPU evaluate_route route {r_idx}: {e}");
                cpu_all_feas = false;
            }
        }
    }

    println!("\nCross-validation:");
    println!("  CPU evaluate_route total travel: {cpu_total} ({})",
             if cpu_all_feas { "all feasible" } else { "WARNING: infeasible" });
    if cpu_total == final_travel && cpu_all_feas {
        println!("  ✓ Phase 4 verified: GPU LS loop produces a feasible solution");
        println!("    that exactly matches CPU evaluate_route.");
    } else {
        println!("  ✗ mismatch ({} vs {})", final_travel, cpu_total);
    }

    Ok(())
}
