//! Phase-3 verification: GPU per-route metrics match CPU `evaluate_route`
//! on a real Solomon instance.
//!
//! Procedure:
//!   1. Load r1_0100.json, solve via brooom CPU.
//!   2. Build per-location problem data (service, demand, TW) keyed by
//!      matrix index.
//!   3. For each route in the solution, extract `[depot, stops..., depot]`
//!      as a Vec<u32> of matrix indices.
//!   4. Upload to a 1-trajectory GpuPopulation, run `precompute_all`.
//!   5. Compare GPU metrics (travel/service/waiting/end_time) to CPU
//!      `evaluate_route`.
//!
//! Run with:
//!   cargo run --release --example gpu_evaluate_correctness
//!   cargo run --release --example gpu_evaluate_correctness -- benchmarks/instances/r1_0200.json

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

    let cfg = SolverConfig {
        max_local_search_passes: 30,
        granular_k: Some(20),
        multi_start: 1,
        ils_iters: 0,
        ils_kick_size: 0.4,
        time_limit_ms: None,
        verbose: false,
        warm_start: None,
    };
    println!("Solving with brooom CPU...");
    let t0 = Instant::now();
    let solved = solve_full(&mut problem, None, cfg)?;
    println!(
        "  cost={:.1} routes={} unassigned={} ({:?})\n",
        solved.solution.summary.cost,
        solved.solution.routes.len(),
        solved.solution.unassigned.len(),
        t0.elapsed()
    );

    let n_loc = solved.matrix.n;
    println!("Matrix dim: {n_loc} locations\n");

    // Build per-location problem data.
    let mut loc_service = vec![0i32; n_loc];
    let mut loc_demand = vec![0i32; n_loc];
    let mut loc_tw_s = vec![0i32; n_loc];
    let mut loc_tw_e = vec![i32::MAX / 4; n_loc];
    for j in &problem.jobs {
        let li = j.location.index.ok_or("job without matrix index")?;
        loc_service[li] = j.service as i32;
        loc_demand[li] = j.delivery.get(0).copied().unwrap_or(0) as i32;
        if let Some(tw) = j.time_windows.first() {
            loc_tw_s[li] = tw.start as i32;
            loc_tw_e[li] = tw.end as i32;
        }
    }

    // Vehicle params (homogeneous fleet — Solomon instances have all vehs equal).
    let veh = &problem.vehicles[0];
    let veh_cap = veh.capacity[0] as i32;
    let veh_tw = veh.time_window();
    let veh_tw_s = veh_tw.start as i32;
    let veh_tw_e = veh_tw.end as i32;
    println!("Vehicle: cap={veh_cap}, TW=[{veh_tw_s}, {veh_tw_e}]");

    // For depot (matrix idx 0): service=0, demand=0, TW=vehicle TW.
    loc_service[0] = 0;
    loc_demand[0] = 0;
    loc_tw_s[0] = veh_tw_s;
    loc_tw_e[0] = veh_tw_e;

    // Build trajectories: 1 trajectory containing all routes from the solution.
    let n_routes = solved.solution.routes.len();
    let max_route_stops = solved.solution.routes.iter()
        .map(|r| r.steps.len())
        .max()
        .unwrap_or(0);
    println!("Solution: {n_routes} routes, longest = {max_route_stops} stops\n");

    let mut tours: Vec<Vec<u32>> = Vec::with_capacity(n_routes);
    for r in &solved.solution.routes {
        let depot = veh.start.as_ref().and_then(|l| l.index).unwrap_or(0) as u32;
        let mut tour = vec![depot];
        for &task in &r.steps {
            let li = task.description(&problem).location.index
                .ok_or("step without matrix index")? as u32;
            tour.push(li);
        }
        tour.push(depot);
        tours.push(tour);
    }

    let max_routes = (n_routes as u32).max(1);
    // Fixed-slot layout: each route gets max_route_len slots in the tour
    // buffer regardless of actual length. Slot size has to accommodate
    // the longest route + headroom for any future grow-ops.
    let max_route_len = tours.iter().map(|r| r.len() as u32).max().unwrap_or(0) + 8;
    let tour_capacity = max_routes * max_route_len;

    let gpu = GpuPopulation::new(
        &solved.matrix.durations,
        n_loc as u32,
        1, // pop_size = 1
        max_routes,
        tour_capacity,
    )?;

    gpu.upload(&[tours.clone()])?;
    gpu.upload_problem_data(&loc_service, &loc_demand, &loc_tw_s, &loc_tw_e)?;
    gpu.upload_vehicle_data(&[veh_cap], &[veh_tw_s], &[veh_tw_e])?;

    let t0 = Instant::now();
    gpu.precompute_all()?;
    let dispatch_us = t0.elapsed().as_secs_f64() * 1e6;
    println!("GPU precompute_all on {n_routes} routes: {:.1} µs", dispatch_us);

    // Compare per-route metrics.
    let gpu_metrics = gpu.read_all_route_metrics()?;
    let traj_metrics = &gpu_metrics[0];

    let mut all_match = true;
    let mut total_travel_cpu: i64 = 0;
    let mut total_travel_gpu: i64 = 0;

    for (r_idx, route) in solved.solution.routes.iter().enumerate() {
        let vehicle = &problem.vehicles[route.vehicle_idx];
        let cpu_metrics = evaluate_route(&problem, &solved.matrix, vehicle, &route.steps)
            .map_err(|e| format!("evaluate_route route {r_idx}: {e}"))?;
        let gpu = &traj_metrics[r_idx];

        total_travel_cpu += cpu_metrics.travel_time;
        total_travel_gpu += gpu.travel_time as i64;

        let cpu_t = cpu_metrics.travel_time;
        let cpu_s = cpu_metrics.service_time;
        let cpu_w = cpu_metrics.waiting_time;
        let cpu_e = cpu_metrics.end_time;

        let mismatch = (cpu_t as i32) != gpu.travel_time
            || (cpu_s as i32) != gpu.service_time
            || (cpu_w as i32) != gpu.waiting_time
            || (cpu_e as i32) != gpu.end_time
            || !gpu.feasible;

        if mismatch {
            all_match = false;
            println!(
                "  ✗ route {r_idx} ({} stops):\n     cpu travel={cpu_t} svc={cpu_s} wait={cpu_w} end={cpu_e} feas=true\n     gpu travel={} svc={} wait={} end={} feas={}",
                route.steps.len(),
                gpu.travel_time, gpu.service_time, gpu.waiting_time, gpu.end_time, gpu.feasible,
            );
        }
    }

    println!("\nTotal travel time: CPU={} s, GPU={} s", total_travel_cpu, total_travel_gpu);

    if all_match {
        println!("\n  ✓ Phase 3 verified: GPU metrics match CPU evaluate_route on all {n_routes} routes.");
    } else {
        return Err("Phase 3 mismatch".into());
    }

    // Bench: how fast is GPU evaluation of all routes?
    let runs = 50;
    let t0 = Instant::now();
    for _ in 0..runs {
        gpu.precompute_all()?;
    }
    let avg_gpu_us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;

    let t0 = Instant::now();
    for _ in 0..runs {
        for route in &solved.solution.routes {
            let vehicle = &problem.vehicles[route.vehicle_idx];
            let _ = evaluate_route(&problem, &solved.matrix, vehicle, &route.steps);
        }
    }
    let avg_cpu_us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;

    println!(
        "\nThroughput on {n_routes} routes: GPU {:.1} µs/dispatch, CPU {:.1} µs/sweep, ratio {:.2}×",
        avg_gpu_us, avg_cpu_us, avg_cpu_us / avg_gpu_us,
    );
    println!("(GPU dispatch cost ~2 ms is the floor; CPU eval_route is microseconds per route.)");

    Ok(())
}
