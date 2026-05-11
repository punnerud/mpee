//! Megakernel batch demo: solve N trajectories of the same problem in
//! a single GPU dispatch via multi-workgroup megakernel (Phase 8).
//!
//! Each trajectory is the same problem but starting from a different
//! perturbed insertion order. After the megakernel converges, best-of-N
//! gives population-mode quality on a single GPU dispatch.
//!
//! Comparison points:
//!   - Sequential megakernel: dispatch once per trajectory, sum wallclock
//!   - Batch megakernel: dispatch once for all trajectories
//!   - CPU sequential: same N solves on CPU (1 thread)
//!
//! Run with:
//!   cargo run --release --example gpu_megakernel_batch
//!   cargo run --release --example gpu_megakernel_batch -- benchmarks/instances/r1_0250.json 32

use std::time::Instant;

use brooom::gpu_population::GpuPopulation;
use brooom::problem::Problem;
use brooom::solution::evaluate_route;
use brooom::solver::{solve_full, SolverConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "benchmarks/instances/r1_0100.json".into());
    let pop_size: u32 = std::env::args().nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    println!("Loading {} (pop_size={pop_size}) ...", path);
    let f = std::fs::File::open(&path)?;
    let mut problem: Problem = brooom::io::parse_input_reader(std::io::BufReader::new(f))?;

    // Get a baseline LS-converged solution to use as the "reference start"
    // for all trajectories. We then perturb each trajectory with a
    // different seed to give the megakernel work to do.
    let cfg = SolverConfig {
        max_local_search_passes: 0,  // insertion-only baseline so megakernel has work
        granular_k: Some(20),
        multi_start: 1,
        ils_iters: 0,
        ils_kick_size: 0.4,
        time_limit_ms: None,
        verbose: false,
        warm_start: None,
    };
    let solved = solve_full(&mut problem, None, cfg)?;
    let initial_travel: i64 = solved.solution.routes.iter().map(|r| r.metrics.travel_time).sum();
    println!("Insertion baseline: cost={:.1} routes={} travel={}",
        solved.solution.summary.cost, solved.solution.routes.len(), initial_travel);

    let n_loc = solved.matrix.n;
    let veh = &problem.vehicles[0];
    let depot = veh.start.as_ref().and_then(|l| l.index).unwrap_or(0) as u32;
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

    // Build trajectory_size copies of the route set, each with a small
    // permutation seed.
    let base_tours: Vec<Vec<u32>> = solved.solution.routes.iter().map(|r| {
        let mut tour = vec![depot];
        for &task in &r.steps {
            tour.push(task.description(&problem).location.index.unwrap() as u32);
        }
        tour.push(depot);
        tour
    }).collect();
    let n_routes = base_tours.len() as u32;

    // For this throughput demo, all trajectories start identical. Real
    // population diversity comes from ILS-kick + reinsert (Phase 7) which
    // preserves feasibility while providing distinct starting points.
    // Random permutations of the tour break TW from the start, leaving
    // the megakernel with no feasible 2-opt moves — so we hold the start
    // fixed here and measure just the dispatch-cost saving.
    let trajectories: Vec<Vec<Vec<u32>>> = (0..pop_size)
        .map(|_| base_tours.clone())
        .collect();

    // Fixed-slot layout with extra headroom for future ILS-kick growth.
    let max_route_len = trajectories[0].iter().map(|r| r.len() as u32).max().unwrap_or(0) + 16;
    let tour_capacity = n_routes * max_route_len;

    let gpu = GpuPopulation::new(
        &solved.matrix.durations,
        n_loc as u32,
        pop_size,
        n_routes.max(1),
        tour_capacity,
    )?;
    gpu.upload(&trajectories)?;
    gpu.upload_problem_data(&loc_service, &loc_demand, &loc_tw_s, &loc_tw_e)?;
    gpu.upload_vehicle_data(
        &vec![veh_cap; pop_size as usize],
        &vec![veh_tw_s; pop_size as usize],
        &vec![veh_tw_e; pop_size as usize],
    )?;

    // ---- Variant A: sequential megakernel calls (one per trajectory) ----
    let t0 = Instant::now();
    let mut seq_results = Vec::with_capacity(pop_size as usize);
    for t in 0..pop_size {
        let r = gpu.run_megakernel_2opt(t, 200)?;
        seq_results.push(r);
    }
    let t_seq = t0.elapsed();

    // Re-upload (each megakernel call mutated tours).
    gpu.upload(&trajectories)?;

    // ---- Variant B: batch megakernel (single dispatch, all trajectories) ----
    let t0 = Instant::now();
    let batch_results = gpu.run_megakernel_2opt_batch(200)?;
    let t_batch = t0.elapsed();

    // ---- Variant C: batch megakernel + ILS-kick at varying intensities ----
    // Test 1, 2, 4 swaps per workgroup to find the sweet spot. Higher
    // kick → more diversity but higher chance of breaking TW.
    let mut kick_variants: Vec<(u32, std::time::Duration, Vec<brooom::gpu_population::MegakernelStatus>)> = Vec::new();
    for &k in &[1u32, 2, 4] {
        gpu.upload(&trajectories)?;
        let t0 = Instant::now();
        let r = gpu.run_megakernel_2opt_batch_with_kick(200, k, 0xCAFEBABE)?;
        kick_variants.push((k, t0.elapsed(), r));
    }
    let t_kick = kick_variants.last().unwrap().1;

    // ---- Print results ----
    println!("\n=== Variant A: sequential megakernel × {pop_size} ===");
    println!("  Wallclock total: {:?}", t_seq);
    println!("  Per trajectory:  {:?}", t_seq / pop_size);
    println!("  Iters total:     {}", seq_results.iter().map(|(i,_,_)| *i).sum::<u32>());

    println!("\n=== Variant B: batch megakernel (single dispatch) ===");
    println!("  Wallclock total: {:?}", t_batch);
    println!("  Per trajectory:  {:?}", t_batch / pop_size);
    println!("  Iters total:     {}", batch_results.iter().map(|s| s.iters).sum::<u32>());
    println!("  Dropped tasks:   {}", batch_results.iter().map(|s| s.dropped).sum::<u32>());

    let speedup = t_seq.as_secs_f64() / t_batch.as_secs_f64().max(1e-9);
    println!("\n  Speedup B vs A: {:.1}×", speedup);

    println!("\n=== Variant C: batch megakernel + ILS-kick (varying intensity) ===");
    for (k, dur, _) in &kick_variants {
        println!("  kick={k}: wallclock={:?}, per traj={:?}", dur, *dur / pop_size);
    }
    let _ = t_kick;

    // ---- Quality check: re-run each kick variant with feasibility check ----
    let location_to_job: std::collections::HashMap<usize, brooom::solution::TaskRef> =
        problem.jobs.iter().enumerate().filter_map(|(idx, j)| {
            j.location.index.map(|li| (li, brooom::solution::TaskRef::Job(idx)))
        }).collect();

    fn evaluate_batch(
        problem: &Problem,
        matrix: &brooom::matrix::Matrix,
        routes: &[brooom::solution::Route],
        location_to_job: &std::collections::HashMap<usize, brooom::solution::TaskRef>,
        final_tours: &[Vec<Vec<u32>>],
    ) -> (usize, Vec<i64>) {
        let mut feas_travels = Vec::new();
        for t in 0..final_tours.len() {
            let mut total = 0i64;
            let mut all_feas = true;
            for (r_idx, route) in routes.iter().enumerate() {
                let vehicle = &problem.vehicles[route.vehicle_idx];
                let mut steps = Vec::new();
                for &li in final_tours[t][r_idx].iter().skip(1).take(final_tours[t][r_idx].len() - 2) {
                    if let Some(&tr) = location_to_job.get(&(li as usize)) {
                        steps.push(tr);
                    }
                }
                match evaluate_route(problem, matrix, vehicle, &steps) {
                    Ok(m) => total += m.travel_time,
                    Err(_) => { all_feas = false; break; }
                }
            }
            if all_feas { feas_travels.push(total); }
        }
        (feas_travels.len(), feas_travels)
    }

    println!("\n=== Quality per variant (best-of-{pop_size}) ===");
    println!("  Initial travel: {initial_travel}");

    // Re-run each variant cleanly to evaluate.
    for &k in &[0u32, 1, 2, 4] {
        gpu.upload(&trajectories)?;
        if k == 0 {
            gpu.run_megakernel_2opt_batch(200)?;
        } else {
            gpu.run_megakernel_2opt_batch_with_kick(200, k, 0xCAFEBABE)?;
        }
        let final_tours = gpu.read_back_all()?;
        let (feas_count, mut travels) =
            evaluate_batch(&problem, &solved.matrix, &solved.solution.routes, &location_to_job, &final_tours);
        travels.sort();
        if !travels.is_empty() {
            let best = travels[0];
            let median = travels[travels.len()/2];
            println!(
                "  kick={k}:  feas={}/{} best={best} ({:+.2}%) median={median} ({:+.2}%)",
                feas_count, pop_size,
                100.0 * (best - initial_travel) as f64 / initial_travel as f64,
                100.0 * (median - initial_travel) as f64 / initial_travel as f64,
            );
        } else {
            println!("  kick={k}:  feas=0/{pop_size} (all trajectories TW-broken — kick too aggressive)");
        }
    }

    Ok(())
}
