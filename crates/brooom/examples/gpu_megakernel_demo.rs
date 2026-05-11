//! Megakernel demo: hele 2-opt LS-loopen i én GPU-dispatch.
//!
//! Sammenligner mot Phase-4 multi-dispatch loop på samme problem og samme
//! starting point, måler:
//!   - Wallclock per LS-loop
//!   - Sluttkostnad (skal være identisk — samme 2-opt-search-rom)
//!   - Antall iterasjoner til konvergens
//!
//! Distance-only 2-opt; TW validates after via precompute_all + rollback
//! if needed (handled in user code, not in megakernel).
//!
//! Run with:
//!   cargo run --release --example gpu_megakernel_demo
//!   cargo run --release --example gpu_megakernel_demo -- benchmarks/instances/r1_0500_s1.json

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

    // Use a non-LS-converged baseline so 2-opt has work to do.
    let cfg = SolverConfig {
        max_local_search_passes: 0,
        granular_k: Some(20),
        multi_start: 1,
        ils_iters: 0,
        ils_kick_size: 0.4,
        time_limit_ms: None,
        verbose: false,
        warm_start: None,
    };
    let solved = solve_full(&mut problem, None, cfg)?;
    println!(
        "Insertion-only baseline: cost={:.1} routes={} ({} stops total)\n",
        solved.solution.summary.cost,
        solved.solution.routes.len(),
        solved.solution.routes.iter().map(|r| r.steps.len()).sum::<usize>(),
    );

    let n_loc = solved.matrix.n;

    // Build location/vehicle data (homogeneous fleet, single TW per loc).
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

    // Build initial tours [depot, stops..., depot] for every route.
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
        n_routes.max(1),
        tour_capacity,
    )?;
    gpu.upload(&[tours.clone()])?;
    gpu.upload_problem_data(&loc_service, &loc_demand, &loc_tw_s, &loc_tw_e)?;
    gpu.upload_vehicle_data(&[veh_cap], &[veh_tw_s], &[veh_tw_e])?;

    // Upload granular K-NN table to enable O(N×K) relocate.
    // K=20 for N=100, K=50 for N≥500 (matches CPU brooom defaults).
    let granular_k = if n_loc >= 500 { 50 } else { 20 };
    let granular = brooom::granular::Granular::build(&solved.matrix, granular_k);
    // Pack into flat array of size n_loc × k (Granular stores it this way).
    let mut near_flat: Vec<u32> = Vec::with_capacity(n_loc * granular.k());
    for i in 0..n_loc {
        let mut found = 0;
        for n in granular.neighbors(i) {
            near_flat.push(n as u32);
            found += 1;
        }
        // Pad to k if fewer neighbours exist.
        while found < granular.k() {
            near_flat.push(i as u32);
            found += 1;
        }
    }
    gpu.upload_granular(&near_flat, granular.k() as u32)?;
    println!("Granular K={} uploaded ({} entries)", granular.k(), near_flat.len());

    // ---- Variant A: Phase-4 multi-dispatch (CPU-orchestrated) ----
    // Skip for large N — Phase-4 dispatch overhead (~7 ms/iter on M3) dominates
    // and makes the comparison meaningless. Set SKIP_VARIANT_A=1 to skip.
    let skip_a = std::env::var("SKIP_VARIANT_A").map(|v| v == "1").unwrap_or(n_loc >= 1000);
    let (t_a, iters_a, applies_a, tours_after_a) = if skip_a {
        println!("(Skipping Variant A: N={n_loc} ≥ 1000 or SKIP_VARIANT_A=1)\n");
        (std::time::Duration::from_secs(0), 0u32, 0u32, Vec::<Vec<u32>>::new())
    } else {
        let t0 = Instant::now();
        let mut iters_a = 0u32;
        let mut applies_a = 0u32;
        let max_iter_safety = 500;
        loop {
            if iters_a >= max_iter_safety { break; }
            iters_a += 1;
            gpu.find_best_2opt_all()?;
            let bests = gpu.read_best_2opt_all()?;
            let mut best: Option<(u32, u32, u32, i32)> = None;
            for r in 0..n_routes as usize {
                let b = bests[r];
                if b.delta >= 0 || b.delta == i32::MAX { continue; }
                // Race in atomic-min find kernel: (i, j) may come from
                // different winning threads — reject obviously invalid pairs.
                if b.j <= b.i + 1 { continue; }
                match best {
                    None => best = Some((r as u32, b.i, b.j, b.delta)),
                    Some((_, _, _, gd)) if b.delta < gd => best = Some((r as u32, b.i, b.j, b.delta)),
                    _ => {}
                }
            }
            let Some((r_idx, i, j, _)) = best else { break; };
            gpu.apply_2opt(0, r_idx, i, j)?;
            applies_a += 1;
        }
        let t_a = t0.elapsed();
        let tours_after_a = gpu.read_back(0)?;
        // Re-upload the original (so megakernel starts from same point).
        gpu.upload(&[tours.clone()])?;
        (t_a, iters_a, applies_a, tours_after_a)
    };

    // ---- Variant B: Megakernel (single dispatch) ----
    let t0 = Instant::now();
    // Higher cap for N≥1000 — small instances converge in ≤200, but N=2000+
    // needs 1000-2000 iterations to fully drain improving moves.
    let max_iter = if n_loc >= 1000 { 2000 } else { 500 };
    // For N≥5000, the single-dispatch megakernel can exceed Metal's GPU
    // watchdog (≈5-10s per command buffer on macOS). Chunk the iteration
    // budget into smaller dispatches that return to CPU between chunks.
    let (iters_b, applies_b, final_delta_b) = if n_loc >= 5000 {
        let chunk = 30;
        println!("  (using chunked dispatch, chunk={chunk} for N≥5000)");
        gpu.run_megakernel_2opt_chunked(0, max_iter, chunk)?
    } else {
        gpu.run_megakernel_2opt(0, max_iter)?
    };
    let t_b = t0.elapsed();
    let tours_after_b = gpu.read_back(0)?;

    // ---- Compare ----
    if !skip_a {
        println!("=== Variant A: Phase-4 multi-dispatch ===");
        println!("  Iterations:    {iters_a}");
        println!("  Applies:       {applies_a}");
        println!("  Wallclock:     {:?}", t_a);
        println!("  Per iteration: {:?}", t_a / iters_a.max(1) as u32);
        println!();
    }
    println!("=== Variant B: Megakernel (single dispatch) ===");
    println!("  Iterations:    {iters_b}");
    println!("  Applies:       {applies_b}");
    println!("  Final Δ:       {final_delta_b}");
    println!("  Wallclock:     {:?}", t_b);
    if iters_b > 0 {
        println!("  Per iteration: {:?}", t_b / iters_b);
    }
    println!();
    if !skip_a {
        let speedup = t_a.as_secs_f64() / t_b.as_secs_f64().max(1e-9);
        println!("  Speedup B vs A: {:.1}×", speedup);

        // Verify: both variants should land on the same final tours (same
        // 2-opt search space, same argmin).
        let mut differ = 0;
        for (i, (a, b)) in tours_after_a.iter().zip(tours_after_b.iter()).enumerate() {
            if a != b {
                differ += 1;
                if differ <= 3 {
                    println!("  ✗ route {i} differs:\n    A = {a:?}\n    B = {b:?}");
                }
            }
        }
        if differ == 0 {
            println!("\n  ✓ Both variants produce identical final tours.");
        } else {
            println!("\n  ⚠ {differ} routes differ — likely tie-breaking variance");
            println!("    (atomic-min races on equal deltas).");
        }
    }

    // Sanity: total customer count must be preserved (no dropped tasks).
    let initial_customers: usize = solved.solution.routes.iter().map(|r| r.steps.len()).sum();
    let mut final_customers: usize = 0;
    for r_idx in 0..tours_after_b.len() {
        // skip start + end depots
        let len = tours_after_b[r_idx].len();
        if len >= 2 { final_customers += len - 2; }
    }
    println!("\nTask preservation check:");
    println!("  Initial customers: {initial_customers}");
    println!("  Final customers:   {final_customers}");
    if initial_customers != final_customers {
        println!("  ✗ TASK LEAK: {} customers lost/duplicated!",
            (initial_customers as i64 - final_customers as i64).abs());
    }

    // Final cost check via CPU evaluate_route.
    let mut total_cpu_travel: i64 = 0;
    let mut all_feasible = true;
    let location_to_job: std::collections::HashMap<usize, brooom::solution::TaskRef> =
        problem.jobs.iter().enumerate().filter_map(|(idx, j)| {
            j.location.index.map(|li| (li, brooom::solution::TaskRef::Job(idx)))
        }).collect();
    for (r_idx, route) in solved.solution.routes.iter().enumerate() {
        let vehicle = &problem.vehicles[route.vehicle_idx];
        let mut steps = Vec::new();
        for &li in tours_after_b[r_idx].iter().skip(1).take(tours_after_b[r_idx].len() - 2) {
            if let Some(&tr) = location_to_job.get(&(li as usize)) {
                steps.push(tr);
            }
        }
        match evaluate_route(&problem, &solved.matrix, vehicle, &steps) {
            Ok(m) => total_cpu_travel += m.travel_time,
            Err(e) => {
                println!("  ⚠ route {r_idx}: CPU evaluate_route failed: {e}");
                all_feasible = false;
            }
        }
    }
    let initial_travel: i64 = solved.solution.routes.iter().map(|r| r.metrics.travel_time).sum();
    println!("\nFinal travel time (CPU evaluate_route): {total_cpu_travel}");
    println!("Initial travel time:                    {initial_travel}");
    println!("Improvement:                            {} ({:.2}%)",
        initial_travel - total_cpu_travel,
        100.0 * (initial_travel - total_cpu_travel) as f64 / initial_travel as f64);
    if !all_feasible {
        println!("⚠ Some routes are TW-infeasible after distance-only 2-opt.");
        println!("  In production, validate each apply via precompute_all and rollback.");
    }

    Ok(())
}
