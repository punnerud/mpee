//! Phase-2 verification: forward+backward TW pass and load simulation
//! on the GPU, validated bit-exact against a CPU reference.
//!
//! Scope (intentionally narrower than `eval::precompute()`):
//!   - single time window per location
//!   - single capacity dim
//!   - speed_factor = 1, no setup
//!   - homogeneous fleet (one vehicle config per trajectory)
//!
//! These match Solomon / Gehring-Homberger VRPTW exactly, which is what
//! we'll need for the full GPU LS loop. Full feature parity with eval.rs
//! comes once the loop closes.
//!
//! Run with:
//!   cargo run --release --example gpu_precompute_correctness

use brooom::gpu_population::{GpuPopulation, GpuRoutePrecomp, TrajectoryTours};

fn random_problem(n_loc: usize, seed: u64) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    // Coords in [0, 100). Symmetric Euclidean distance, rounded to int.
    let xs: Vec<i32> = (0..n_loc).map(|_| (next() % 100) as i32).collect();
    let ys: Vec<i32> = (0..n_loc).map(|_| (next() % 100) as i32).collect();
    let mut matrix = vec![0i32; n_loc * n_loc];
    for i in 0..n_loc {
        for j in 0..n_loc {
            if i != j {
                let dx = xs[i] - xs[j];
                let dy = ys[i] - ys[j];
                let d = ((dx * dx + dy * dy) as f64).sqrt().round() as i32;
                matrix[i * n_loc + j] = d.max(1);
            }
        }
    }
    // Service: 10 at customers, 0 at depot (loc 0).
    let mut service: Vec<i32> = vec![10; n_loc];
    service[0] = 0;
    // Demand: ~1..15 at customers, 0 at depot.
    let mut demand: Vec<i32> = (0..n_loc).map(|_| ((next() % 15) + 1) as i32).collect();
    demand[0] = 0;
    // Time windows: depot is wide [0, 1000]; customers each have a
    // window of width ~200 inside [0, 1000].
    let mut tw_start = vec![0i32; n_loc];
    let mut tw_end = vec![1000i32; n_loc];
    for i in 1..n_loc {
        let center = (next() % 800) as i32 + 100;
        tw_start[i] = (center - 100).max(0);
        tw_end[i] = (center + 100).min(1000);
    }
    (matrix, service, demand, tw_start, tw_end)
}

fn random_routes(
    n_loc: u32,
    n_routes: usize,
    avg_len: u32,
    seed: u64,
) -> TrajectoryTours {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut routes = Vec::with_capacity(n_routes);
    for _ in 0..n_routes {
        let jitter = (next() % 7) as i32 - 3;
        let len = ((avg_len as i32) + jitter).max(2) as u32;
        // Each route is depot, customers..., depot. Customers in 1..n_loc.
        let mut r = Vec::with_capacity(len as usize + 2);
        r.push(0u32); // start depot
        for _ in 0..len {
            r.push((next() % (n_loc - 1)) + 1);
        }
        r.push(0u32); // end depot
        routes.push(r);
    }
    routes
}

/// CPU reference matching the same simplified semantics as the kernel.
fn cpu_precompute(
    route: &[u32],
    matrix: &[i32],
    md: usize,
    service: &[i32],
    demand: &[i32],
    tw_start: &[i32],
    tw_end: &[i32],
    veh_cap: i32,
    veh_tw_s: i32,
    veh_tw_e: i32,
) -> GpuRoutePrecomp {
    let len = route.len();
    let mut depart = vec![0i32; len];
    let mut latest = vec![0i32; len];
    let mut load_at = vec![0i32; len];
    let mut feas = true;

    // Initial load = sum demands of interior stops.
    let mut init_load: i32 = 0;
    for k in 1..len - 1 {
        init_load += demand[route[k] as usize];
    }
    if init_load > veh_cap { feas = false; }

    // Forward.
    let mut t = veh_tw_s;
    let mut cur_load = init_load;
    depart[0] = t;
    load_at[0] = cur_load;
    let mut prev = route[0] as usize;

    for k in 1..len {
        let here = route[k] as usize;
        t += matrix[prev * md + here];
        if k + 1 < len {
            // customer
            if t < tw_start[here] { t = tw_start[here]; }
            if t > tw_end[here] { feas = false; }
            cur_load -= demand[here];
            if cur_load < 0 { feas = false; }
            load_at[k] = cur_load;
            t += service[here];
        } else {
            load_at[k] = cur_load;
        }
        depart[k] = t;
        prev = here;
    }
    if t > veh_tw_e { feas = false; }

    // Backward.
    latest[len - 1] = veh_tw_e;
    if len >= 2 {
        for k in (0..len - 1).rev() {
            let here = route[k] as usize;
            let next_loc = route[k + 1] as usize;
            let edge = matrix[here * md + next_loc];
            let s_here = if k > 0 && k + 1 < len { service[here] } else { 0 };
            let chain = latest[k + 1] - s_here - edge;
            let tw_e_here = if k > 0 && k + 1 < len { tw_end[here] } else { veh_tw_e };
            latest[k] = chain.min(tw_e_here);
        }
    }

    GpuRoutePrecomp { depart, latest_arrival: latest, load_at, feasible: feas }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_loc = 100usize;
    let pop_size = 4usize;
    let n_routes = 12usize;
    let avg_len = 12u32;
    let max_routes = 16u32;
    let tour_capacity = 400u32;

    let veh_cap = 100i32;
    let veh_tw_s = 0i32;
    let veh_tw_e = 1000i32;

    println!("Phase-2 precompute correctness: n_loc={n_loc}, pop={pop_size}, routes={n_routes}");
    let (matrix, service, demand, tw_start, tw_end) = random_problem(n_loc, 42);

    let pop: Vec<TrajectoryTours> = (0..pop_size)
        .map(|t| random_routes(n_loc as u32, n_routes, avg_len, 100 + t as u64 * 7))
        .collect();

    let gpu = GpuPopulation::new(
        &matrix,
        n_loc as u32,
        pop_size as u32,
        max_routes,
        tour_capacity,
    )?;

    gpu.upload(&pop)?;
    gpu.upload_problem_data(&service, &demand, &tw_start, &tw_end)?;
    gpu.upload_vehicle_data(
        &vec![veh_cap; pop_size],
        &vec![veh_tw_s; pop_size],
        &vec![veh_tw_e; pop_size],
    )?;

    let t0 = std::time::Instant::now();
    gpu.precompute_all()?;
    let dispatch_us = t0.elapsed().as_secs_f64() * 1e6;
    println!("\nGPU precompute_all (pop={pop_size} × routes={n_routes} = {} routes): {:.1} µs",
             pop_size * n_routes, dispatch_us);

    // Compare bit-exact against CPU reference.
    let mut total_compared = 0;
    let mut depart_mm = 0;
    let mut latest_mm = 0;
    let mut load_mm = 0;
    let mut feas_mm = 0;
    let mut feasible_routes = 0;
    let mut infeasible_routes = 0;

    for t in 0..pop_size {
        for r in 0..n_routes {
            let cpu = cpu_precompute(
                &pop[t][r], &matrix, n_loc, &service, &demand, &tw_start, &tw_end,
                veh_cap, veh_tw_s, veh_tw_e,
            );
            let gpu_r = gpu.read_precompute(t as u32, r as u32)?;

            if cpu.depart != gpu_r.depart {
                depart_mm += 1;
                if depart_mm <= 2 {
                    println!("  ✗ traj {t} route {r} depart mismatch:");
                    println!("    cpu = {:?}", cpu.depart);
                    println!("    gpu = {:?}", gpu_r.depart);
                }
            }
            if cpu.latest_arrival != gpu_r.latest_arrival {
                latest_mm += 1;
                if latest_mm <= 2 {
                    println!("  ✗ traj {t} route {r} latest_arrival mismatch:");
                    println!("    cpu = {:?}", cpu.latest_arrival);
                    println!("    gpu = {:?}", gpu_r.latest_arrival);
                }
            }
            if cpu.load_at != gpu_r.load_at {
                load_mm += 1;
                if load_mm <= 2 {
                    println!("  ✗ traj {t} route {r} load_at mismatch:");
                    println!("    cpu = {:?}", cpu.load_at);
                    println!("    gpu = {:?}", gpu_r.load_at);
                }
            }
            if cpu.feasible != gpu_r.feasible {
                feas_mm += 1;
            }
            if cpu.feasible { feasible_routes += 1; } else { infeasible_routes += 1; }
            total_compared += 1;
        }
    }

    println!("\nCompared {total_compared} routes ({feasible_routes} feasible, {infeasible_routes} infeasible).");
    println!("  depart mismatches:        {depart_mm}");
    println!("  latest_arrival mismatches: {latest_mm}");
    println!("  load_at mismatches:        {load_mm}");
    println!("  feasibility mismatches:    {feas_mm}");

    if depart_mm == 0 && latest_mm == 0 && load_mm == 0 && feas_mm == 0 {
        println!("\n  ✓ Phase 2 verified: GPU precompute matches CPU reference bit-exactly.");
    } else {
        return Err("Phase 2 mismatch".into());
    }

    // Run again 50× to measure steady-state dispatch cost.
    let runs = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..runs {
        gpu.precompute_all()?;
    }
    let avg_us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;
    println!("\nSteady-state precompute_all: {:.1} µs/dispatch (averaged over {runs} runs)",
             avg_us);

    // ---- Second scenario: wide TW + ample capacity → routes feasible by
    //      construction, so the feasible flag and depart values both
    //      cover the happy path.
    println!("\n=== Feasible-routes scenario (wide TW, ample capacity) ===");
    let wide_tw_start = vec![0i32; n_loc];
    let wide_tw_end = vec![10_000i32; n_loc];
    let zero_demand = vec![0i32; n_loc]; // capacity trivially satisfied
    gpu.upload_problem_data(&service, &zero_demand, &wide_tw_start, &wide_tw_end)?;
    gpu.upload_vehicle_data(
        &vec![10_000i32; pop_size],
        &vec![0i32; pop_size],
        &vec![10_000i32; pop_size],
    )?;
    gpu.precompute_all()?;

    let mut feas_ok = 0;
    let mut feas_bad = 0;
    let mut depart_mm2 = 0;
    for t in 0..pop_size {
        for r in 0..n_routes {
            let cpu = cpu_precompute(
                &pop[t][r], &matrix, n_loc, &service, &zero_demand,
                &wide_tw_start, &wide_tw_end, 10_000, 0, 10_000,
            );
            let gpu_r = gpu.read_precompute(t as u32, r as u32)?;
            if gpu_r.feasible { feas_ok += 1; } else { feas_bad += 1; }
            if cpu.depart != gpu_r.depart { depart_mm2 += 1; }
            if cpu.feasible != gpu_r.feasible {
                println!("  ✗ traj {t} route {r} feasibility disagreement: cpu={} gpu={}",
                         cpu.feasible, gpu_r.feasible);
            }
        }
    }
    println!("Feasible: {feas_ok}, infeasible: {feas_bad}, depart mismatches: {depart_mm2}");
    if feas_ok == pop_size * n_routes && depart_mm2 == 0 {
        println!("  ✓ all routes flagged feasible, depart matches CPU");
    } else {
        return Err("happy-path scenario failed".into());
    }

    Ok(())
}
