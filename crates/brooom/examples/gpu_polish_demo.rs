//! Demo: GPU-accelerated 2-opt polish on a single brooom-produced route.
//!
//! Shows the integration pattern: GPU finds the most-improving 2-opt move
//! based on distance only, CPU validates TW + capacity feasibility before
//! applying. Repeat until no improving feasible move remains.
//!
//! Run with:
//!   cargo run --release --example gpu_polish_demo -- benchmarks/instances/r1_1000.json
//!
//! For routes ≥ 1500 stops the GPU wins on per-iteration time. For shorter
//! routes (typical VRPLT) the CPU sweep is faster — we apply the GPU only
//! when a route is long enough to benefit.

use std::time::Instant;

use brooom::gpu_sweep::{GpuSweep, BestMove};
use brooom::matrix::Matrix;
use brooom::solver::{solve_full, SolverConfig};
use brooom::solution::evaluate_route;
use brooom::problem::Problem;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "benchmarks/instances/r1_1000.json".into()
    });

    println!("Loading {} ...", path);
    let f = std::fs::File::open(&path)?;
    let mut problem: Problem = brooom::io::parse_input_reader(std::io::BufReader::new(f))?;

    println!("Initial solve (auto-K + auto-decompose disabled, m=1, ils=0)...");
    let t0 = Instant::now();
    let cfg = SolverConfig {
        max_local_search_passes: 50,
        granular_k: Some(40),
        multi_start: 1,
        ils_iters: 0,
        ils_kick_size: 0.4,
        time_limit_ms: None,
        verbose: false,
        warm_start: None,
            ..Default::default()
    };
    let solved = solve_full(&mut problem, None, cfg)?;
    println!(
        "  initial: cost={:.1} routes={} unassigned={} ({:?})",
        solved.solution.summary.cost,
        solved.solution.routes.len(),
        solved.solution.unassigned.len(),
        t0.elapsed()
    );

    // Pick the longest route to demonstrate GPU's strength.
    let (longest_idx, longest_len) = solved
        .solution
        .routes
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.steps.len()))
        .max_by_key(|&(_, len)| len)
        .ok_or("no routes")?;
    println!(
        "\nLongest route: index {} with {} stops",
        longest_idx, longest_len
    );

    // Build a tour-as-location-indices array for the GPU.
    let route = &solved.solution.routes[longest_idx];
    let vehicle = &problem.vehicles[route.vehicle_idx];
    // Tour is depot → stop1 → stop2 → ... → depot.
    let depot = vehicle.start.as_ref().and_then(|l| l.index).unwrap_or(0);
    let mut tour: Vec<u32> = std::iter::once(depot as u32).collect();
    for &task in &route.steps {
        if let Some(li) = task.description(&problem).location.index {
            tour.push(li as u32);
        }
    }
    tour.push(depot as u32);
    println!("  tour = {} stops (incl depots)", tour.len());

    println!("\nInit GPU device + upload matrix...");
    let t0 = Instant::now();
    let n = solved.matrix.n as u32;
    let gpu = GpuSweep::new(&solved.matrix.durations, n)?;
    println!("  GPU ready in {:?}", t0.elapsed());

    // GPU finds the best 2-opt move on this tour.
    let t0 = Instant::now();
    let mv: BestMove = gpu.best_2opt(&tour)?;
    println!(
        "\nGPU best_2opt: delta={} i={} j={} ({:?})",
        mv.delta, mv.i, mv.j, t0.elapsed()
    );

    if mv.delta >= 0 {
        println!("  no improving 2-opt move found — route is already optimal under distance.");
        return Ok(());
    }

    // Apply the swap on a candidate tour and check TW feasibility.
    let mut candidate_tour = tour.clone();
    let i = mv.i as usize;
    let j = mv.j as usize;
    candidate_tour[i + 1..=j].reverse();

    // Build TaskRefs from the candidate tour (skip depot endpoints).
    let mut steps: Vec<_> = Vec::new();
    let location_to_taskref: std::collections::HashMap<usize, brooom::solution::TaskRef> =
        problem.jobs.iter().enumerate().filter_map(|(idx, j)| {
            j.location.index.map(|li| (li, brooom::solution::TaskRef::Job(idx)))
        }).collect();
    for &li in candidate_tour.iter().skip(1).take(candidate_tour.len() - 2) {
        if let Some(&tr) = location_to_taskref.get(&(li as usize)) {
            steps.push(tr);
        }
    }

    let tw_check = evaluate_route(&problem, &solved.matrix, vehicle, &steps);
    match tw_check {
        Ok(metrics) => {
            let old_cost = route.metrics.cost;
            let new_cost = metrics.cost;
            println!(
                "\nTW-check: ✓ feasible. old_cost={:.1} new_cost={:.1} delta={:.1}",
                old_cost, new_cost, new_cost - old_cost
            );
            println!("  ⇒ GPU-proposed move is APPLY-ABLE (would reduce route cost by {:.1})",
                old_cost - new_cost);
        }
        Err(reason) => {
            println!("\nTW-check: ✗ infeasible — {reason}");
            println!("  GPU proposed a distance-improving move that violates TW;");
            println!("  in production LS we'd fall back to the next-best move.");
        }
    }

    Ok(())
}
