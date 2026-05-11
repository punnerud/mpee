//! Etappe-1 demo: replace each ≤ 12-stop route in a CPU-converged
//! solution with the *exact* optimal ordering. Reports per-route deltas
//! and total improvement.
//!
//! This is the polish-pass companion: cluster_decompose finds a good
//! partition, LS local-optimises within each route, and `route_exact`
//! finishes the job by guaranteeing every short route is the literal
//! shortest feasible ordering.
//!
//! Run with:
//!   cargo run --release --example exact_polish_demo
//!   cargo run --release --example exact_polish_demo -- benchmarks/instances/r1_0500_s1.json

use std::time::Instant;

use brooom::problem::Problem;
use brooom::route_exact::{solve_route_exact, MAX_EXACT_LEN};
use brooom::solution::evaluate_route;
use brooom::solver::{solve_full, SolverConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "benchmarks/instances/r1_0100.json".into());

    println!("Loading {} ...", path);
    let f = std::fs::File::open(&path)?;
    let mut problem: Problem = brooom::io::parse_input_reader(std::io::BufReader::new(f))?;

    // First arg: max LS passes (default 30). Pass 0 for "raw insertion only"
    // baseline, which lets exact-solver demonstrate its global-optimum win.
    let max_ls = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(30);

    let cfg = SolverConfig {
        max_local_search_passes: max_ls,
        granular_k: Some(20),
        multi_start: 1,
        ils_iters: 0,
        ils_kick_size: 0.4,
        time_limit_ms: None,
        verbose: false,
        warm_start: None,
    };
    println!("Solving with brooom CPU (max_ls={max_ls}) ...");
    let t0 = Instant::now();
    let mut solved = solve_full(&mut problem, None, cfg)?;
    let solve_t = t0.elapsed();
    let initial_cost = solved.solution.summary.cost;
    let initial_distance: i64 = solved.solution.routes.iter().map(|r| r.metrics.distance).sum();
    let initial_travel: i64 = solved.solution.routes.iter().map(|r| r.metrics.travel_time).sum();
    println!(
        "  baseline cost={:.1} distance={} travel={} routes={} ({:?})",
        initial_cost,
        initial_distance,
        initial_travel,
        solved.solution.routes.len(),
        solve_t
    );

    // Polish-pass: for each route ≤ MAX_EXACT_LEN, run exact solver.
    println!("\nExact-polishing routes ≤ {MAX_EXACT_LEN} stops:");
    let mut improved = 0;
    let mut tried = 0;
    let mut total_cost_delta = 0.0_f64;
    let mut total_distance_delta: i64 = 0;
    let mut total_solver_us = 0.0;
    let mut largest_improvement = 0.0_f64;

    let n_routes = solved.solution.routes.len();
    for r_idx in 0..n_routes {
        let route_len = solved.solution.routes[r_idx].steps.len();
        if route_len == 0 || route_len > MAX_EXACT_LEN {
            continue;
        }
        tried += 1;
        let vehicle_idx = solved.solution.routes[r_idx].vehicle_idx;
        let vehicle = problem.vehicles[vehicle_idx].clone();
        let steps: Vec<_> = solved.solution.routes[r_idx].steps.clone();
        let baseline_cost = solved.solution.routes[r_idx].metrics.cost;
        let baseline_dist = solved.solution.routes[r_idx].metrics.distance;

        let t0 = Instant::now();
        let result = solve_route_exact(&problem, &solved.matrix, &vehicle, &steps);
        total_solver_us += t0.elapsed().as_secs_f64() * 1e6;

        if let Some(exact) = result {
            let dc = exact.metrics.cost - baseline_cost;
            let dd = exact.metrics.distance - baseline_dist;
            if dc < -1e-6 {
                improved += 1;
                total_cost_delta += dc;
                total_distance_delta += dd;
                if dc < largest_improvement {
                    largest_improvement = dc;
                }
                if improved <= 5 {
                    println!(
                        "  ✓ route {r_idx} ({route_len} stops): cost {:.2} → {:.2} (Δ={:+.2}), dist {} → {} (Δ={:+})",
                        baseline_cost, exact.metrics.cost, dc,
                        baseline_dist, exact.metrics.distance, dd
                    );
                }
                // Apply.
                solved.solution.routes[r_idx].steps = exact.steps;
                solved.solution.routes[r_idx].metrics = exact.metrics;
            }
        }
    }

    solved.solution.recompute_summary();
    let final_cost = solved.solution.summary.cost;
    let final_distance: i64 = solved.solution.routes.iter().map(|r| r.metrics.distance).sum();
    let final_travel: i64 = solved.solution.routes.iter().map(|r| r.metrics.travel_time).sum();

    println!("\n=== Summary ===");
    println!("Tried exact:        {tried}/{n_routes} routes (others > {MAX_EXACT_LEN} stops)");
    println!("Improved:           {improved}");
    println!("Avg solver time:    {:.1} µs/route", total_solver_us / tried.max(1) as f64);
    println!("Total solver time:  {:.1} ms", total_solver_us / 1000.0);
    println!("Cost change:        {:+.2}  ({:+.4}%)",
        final_cost - initial_cost,
        100.0 * (final_cost - initial_cost) / initial_cost);
    println!("Distance change:    {:+}", total_distance_delta);
    println!("Travel change:      {} → {}", initial_travel, final_travel);
    if largest_improvement < 0.0 {
        println!("Largest single-route saving: {:.2}", -largest_improvement);
    }

    // Sanity-check: re-evaluate every route with evaluate_route to confirm
    // we didn't accidentally break feasibility.
    let mut all_feasible = true;
    for (r_idx, route) in solved.solution.routes.iter().enumerate() {
        let vehicle = &problem.vehicles[route.vehicle_idx];
        match evaluate_route(&problem, &solved.matrix, vehicle, &route.steps) {
            Ok(m) => {
                if (m.cost - route.metrics.cost).abs() > 1e-3 {
                    println!("  ✗ route {r_idx}: stored cost {:.2} != recomputed {:.2}",
                             route.metrics.cost, m.cost);
                    all_feasible = false;
                }
            }
            Err(e) => {
                println!("  ✗ route {r_idx}: re-evaluate failed: {e}");
                all_feasible = false;
            }
        }
    }
    if all_feasible {
        println!("\n  ✓ All routes pass post-polish feasibility check.");
    }

    Ok(())
}
