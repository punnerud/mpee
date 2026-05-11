//! Smoke-test the exact route solver: take an LS-converged route, scramble
//! the customer order, exact-solve, verify cost ≤ original. If LS already
//! found the global optimum, exact-solver should return the same cost.
//!
//! Then deliberately scramble and verify the solver finds an ordering at
//! least as good as the LS-converged baseline.
//!
//! Run with:
//!   cargo run --release --example exact_solver_verify

use std::time::Instant;

use brooom::problem::Problem;
use brooom::route_exact::{solve_route_exact, MAX_EXACT_LEN};
use brooom::solution::evaluate_route;
use brooom::solver::{solve_full, SolverConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "benchmarks/instances/r1_0100.json";
    println!("Loading {path} ...");
    let f = std::fs::File::open(path)?;
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
    println!("Solving CPU baseline...");
    let solved = solve_full(&mut problem, None, cfg)?;
    println!("  routes={} cost={:.1}\n", solved.solution.routes.len(), solved.solution.summary.cost);

    // Pick the longest route ≤ MAX_EXACT_LEN.
    let candidate = solved.solution.routes
        .iter()
        .enumerate()
        .filter(|(_, r)| r.steps.len() >= 5 && r.steps.len() <= MAX_EXACT_LEN)
        .max_by_key(|(_, r)| r.steps.len());

    let Some((r_idx, route)) = candidate else {
        println!("No route in size range [5, {MAX_EXACT_LEN}] — bump MAX_EXACT_LEN or use a different instance.");
        return Ok(());
    };

    let vehicle = problem.vehicles[route.vehicle_idx].clone();
    let baseline = route.metrics;
    let original_steps = route.steps.clone();
    println!("Test route {r_idx}: {} stops, baseline cost={:.2}", original_steps.len(), baseline.cost);

    // 1. Sanity: pass LS-converged ordering to the exact solver.
    let t0 = Instant::now();
    let on_optimal = solve_route_exact(&problem, &solved.matrix, &vehicle, &original_steps);
    let solver_t1 = t0.elapsed();

    match &on_optimal {
        None => println!("  ✗ exact_solve(LS-converged) returned None"),
        Some(r) => {
            println!("  exact-solve on LS-converged: cost={:.2} ({:?})", r.metrics.cost, solver_t1);
            if r.metrics.cost > baseline.cost + 1e-6 {
                println!("    ✗ exact result is *worse* than LS — bug");
            } else if (r.metrics.cost - baseline.cost).abs() < 1e-6 {
                println!("    ✓ exact agrees with LS (LS found global optimum on this route)");
            } else {
                println!("    ✓ exact found IMPROVEMENT of {:.2} over LS-converged",
                    baseline.cost - r.metrics.cost);
            }
        }
    }

    // 2. Scramble the order, then exact-solve. Result must be ≤ baseline.
    let mut scrambled = original_steps.clone();
    // Cyclic rotation by 2 (cheap deterministic scramble).
    if scrambled.len() >= 4 {
        scrambled.rotate_left(2);
    }
    let scrambled_metrics = evaluate_route(&problem, &solved.matrix, &vehicle, &scrambled);
    let scrambled_cost_str = match &scrambled_metrics {
        Ok(m) => format!("{:.2}", m.cost),
        Err(e) => format!("infeasible ({e})"),
    };
    println!("\nScrambled order ({}-rotation): cost={scrambled_cost_str}",
             if scrambled.len() >= 4 { "left-2" } else { "none" });

    let t0 = Instant::now();
    let on_scrambled = solve_route_exact(&problem, &solved.matrix, &vehicle, &scrambled);
    let solver_t2 = t0.elapsed();

    match on_scrambled {
        None => println!("  ✗ no feasible permutation of scrambled — would mean instance constraints rule it out"),
        Some(r) => {
            println!("  exact-solve on scrambled:    cost={:.2} ({:?})", r.metrics.cost, solver_t2);
            if r.metrics.cost > baseline.cost + 1e-6 {
                println!("    ✗ exact didn't recover the LS-converged optimum (bug or infeasibility under permutation)");
            } else if (r.metrics.cost - baseline.cost).abs() < 1e-6 {
                println!("    ✓ exact recovered the LS optimum from scrambled input");
            } else {
                println!("    ✓ exact found a *better* ordering than LS (improvement {:.2})",
                    baseline.cost - r.metrics.cost);
            }
        }
    }

    Ok(())
}
