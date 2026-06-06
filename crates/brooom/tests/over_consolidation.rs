//! Isolated regression test for the R2/RC2 over-consolidation gap.
//!
//! Diagnosis: on wide-window instances brooom packs stops into too FEW routes
//! (rc208: 3 routes vs PyVRP's 4; r211: 3 vs 4), which raises total distance.
//! These tests pin the worst cases so we can develop the route-opening fix
//! against a fast, deterministic target — and prove the fix without the full
//! benchmark. PyVRP reference: rc208 ≈ 77892 (4 routes), r211 ≈ 75523 (4 routes).
//!
//! Run: `cargo test -p brooom --test over_consolidation -- --nocapture`

use std::path::Path;

use brooom::io::parse_input;
use brooom::solver::{solve, SolverConfig};

fn solve_instance(name: &str) -> (usize, f64, usize) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks/instances_solomon")
        .join(format!("{name}.json"));
    let json = std::fs::read_to_string(&path).expect("read instance");
    let mut problem = parse_input(&json).expect("parse");
    brooom::solution::eval_cache_invalidate();
    let mut cfg = SolverConfig::default();
    // Time-bounded so the test is fast; the route-opening LAHC trajectory runs
    // to this deadline. (Deterministic fixed-iter would be far too slow here.)
    cfg.multi_start = 8;
    cfg.time_limit_ms = Some(6000);
    let sol = solve(&mut problem, None, cfg).expect("solve");
    let routes = sol.routes.iter().filter(|r| !r.steps.is_empty()).count();
    let jobs: usize = sol.routes.iter().map(|r| r.steps.len()).sum();
    (routes, sol.summary.cost, jobs + sol.unassigned.len())
}

// Timing-sensitive (run the route-opening LAHC trajectory to a deadline), so
// `#[ignore]`d by default — `cargo test` runs tests in parallel and the core
// contention starves the budget. Run manually:
//   cargo test --release -p brooom --test over_consolidation -- --ignored --test-threads=1 --nocapture
// The authoritative R2/RC2 gate is the binary benchmark (benchmarks/results/beat_pyvrp.md).
#[test]
#[ignore]
fn rc208_route_count_and_cost() {
    let (routes, cost, _) = solve_instance("rc208");
    eprintln!("rc208: {routes} routes, cost {cost:.0} (PyVRP: 4 routes, 77892)");
    // Goal of the fix: reach PyVRP's 4 routes and get the cost down. Baseline on
    // `main` is 3 routes / ~88000; assert the fix opens the 4th route and lands
    // well under the baseline.
    // Baseline on `main` is 3 routes / ~88000; the route-opening fix reaches ≥4
    // routes and lands well under the baseline (≈80000, vs PyVRP 77892).
    assert!(routes >= 4, "expected ≥4 routes (was 3), got {routes}");
    assert!(cost <= 83000.0, "expected cost ≤ 83000 (was ~88000), got {cost:.0}");
}

#[test]
#[ignore]
fn r211_route_count_and_cost() {
    let (routes, cost, _) = solve_instance("r211");
    eprintln!("r211: {routes} routes, cost {cost:.0} (PyVRP: 4 routes, 75523)");
    assert!(routes >= 4, "expected ≥4 routes (was 3), got {routes}");
    assert!(cost <= 81000.0, "expected cost ≤ 81000 (was ~81300), got {cost:.0}");
}

#[test]
fn tight_window_unaffected() {
    // c101/r101 are tight-window — the fix must NOT change these (they already
    // tie PyVRP). Just assert they solve fully with a sane route count.
    let (rc, _, _) = solve_instance("c101");
    assert!(rc >= 10, "c101 should use ~10 routes, got {rc}");
    let (rr, _, _) = solve_instance("r101");
    assert!(rr >= 19, "r101 should use ~20 routes, got {rr}");
}
