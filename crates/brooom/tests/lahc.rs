//! The small-N LAHC boost is ADDITIVE: it runs extra Late-Acceptance + TW-granular
//! multi-start variants alongside the proven greedy ones and keeps the best, so it
//! can only improve — never regress. This test pins that property.

use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::solver::{solve, SolverConfig};

const TW: &str = r#"{
    "vehicles": [
        {"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [50], "time_window": [0, 100000]},
        {"id": 2, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [50], "time_window": [0, 100000]},
        {"id": 3, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [50], "time_window": [0, 100000]}
    ],
    "jobs": [
        {"id": 10, "location": [10.05,60.0], "delivery": [5], "time_windows": [[0, 6000]]},
        {"id": 20, "location": [10.10,60.0], "delivery": [5], "time_windows": [[0, 9000]]},
        {"id": 30, "location": [10.15,60.0], "delivery": [5], "time_windows": [[0, 14000]]},
        {"id": 40, "location": [10.02,60.05], "delivery": [5], "time_windows": [[0, 7000]]},
        {"id": 50, "location": [10.12,60.04], "delivery": [5], "time_windows": [[0, 16000]]},
        {"id": 60, "location": [10.18,60.02], "delivery": [5], "time_windows": [[0, 18000]]}
    ]
}"#;

fn solve_cost(allow_lahc: bool) -> f64 {
    brooom::solution::eval_cache_invalidate();
    let mut p = parse_input(TW).unwrap();
    let mut cfg = SolverConfig::default();
    cfg.allow_lahc = allow_lahc;
    // Deterministic: fixed multi-start + ils iters, no time limit.
    cfg.multi_start = 4;
    cfg.ils_iters = 20;
    let sol = solve(&mut p, Some(&HaversineMatrix::default()), cfg).unwrap();
    assert_eq!(sol.unassigned.len(), 0, "instance is feasible");
    sol.summary.cost
}

#[test]
fn lahc_boost_never_regresses() {
    // allow_lahc adds variants alongside the greedy ones and keeps the best, so
    // the cost with the boost on must be <= the cost with it off.
    let off = solve_cost(false);
    let on = solve_cost(true);
    assert!(
        on <= off + 1e-6,
        "additive LAHC boost regressed: on={on} off={off}"
    );
}
