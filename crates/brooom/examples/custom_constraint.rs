//! Custom-constraints-as-code demo (runnable, no map cache needed).
//!
//!     cargo run -p brooom --example custom_constraint
//!
//! Shows the `brooom::constraint` hook end to end on a haversine matrix:
//!   1. solve with no constraint  → every job assigned
//!   2. a HARD constraint forbidding job 20 → job 20 is dropped
//!   3. a SOFT penalty on long routes → search is biased, nothing rejected

use std::sync::Arc;

use brooom::constraint::{ConstraintGuard, RouteView, Verdict};
use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::solver::{solve, SolverConfig};

const PROBLEM: &str = r#"{
    "vehicles": [
        {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}
    ],
    "jobs": [
        {"id": 10, "location": [10.10, 60.0], "delivery": [1]},
        {"id": 20, "location": [10.20, 60.0], "delivery": [1]},
        {"id": 30, "location": [10.30, 60.0], "delivery": [1]}
    ]
}"#;

fn served_ids(sol: &brooom::Solution, problem: &brooom::Problem) -> Vec<u64> {
    sol.routes
        .iter()
        .flat_map(|r| r.steps.iter().map(|s| s.description(problem).id))
        .collect()
}

fn run() -> Vec<u64> {
    let mut problem = parse_input(PROBLEM).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    served_ids(&sol, &problem)
}

fn main() {
    // 1. Baseline — no custom constraint.
    println!("baseline served jobs:        {:?}", run());

    // 2. Hard constraint: no route may visit job 20.
    {
        let c: Arc<brooom::constraint::CustomConstraintFn> = Arc::new(|r: &RouteView| {
            if r.stop_ids().contains(&20) { Verdict::Infeasible } else { Verdict::Feasible }
        });
        let _guard = ConstraintGuard::install(vec![c]);
        let served = run();
        println!("with 'reject job 20' served: {served:?}");
        assert!(!served.contains(&20), "job 20 must be dropped");
        println!("  → job 20 correctly left unassigned ✓");
    } // guard drops here → constraint cleared

    // 3. Soft penalty: discourage (but don't forbid) routes longer than 1 km.
    {
        let c: Arc<brooom::constraint::CustomConstraintFn> = Arc::new(|r: &RouteView| {
            if r.metrics.distance > 1_000 { Verdict::Penalty(250.0) } else { Verdict::Feasible }
        });
        let _guard = ConstraintGuard::install(vec![c]);
        println!("with soft penalty served:    {:?}", run());
        println!("  → all jobs still served; the penalty only re-shaped cost ✓");
    }

    // 4. Back to normal once constraints are cleared.
    assert!(!brooom::constraint::has_constraints());
    println!("constraints cleared:         {:?}", run());
}
