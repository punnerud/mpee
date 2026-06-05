//! Custom-constraints-as-code suite.
//!
//! Exercises the `brooom::constraint` hook: register a Rust closure that the
//! solver calls on every completed route, returning a hard rejection or a soft
//! penalty. Custom constraints live in process-global state, so every test
//! here takes a shared lock to stay isolated (this file is its own test
//! binary, so it never races the other suites).

use std::sync::{Arc, Mutex};

use brooom::constraint::{ConstraintGuard, RouteView, Verdict};
use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::solution::{eval_cache_invalidate, evaluate_route, TaskRef};
use brooom::solver::{build_matrix, solve, SolverConfig};

static LOCK: Mutex<()> = Mutex::new(());

/// Serialize tests (shared global registry) and tolerate a poisoned lock from a
/// previously-panicking test.
fn guard() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn prep(json: &str) -> (brooom::Problem, brooom::Matrix) {
    eval_cache_invalidate();
    let mut problem = parse_input(json).unwrap();
    let matrix = build_matrix(&mut problem, Some(&HaversineMatrix::default())).unwrap();
    (problem, matrix)
}

const TWO_JOBS: &str = r#"{
    "vehicles": [
        {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}
    ],
    "jobs": [
        {"id": 10, "location": [10.10, 60.0], "delivery": [1]},
        {"id": 20, "location": [10.20, 60.0], "delivery": [1]}
    ]
}"#;

#[test]
fn custom_constraint_hard_reject_in_evaluator() {
    let _lock = guard();
    let (problem, matrix) = prep(TWO_JOBS);
    let veh = &problem.vehicles[0];

    // Reject any route that visits job id 20.
    let c: Arc<brooom::constraint::CustomConstraintFn> = Arc::new(|v: &RouteView| {
        if v.stop_ids().contains(&20) { Verdict::Infeasible } else { Verdict::Feasible }
    });
    let _g = ConstraintGuard::install(vec![c]);

    assert!(
        evaluate_route(&problem, &matrix, veh, &[TaskRef::Job(0)]).is_ok(),
        "a route without job 20 is fine"
    );
    assert_eq!(
        evaluate_route(&problem, &matrix, veh, &[TaskRef::Job(1)]).err(),
        Some("custom constraint violated"),
        "a route visiting job 20 is rejected by the custom constraint"
    );
}

#[test]
fn custom_constraint_soft_penalty_adds_cost() {
    let _lock = guard();
    let (problem, matrix) = prep(TWO_JOBS);
    let veh = &problem.vehicles[0];
    let steps = [TaskRef::Job(0), TaskRef::Job(1)];

    // Baseline cost with no constraint installed.
    let base = evaluate_route(&problem, &matrix, veh, &steps).unwrap().cost;

    // A flat soft penalty of 1000 on every route.
    let c: Arc<brooom::constraint::CustomConstraintFn> =
        Arc::new(|_v: &RouteView| Verdict::Penalty(1000.0));
    let _g = ConstraintGuard::install(vec![c]);

    let penalized = evaluate_route(&problem, &matrix, veh, &steps).unwrap().cost;
    assert!(
        (penalized - (base + 1000.0)).abs() < 1e-6,
        "soft penalty should add exactly 1000 to the route cost ({penalized} vs {base}+1000)"
    );
}

#[test]
fn custom_constraint_drops_job_during_solve() {
    let _lock = guard();
    // Reject any route containing job id 20 ⇒ the solver must leave it unassigned.
    let c: Arc<brooom::constraint::CustomConstraintFn> = Arc::new(|v: &RouteView| {
        if v.stop_ids().contains(&20) { Verdict::Infeasible } else { Verdict::Feasible }
    });
    let _g = ConstraintGuard::install(vec![c]);

    let mut problem = parse_input(TWO_JOBS).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();

    let served: Vec<u64> = sol.routes.iter()
        .flat_map(|r| r.steps.iter().map(|s| s.description(&problem).id))
        .collect();
    assert!(!served.contains(&20), "job 20 must never be scheduled");
    assert!(served.contains(&10), "job 10 has no constraint and should be served");
    assert_eq!(sol.unassigned.len(), 1, "exactly job 20 is unassigned");
}

#[test]
fn cleared_constraint_restores_plain_behaviour() {
    let _lock = guard();
    // Install then drop a rejecting constraint; afterwards the route is fine.
    {
        let c: Arc<brooom::constraint::CustomConstraintFn> =
            Arc::new(|_v: &RouteView| Verdict::Infeasible);
        let _g = ConstraintGuard::install(vec![c]);
        assert!(brooom::constraint::has_constraints());
    }
    assert!(!brooom::constraint::has_constraints(), "guard clears on drop");

    let (problem, matrix) = prep(TWO_JOBS);
    assert!(
        evaluate_route(&problem, &matrix, &problem.vehicles[0], &[TaskRef::Job(0), TaskRef::Job(1)]).is_ok(),
        "with constraints cleared, the route evaluates normally"
    );
}
