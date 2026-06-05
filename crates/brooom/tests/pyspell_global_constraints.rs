//! Integration tests for **global** (cross-route) constraints written in the
//! pyspell DSL (`--features pyspell`). A user can now express a solution-level
//! rule over `solution.*` fields — the same capability the built-in
//! `max_vehicles` closure provides, but authored in text and compiled at
//! install time.
//!
//! Globals run at `recompute_summary` (the cold path), not in the insertion
//! probe. The whole file is gated on `pyspell` and takes a process-global lock
//! (own test binary) because global constraints live in a shared registry.
#![cfg(feature = "pyspell")]

use std::sync::Mutex;

use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::solver::{solve, SolverConfig};

static LOCK: Mutex<()> = Mutex::new(());

fn guard() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn vehicles_used(sol: &brooom::Solution) -> usize {
    sol.routes.iter().filter(|r| !r.steps.is_empty()).count()
}

// Two single-seat vehicles, two jobs each needing a seat. Uncapped this serves
// both on two routes.
const TWO_SEATS_TWO_JOBS: &str = r#"{
    "vehicles": [
        {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [1]},
        {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [1]}
    ],
    "jobs": [
        {"id": 1, "location": [10.10, 60.0], "delivery": [1]},
        {"id": 2, "location": [10.50, 60.0], "delivery": [1]}
    ]
}"#;

#[test]
fn dsl_global_penalty_is_added_to_the_summary_cost() {
    let _lock = guard();
    // A user-authored global over the cross-route namespace: "use at most one
    // vehicle". This is the capability the per-route DSL cannot express. We
    // assert it directly on `recompute_summary` (the cold path where globals
    // run): a 2-vehicle solution must carry the hard penalty, a 1-vehicle one
    // must not. The metaheuristic consults exactly this cost during search.
    let mut problem = parse_input(TWO_SEATS_TWO_JOBS).unwrap();
    // First solve a baseline with no global to get a real 2-vehicle solution.
    let mut two = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(vehicles_used(&two), 2, "baseline serves both jobs on two routes");

    let _g = brooom::pyspell::install_global_rust(&["solution.vehicles_used <= 1"]).unwrap();
    assert!(brooom::global_constraint::has_global(), "global must be registered");

    // Recompute the 2-vehicle solution's summary under the installed global: the
    // hard violation (used=2 > cap=1) must dominate its cost.
    two.recompute_summary(&problem);
    assert!(
        two.summary.cost >= brooom::global_constraint::HARD,
        "a 2-vehicle solution must carry the global hard penalty, cost = {}",
        two.summary.cost
    );

    // A feasible 1-vehicle solution (serve only job 1) must carry no penalty.
    let mut one = two.clone();
    one.routes[1].steps.clear();
    one.recompute_summary(&problem);
    assert!(
        one.summary.cost < brooom::global_constraint::HARD,
        "a 1-vehicle solution must satisfy the DSL global, cost = {}",
        one.summary.cost
    );
}

#[test]
fn dsl_global_cleared_on_guard_drop() {
    let _lock = guard();
    {
        let _g = brooom::pyspell::install_global_rust(&["solution.unassigned_count <= 0"]).unwrap();
        assert!(brooom::global_constraint::has_global());
    }
    // The RAII guard must clear the registry so the next solve is unconstrained.
    assert!(!brooom::global_constraint::has_global(), "guard drop must clear globals");
}

#[test]
fn dsl_global_soft_penalty_keeps_all_jobs() {
    let _lock = guard();
    // A soft cross-route penalty (a positive number, not a bool) discourages but
    // never rejects. With a generous cap both jobs still fit.
    let _g = brooom::pyspell::install_global_rust(&[
        "if solution.vehicles_used > 5 { 1000 } else { 0 }",
    ])
    .unwrap();

    let mut problem = parse_input(TWO_SEATS_TWO_JOBS).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(sol.unassigned.len(), 0, "a soft global penalty must not drop jobs");
}

#[test]
fn dsl_global_rejects_route_namespace() {
    let _lock = guard();
    // A global program reads `solution.*`; `route.*` is a different evaluation
    // context. Installing compiles fine (the field exists in the schema), but at
    // eval time the route namespace is unavailable, so the program errors and is
    // treated as a hard violation — the solver stays correct and never panics.
    let _g = brooom::pyspell::install_global_rust(&["route.distance <= 100"]).unwrap();
    let mut problem = parse_input(TWO_SEATS_TWO_JOBS).unwrap();
    // Must not panic; a solution is still produced.
    let _sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
}

#[test]
fn dsl_global_compile_error_is_returned_not_panicked() {
    let _lock = guard();
    assert!(brooom::pyspell::install_global_rust(&["solution.nonsense_field > 1"]).is_err());
    // A failed install must not leave the registry populated.
    assert!(!brooom::global_constraint::has_global());
}
