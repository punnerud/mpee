//! Integration tests for the constraint DSL (`--features pyspell`).
//!
//! Compile a constraint from Rust-expression text, install it, and check it
//! shapes a real `solve`. The whole file is gated on the `pyspell` feature and
//! takes a process-global lock (own test binary) like `custom_constraints.rs`.
#![cfg(feature = "pyspell")]

use std::sync::Mutex;

use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::solver::{solve, SolverConfig};

static LOCK: Mutex<()> = Mutex::new(());

fn guard() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
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

fn served_ids(sol: &brooom::Solution, problem: &brooom::Problem) -> Vec<u64> {
    sol.routes
        .iter()
        .flat_map(|r| r.steps.iter().map(|s| s.description(problem).id))
        .collect()
}

#[test]
fn dsl_hard_reject_drops_job() {
    let _lock = guard();
    let _g = brooom::pyspell::install_rust(&["!route.job_ids.contains(20)"]).unwrap();

    let mut problem = parse_input(TWO_JOBS).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();

    let served = served_ids(&sol, &problem);
    assert!(!served.contains(&20), "DSL constraint must drop job 20");
    assert!(served.contains(&10), "job 10 is unconstrained and should be served");
    assert_eq!(sol.unassigned.len(), 1);
}

#[test]
fn dsl_soft_penalty_keeps_all_jobs() {
    let _lock = guard();
    // A soft penalty on any non-trivial distance — discourages, never rejects.
    let _g = brooom::pyspell::install_rust(&["if route.distance > 1 { 500 } else { 0 }"]).unwrap();

    let mut problem = parse_input(TWO_JOBS).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();

    assert_eq!(sol.unassigned.len(), 0, "a soft penalty must not drop jobs");
}

#[test]
fn dsl_probe_mirrored_bound_solves() {
    let _lock = guard();
    // A generous probe-safe hard bound (route.distance <= huge): mirrored into
    // the insertion probe, every job still fits.
    let _g = brooom::pyspell::install_rust(&["route.distance <= 100000000"]).unwrap();
    assert!(brooom::constraint::has_probe_bounds(), "bound should be mirrored to the probe");

    let mut problem = parse_input(TWO_JOBS).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(sol.unassigned.len(), 0);
}

#[test]
fn dsl_precedence_before_orders_stops() {
    let _lock = guard();
    // Require job 20 before job 10 on any route serving both (guarded so routes
    // lacking either job are unaffected). No route may then have 10 before 20.
    let _g = brooom::pyspell::install_rust(&[
        "!route.job_ids.contains(10) || !route.job_ids.contains(20) || before(route.job_ids, 20, 10)",
    ])
    .unwrap();

    let mut problem = parse_input(TWO_JOBS).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    for r in &sol.routes {
        let ids: Vec<u64> = r.steps.iter().map(|s| s.description(&problem).id).collect();
        let p10 = ids.iter().position(|&x| x == 10);
        let p20 = ids.iter().position(|&x| x == 20);
        if let (Some(a), Some(b)) = (p10, p20) {
            assert!(b < a, "constraint requires job 20 before job 10 on a shared route");
        }
    }
}

#[test]
fn dsl_compile_error_is_returned_not_panicked() {
    let _lock = guard();
    assert!(brooom::pyspell::install_rust(&["route.nonsense_field > 1"]).is_err());
    assert!(brooom::pyspell::install_rust(&["std::process::exit(0)"]).is_err());
    // A failed install must not leave global state set.
    assert!(!brooom::constraint::has_constraints());
}
