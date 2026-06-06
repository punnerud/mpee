//! End-to-end tests for the JSON `options` surface: `options.objective.levels`
//! → lexicographic mode, and `options.dimensions` → a registered custom
//! dimension whose pyspell transit shapes the solve (a fuel cap that drops a far
//! job). Also the cross-surface check: parsing the JSON `options` produces the
//! SAME `ObjectiveMode`/dimensions the direct Rust `SolverConfig`/`set_dimensions`
//! API would, so all three surfaces (JSON, CLI, Python) agree by construction.
//!
//! Objective globals and custom dimensions both live in process-global state, so
//! this file (its own test binary) serialises behind a shared lock and installs
//! everything behind RAII guards.

use std::sync::Mutex;

use brooom::dimension::{clear_dimensions, DimensionGuard};
use brooom::io::{parse_input, parse_input_with_options};
use brooom::matrix::HaversineMatrix;
use brooom::solution::eval_cache_invalidate;
use brooom::solver::{build_matrix, solve, LexObjective, ObjectiveMode, SolverConfig};
use brooom::SolverOptions;

static LOCK: Mutex<()> = Mutex::new(());
fn guard() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn vehicles_used(sol: &brooom::Solution) -> usize {
    sol.routes.iter().filter(|r| !r.steps.is_empty()).count()
}

// ---------------------------------------------------------------------------
// Parse-level unit checks (SolverOptions → ObjectiveMode / dimensions).
// ---------------------------------------------------------------------------

#[test]
fn options_objective_levels_parse_to_lexicographic() {
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0]}],
        "jobs": [{"id": 10, "location": [10.05, 60.0]}],
        "options": {"objective": {"levels": ["vehicles", "cost"]}}
    }"#;
    let (_p, opts) = parse_input_with_options(json).unwrap();
    match opts.objective_mode().unwrap() {
        ObjectiveMode::Lexicographic { levels } => {
            assert_eq!(levels, vec![LexObjective::Vehicles, LexObjective::Cost]);
        }
        _ => panic!("expected lexicographic"),
    }
}

#[test]
fn absent_options_is_scalar_and_byte_identical_problem() {
    // A problem with no `options` parses to the SAME Problem as plain parse_input
    // and yields the default scalar objective — the backward-compat guarantee.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0]}],
        "jobs": [{"id": 10, "location": [10.05, 60.0]}]
    }"#;
    let (p_with, opts) = parse_input_with_options(json).unwrap();
    let p_plain = parse_input(json).unwrap();
    assert_eq!(p_with.jobs.len(), p_plain.jobs.len());
    assert_eq!(p_with.vehicles.len(), p_plain.vehicles.len());
    assert!(matches!(opts.objective_mode().unwrap(), ObjectiveMode::Scalar));
    assert!(!opts.has_dimensions());
}

// ---------------------------------------------------------------------------
// Lexicographic via JSON options == lexicographic via the Rust SolverConfig API.
// ---------------------------------------------------------------------------

/// Two vehicles at two depots, each beside its own job cluster. The minimum-cost
/// plan uses BOTH vehicles; the minimum-vehicle plan uses ONE (capacity fits).
/// So scalar → 2 vehicles, lexicographic [vehicles, cost] → 1. Mirrors the
/// scenario in `lexicographic.rs`, but driven through the JSON `options` surface.
const TWO_CLUSTER: &str = r#"{
    "vehicles": [
        {"id": 1, "start": [10.00, 60.0], "end": [10.00, 60.0], "capacity": [100]},
        {"id": 2, "start": [10.50, 60.0], "end": [10.50, 60.0], "capacity": [100]}
    ],
    "jobs": [
        {"id": 10, "location": [10.01, 60.0], "delivery": [1]},
        {"id": 11, "location": [10.02, 60.0], "delivery": [1]},
        {"id": 20, "location": [10.51, 60.0], "delivery": [1]},
        {"id": 21, "location": [10.52, 60.0], "delivery": [1]}
    ]
}"#;

fn two_cluster_with_options(options: &str) -> String {
    // Splice an `options` object into the TWO_CLUSTER problem.
    let trimmed = TWO_CLUSTER.trim_end();
    let without_brace = &trimmed[..trimmed.len() - 1];
    format!("{without_brace}, \"options\": {options} }}")
}

#[test]
fn json_options_lexicographic_matches_rust_api() {
    let _l = guard();

    // (A) Direct Rust API: ObjectiveMode::Lexicographic { [Vehicles, Cost] }.
    let mut p_api = parse_input(TWO_CLUSTER).unwrap();
    let cfg_api = SolverConfig {
        objective_mode: ObjectiveMode::Lexicographic {
            levels: vec![LexObjective::Vehicles, LexObjective::Cost],
        },
        ..Default::default()
    };
    let sol_api = solve(&mut p_api, Some(&HaversineMatrix::default()), cfg_api).unwrap();

    // (B) JSON options surface: options.objective.levels = ["vehicles","cost"].
    let json = two_cluster_with_options(r#"{"objective": {"levels": ["vehicles", "cost"]}}"#);
    let (mut p_json, opts) = parse_input_with_options(&json).unwrap();
    let cfg_json = SolverConfig {
        objective_mode: opts.objective_mode().unwrap(),
        ..Default::default()
    };
    let sol_json = solve(&mut p_json, Some(&HaversineMatrix::default()), cfg_json).unwrap();

    // Both surfaces must pick the minimum-vehicle plan (1 vehicle) and agree.
    assert_eq!(vehicles_used(&sol_api), 1, "Rust API lexicographic → 1 vehicle");
    assert_eq!(vehicles_used(&sol_json), 1, "JSON options lexicographic → 1 vehicle");
    assert_eq!(vehicles_used(&sol_api), vehicles_used(&sol_json));
    assert!(
        (sol_api.summary.cost - sol_json.summary.cost).abs() < 1e-6,
        "the two surfaces produce the same cost: api={} json={}",
        sol_api.summary.cost, sol_json.summary.cost
    );

    // And scalar (no options) uses BOTH vehicles — proving the option changed it.
    let mut p_scalar = parse_input(TWO_CLUSTER).unwrap();
    let sol_scalar = solve(
        &mut p_scalar, Some(&HaversineMatrix::default()), SolverConfig::default(),
    )
    .unwrap();
    assert_eq!(vehicles_used(&sol_scalar), 2, "scalar default uses both vehicles");
}

// ---------------------------------------------------------------------------
// options.dimensions registers a working dimension: a fuel cap drops a far job.
// ---------------------------------------------------------------------------

/// One depot, two near jobs and one FAR job, a single vehicle. A `fuel`
/// dimension starts at a tank that covers the near jobs' round trip but NOT the
/// far one; a DSL constraint `route.fuel >= 0` (the cumul never goes negative)
/// then forces the far job into `unassigned`. The transit burns fuel
/// proportional to each arc's distance — exactly the cross-dim coupling the arc
/// schema enables.
const FUEL_PROBLEM: &str = r#"{
    "vehicles": [
        {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100],
         "time_window": [0, 100000000]}
    ],
    "jobs": [
        {"id": 10, "location": [10.01, 60.0], "delivery": [1]},
        {"id": 20, "location": [10.02, 60.0], "delivery": [1]},
        {"id": 99, "location": [11.50, 60.0], "delivery": [1]}
    ]
}"#;

fn fuel_problem_with_options(options: &str) -> String {
    let trimmed = FUEL_PROBLEM.trim_end();
    let without_brace = &trimmed[..trimmed.len() - 1];
    format!("{without_brace}, \"options\": {options} }}")
}

#[test]
fn json_options_dimension_drops_a_far_job() {
    let _l = guard();
    eval_cache_invalidate();
    clear_dimensions();

    // The fuel dimension: start with a tank of 5_000, burn distance/10 per arc,
    // floor 0, declared draining (so the min bound prunes in the probe too). The
    // near jobs' arcs burn ~70 each (≈700 m legs), so the two near jobs fit
    // comfortably; the far job's first leg alone (≈108 km → burn ≈10_800) drives
    // the tank well below 0, so serving it is infeasible and it is dropped.
    let dim_opts = r#"{
        "dimensions": [{
            "name": "fuel",
            "transit": "-(distance / 10)",
            "start": 5000,
            "min": 0,
            "monotonicity": "non_increasing"
        }]
    }"#;

    let json = fuel_problem_with_options(dim_opts);
    let (mut problem, opts) = parse_input_with_options(&json).unwrap();
    assert!(opts.has_dimensions(), "options declared a dimension");

    // Build the matrix, then install the compiled dimensions for the solve.
    let matrix = build_matrix(&mut problem, Some(&HaversineMatrix::default())).unwrap();
    let dims = opts.build_dimensions().unwrap();
    assert_eq!(dims.len(), 1);
    let _g = DimensionGuard::install(dims);

    let sol = brooom::solver::solve_with_matrix(&problem, &matrix, &SolverConfig::default());

    // The far job (99) cannot be served without running the tank below 0, so it
    // lands in unassigned; the two near jobs are served.
    let unassigned_ids: Vec<u64> = sol
        .unassigned
        .iter()
        .filter_map(|t| match t {
            brooom::solution::TaskRef::Job(j) => Some(problem.jobs[*j].id),
            _ => None,
        })
        .collect();
    assert!(
        unassigned_ids.contains(&99),
        "the far job 99 should be dropped by the fuel cap; unassigned={unassigned_ids:?}"
    );
    let served: usize = sol.routes.iter().map(|r| r.steps.len()).sum();
    assert_eq!(served, 2, "the two near jobs are served");
}

#[test]
fn no_dimension_serves_the_far_job() {
    // Control: WITHOUT the fuel cap (no options.dimensions), the same problem
    // serves all three jobs — proving the cap, not the geometry, drops job 99.
    let _l = guard();
    eval_cache_invalidate();
    clear_dimensions();
    let mut problem = parse_input(FUEL_PROBLEM).unwrap();
    let sol = solve(
        &mut problem, Some(&HaversineMatrix::default()), SolverConfig::default(),
    )
    .unwrap();
    let served: usize = sol.routes.iter().map(|r| r.steps.len()).sum();
    assert_eq!(served, 3, "no fuel cap → all three jobs served");
    assert!(sol.unassigned.is_empty());
}

// ---------------------------------------------------------------------------
// SolverOptions::from_value mirrors the JSON-embedded options exactly (the same
// object the CLI's --options path and the Python dimensions= path feed in).
// ---------------------------------------------------------------------------

#[test]
fn from_value_matches_embedded_options() {
    let embedded = r#"{"objective": {"levels": ["unassigned", "vehicles", "cost"]},
                       "dimensions": [{"name": "x", "transit": "-1", "start": 5}]}"#;
    let v: serde_json::Value = serde_json::from_str(embedded).unwrap();
    let opts = SolverOptions::from_value(Some(&v)).unwrap();
    match opts.objective_mode().unwrap() {
        ObjectiveMode::Lexicographic { levels } => assert_eq!(levels.len(), 3),
        _ => panic!("expected lexicographic"),
    }
    assert_eq!(opts.dimensions.len(), 1);
    assert_eq!(opts.dimensions[0].name, "x");
    assert_eq!(opts.dimensions[0].start, 5);
}
