//! Multi-objective cost shaping: per-vehicle span_cost / distance_weight /
//! time_weight, the decomposed RouteMetrics cost components, and the optional
//! SolverConfig::objective_weights global multiplier.
//!
//! HONEST CAVEAT (matches the source comments): everything here is *weighted
//! scalarization*, not true lexicographic multi-objective search. The local
//! search still minimises one aggregated scalar; these knobs only shape it.
//!
//! The solver installs global constraints (incl. objective_weights) into a
//! process-global registry for the duration of each solve, so the solve-level
//! tests take a shared lock to avoid leaking globals across tests.

use std::sync::Mutex;

use brooom::io::parse_input;
use brooom::matrix::{HaversineMatrix, Matrix};
use brooom::problem::{ProvidedMatrix, Vehicle};
use brooom::solution::{evaluate_route, TaskRef};
use brooom::solver::{solve, SolverConfig};

static LOCK: Mutex<()> = Mutex::new(());
fn guard() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A 2-node problem (depot=0, one customer=1) with an explicit matrix so we can
/// drive `evaluate_route` directly and reason about exact numbers.
///
/// duration(0,1)=duration(1,0)=600s (10 min each way → 1200s round trip).
/// distance(0,1)=distance(1,0)=5000m  (10 km round trip).
fn two_node_problem(modify: impl FnOnce(&mut Vehicle)) -> (brooom::Problem, Matrix) {
    let json = r#"{
        "vehicles": [{"id": 1, "start_index": 0, "end_index": 0}],
        "jobs": [{"id": 10, "location_index": 1, "service": 30}]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let pm = ProvidedMatrix {
        durations: vec![vec![0, 600], vec![600, 0]],
        distances: Some(vec![vec![0, 5000], vec![5000, 0]]),
    };
    let matrix = Matrix::from_provided(&pm).unwrap();
    modify(&mut problem.vehicles[0]);
    (problem, matrix)
}

#[test]
fn defaults_reproduce_historical_cost_exactly() {
    // With span_cost=0, distance_weight=0, time_weight=1 (the serde defaults),
    // the cost must equal the historical formula:
    //   fixed + travel_time*(per_hour/3600) + service_time*1e-6.
    let (problem, matrix) = two_node_problem(|_v| {});
    let v = &problem.vehicles[0];
    assert_eq!(v.span_cost, 0.0);
    assert_eq!(v.distance_weight, 0.0);
    assert_eq!(v.time_weight, 1.0);

    let steps = [TaskRef::Job(0)];
    let m = evaluate_route(&problem, &matrix, v, &steps).unwrap();

    // travel_time = 1200s (round trip), per_hour default 3600 → 1.0 per second.
    assert_eq!(m.travel_time, 1200);
    let expected = v.fixed
        + 1200.0 * (v.per_hour / 3600.0)
        + (m.service_time as f64) * 1e-6;
    assert!((m.cost - expected).abs() < 1e-9, "cost {} != historical {}", m.cost, expected);

    // Decomposition: span/custom are zero, travel carries everything, and the
    // components sum to the scalar.
    assert_eq!(m.cost_span, 0.0);
    assert_eq!(m.cost_custom, 0.0);
    assert!((m.cost_travel - m.cost).abs() < 1e-9);
    assert!((m.cost - (m.cost_travel + m.cost_span + m.cost_custom)).abs() < 1e-12);
}

#[test]
fn span_cost_adds_a_span_component() {
    // span_cost charges per second of span (end - start). The single-customer
    // route spans travel(1200) + service(30) = 1230s.
    let (problem, matrix) = two_node_problem(|v| v.span_cost = 2.0);
    let v = &problem.vehicles[0];
    let steps = [TaskRef::Job(0)];
    let m = evaluate_route(&problem, &matrix, v, &steps).unwrap();

    let span = m.end_time - m.start_time;
    assert_eq!(span, 1230);
    assert!((m.cost_span - (span as f64) * 2.0).abs() < 1e-9, "cost_span={}", m.cost_span);
    // Travel component is unchanged from default; the total grew by exactly the
    // span component.
    let (base_problem, base_matrix) = two_node_problem(|_v| {});
    let base = evaluate_route(&base_problem, &base_matrix, &base_problem.vehicles[0], &steps).unwrap();
    assert!((m.cost_travel - base.cost_travel).abs() < 1e-9);
    assert!((m.cost - (base.cost + m.cost_span)).abs() < 1e-9);
    // Components still sum to the scalar.
    assert!((m.cost - (m.cost_travel + m.cost_span + m.cost_custom)).abs() < 1e-9);
}

#[test]
fn distance_weight_and_time_weight_shape_the_travel_component() {
    // distance_weight bills per metre; time_weight scales the per-hour time term.
    let (problem, matrix) = two_node_problem(|v| {
        v.distance_weight = 0.001; // 0.001 cost per metre
        v.time_weight = 0.5; // halve the time term
    });
    let v = &problem.vehicles[0];
    let steps = [TaskRef::Job(0)];
    let m = evaluate_route(&problem, &matrix, v, &steps).unwrap();

    // distance round trip = 10000m → +10.0 ; time term halved.
    let expected_travel = v.fixed
        + 1200.0 * (v.per_hour / 3600.0) * 0.5
        + 10000.0 * 0.001
        + (m.service_time as f64) * 1e-6;
    assert!((m.cost_travel - expected_travel).abs() < 1e-9, "cost_travel={}", m.cost_travel);
    assert!((m.cost - m.cost_travel).abs() < 1e-9, "no span/custom here");
}

#[test]
fn objective_weights_default_is_a_noop() {
    let _lock = guard();
    // An all-1.0 objective_weights set (and None) must leave the solution and its
    // cost identical to today's behaviour.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.20, 60.0], "delivery": [1]}
        ]
    }"#;

    let mut p0 = parse_input(json).unwrap();
    let baseline = solve(&mut p0, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();

    let cfg = SolverConfig {
        objective_weights: Some(brooom::global_constraint::ObjectiveWeights::default()),
        ..Default::default()
    };
    let mut p1 = parse_input(json).unwrap();
    let weighted = solve(&mut p1, Some(&HaversineMatrix::default()), cfg).unwrap();

    assert_eq!(weighted.unassigned.len(), baseline.unassigned.len());
    assert!(
        (weighted.summary.cost - baseline.summary.cost).abs() < 1e-6,
        "identity weights changed cost: {} vs {}",
        weighted.summary.cost,
        baseline.summary.cost
    );
}

#[test]
fn objective_weights_inflate_the_global_cost() {
    let _lock = guard();
    // Weighting travel by 3.0 must raise the reported objective above the
    // baseline (the travel component dominates here) while keeping every job
    // served — it is a soft re-scaling, not a hard constraint.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.20, 60.0], "delivery": [1]}
        ]
    }"#;

    let mut p0 = parse_input(json).unwrap();
    let baseline = solve(&mut p0, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();

    let cfg = SolverConfig {
        objective_weights: Some(brooom::global_constraint::ObjectiveWeights {
            travel: 3.0,
            span: 1.0,
            custom: 1.0,
        }),
        ..Default::default()
    };
    let mut p1 = parse_input(json).unwrap();
    let weighted = solve(&mut p1, Some(&HaversineMatrix::default()), cfg).unwrap();

    assert_eq!(weighted.unassigned.len(), 0, "soft re-weighting must not drop jobs");
    assert!(
        weighted.summary.cost > baseline.summary.cost + 1e-6,
        "travel weight 3.0 should inflate the objective: {} !> {}",
        weighted.summary.cost,
        baseline.summary.cost
    );
}
