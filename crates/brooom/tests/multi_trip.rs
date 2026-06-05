//! Multi-trip / reloading: a vehicle returns to its depot mid-route, reloads,
//! and continues. Load resets per trip; time/distance accumulate over the shift.

use brooom::io::{parse_input, to_output};
use brooom::matrix::HaversineMatrix;
use brooom::solution::{eval_cache_invalidate, evaluate_route, StepKind, TaskRef};
use brooom::solver::{build_matrix, solve, SolverConfig};

fn prep(json: &str) -> (brooom::Problem, brooom::Matrix) {
    eval_cache_invalidate();
    let mut problem = parse_input(json).unwrap();
    let matrix = build_matrix(&mut problem, Some(&HaversineMatrix::default())).unwrap();
    (problem, matrix)
}

const TWO: &str = r#"{
    "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100]}],
    "jobs": [
        {"id": 1, "location": [10.10, 60.0], "delivery": [1]},
        {"id": 2, "location": [10.20, 60.0], "delivery": [1]}
    ]
}"#;

#[test]
fn reload_route_equals_two_single_trips() {
    let (p, m) = prep(TWO);
    let veh = &p.vehicles[0];
    let a = evaluate_route(&p, &m, veh, &[TaskRef::Job(0)]).unwrap();
    let b = evaluate_route(&p, &m, veh, &[TaskRef::Job(1)]).unwrap();
    let ab = evaluate_route(&p, &m, veh, &[TaskRef::Job(0), TaskRef::Reload, TaskRef::Job(1)]).unwrap();
    // A reload makes [A | reload | B] two independent depot round-trips, so its
    // travel/distance equal the sum of the two single-trip routes.
    assert_eq!(ab.travel_time, a.travel_time + b.travel_time, "travel = trip1 + trip2");
    assert_eq!(ab.distance, a.distance + b.distance, "distance = trip1 + trip2");
}

#[test]
fn reload_resets_load_so_a_capacity_one_vehicle_serves_two() {
    // Capacity 1, two unit-demand jobs: a single trip can carry only one, but
    // with max_trips=2 the vehicle reloads and serves both.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [1], "max_trips": 2}],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.20, 60.0], "delivery": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(sol.unassigned.len(), 0, "both jobs served across two trips");
    assert_eq!(sol.routes.len(), 1, "one vehicle, multi-trip");
    let reloads = sol.routes[0].steps.iter().filter(|s| s.is_reload()).count();
    assert_eq!(reloads, 1, "exactly one reload between the two trips");
}

#[test]
fn single_trip_capacity_one_drops_the_second() {
    // Same instance but single-trip (max_trips defaults to 1): only one fits.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [1]}],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.20, 60.0], "delivery": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(sol.unassigned.len(), 1, "single trip can carry only one unit");
}

#[test]
fn reload_renders_in_output() {
    let (mut p, m) = prep(TWO);
    let sol = brooom::Solution {
        routes: vec![brooom::Route {
            vehicle_idx: 0,
            steps: vec![TaskRef::Job(0), TaskRef::Reload, TaskRef::Job(1)],
            metrics: evaluate_route(&p, &m, &p.vehicles[0], &[TaskRef::Job(0), TaskRef::Reload, TaskRef::Job(1)]).unwrap(),
        }],
        unassigned: vec![],
        summary: Default::default(),
    };
    let _ = &mut p;
    let out = to_output(&p, &sol, Some(&m));
    let reloads = out.routes[0].steps.iter().filter(|s| s.kind == StepKind::Reload).count();
    assert_eq!(reloads, 1, "the reload appears as a `reload` step in the output");
}
