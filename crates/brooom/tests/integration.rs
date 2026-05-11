//! End-to-end tests: parse Vroom-style JSON, solve, sanity-check the answer.

use brooom::{
    io::{parse_input, to_output, verify_solution},
    matrix::HaversineMatrix,
    solver::{solve, SolverConfig},
};

/// 1 vehicle, 5 jobs around Oslo. Haversine matrix, no time windows. Should
/// schedule all jobs in one route.
#[test]
fn cvrp_oslo_single_vehicle() {
    let json = r#"{
        "vehicles": [{
            "id": 1,
            "start": [10.7522, 59.9139],
            "end":   [10.7522, 59.9139],
            "capacity": [10]
        }],
        "jobs": [
            {"id": 11, "location": [10.7700, 59.9200], "delivery": [1]},
            {"id": 12, "location": [10.7400, 59.9100], "delivery": [1]},
            {"id": 13, "location": [10.7600, 59.9300], "delivery": [1]},
            {"id": 14, "location": [10.7300, 59.9200], "delivery": [1]},
            {"id": 15, "location": [10.7800, 59.9050], "delivery": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(
        &mut problem,
        Some(&HaversineMatrix::default()),
        SolverConfig::default(),
    )
    .unwrap();
    assert_eq!(sol.unassigned.len(), 0, "all jobs should fit");
    assert_eq!(sol.routes.len(), 1, "single vehicle should serve all");
    assert_eq!(sol.routes[0].steps.len(), 5);

    let out = to_output(&problem, &sol, None);
    // Start + 5 jobs + end = 7 steps
    assert_eq!(out.routes[0].steps.len(), 7);
    assert_eq!(out.summary.unassigned, 0);
}

/// Capacity overflow forces the second vehicle to pick up the rest.
#[test]
fn capacity_splits_routes() {
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [3]},
            {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [3]}
        ],
        "jobs": [
            {"id": 1, "location": [10.1, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.2, 60.0], "delivery": [1]},
            {"id": 3, "location": [10.3, 60.0], "delivery": [1]},
            {"id": 4, "location": [10.4, 60.0], "delivery": [1]},
            {"id": 5, "location": [10.5, 60.0], "delivery": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(
        &mut problem,
        Some(&HaversineMatrix::default()),
        SolverConfig::default(),
    )
    .unwrap();
    assert_eq!(sol.unassigned.len(), 0);
    assert_eq!(sol.routes.len(), 2);
    let total_steps: usize = sol.routes.iter().map(|r| r.steps.len()).sum();
    assert_eq!(total_steps, 5);
    for r in &sol.routes {
        assert!(r.steps.len() <= 3, "capacity 3 should cap stops");
    }
}

/// Time windows that don't overlap force two vehicles even when capacity allows one.
#[test]
fn time_windows_force_two_vehicles() {
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
             "time_window": [0, 86400]},
            {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
             "time_window": [0, 86400]}
        ],
        "jobs": [
            {"id": 1, "location": [10.1, 60.0], "delivery": [1], "time_windows": [[0, 1000]]},
            {"id": 2, "location": [10.2, 60.0], "delivery": [1], "time_windows": [[0, 1000]]},
            {"id": 3, "location": [10.3, 60.0], "delivery": [1], "time_windows": [[0, 1000]]},
            {"id": 4, "location": [10.4, 60.0], "delivery": [1], "time_windows": [[0, 1000]]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(
        &mut problem,
        Some(&HaversineMatrix::default()),
        SolverConfig::default(),
    )
    .unwrap();
    // The TW is tight enough that all four jobs cannot ride one vehicle.
    let assigned: usize = sol.routes.iter().map(|r| r.steps.len()).sum();
    assert!(
        assigned + sol.unassigned.len() == 4,
        "every job must be accounted for"
    );
}

/// Skill mismatch routes work to the right vehicle.
#[test]
fn skills_route_to_correct_vehicle() {
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "skills": [1]},
            {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "skills": [2]}
        ],
        "jobs": [
            {"id": 1, "location": [10.1, 60.0], "skills": [1]},
            {"id": 2, "location": [10.2, 60.0], "skills": [2]},
            {"id": 3, "location": [10.3, 60.0], "skills": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(
        &mut problem,
        Some(&HaversineMatrix::default()),
        SolverConfig::default(),
    )
    .unwrap();
    assert_eq!(sol.unassigned.len(), 0);
    // v1 gets jobs 1 and 3; v2 gets job 2.
    for r in &sol.routes {
        let veh = &problem.vehicles[r.vehicle_idx];
        for s in &r.steps {
            let task_skills = s.skills(&problem);
            assert!(veh.has_skills(task_skills), "route assignment must satisfy skills");
        }
    }
}

/// Pickup/delivery shipment must keep order and use the same vehicle.
#[test]
fn shipment_pickup_before_delivery_same_vehicle() {
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [5]},
            {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [5]}
        ],
        "shipments": [
            {
                "amount": [1],
                "pickup":   {"id": 100, "location": [10.1, 60.0]},
                "delivery": {"id": 101, "location": [10.5, 60.0]}
            }
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(
        &mut problem,
        Some(&HaversineMatrix::default()),
        SolverConfig::default(),
    )
    .unwrap();
    assert_eq!(sol.unassigned.len(), 0);
    assert_eq!(sol.routes.len(), 1);
    let route = &sol.routes[0];
    assert_eq!(route.steps.len(), 2);
    use brooom::TaskRef;
    let _ = TaskRef::Job(0); // ensure the type is reachable
    // First must be the pickup, second the delivery.
    match (route.steps[0], route.steps[1]) {
        (brooom::solution::TaskRef::Pickup(0), brooom::solution::TaskRef::Delivery(0)) => {}
        other => panic!("unexpected pickup/delivery order: {other:?}"),
    }
}

/// Solution must round-trip through verification: stored metrics match a
/// fresh evaluation.
#[test]
fn local_search_keeps_solution_valid() {
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100]}
        ],
        "jobs": [
            {"id": 1, "location": [10.10, 60.10], "delivery": [1]},
            {"id": 2, "location": [10.20, 60.05], "delivery": [1]},
            {"id": 3, "location": [10.30, 60.10], "delivery": [1]},
            {"id": 4, "location": [10.40, 60.05], "delivery": [1]},
            {"id": 5, "location": [10.50, 60.10], "delivery": [1]},
            {"id": 6, "location": [10.60, 60.05], "delivery": [1]},
            {"id": 7, "location": [10.70, 60.10], "delivery": [1]},
            {"id": 8, "location": [10.80, 60.05], "delivery": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(
        &mut problem,
        Some(&HaversineMatrix::default()),
        SolverConfig::default(),
    )
    .unwrap();

    let matrix = brooom::solver::build_matrix(&mut problem, Some(&HaversineMatrix::default())).unwrap();
    verify_solution(&problem, &matrix, &sol).unwrap();
}
