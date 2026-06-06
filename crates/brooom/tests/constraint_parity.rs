//! New first-class constraints added to close the OR-Tools "constraints" gap:
//!   * native precedence (job A before job B on the same route),
//!   * HARD balance (cap the spread of route duration/load), and
//!   * k-of-N client-group cardinality (not just exactly-one).
//!
//! Each is a field/option, no DSL required — proving the README matrix rows are
//! real, not aspirational.

use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::solver::{solve, SolverConfig};
use brooom::solution::TaskRef;

fn job_ids_in_routes(sol: &brooom::Solution, problem: &brooom::Problem) -> Vec<Vec<u64>> {
    sol.routes
        .iter()
        .map(|r| {
            r.steps
                .iter()
                .filter(|s| matches!(s, TaskRef::Job(_)))
                .map(|s| s.description(problem).id)
                .collect()
        })
        .collect()
}

#[test]
fn native_precedence_orders_a_before_b() {
    // Three colinear stops east of the depot. The cheapest visiting order is by
    // distance: 10, 20, 30. We force job 30 BEFORE job 10 — the route must then
    // visit 30 ahead of 10 even though it costs more.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100]}],
        "jobs": [
            {"id": 10, "location": [10.05, 60.0], "delivery": [1]},
            {"id": 20, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 30, "location": [10.15, 60.0], "delivery": [1]}
        ],
        "precedence": [[30, 10]]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();

    assert_eq!(sol.unassigned.len(), 0, "all jobs should be served");
    let route = job_ids_in_routes(&sol, &problem).into_iter().find(|r| !r.is_empty()).unwrap();
    let p30 = route.iter().position(|&j| j == 30).unwrap();
    let p10 = route.iter().position(|&j| j == 10).unwrap();
    assert!(p30 < p10, "precedence [30,10] must put 30 before 10, got {route:?}");
}

#[test]
fn precedence_absent_is_unconstrained() {
    // Same instance without precedence: the natural (cheapest) order is 10,20,30.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100]}],
        "jobs": [
            {"id": 10, "location": [10.05, 60.0], "delivery": [1]},
            {"id": 20, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 30, "location": [10.15, 60.0], "delivery": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    // Without precedence all three are served on one route (the out-and-back order
    // on a symmetric line is a free tie-break, so we don't assert a direction).
    assert_eq!(sol.unassigned.len(), 0);
    let route = job_ids_in_routes(&sol, &problem).into_iter().find(|r| !r.is_empty()).unwrap();
    assert_eq!(route.len(), 3, "all three jobs on one route, got {route:?}");
}

#[test]
fn hard_balance_caps_route_spread() {
    // Two tight clusters far apart, two vehicles. Without balancing the solver may
    // load one route much heavier than the other. A HARD spread cap on load forces
    // a balanced split.
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100]},
            {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100]}
        ],
        "jobs": [
            {"id": 1, "location": [10.20, 60.0], "delivery": [10]},
            {"id": 2, "location": [10.205, 60.0], "delivery": [10]},
            {"id": 3, "location": [10.21, 60.0], "delivery": [10]},
            {"id": 4, "location": [9.80, 60.0], "delivery": [10]},
            {"id": 5, "location": [9.795, 60.0], "delivery": [10]},
            {"id": 6, "location": [9.79, 60.0], "delivery": [10]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let mut cfg = SolverConfig::default();
    cfg.fairness_metric = brooom::FairnessMetric::Load;
    cfg.balance_spread = Some(0); // identical loads required
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), cfg).unwrap();

    assert_eq!(sol.unassigned.len(), 0);
    let loads: Vec<i64> = sol
        .routes
        .iter()
        .filter(|r| !r.steps.is_empty())
        .map(|r| {
            r.steps
                .iter()
                .filter(|s| matches!(s, TaskRef::Job(_)))
                .map(|s| s.description(&problem).delivery.first().copied().unwrap_or(0))
                .sum::<i64>()
        })
        .collect();
    assert!(loads.len() >= 2, "balance should keep both routes in use, got {loads:?}");
    let spread = loads.iter().max().unwrap() - loads.iter().min().unwrap();
    assert_eq!(spread, 0, "hard balance(spread=0) must equalise loads, got {loads:?}");
}

#[test]
fn group_cardinality_k_of_n() {
    // Four jobs in one group. Default would serve exactly one; group_cardinality
    // (2, 2) must serve exactly two of them.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100]}],
        "jobs": [
            {"id": 1, "location": [10.05, 60.0], "delivery": [1], "group": 7, "prize": 100.0},
            {"id": 2, "location": [10.06, 60.0], "delivery": [1], "group": 7, "prize": 100.0},
            {"id": 3, "location": [10.07, 60.0], "delivery": [1], "group": 7, "prize": 100.0},
            {"id": 4, "location": [10.08, 60.0], "delivery": [1], "group": 7, "prize": 100.0}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let mut cfg = SolverConfig::default();
    cfg.group_cardinality = Some((2, 2));
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), cfg).unwrap();

    let served: usize = sol
        .routes
        .iter()
        .flat_map(|r| r.steps.iter())
        .filter(|s| matches!(s, TaskRef::Job(_)))
        .filter(|s| s.description(&problem).group == Some(7))
        .count();
    assert_eq!(served, 2, "group_cardinality (2,2) must serve exactly 2 of the group");
}

#[test]
fn group_default_is_exactly_one() {
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100]}],
        "jobs": [
            {"id": 1, "location": [10.05, 60.0], "delivery": [1], "group": 7, "prize": 100.0},
            {"id": 2, "location": [10.06, 60.0], "delivery": [1], "group": 7, "prize": 100.0},
            {"id": 3, "location": [10.07, 60.0], "delivery": [1], "group": 7, "prize": 100.0}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    let served: usize = sol
        .routes
        .iter()
        .flat_map(|r| r.steps.iter())
        .filter(|s| matches!(s, TaskRef::Job(_)))
        .filter(|s| s.description(&problem).group == Some(7))
        .count();
    assert_eq!(served, 1, "default group behaviour is exactly-one");
}
