//! Prize-collecting + cross-route (global) constraints: max-vehicles,
//! client-groups, fairness. The solver installs global constraints into a
//! process-global registry for the duration of each solve, so these tests take
//! a shared lock to avoid one solve's globals leaking into another's.

use std::sync::Mutex;

use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::solver::{solve, SolverConfig};

static LOCK: Mutex<()> = Mutex::new(());
fn guard() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn served(sol: &brooom::Solution, problem: &brooom::Problem) -> Vec<u64> {
    sol.routes
        .iter()
        .flat_map(|r| r.steps.iter().map(|s| s.description(problem).id))
        .collect()
}
fn vehicles_used(sol: &brooom::Solution) -> usize {
    sol.routes.iter().filter(|r| !r.steps.is_empty()).count()
}

#[test]
fn prize_collecting_drops_the_cheaper_optional_job() {
    let _lock = guard();
    // One seat (capacity 1). Job 1 is optional (small prize 5); job 2 keeps the
    // default sentinel prize (effectively mandatory). The objective drops the
    // job whose prize is cheapest to forgo → job 1.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [1]}],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1], "prize": 5},
            {"id": 2, "location": [10.12, 60.0], "delivery": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(served(&sol, &problem), vec![2], "the mandatory job 2 is kept; optional job 1 dropped");
    assert_eq!(sol.unassigned.len(), 1);
}

#[test]
fn max_vehicles_cap_forces_a_drop() {
    let _lock = guard();
    // Two single-seat vehicles, two jobs that each need a seat. Uncapped this
    // serves both (2 routes); capping to 1 vehicle means only one can be served
    // (opening a 2nd vehicle costs HARD ≫ a dropped job's prize).
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [1]},
            {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [1]}
        ],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.50, 60.0], "delivery": [1]}
        ]
    }"#;
    let cfg = SolverConfig { max_vehicles: Some(1), ..Default::default() };
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), cfg).unwrap();
    assert!(vehicles_used(&sol) <= 1, "cap of 1 vehicle must hold, used {}", vehicles_used(&sol));
    assert_eq!(sol.unassigned.len(), 1, "one job can't be placed within the 1-vehicle cap");
}

#[test]
fn client_group_serves_exactly_one() {
    let _lock = guard();
    // Two alternative jobs in group 1 — exactly one must be served.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1], "group": 1},
            {"id": 2, "location": [10.20, 60.0], "delivery": [1], "group": 1}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    let served = served(&sol, &problem);
    let group_served = served.iter().filter(|&&id| id == 1 || id == 2).count();
    assert_eq!(group_served, 1, "exactly one group member served, got {served:?}");
}

#[test]
fn fairness_weight_keeps_a_valid_solution() {
    let _lock = guard();
    // Fairness is a soft balancing term; it must not drop jobs or break solving.
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]},
            {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}
        ],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.20, 60.0], "delivery": [1]},
            {"id": 3, "location": [10.30, 60.0], "delivery": [1]},
            {"id": 4, "location": [10.40, 60.0], "delivery": [1]}
        ]
    }"#;
    let cfg = SolverConfig {
        fairness_weight: 1000.0,
        fairness_metric: brooom::FairnessMetric::Duration,
        ..Default::default()
    };
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), cfg).unwrap();
    assert_eq!(sol.unassigned.len(), 0, "a soft fairness term must not drop jobs");
}
