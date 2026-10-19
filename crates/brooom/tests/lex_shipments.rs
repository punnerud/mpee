//! Lexicographic objectives on pickup&delivery shipments.
//!
//! Regressions for three bugs found while driving brooom from a warehouse
//! simulation (many bots, one shipment per item, explicit matrix):
//! 1. a route could hold a shipment's pickup while its delivery was unassigned;
//! 2. the `unassigned` level ignored shipments, so a later `makespan` level
//!    could drop whole shipments to shorten routes;
//! 3. `makespan` / `completion` could not move a whole pickup+delivery pair to
//!    an idle vehicle, so work stayed bundled on one vehicle.

use std::collections::HashSet;
use std::sync::Mutex;

use brooom::io::parse_input;
use brooom::solution::TaskRef;
use brooom::solver::{solve, LexObjective, ObjectiveMode, SolverConfig};

static LOCK: Mutex<()> = Mutex::new(());
fn guard() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Two vehicles, three shipments. Total travel is lowest when vehicle 0 does
/// everything (cost 18); spreading over both vehicles shortens the longest
/// route and finishes the deliveries earlier.
const TWO_BOTS: &str = r#"{
  "vehicles": [{"id": 0, "start_index": 0, "capacity": [3]},
               {"id": 1, "start_index": 1, "capacity": [3]}],
  "shipments": [
    {"id": 10, "amount": [1], "pickup": {"id": 100, "location_index": 2}, "delivery": {"id": 200, "location_index": 4}},
    {"id": 11, "amount": [1], "pickup": {"id": 101, "location_index": 3}, "delivery": {"id": 201, "location_index": 4}},
    {"id": 12, "amount": [1], "pickup": {"id": 102, "location_index": 2}, "delivery": {"id": 202, "location_index": 5}}],
  "matrices": {"car": {"durations": [[0,5,3,6,4,8],[5,0,4,2,6,3],[3,4,0,3,2,5],[6,2,3,0,4,2],[4,6,2,4,0,6],[8,3,5,2,6,0]]}}
}"#;

fn cfg(levels: Vec<LexObjective>) -> SolverConfig {
    SolverConfig {
        objective_mode: ObjectiveMode::Lexicographic { levels },
        time_limit_ms: Some(200),
        ..Default::default()
    }
}

fn solve_with(levels: Vec<LexObjective>) -> (brooom::Solution, brooom::problem::Problem) {
    let mut problem = parse_input(TWO_BOTS).unwrap();
    let sol = solve(&mut problem, None, cfg(levels)).unwrap();
    (sol, problem)
}

/// Every shipment is either fully routed (pickup before delivery, same route)
/// or fully unassigned.
fn assert_pairs_consistent(sol: &brooom::Solution) {
    for r in &sol.routes {
        let mut open = HashSet::new();
        for s in &r.steps {
            match s {
                TaskRef::Pickup(i) => assert!(open.insert(*i), "pickup {i} twice"),
                TaskRef::Delivery(i) => assert!(open.remove(i), "delivery {i} without its pickup first"),
                _ => {}
            }
        }
        assert!(open.is_empty(), "pickups {open:?} without their deliveries on a route");
    }
    let un: HashSet<_> = sol.unassigned.iter().copied().collect();
    for t in &sol.unassigned {
        match t {
            TaskRef::Pickup(i) => assert!(un.contains(&TaskRef::Delivery(*i)), "only pickup {i} unassigned"),
            TaskRef::Delivery(i) => assert!(un.contains(&TaskRef::Pickup(*i)), "only delivery {i} unassigned"),
            _ => {}
        }
    }
}

fn makespan(sol: &brooom::Solution) -> i64 {
    sol.routes.iter().filter(|r| !r.steps.is_empty())
        .map(|r| r.metrics.end_time - r.metrics.start_time).max().unwrap_or(0)
}

#[test]
fn makespan_level_never_splits_or_drops_shipments() {
    let _lock = guard();
    let (sol, _) = solve_with(vec![LexObjective::Makespan, LexObjective::Cost]);
    assert_pairs_consistent(&sol);
    assert!(sol.unassigned.is_empty(), "makespan level dropped tasks: {:?}", sol.unassigned);
}

#[test]
fn unassigned_then_makespan_spreads_over_both_vehicles() {
    let _lock = guard();
    let (sol, _) = solve_with(vec![LexObjective::UnassignedCount, LexObjective::Makespan, LexObjective::Cost]);
    assert_pairs_consistent(&sol);
    assert!(sol.unassigned.is_empty());
    assert!(makespan(&sol) < 18, "makespan {} not below the one-vehicle plan's 18", makespan(&sol));
}

#[test]
fn completion_level_finishes_deliveries_earlier_than_cost() {
    let _lock = guard();
    let (cost_sol, _) = solve_with(vec![LexObjective::Cost]);
    let (sol, _) = solve_with(vec![LexObjective::UnassignedCount, LexObjective::Completion, LexObjective::Cost]);
    assert_pairs_consistent(&sol);
    assert!(sol.unassigned.is_empty(), "completion level dropped tasks: {:?}", sol.unassigned);
    let completion = |s: &brooom::Solution| s.routes.iter().map(|r| r.metrics.completion).sum::<i64>();
    assert!(
        completion(&sol) < completion(&cost_sol),
        "completion {} not below cost-only plan's {}",
        completion(&sol),
        completion(&cost_sol)
    );
}
