//! Penalty-managed soft constraints (PyVRP-style time-warp).
//!
//! Proves, at two levels:
//!   * EVAL: with soft mode armed on the thread, `evaluate_route` no longer
//!     hard-rejects a route that misses a time window, exceeds capacity, or
//!     overruns its duration — it returns `Ok` with the violation recorded and a
//!     proportional penalty folded into the cost. With soft mode off the very
//!     same route is rejected (and the route-eval cache never leaks a soft
//!     result into a hard query).
//!   * SEARCH: a full solve in soft mode always returns a HARD-feasible plan,
//!     never assigns fewer jobs than the feasible-only baseline, and a problem
//!     without time windows auto-disables soft mode (byte-identical to before).
//!
//! Soft state is thread-local, so these tests serialise on a lock to avoid one
//! test arming penalties while another evaluates in hard mode on the same thread.

use std::sync::Mutex;

use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::solution::{
    eval_cache_invalidate, evaluate_route, set_soft_penalties, SoftWeights, TaskRef,
};
use brooom::solver::{build_matrix, solve, SolverConfig};

static LOCK: Mutex<()> = Mutex::new(());

fn guard() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn prep(json: &str) -> (brooom::Problem, brooom::Matrix) {
    eval_cache_invalidate();
    let mut problem = parse_input(json).unwrap();
    let matrix = build_matrix(&mut problem, Some(&HaversineMatrix::default())).unwrap();
    (problem, matrix)
}

// One depot, generous shift, capacity 1. Two delivery jobs of 1 unit each on a
// line east of the depot. A route serving BOTH overloads the capacity-1 vehicle
// (load 2 at start) — a hard reject — and the far job also has a tight window
// the second stop cannot meet, so it is a time-warp violation too.
const TIGHT: &str = r#"{
    "vehicles": [
        {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [1],
         "time_window": [0, 100000]}
    ],
    "jobs": [
        {"id": 10, "location": [10.10, 60.0], "delivery": [1], "service": 10},
        {"id": 20, "location": [10.20, 60.0], "delivery": [1], "service": 10,
         "time_windows": [[0, 1]]}
    ]
}"#;

const SOFT_W: SoftWeights = SoftWeights { tw: 1.0, load: 1.0, dur: 1.0 };

#[test]
fn eval_hard_rejects_what_soft_accepts_with_penalty() {
    let _lock = guard();
    let (problem, matrix) = prep(TIGHT);
    let veh = &problem.vehicles[0];
    let route = [TaskRef::Job(0), TaskRef::Job(1)];

    // Hard mode (default): the route is rejected outright.
    set_soft_penalties(None);
    eval_cache_invalidate();
    let hard = evaluate_route(&problem, &matrix, veh, &route);
    assert!(hard.is_err(), "hard mode must reject the overloaded/late route");

    // Soft mode: the route is feasible-with-penalty.
    set_soft_penalties(Some(SOFT_W));
    eval_cache_invalidate();
    let soft = evaluate_route(&problem, &matrix, veh, &route)
        .expect("soft mode must accept the route");
    assert!(soft.load_excess > 0.0, "capacity overload should be recorded");
    assert!(soft.tw_excess > 0.0, "time warp should be recorded on the late stop");
    assert!(soft.violation() > 0.0);
    // The penalty pushes the cost above the bare travel/span cost.
    assert!(
        soft.cost > soft.cost_travel + soft.cost_span + 1e-9,
        "soft penalty must be folded into the route cost"
    );

    set_soft_penalties(None);
}

#[test]
fn eval_cache_never_leaks_soft_result_into_hard_query() {
    let _lock = guard();
    let (problem, matrix) = prep(TIGHT);
    let veh = &problem.vehicles[0];
    let route = [TaskRef::Job(0), TaskRef::Job(1)];
    eval_cache_invalidate();

    // Populate the cache in soft mode...
    set_soft_penalties(Some(SOFT_W));
    assert!(evaluate_route(&problem, &matrix, veh, &route).is_ok());

    // ...then query in hard mode WITHOUT bumping the eval epoch. The soft
    // generation folded into the cache key must keep the hard query from
    // reusing the soft `Ok`.
    set_soft_penalties(None);
    assert!(
        evaluate_route(&problem, &matrix, veh, &route).is_err(),
        "hard query must re-reject, not reuse the cached soft Ok"
    );

    set_soft_penalties(None);
}

#[test]
fn larger_weight_charges_more_for_the_same_violation() {
    let _lock = guard();
    let (problem, matrix) = prep(TIGHT);
    let veh = &problem.vehicles[0];
    let route = [TaskRef::Job(0), TaskRef::Job(1)];

    set_soft_penalties(Some(SoftWeights { tw: 1.0, load: 1.0, dur: 1.0 }));
    eval_cache_invalidate();
    let lo = evaluate_route(&problem, &matrix, veh, &route).unwrap().cost;

    set_soft_penalties(Some(SoftWeights { tw: 100.0, load: 100.0, dur: 100.0 }));
    eval_cache_invalidate();
    let hi = evaluate_route(&problem, &matrix, veh, &route).unwrap().cost;

    assert!(hi > lo + 1e-6, "a heavier weight must cost more for the same violation");
    set_soft_penalties(None);
}

#[test]
fn soft_solve_returns_hard_feasible_plan() {
    let _lock = guard();
    // A feasible-but-tight TW instance. Soft search may wander through infeasible
    // space, but the returned plan must contain only hard-feasible routes.
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
             "time_window": [0, 100000]},
            {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
             "time_window": [0, 100000]}
        ],
        "jobs": [
            {"id": 10, "location": [10.05, 60.0], "delivery": [1], "time_windows": [[0, 5000]]},
            {"id": 20, "location": [10.10, 60.0], "delivery": [1], "time_windows": [[0, 8000]]},
            {"id": 30, "location": [10.15, 60.0], "delivery": [1], "time_windows": [[0, 12000]]},
            {"id": 40, "location": [10.20, 60.0], "delivery": [1], "time_windows": [[0, 16000]]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    eval_cache_invalidate();
    let mut cfg = SolverConfig::default();
    cfg.soft_search = Some(true);
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), cfg).unwrap();

    // Every returned route must be hard-feasible when re-checked with soft off.
    let (problem2, matrix2) = prep(json);
    set_soft_penalties(None);
    for r in &sol.routes {
        let veh = &problem2.vehicles[r.vehicle_idx];
        assert!(
            evaluate_route(&problem2, &matrix2, veh, &r.steps).is_ok(),
            "soft search returned a route that is not hard-feasible"
        );
        for s in &r.steps {
            assert!(r.metrics.violation() == 0.0);
            let _ = s;
        }
    }
}

#[test]
fn soft_does_not_assign_fewer_jobs_than_hard() {
    let _lock = guard();
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [5],
             "time_window": [0, 100000]}
        ],
        "jobs": [
            {"id": 10, "location": [10.05, 60.0], "delivery": [1], "time_windows": [[0, 6000]]},
            {"id": 20, "location": [10.10, 60.0], "delivery": [1], "time_windows": [[0, 9000]]},
            {"id": 30, "location": [10.15, 60.0], "delivery": [1], "time_windows": [[0, 13000]]}
        ]
    }"#;

    let mut p_hard = parse_input(json).unwrap();
    eval_cache_invalidate();
    let mut hard_cfg = SolverConfig::default();
    hard_cfg.soft_search = Some(false);
    let hard = solve(&mut p_hard, Some(&HaversineMatrix::default()), hard_cfg).unwrap();

    let mut p_soft = parse_input(json).unwrap();
    eval_cache_invalidate();
    let mut soft_cfg = SolverConfig::default();
    soft_cfg.soft_search = Some(true);
    let soft = solve(&mut p_soft, Some(&HaversineMatrix::default()), soft_cfg).unwrap();

    assert!(
        soft.unassigned.len() <= hard.unassigned.len(),
        "soft search assigned fewer jobs ({} unassigned) than hard ({} unassigned)",
        soft.unassigned.len(),
        hard.unassigned.len()
    );
}

#[test]
fn no_time_windows_auto_disables_soft_and_is_deterministic() {
    let _lock = guard();
    // No job/vehicle time windows ⇒ AUTO (soft_search = None) must resolve to
    // OFF and be byte-identical to an explicit hard solve, run-to-run.
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}
        ],
        "jobs": [
            {"id": 10, "location": [10.05, 60.0], "delivery": [1]},
            {"id": 20, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 30, "location": [10.15, 60.0], "delivery": [1]}
        ]
    }"#;

    let run = |soft: Option<bool>| {
        let mut p = parse_input(json).unwrap();
        eval_cache_invalidate();
        let mut cfg = SolverConfig::default();
        cfg.soft_search = soft;
        solve(&mut p, Some(&HaversineMatrix::default()), cfg).unwrap().summary.cost
    };

    let auto = run(None);
    let forced_off = run(Some(false));
    assert!(
        (auto - forced_off).abs() < 1e-6,
        "AUTO must equal hard on a no-TW instance: auto={auto} off={forced_off}"
    );
    // Determinism: AUTO again gives the same cost.
    assert!((auto - run(None)).abs() < 1e-6);
}
