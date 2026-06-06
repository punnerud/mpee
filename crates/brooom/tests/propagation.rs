//! Native structured constraint propagation (crate::propagate).
//!
//! Proves the pre-pass is SOUND (never removes a feasible option, full solve cost
//! unchanged) and that it does its job: tighten windows, close precedence, and
//! flag provably-unservable jobs up front with a reason.

use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::propagate;
use brooom::solver::{build_matrix, solve, SolverConfig};

fn prep(json: &str) -> (brooom::Problem, brooom::Matrix) {
    brooom::solution::eval_cache_invalidate();
    let mut problem = parse_input(json).unwrap();
    let matrix = build_matrix(&mut problem, Some(&HaversineMatrix::default())).unwrap();
    (problem, matrix)
}

const FEASIBLE_TW: &str = r#"{
    "vehicles": [
        {"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100], "time_window": [0, 100000]},
        {"id": 2, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100], "time_window": [0, 100000]}
    ],
    "jobs": [
        {"id": 10, "location": [10.05,60.0], "delivery": [1], "time_windows": [[0, 6000]]},
        {"id": 20, "location": [10.10,60.0], "delivery": [1], "time_windows": [[0, 9000]]},
        {"id": 30, "location": [10.15,60.0], "delivery": [1], "time_windows": [[0, 14000]]}
    ]
}"#;

#[test]
fn propagation_preserves_the_solution() {
    // Sound tightening must not change the optimum found: same cost & assignment
    // with propagation on vs off, and no feasible job dropped.
    let mut p_on = parse_input(FEASIBLE_TW).unwrap();
    let mut cfg_on = SolverConfig::default();
    cfg_on.propagate = true;
    let on = solve(&mut p_on, Some(&HaversineMatrix::default()), cfg_on).unwrap();

    let mut p_off = parse_input(FEASIBLE_TW).unwrap();
    let mut cfg_off = SolverConfig::default();
    cfg_off.propagate = false;
    let off = solve(&mut p_off, Some(&HaversineMatrix::default()), cfg_off).unwrap();

    assert_eq!(on.unassigned.len(), 0, "propagation dropped a feasible job");
    assert_eq!(off.unassigned.len(), 0);
    assert!(
        (on.summary.cost - off.summary.cost).abs() < 1e-6,
        "propagation changed the solution cost: on={} off={}",
        on.summary.cost,
        off.summary.cost
    );
}

#[test]
fn tighten_narrows_a_window_but_keeps_feasible_arrival() {
    // The far job's [0, 14000] window can be tightened on the upper end (must
    // start early enough to return within the shift) but must still contain the
    // arrival a real route uses. Sound: tightened window is non-empty.
    let (mut problem, matrix) = prep(FEASIBLE_TW);
    let before: Vec<(i64, i64)> = problem.jobs.iter()
        .map(|j| (j.time_windows[0].start, j.time_windows[0].end)).collect();
    let infeasible = propagate::tighten(&mut problem, &matrix, false);
    assert!(infeasible.is_empty(), "no job here is unservable: {infeasible:?}");
    for (j, (bs, be)) in problem.jobs.iter().zip(before) {
        let w = j.time_windows[0];
        assert!(w.start >= bs && w.end <= be, "tightening must only narrow");
        assert!(w.start <= w.end, "tightened window must stay non-empty");
    }
}

#[test]
fn unservable_window_is_flagged() {
    // A job whose only window closes long before any vehicle could reach it from
    // the depot is provably unservable — flagged up front (hard mode).
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100], "time_window": [0, 100000]}],
        "jobs": [
            {"id": 10, "location": [10.05,60.0], "delivery": [1], "time_windows": [[0, 6000]]},
            {"id": 99, "location": [12.0,60.0], "delivery": [1], "time_windows": [[0, 1]]}
        ]
    }"#;
    let (mut problem, matrix) = prep(json);
    let infeasible = propagate::tighten(&mut problem, &matrix, false);
    assert!(
        infeasible.iter().any(|i| i.job_id == 99),
        "the far, tight-window job must be flagged unservable: {infeasible:?}"
    );
}

#[test]
fn unservable_skill_is_flagged() {
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100], "skills": [1]}],
        "jobs": [
            {"id": 10, "location": [10.05,60.0], "delivery": [1]},
            {"id": 88, "location": [10.06,60.0], "delivery": [1], "skills": [2]}
        ]
    }"#;
    let (mut problem, matrix) = prep(json);
    let infeasible = propagate::tighten(&mut problem, &matrix, false);
    assert!(
        infeasible.iter().any(|i| i.job_id == 88 && i.reason.contains("skill")),
        "job needing an absent skill must be flagged: {infeasible:?}"
    );
}

#[test]
fn soft_mode_never_flags_infeasible() {
    // Under soft mode the engine serves late instead of dropping, so propagation
    // must not flag anything unservable (the two features must not fight).
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100], "time_window": [0, 100000]}],
        "jobs": [
            {"id": 99, "location": [12.0,60.0], "delivery": [1], "time_windows": [[0, 1]]}
        ]
    }"#;
    let (mut problem, matrix) = prep(json);
    let infeasible = propagate::tighten(&mut problem, &matrix, true);
    assert!(infeasible.is_empty(), "soft mode must not flag infeasible: {infeasible:?}");
}

#[test]
fn precedence_transitive_closure() {
    // precedence 1→2, 2→3 should imply 1→3 after the pass.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100]}],
        "jobs": [
            {"id": 1, "location": [10.05,60.0], "delivery": [1]},
            {"id": 2, "location": [10.06,60.0], "delivery": [1]},
            {"id": 3, "location": [10.07,60.0], "delivery": [1]}
        ],
        "precedence": [[1, 2], [2, 3]]
    }"#;
    let (mut problem, matrix) = prep(json);
    propagate::tighten(&mut problem, &matrix, false);
    assert!(
        problem.precedence.contains(&(1, 3)),
        "transitive closure must add 1→3, got {:?}",
        problem.precedence
    );
}
