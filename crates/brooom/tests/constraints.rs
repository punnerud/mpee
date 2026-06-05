//! Constraint conformance suite.
//!
//! One focused test per constraint the engine claims to support, so the
//! README's constraint table is backed by executable proof rather than prose.
//! `tests/integration.rs` already covers CVRP, capacity splitting, time
//! windows, skills and PDPTW shipments; this file adds the rest (backhaul,
//! driver breaks, multi-depot, max travel/distance/tasks, mixed fleet via
//! speed_factor, multi-dimensional capacity, priority) and finishes with a
//! round-trip `verify_solution` over a multi-constraint instance.

use brooom::io::{parse_input, to_output, verify_solution};
use brooom::matrix::HaversineMatrix;
use brooom::solution::{eval_cache_invalidate, evaluate_route, StepKind, TaskRef};
use brooom::solver::{build_matrix, solve, SolverConfig};

/// Build the haversine matrix for a parsed problem, assigning every location a
/// matrix index as a side effect (so `evaluate_route` can be called directly).
fn prep(json: &str) -> (brooom::Problem, brooom::Matrix) {
    eval_cache_invalidate();
    let mut problem = parse_input(json).unwrap();
    let matrix = build_matrix(&mut problem, Some(&HaversineMatrix::default())).unwrap();
    (problem, matrix)
}

// ---------------------------------------------------------------------------
// Release times: service may not begin before a job's release; the vehicle
// waits. release=0 (default) is a no-op.
// ---------------------------------------------------------------------------
#[test]
fn release_time_delays_service() {
    let with_release = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
                      "time_window": [0, 100000]}],
        "jobs": [{"id": 1, "location": [10.05, 60.0], "delivery": [1], "release": 5000}]
    }"#;
    let no_release = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
                      "time_window": [0, 100000]}],
        "jobs": [{"id": 1, "location": [10.05, 60.0], "delivery": [1]}]
    }"#;
    let (pr, mr) = prep(with_release);
    let m_rel = evaluate_route(&pr, &mr, &pr.vehicles[0], &[TaskRef::Job(0)]).unwrap();
    let (pn, mn) = prep(no_release);
    let m_no = evaluate_route(&pn, &mn, &pn.vehicles[0], &[TaskRef::Job(0)]).unwrap();

    // The job's arrival without release is well under 5000s, so the release
    // forces a wait until 5000 before the (instant) service + return leg.
    assert!(m_rel.waiting_time >= 4000, "release should add a long wait: {}", m_rel.waiting_time);
    assert!(m_rel.end_time > m_no.end_time, "release pushes the route end out");
    assert_eq!(m_no.waiting_time, 0, "no release ⇒ no wait (regression guard)");
}

// ---------------------------------------------------------------------------
// Backhaul: every linehaul (delivery) stop must precede any backhaul (pickup-
// only) stop on the same route.
// ---------------------------------------------------------------------------
#[test]
fn backhaul_linehaul_must_precede_backhaul() {
    // job 0 = delivery (linehaul), job 1 = pickup-only (backhaul).
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}
        ],
        "jobs": [
            {"id": 10, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 11, "location": [10.20, 60.0], "pickup":   [1]}
        ]
    }"#;
    let (problem, matrix) = prep(json);
    let veh = &problem.vehicles[0];

    // Linehaul-then-backhaul is allowed.
    let ok = evaluate_route(&problem, &matrix, veh, &[TaskRef::Job(0), TaskRef::Job(1)]);
    assert!(ok.is_ok(), "delivery before backhaul must be feasible: {ok:?}");

    // Backhaul-then-linehaul is rejected.
    let bad = evaluate_route(&problem, &matrix, veh, &[TaskRef::Job(1), TaskRef::Job(0)]);
    assert_eq!(bad.err(), Some("linehaul after backhaul"));
}

#[test]
fn backhaul_solver_orders_linehaul_first() {
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}
        ],
        "jobs": [
            {"id": 10, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 11, "location": [10.20, 60.0], "delivery": [1]},
            {"id": 12, "location": [10.30, 60.0], "pickup":   [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(sol.unassigned.len(), 0, "all jobs should fit");
    // In every route, no delivery job may appear after a backhaul (pickup-only) job.
    for r in &sol.routes {
        let mut seen_backhaul = false;
        for s in &r.steps {
            let j = s.description(&problem);
            let is_backhaul = !j.pickup.is_empty() && j.delivery.is_empty();
            if is_backhaul {
                seen_backhaul = true;
            } else if !j.delivery.is_empty() {
                assert!(!seen_backhaul, "delivery job {} served after a backhaul", j.id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Driver breaks: a mandatory rest that must be taken within its window and
// pushes the route timeline.
// ---------------------------------------------------------------------------
#[test]
fn break_pushes_timeline_by_its_duration() {
    let with_break = r#"{
        "vehicles": [{
            "id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
            "time_window": [0, 100000],
            "breaks": [{"id": 99, "service": 600, "time_windows": [[0, 100000]]}]
        }],
        "jobs": [
            {"id": 1, "location": [10.05, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.10, 60.0], "delivery": [1]}
        ]
    }"#;
    let without_break = r#"{
        "vehicles": [{
            "id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
            "time_window": [0, 100000]
        }],
        "jobs": [
            {"id": 1, "location": [10.05, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.10, 60.0], "delivery": [1]}
        ]
    }"#;

    let (pb, mb) = prep(with_break);
    let mwith = evaluate_route(&pb, &mb, &pb.vehicles[0], &[TaskRef::Job(0), TaskRef::Job(1)]).unwrap();

    let (pn, mn) = prep(without_break);
    let mno = evaluate_route(&pn, &mn, &pn.vehicles[0], &[TaskRef::Job(0), TaskRef::Job(1)]).unwrap();

    assert_eq!(
        mwith.end_time,
        mno.end_time + 600,
        "a 600s break must push the route end by exactly 600s"
    );
    // Travel time is unaffected — a break is rest, not driving.
    assert_eq!(mwith.travel_time, mno.travel_time);
}

#[test]
fn break_is_emitted_in_vroom_output() {
    let json = r#"{
        "vehicles": [{
            "id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
            "time_window": [0, 100000],
            "breaks": [{"id": 77, "service": 600, "time_windows": [[0, 100000]]}]
        }],
        "jobs": [
            {"id": 1, "location": [10.05, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.10, 60.0], "delivery": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    let matrix = build_matrix(&mut problem, Some(&HaversineMatrix::default())).unwrap();
    let out = to_output(&problem, &sol, Some(&matrix));

    let breaks: Vec<_> = out.routes.iter()
        .flat_map(|r| r.steps.iter())
        .filter(|s| s.kind == StepKind::Break)
        .collect();
    assert_eq!(breaks.len(), 1, "exactly one break step should be emitted");
    assert_eq!(breaks[0].job_id, Some(77), "break step carries its break id");
    assert_eq!(breaks[0].service, 600, "break step carries its duration");
}

#[test]
fn break_with_unreachable_window_is_infeasible() {
    // A break window of [0,0] can never be honoured once the vehicle has driven
    // to its first stop (arrival time > 0), so the route is infeasible.
    let json = r#"{
        "vehicles": [{
            "id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
            "time_window": [0, 100000],
            "breaks": [{"id": 99, "service": 600, "time_windows": [[0, 0]]}]
        }],
        "jobs": [
            {"id": 1, "location": [10.05, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.10, 60.0], "delivery": [1]}
        ]
    }"#;
    let (p, m) = prep(json);
    let res = evaluate_route(&p, &m, &p.vehicles[0], &[TaskRef::Job(0), TaskRef::Job(1)]);
    assert_eq!(res.err(), Some("break time window missed"));
}

// ---------------------------------------------------------------------------
// Multi-depot: per-vehicle distinct start/end locations (this *is* multi-depot
// in the Vroom sense).
// ---------------------------------------------------------------------------
#[test]
fn multi_depot_distinct_start_locations() {
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]},
            {"id": 2, "start": [11.0, 60.0], "end": [11.0, 60.0], "capacity": [10]}
        ],
        "jobs": [
            {"id": 1, "location": [10.02, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.04, 60.0], "delivery": [1]},
            {"id": 3, "location": [11.02, 60.0], "delivery": [1]},
            {"id": 4, "location": [11.04, 60.0], "delivery": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(sol.unassigned.len(), 0, "all jobs reachable from some depot");
    // Each route departs from its own vehicle's distinct start index.
    let start0 = problem.vehicles[0].start.as_ref().and_then(|l| l.index);
    let start1 = problem.vehicles[1].start.as_ref().and_then(|l| l.index);
    assert_ne!(start0, start1, "the two depots must be distinct matrix nodes");
}

// ---------------------------------------------------------------------------
// Route caps: max_travel_time / max_distance / max_tasks.
// ---------------------------------------------------------------------------
#[test]
fn max_tasks_rejects_overlong_route() {
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10], "max_tasks": 2}
        ],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.20, 60.0], "delivery": [1]},
            {"id": 3, "location": [10.30, 60.0], "delivery": [1]}
        ]
    }"#;
    let (problem, matrix) = prep(json);
    let veh = &problem.vehicles[0];
    let three = evaluate_route(&problem, &matrix, veh, &[TaskRef::Job(0), TaskRef::Job(1), TaskRef::Job(2)]);
    assert_eq!(three.err(), Some("max_tasks exceeded"));
    let two = evaluate_route(&problem, &matrix, veh, &[TaskRef::Job(0), TaskRef::Job(1)]);
    assert!(two.is_ok(), "two tasks must be within the cap: {two:?}");
}

#[test]
fn max_distance_rejects_overlong_route() {
    // A 1-metre cap is impossible for any real leg.
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10], "max_distance": 1}
        ],
        "jobs": [
            {"id": 1, "location": [10.30, 60.0], "delivery": [1]}
        ]
    }"#;
    let (problem, matrix) = prep(json);
    let res = evaluate_route(&problem, &matrix, &problem.vehicles[0], &[TaskRef::Job(0)]);
    assert_eq!(res.err(), Some("max_distance exceeded"));
}

#[test]
fn max_travel_time_rejects_overlong_route() {
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10], "max_travel_time": 1}
        ],
        "jobs": [
            {"id": 1, "location": [10.30, 60.0], "delivery": [1]}
        ]
    }"#;
    let (problem, matrix) = prep(json);
    let res = evaluate_route(&problem, &matrix, &problem.vehicles[0], &[TaskRef::Job(0)]);
    assert_eq!(res.err(), Some("max_travel_time exceeded"));
}

// ---------------------------------------------------------------------------
// Mixed fleet: speed_factor scales a vehicle's travel time.
// ---------------------------------------------------------------------------
#[test]
fn speed_factor_scales_travel_time() {
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10], "speed_factor": 1.0},
            {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10], "speed_factor": 2.0}
        ],
        "jobs": [
            {"id": 1, "location": [10.20, 60.0], "delivery": [1]}
        ]
    }"#;
    let (problem, matrix) = prep(json);
    let fast = evaluate_route(&problem, &matrix, &problem.vehicles[0], &[TaskRef::Job(0)]).unwrap();
    let slow = evaluate_route(&problem, &matrix, &problem.vehicles[1], &[TaskRef::Job(0)]).unwrap();
    assert!(
        slow.travel_time > fast.travel_time,
        "speed_factor 2.0 should take longer than 1.0 ({} vs {})",
        slow.travel_time, fast.travel_time
    );
}

// ---------------------------------------------------------------------------
// Multi-dimensional capacity: the second dimension (e.g. volume) can be the
// binding constraint independently of the first (e.g. weight).
// ---------------------------------------------------------------------------
#[test]
fn multi_dimensional_capacity_second_dim_binds() {
    // Each vehicle has plenty of dim-0 (weight) but only room for one unit of
    // dim-1 (volume), so two unit-volume jobs cannot share a vehicle.
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100, 1]},
            {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100, 1]}
        ],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1, 1]},
            {"id": 2, "location": [10.20, 60.0], "delivery": [1, 1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(sol.unassigned.len(), 0);
    assert_eq!(sol.routes.len(), 2, "dim-1 volume cap forces a 2-vehicle split");
}

// ---------------------------------------------------------------------------
// Priority: a soft insertion-order bias (higher priority is inserted first).
// With two otherwise-identical jobs competing for one seat, the higher-priority
// one claims it. (Priority is a hint, not a hard prize-collecting objective —
// the unassigned penalty itself is uniform; see README.)
// ---------------------------------------------------------------------------
#[test]
fn priority_breaks_ties_for_the_last_seat() {
    // Same location for both jobs ⇒ identical routing cost either way, so the
    // priority insertion order is the deciding factor for the single seat.
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [1]}
        ],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1], "priority": 0},
            {"id": 2, "location": [10.10, 60.0], "delivery": [1], "priority": 100}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(sol.unassigned.len(), 1, "only one job fits");
    let served: Vec<u64> = sol.routes.iter()
        .flat_map(|r| r.steps.iter().map(|s| s.description(&problem).id))
        .collect();
    assert_eq!(served, vec![2], "the priority-100 job wins the tie-break for the seat");
}

// ---------------------------------------------------------------------------
// Round-trip: a multi-constraint instance solves and its stored metrics match
// a fresh evaluation (so backhaul + breaks + caps are all internally consistent).
// ---------------------------------------------------------------------------
#[test]
fn multi_constraint_solution_round_trips() {
    let json = r#"{
        "vehicles": [{
            "id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [100],
            "time_window": [0, 200000],
            "breaks": [{"id": 1, "service": 300, "time_windows": [[0, 200000]]}]
        }],
        "jobs": [
            {"id": 1, "location": [10.05, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.10, 60.0], "delivery": [1]},
            {"id": 3, "location": [10.15, 60.0], "pickup":   [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();

    let matrix = build_matrix(&mut problem, Some(&HaversineMatrix::default())).unwrap();
    verify_solution(&problem, &matrix, &sol).unwrap();
}
