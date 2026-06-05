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

    // The finished route exposes how many breaks ran and their total duration —
    // this is what the DSL `route.has_break` / `route.break_count` reads.
    assert_eq!(mwith.break_count, 1, "one break was scheduled");
    assert_eq!(mwith.break_duration, 600, "its duration is recorded");
    assert_eq!(mno.break_count, 0, "no breaks on the break-free vehicle");
    assert_eq!(mno.break_duration, 0);
}

// ---------------------------------------------------------------------------
// Vehicle allowlist (`allowed_vehicles`): a job may only ride a listed vehicle.
// Unset (None) leaves the job servable by any eligible vehicle.
// ---------------------------------------------------------------------------
#[test]
fn allowed_vehicles_pins_a_job_to_its_vehicle() {
    // Two identical vehicles. Job 1 is pinned to vehicle 2; job 2 to vehicle 1.
    // The depot is shared, so without the allowlist either vehicle could take
    // either job. With it, each job must land on its allowed vehicle.
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]},
            {"id": 2, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}
        ],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1], "allowed_vehicles": [2]},
            {"id": 2, "location": [10.20, 60.0], "delivery": [1], "allowed_vehicles": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(sol.unassigned.len(), 0, "both jobs must be placed on their allowed vehicle");
    for r in &sol.routes {
        let veh_id = problem.vehicles[r.vehicle_idx].id;
        for s in &r.steps {
            let job = s.description(&problem);
            assert!(
                job.allows_vehicle(veh_id),
                "job {} landed on vehicle {} which is not in its allowlist",
                job.id, veh_id
            );
        }
    }
}

#[test]
fn allowed_vehicles_rejected_directly_by_evaluate_route() {
    // Direct evaluator check: a job pinned away from this vehicle is rejected.
    let json = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}
        ],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1], "allowed_vehicles": [99]}
        ]
    }"#;
    let (p, m) = prep(json);
    let r = evaluate_route(&p, &m, &p.vehicles[0], &[TaskRef::Job(0)]);
    assert_eq!(r.err(), Some("job not allowed on this vehicle"));

    // Sanity: the same job with no allowlist evaluates fine.
    let json_ok = r#"{
        "vehicles": [
            {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]}
        ],
        "jobs": [
            {"id": 1, "location": [10.10, 60.0], "delivery": [1]}
        ]
    }"#;
    let (p2, m2) = prep(json_ok);
    assert!(evaluate_route(&p2, &m2, &p2.vehicles[0], &[TaskRef::Job(0)]).is_ok());
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

// ---------------------------------------------------------------------------
// Disjunctions / optional visits with an explicit drop penalty (OR-Tools
// `AddDisjunction([node], penalty)` semantics). `disjunction_penalty` is
// charged *on top of* `prize` for every unassigned job, so it shows up in the
// objective and lets local search trade dropping against routing cost.
// ---------------------------------------------------------------------------

// A finite drop penalty on an unassigned job is added to the summary cost.
#[test]
fn disjunction_penalty_charged_on_unassigned() {
    // One vehicle with capacity 1 and two equal jobs: exactly one fits, so the
    // other is forced unassigned regardless of heuristics. The far job carries a
    // big drop penalty; the near job none — and both have a small finite prize so
    // neither is "mandatory" via the sentinel.
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [1],
                      "time_window": [0, 1000000]}],
        "jobs": [
            {"id": 1, "location": [10.02, 60.0], "delivery": [1], "prize": 50.0},
            {"id": 2, "location": [10.50, 60.0], "delivery": [1], "prize": 50.0,
             "disjunction_penalty": 100000.0}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();

    // Exactly one job is unassigned (capacity 1).
    assert_eq!(sol.unassigned.len(), 1, "only one of the two jobs fits");

    // Because job 2's drop penalty (100000) dwarfs job 1's (0), the solver keeps
    // job 2 served and drops the cheap, penalty-free job 1.
    let unassigned_ids: Vec<u64> = sol
        .unassigned
        .iter()
        .map(|t| t.description(&problem).id)
        .collect();
    assert_eq!(unassigned_ids, vec![1], "the penalty-free job is the one dropped");

    // And the objective must reflect the dropped job's prize but NOT job 2's
    // penalty (since job 2 stayed served): the only unassigned charge is job 1's
    // prize of 50.
    let routed: f64 = sol.routes.iter().map(|r| r.metrics.cost).sum();
    assert!(
        (sol.summary.cost - (routed + 50.0)).abs() < 1e-6,
        "summary should be routing cost + dropped job's prize (50), got {} (routed {})",
        sol.summary.cost,
        routed
    );
}

// Under capacity contention the drop penalty decides *which* job is sacrificed:
// flipping which job carries the big penalty flips which one is kept. This is
// the core "trade a drop penalty against routing cost / value" behaviour and
// exercises the prize-swap pass that now accounts for the penalty.
//
// (Note: the engine has no *voluntary* single-job drop operator — a feasible
// job is never dropped just because its routing cost exceeds its value, the
// same as for `prize` today. The penalty steers contention, not free capacity.)
#[test]
fn disjunction_penalty_drives_which_job_is_dropped() {
    // Capacity 1 ⇒ exactly one of two equidistant jobs is served. Both carry the
    // same small finite prize, so only the disjunction penalty breaks the tie.
    let make = |pen1: f64, pen2: f64| {
        format!(
            r#"{{
                "vehicles": [{{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0],
                              "capacity": [1], "time_window": [0, 100000000]}}],
                "jobs": [
                    {{"id": 1, "location": [10.05, 60.0], "delivery": [1], "prize": 10.0,
                     "disjunction_penalty": {pen1}}},
                    {{"id": 2, "location": [10.05, 60.10], "delivery": [1], "prize": 10.0,
                     "disjunction_penalty": {pen2}}}
                ]
            }}"#
        )
    };

    // Job 1 has the big penalty ⇒ keep job 1, drop job 2.
    let mut p_a = parse_input(&make(100000.0, 0.0)).unwrap();
    let sol_a = solve(&mut p_a, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    let dropped_a: Vec<u64> = sol_a.unassigned.iter().map(|t| t.description(&p_a).id).collect();
    assert_eq!(dropped_a, vec![2], "big penalty on job 1 ⇒ job 2 is the one dropped");

    // Flip the penalty onto job 2 ⇒ the choice flips: keep job 2, drop job 1.
    let mut p_b = parse_input(&make(0.0, 100000.0)).unwrap();
    let sol_b = solve(&mut p_b, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    let dropped_b: Vec<u64> = sol_b.unassigned.iter().map(|t| t.description(&p_b).id).collect();
    assert_eq!(dropped_b, vec![1], "flipping the penalty flips which job is dropped");
}

// Backward-compat guard: with the field omitted, behaviour is identical to a
// pre-disjunction input (sentinel prize, no extra charge).
#[test]
fn disjunction_penalty_absent_is_backward_compatible() {
    let json = r#"{
        "vehicles": [{"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
                      "time_window": [0, 1000000]}],
        "jobs": [
            {"id": 1, "location": [10.02, 60.0], "delivery": [1]},
            {"id": 2, "location": [10.04, 60.0], "delivery": [1]}
        ]
    }"#;
    let mut problem = parse_input(json).unwrap();
    // The field must deserialize to None and parsing must succeed unchanged.
    assert!(problem.jobs.iter().all(|j| j.disjunction_penalty.is_none()));
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();
    assert_eq!(sol.unassigned.len(), 0, "both reachable jobs served as before");
}
