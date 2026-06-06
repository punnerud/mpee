//! P5: custom accumulator dimensions & stateful (per-arc) callbacks.
//!
//! Registers a user-defined `fuel` dimension that drains per arc, and proves:
//!   1. the solver accumulates it along a route into per-position cumuls,
//!   2. a registered min/max bound is honoured at full route evaluation (the
//!      route is rejected — not pruned in the O(1) probe, by design), and
//!   3. a pyspell DSL constraint reads `route.fuel` / `route.fuel[k]` and shapes
//!      a real solve, dropping the jobs that would run the tank dry.
//!
//! Dimensions live in process-global state, so this file (its own test binary)
//! takes a shared lock and installs everything behind RAII guards.

use std::sync::{Arc, Mutex};

use brooom::dimension::{ArcCtx, CustomDimension, DimensionGuard};
use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::solution::{eval_cache_invalidate, evaluate_route, TaskRef};
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

const THREE_JOBS: &str = r#"{
    "vehicles": [
        {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10],
         "time_window": [0, 1000000]}
    ],
    "jobs": [
        {"id": 10, "location": [10.05, 60.0], "delivery": [1]},
        {"id": 20, "location": [10.10, 60.0], "delivery": [1]},
        {"id": 30, "location": [10.15, 60.0], "delivery": [1]}
    ]
}"#;

/// A fuel dimension that starts full (100) and burns a fixed 40 per traversed
/// arc, with a hard floor of 0. The transit ignores the actual nodes — it is a
/// deterministic function of the arc count, which is enough to prove
/// accumulation and bound enforcement.
fn fuel_dim() -> CustomDimension {
    CustomDimension::new("fuel", Arc::new(|_: &ArcCtx| -40))
        .with_start(100)
        .with_min(0)
}

#[test]
fn dimension_accumulates_into_per_position_cumuls() {
    let _lock = guard();
    let (problem, matrix) = prep(THREE_JOBS);
    let veh = &problem.vehicles[0];

    let _g = DimensionGuard::install(vec![fuel_dim()]);
    assert!(brooom::dimension::has_dimensions());

    // Recompute cumuls for a one-stop route: start -> job10 -> end = 2 arcs.
    // Tank 100, burn 40/arc → [100, 60, 20].
    let cumuls = brooom::solution::dimension_cumuls_for_route(
        &problem,
        &matrix,
        veh,
        &[TaskRef::Job(0)],
    );
    assert_eq!(cumuls.cumuls_of(0), &[100, 60, 20]);
    assert_eq!(cumuls.at(0, 1), 60);
    // Whole-route aggregate (peak) is the full starting tank.
    assert_eq!(cumuls.aggregate_max(0), 100);
    assert!(!cumuls.bound_violated, "a single stop stays above the floor");
}

#[test]
fn dimension_bound_rejects_at_full_eval() {
    let _lock = guard();
    let (problem, matrix) = prep(THREE_JOBS);
    let veh = &problem.vehicles[0];

    let _g = DimensionGuard::install(vec![fuel_dim()]);

    // One stop = 2 arcs = 80 burned, ends at 20 ≥ 0 → feasible.
    assert!(
        evaluate_route(&problem, &matrix, veh, &[TaskRef::Job(0)]).is_ok(),
        "a one-stop route keeps fuel above zero"
    );
    // Two stops = 3 arcs = 120 burned, would hit -20 < 0 → the dimension's
    // min bound is exceeded and the route is rejected at FULL evaluation
    // (honest: this is not pruned in the O(1) insertion probe).
    assert_eq!(
        evaluate_route(&problem, &matrix, veh, &[TaskRef::Job(0), TaskRef::Job(1)]).err(),
        Some("custom dimension bound exceeded"),
        "a two-stop route runs the tank dry and is rejected"
    );
}

#[test]
fn cleared_dimension_restores_plain_behaviour() {
    let _lock = guard();
    {
        let _g = DimensionGuard::install(vec![fuel_dim()]);
        assert!(brooom::dimension::has_dimensions());
    }
    assert!(!brooom::dimension::has_dimensions(), "guard clears on drop");

    // With no dimension registered, a two-stop route is fine again (no fuel).
    let (problem, matrix) = prep(THREE_JOBS);
    assert!(
        evaluate_route(
            &problem,
            &matrix,
            &problem.vehicles[0],
            &[TaskRef::Job(0), TaskRef::Job(1)]
        )
        .is_ok(),
        "cleared dimension means the route evaluates normally"
    );
}

/// A MONOTONE "load" resource: it accrues a fixed +30 per traversed arc and is
/// capped at 65. Declared `monotone` so its max bound is mirrored into the O(1)
/// insertion probe (the spike). Two arcs (start->job->end) peak at 60 ≤ 65 →
/// feasible; three arcs peak at 90 > 65 → infeasible.
fn load_dim() -> CustomDimension {
    CustomDimension::new("load", Arc::new(|_: &ArcCtx| 30))
        .with_start(0)
        .with_max(65)
        .monotone()
}

/// The spike: a monotone resource's max bound prunes a tempting insertion in the
/// fast `precompute` probe BEFORE the full `evaluate_route`, and the two paths
/// agree exactly (the probe never prunes something the evaluator would accept,
/// and rejects exactly what the evaluator rejects).
#[test]
fn monotone_resource_prunes_in_the_probe_and_matches_full_eval() {
    let _lock = guard();
    let (problem, matrix) = prep(THREE_JOBS);
    let veh = &problem.vehicles[0];

    let _g = DimensionGuard::install(vec![load_dim()]);
    assert!(
        brooom::dimension::has_probe_dimensions(),
        "a monotone+max dimension is probe-mirrorable"
    );

    // One stop = 2 arcs → load cumuls [0,30,60], peak 60 ≤ 65: FEASIBLE on both
    // paths. The probe returns Some(_) (no prune) and the evaluator returns Ok.
    let one = [TaskRef::Job(0)];
    assert!(
        brooom::eval::precompute(&problem, &matrix, veh, 0, &one).is_some(),
        "the probe accepts a one-stop route (peak 60 ≤ 65)"
    );
    assert!(
        evaluate_route(&problem, &matrix, veh, &one).is_ok(),
        "full eval agrees the one-stop route is feasible"
    );

    // Two stops = 3 arcs → load cumuls [0,30,60,90], peak 90 > 65: the tempting
    // second insertion breaches the resource max. The PROBE must reject it early
    // (returns None) — this is the proactive prune the spike adds — AND the full
    // evaluator rejects it with the dimension-bound error, proving correctness is
    // identical to the P5 full-eval path.
    let two = [TaskRef::Job(0), TaskRef::Job(1)];
    assert!(
        brooom::eval::precompute(&problem, &matrix, veh, 0, &two).is_none(),
        "the probe PRUNES the breaching insertion before full eval (the spike)"
    );
    assert_eq!(
        evaluate_route(&problem, &matrix, veh, &two).err(),
        Some("custom dimension bound exceeded"),
        "full eval rejects the same route with the dimension-bound error"
    );
}

/// Honest-caveat guard: a NON-monotone dimension with a max bound is NOT
/// probe-mirrorable, so `precompute` does not prune it (it stays on the full-eval
/// fallback, exactly as before the spike). We assert the probe does NOT prune a
/// route that the full evaluator nonetheless rejects — i.e. the caveat is real.
#[test]
fn non_monotone_max_still_falls_back_to_full_eval() {
    let _lock = guard();
    let (problem, matrix) = prep(THREE_JOBS);
    let veh = &problem.vehicles[0];

    // Same +30/arc accrual and max 65, but NOT declared monotone. Because the
    // flag is absent, the probe mirror is inert for this dimension.
    let non_mono = CustomDimension::new("load", Arc::new(|_: &ArcCtx| 30))
        .with_start(0)
        .with_max(65);
    let _g = DimensionGuard::install(vec![non_mono]);
    assert!(
        !brooom::dimension::has_probe_dimensions(),
        "a non-monotone dimension is not probe-mirrorable"
    );

    let two = [TaskRef::Job(0), TaskRef::Job(1)];
    // The probe does NOT prune (it never sees this dimension): it returns Some.
    assert!(
        brooom::eval::precompute(&problem, &matrix, veh, 0, &two).is_some(),
        "non-monotone dimension is invisible to the probe (no early prune)"
    );
    // But the full evaluator STILL rejects it — correctness preserved, just no
    // proactive prune. This is the documented residual caveat.
    assert_eq!(
        evaluate_route(&problem, &matrix, veh, &two).err(),
        Some("custom dimension bound exceeded"),
        "full eval is still the authority for non-monotone dimensions"
    );
}

// ---------------------------------------------------------------------------
// Phase 1 extensions: draining/min probe, soft bounds, coupling, shared resource.
// ---------------------------------------------------------------------------

/// A DRAINING fuel dimension (the dual of `load_dim`): starts at 100, burns a
/// fixed 40 per arc, floor 0, declared `draining()` so its MIN bound is mirrored
/// into the O(1) probe. Two arcs trough at 20 ≥ 0 → feasible; three arcs trough at
/// −20 < 0 → infeasible and pruned in the probe.
fn draining_fuel_dim() -> CustomDimension {
    CustomDimension::new("fuel", Arc::new(|_: &ArcCtx| -40))
        .with_start(100)
        .with_min(0)
        .draining()
}

/// The draining dual of the monotone-max spike: a draining resource's MIN bound
/// (its floor) prunes a breaching insertion in the fast `precompute` probe BEFORE
/// the full `evaluate_route`, and the two paths agree exactly.
#[test]
fn draining_resource_prunes_on_min_in_the_probe_and_matches_full_eval() {
    let _lock = guard();
    let (problem, matrix) = prep(THREE_JOBS);
    let veh = &problem.vehicles[0];

    let _g = DimensionGuard::install(vec![draining_fuel_dim()]);
    assert!(
        brooom::dimension::has_probe_dimensions(),
        "a draining+min dimension is probe-mirrorable (the dual direction)"
    );

    // One stop = 2 arcs → fuel cumuls [100,60,20], trough 20 ≥ 0: FEASIBLE on both
    // paths. The probe returns Some(_) (no prune) and the evaluator returns Ok.
    let one = [TaskRef::Job(0)];
    assert!(
        brooom::eval::precompute(&problem, &matrix, veh, 0, &one).is_some(),
        "the probe accepts a one-stop route (trough 20 ≥ 0)"
    );
    assert!(
        evaluate_route(&problem, &matrix, veh, &one).is_ok(),
        "full eval agrees the one-stop route is feasible"
    );

    // Two stops = 3 arcs → fuel cumuls [100,60,20,-20], trough −20 < 0: the second
    // insertion drains the tank below the floor. The PROBE prunes it early
    // (returns None) via `probe_breaches_monotone_min`, AND full eval rejects it
    // with the dimension-bound error — correctness identical to the P5 path.
    let two = [TaskRef::Job(0), TaskRef::Job(1)];
    assert!(
        brooom::eval::precompute(&problem, &matrix, veh, 0, &two).is_none(),
        "the probe PRUNES the draining breach before full eval (the dual spike)"
    );
    assert_eq!(
        evaluate_route(&problem, &matrix, veh, &two).err(),
        Some("custom dimension bound exceeded"),
        "full eval rejects the same route with the dimension-bound error"
    );
}

/// A SOFT cumul bound adds a penalty (it does NOT reject): a load resource with a
/// hard max far above the route's peak but a soft max it does exceed is still
/// feasible, and the per-route cost carries the soft penalty in `cost_custom`.
#[test]
fn soft_cumul_bound_penalises_instead_of_rejecting() {
    let _lock = guard();
    let (problem, matrix) = prep(THREE_JOBS);
    let veh = &problem.vehicles[0];

    // +30/arc, hard max 1000 (never breached), soft max 40 with weight 2.
    // One stop = 2 arcs → load cumuls [0,30,60]: only position 2 (60) breaches the
    // soft band, by 20 → penalty 2 × 20 = 40.
    let dim = CustomDimension::new("load", Arc::new(|_: &ArcCtx| 30))
        .with_start(0)
        .with_max(1000)
        .with_soft_max(40, 2.0)
        .monotone();
    let _g = DimensionGuard::install(vec![dim]);

    let m = evaluate_route(&problem, &matrix, veh, &[TaskRef::Job(0)])
        .expect("soft-bound breach is feasible (hard max not exceeded)");
    assert!(m.cost_custom >= 40.0, "the soft penalty is folded into cost_custom: {}", m.cost_custom);

    // Sanity: the SAME route with no soft band has no such penalty.
    drop(_g);
    let plain = CustomDimension::new("load", Arc::new(|_: &ArcCtx| 30))
        .with_start(0)
        .with_max(1000)
        .monotone();
    let _g2 = DimensionGuard::install(vec![plain]);
    let m0 = evaluate_route(&problem, &matrix, veh, &[TaskRef::Job(0)]).unwrap();
    assert_eq!(m0.cost_custom, 0.0, "no soft band → no custom penalty");
}

/// A CROSS-DIMENSION coupling: a fuel dimension whose per-arc burn is a function
/// of the arc's physical DISTANCE (`ArcCtx::distance`), proving the callback can
/// express `fuel = distance × factor` without re-querying the matrix. We just
/// assert the cumuls fall with distance and the longer route burns strictly more.
#[test]
fn coupling_dimension_reads_arc_distance() {
    let _lock = guard();
    let (problem, matrix) = prep(THREE_JOBS);
    let veh = &problem.vehicles[0];

    // Burn proportional to distance. The Haversine matrix gives positive arc
    // distances, so any served stop drains the tank below the 100 start.
    let dim = CustomDimension::new(
        "fuel",
        Arc::new(|c: &ArcCtx| -(c.distance / 100)),
    )
    .with_start(100_000);
    let _g = DimensionGuard::install(vec![dim]);

    let cumuls = brooom::solution::dimension_cumuls_for_route(
        &problem,
        &matrix,
        veh,
        &[TaskRef::Job(0)],
    );
    let row = cumuls.cumuls_of(0);
    assert_eq!(row.len(), 3, "start -> job -> end = 3 positions");
    assert_eq!(row[0], 100_000, "starts full");
    assert!(row[1] < row[0], "the outbound arc burns a distance-proportional amount");
    assert!(row[2] < row[1], "the return arc burns more still");

    // A two-stop route covers more distance, so its end-of-route fuel is strictly
    // lower than the one-stop route's — coupling genuinely tracks distance.
    let two = brooom::solution::dimension_cumuls_for_route(
        &problem,
        &matrix,
        veh,
        &[TaskRef::Job(0), TaskRef::Job(1)],
    );
    let one_end = *row.last().unwrap();
    let two_end = *two.cumuls_of(0).last().unwrap();
    assert!(two_end < one_end, "the longer route burns strictly more fuel");
}

/// The SHARED cross-route resource: a loading-dock global penalises a solution
/// whose routes all load at the depot at the same instant (overbooking the dock),
/// while a staggered solution pays nothing.
#[test]
fn shared_resource_global_penalises_overbooked_dock() {
    let _lock = guard();
    use brooom::global_constraint::{shared_resource_dock, SolutionView};
    use brooom::solution::{Route, RouteMetrics};

    let (problem, matrix) = prep(THREE_JOBS);

    // A dimension whose DEPOT cumul (position 0) is the amount loaded at the dock:
    // start 30 means each route holds 30 units at the depot. (The per-arc transit
    // is irrelevant to the dock reading, which uses cumul[dim][0].)
    let _g = DimensionGuard::install(vec![
        CustomDimension::new("dock", Arc::new(|_: &ArcCtx| 0)).with_start(30),
    ]);

    let mk_route = |start: i64| Route {
        vehicle_idx: 0,
        steps: vec![TaskRef::Job(0)],
        metrics: RouteMetrics { start_time: start, end_time: start + 1, ..Default::default() },
    };

    // Two routes both leaving at t=0 → both occupy a 100s dock window together,
    // peak concurrent load = 60. Capacity 50, weight 1 → penalty 1 × (60−50) = 10.
    let overbooked = vec![mk_route(0), mk_route(0)];
    let view = SolutionView { problem: &problem, routes: &overbooked, unassigned: &[] };
    let pen = shared_resource_dock(&view, &matrix, 0, 100, 50, 1.0);
    assert_eq!(pen, 10.0, "two overlapping 30-unit loads peak at 60, 10 over capacity 50");

    // Stagger the second route past the 100s dock window → never concurrent, peak
    // 30 ≤ 50 → no penalty.
    let staggered = vec![mk_route(0), mk_route(200)];
    let view2 = SolutionView { problem: &problem, routes: &staggered, unassigned: &[] };
    let pen2 = shared_resource_dock(&view2, &matrix, 0, 100, 50, 1.0);
    assert_eq!(pen2, 0.0, "staggered loads never overlap, peak 30 ≤ 50");
}

#[cfg(feature = "pyspell")]
#[test]
fn dsl_reads_route_dimension_and_drops_thirsty_jobs() {
    let _lock = guard();
    // Register the fuel dimension FIRST so the DSL can resolve `route.fuel`.
    let _dg = DimensionGuard::install(vec![fuel_dim()]);

    // A hard DSL constraint reading the dimension: the LAST cumul (end depot)
    // must stay at or above 0 — i.e. the tank may not run dry. Equivalent to the
    // built-in min bound, but authored as text and proving `route.fuel[k]` reads.
    // `route.fuel` is the aggregate (peak); `route.fuel[len-1]` is the tank left
    // at the end. We use the explicit end index via len().
    let _g = brooom::pyspell::install_rust(&[
        "route.fuel[len(route.fuel) - 1] >= 0",
    ])
    .unwrap();

    let mut problem = parse_input(THREE_JOBS).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), SolverConfig::default()).unwrap();

    // Each non-empty route may serve at most one job (2 arcs = 80 fuel); a
    // second stop (3 arcs = 120) would breach the floor. With a single vehicle,
    // only one job can be served and the other two are dropped.
    for r in &sol.routes {
        assert!(
            r.steps.len() <= 1,
            "no route may serve more than one job under the fuel constraint"
        );
    }
    assert!(
        sol.routes.iter().map(|r| r.steps.len()).sum::<usize>() + sol.unassigned.len() == 3,
        "all three jobs are accounted for (served or unassigned)"
    );
    assert!(sol.unassigned.len() >= 2, "two jobs cannot fit on fuel and are dropped");
}

#[cfg(feature = "pyspell")]
#[test]
fn dsl_aggregate_and_indexed_forms_compile() {
    let _lock = guard();
    let _dg = DimensionGuard::install(vec![fuel_dim()]);
    // Both the aggregate scalar and the indexed cumul form must compile.
    assert!(brooom::pyspell::install_rust(&["route.fuel >= 0"]).is_ok());
    assert!(brooom::pyspell::install_rust(&["route.fuel[0] == 100"]).is_ok());
    // An unregistered dimension name is a compile error, not a panic.
    drop(_dg);
    assert!(brooom::pyspell::install_rust(&["route.fuel >= 0"]).is_err());
}
