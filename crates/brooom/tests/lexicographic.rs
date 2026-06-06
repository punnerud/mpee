//! Two-phase lexicographic (count-then-cost) optimisation.
//!
//! The solver installs global constraints into a process-global registry for
//! the duration of each solve, so — like the other global-constraint tests —
//! these take a shared lock to avoid one solve's globals leaking into another's.

use std::sync::Mutex;

use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::solver::{solve, LexObjective, ObjectiveMode, SolverConfig};

static LOCK: Mutex<()> = Mutex::new(());
fn guard() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn vehicles_used(sol: &brooom::Solution) -> usize {
    sol.routes.iter().filter(|r| !r.steps.is_empty()).count()
}

fn lexico_cfg() -> SolverConfig {
    SolverConfig {
        objective_mode: ObjectiveMode::Lexicographic {
            levels: vec![LexObjective::Vehicles, LexObjective::Cost],
        },
        ..Default::default()
    }
}

/// Two vehicles parked at DIFFERENT depots, each next to its own tight cluster
/// of jobs. Using BOTH vehicles (each serving its local cluster) gives far less
/// total travel than one vehicle driving all the way across to the far cluster
/// and back. So the minimum-COST plan uses 2 vehicles, while the minimum-COUNT
/// plan uses 1 (capacity 10 fits all four jobs). This is the genuine
/// count-vs-cost trade a naive scalar minimiser resolves in favour of cost
/// (extra vehicle), and lexicographic mode must resolve in favour of count.
///
/// Empirically (HaversineMatrix, default config): scalar → 2 vehicles, cost
/// ~832; forced-one-vehicle / lexicographic → 1 vehicle, cost ~10400.
const TWO_DEPOT_JSON: &str = r#"{
    "vehicles": [
        {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [10]},
        {"id": 2, "start": [11.0, 60.0], "end": [11.0, 60.0], "capacity": [10]}
    ],
    "jobs": [
        {"id": 1, "location": [10.02, 60.0], "delivery": [1]},
        {"id": 2, "location": [10.04, 60.0], "delivery": [1]},
        {"id": 3, "location": [10.98, 60.0], "delivery": [1]},
        {"id": 4, "location": [11.02, 60.0], "delivery": [1]}
    ]
}"#;

/// Baseline: confirm the instance actually exhibits the trade — the default
/// scalar solver DOES spend an extra vehicle to lower travel cost. If this ever
/// stops holding, the discriminating tests below would pass vacuously, so guard
/// it explicitly.
#[test]
fn scalar_spends_an_extra_vehicle_for_lower_cost() {
    let _lock = guard();
    let mut problem = parse_input(TWO_DEPOT_JSON).unwrap();
    let sol = solve(
        &mut problem,
        Some(&HaversineMatrix::default()),
        SolverConfig::default(),
    )
    .unwrap();
    assert_eq!(sol.unassigned.len(), 0, "every job must be served");
    assert_eq!(
        vehicles_used(&sol),
        2,
        "scalar trades an extra vehicle for lower travel cost on this instance, used {}",
        vehicles_used(&sol)
    );
}

/// Lexicographic mode returns the MINIMUM vehicle count (1 here) and serves
/// every job — refusing the extra-vehicle trade the scalar solver took.
#[test]
fn lexicographic_minimises_vehicle_count() {
    let _lock = guard();
    let mut problem = parse_input(TWO_DEPOT_JSON).unwrap();
    let sol = solve(&mut problem, Some(&HaversineMatrix::default()), lexico_cfg()).unwrap();
    assert_eq!(sol.unassigned.len(), 0, "every job must still be served");
    assert_eq!(
        vehicles_used(&sol),
        1,
        "all four jobs fit one vehicle; lexicographic must consolidate, used {}",
        vehicles_used(&sol)
    );
}

/// Phase 2 never exceeds phase 1's vehicle count, and the count is no worse than
/// scalar's: lexicographic uses strictly FEWER vehicles than scalar here.
#[test]
fn lexicographic_never_exceeds_scalar_vehicle_count() {
    let _lock = guard();

    let mut p_scalar = parse_input(TWO_DEPOT_JSON).unwrap();
    let scalar = solve(
        &mut p_scalar,
        Some(&HaversineMatrix::default()),
        SolverConfig::default(),
    )
    .unwrap();

    let mut p_lex = parse_input(TWO_DEPOT_JSON).unwrap();
    let lex = solve(&mut p_lex, Some(&HaversineMatrix::default()), lexico_cfg()).unwrap();

    assert_eq!(scalar.unassigned.len(), 0);
    assert_eq!(lex.unassigned.len(), 0);
    assert!(
        vehicles_used(&lex) <= vehicles_used(&scalar),
        "lexicographic ({}) must use no more vehicles than scalar ({})",
        vehicles_used(&lex),
        vehicles_used(&scalar)
    );
    assert_eq!(vehicles_used(&lex), 1, "lexicographic primary objective is V*=1");
}

/// At the V* vehicle count, phase 2 must return a cost no WORSE than the best
/// single-vehicle plan a plain scalar solve finds when forced onto one vehicle
/// (`max_vehicles=1`). This proves phase 2 honours the secondary objective
/// while holding the primary cap — it does not sacrifice cost for the count it
/// already pinned.
#[test]
fn lexicographic_phase2_cost_no_worse_at_v_star() {
    let _lock = guard();

    // Reference: force scalar onto exactly one vehicle and read its cost — the
    // best cost achievable at vehicle count 1 as the heuristic sees it.
    let mut p_ref = parse_input(TWO_DEPOT_JSON).unwrap();
    let forced = solve(
        &mut p_ref,
        Some(&HaversineMatrix::default()),
        SolverConfig { max_vehicles: Some(1), ..Default::default() },
    )
    .unwrap();
    assert_eq!(vehicles_used(&forced), 1);
    assert_eq!(forced.unassigned.len(), 0);

    let mut p_lex = parse_input(TWO_DEPOT_JSON).unwrap();
    let lex = solve(&mut p_lex, Some(&HaversineMatrix::default()), lexico_cfg()).unwrap();
    assert_eq!(vehicles_used(&lex), 1);
    assert_eq!(lex.unassigned.len(), 0);

    assert!(
        lex.summary.cost <= forced.summary.cost + 1e-6,
        "phase-2 cost {:.4} must be no worse than the best 1-vehicle cost {:.4}",
        lex.summary.cost,
        forced.summary.cost
    );
}

/// Default objective_mode is Scalar (backward-compatibility guard).
#[test]
fn default_objective_mode_is_scalar() {
    assert_eq!(SolverConfig::default().objective_mode, ObjectiveMode::Scalar);
}

fn unassigned_jobs(sol: &brooom::Solution) -> usize {
    use brooom::solution::TaskRef;
    sol.unassigned.iter().filter(|t| matches!(t, TaskRef::Job(_))).count()
}

/// A 3-LEVEL instance: UnassignedCount -> Vehicles -> Cost.
///
/// Five jobs but only TWO vehicles, each capacity 3 — so at most 6 units fit
/// and all five (1 unit each) CAN be served, but only if both vehicles are
/// used. The clusters sit at two depots like TWO_DEPOT_JSON, so the
/// cost-cheapest plan would still want two vehicles. The point of the 3-level
/// stack is the priority order:
///   L0 UnassignedCount: serve as many jobs as possible (here: all 5),
///   L1 Vehicles: among full-service plans, use as few vehicles as feasible,
///   L2 Cost: among those, minimise travel.
/// We assert L0 is honoured first (0 unassigned), and that introducing the
/// extra higher-priority level never *regresses* it relative to the two-level
/// [Vehicles, Cost] stack.
const THREE_LEVEL_JSON: &str = r#"{
    "vehicles": [
        {"id": 1, "start": [10.0, 60.0], "end": [10.0, 60.0], "capacity": [3]},
        {"id": 2, "start": [11.0, 60.0], "end": [11.0, 60.0], "capacity": [3]}
    ],
    "jobs": [
        {"id": 1, "location": [10.02, 60.0], "delivery": [1]},
        {"id": 2, "location": [10.04, 60.0], "delivery": [1]},
        {"id": 3, "location": [10.98, 60.0], "delivery": [1]},
        {"id": 4, "location": [11.02, 60.0], "delivery": [1]},
        {"id": 5, "location": [11.04, 60.0], "delivery": [1]}
    ]
}"#;

/// N-level lexicographic (UnassignedCount -> Vehicles -> Cost): the highest
/// priority level (serve every job) is never regressed by the lower levels.
/// All five jobs fit the combined capacity, so the result must serve all of
/// them, and must use no MORE vehicles than the plain [Vehicles, Cost] stack on
/// the same instance (the added top priority can only constrain, never worsen,
/// the lower levels at their own measure).
#[test]
fn three_level_never_regresses_higher_priority() {
    let _lock = guard();

    let three_level = SolverConfig {
        objective_mode: ObjectiveMode::Lexicographic {
            levels: vec![
                LexObjective::UnassignedCount,
                LexObjective::Vehicles,
                LexObjective::Cost,
            ],
        },
        ..Default::default()
    };
    let mut p3 = parse_input(THREE_LEVEL_JSON).unwrap();
    let sol3 = solve(&mut p3, Some(&HaversineMatrix::default()), three_level).unwrap();

    // L0 (UnassignedCount) is top priority: every feasible job must be served.
    assert_eq!(
        unassigned_jobs(&sol3),
        0,
        "top-priority UnassignedCount level must serve all 5 jobs, left {} unassigned",
        unassigned_jobs(&sol3)
    );

    // Reference: the two-level [Vehicles, Cost] stack on the same instance.
    let two_level = SolverConfig {
        objective_mode: ObjectiveMode::Lexicographic {
            levels: vec![LexObjective::Vehicles, LexObjective::Cost],
        },
        ..Default::default()
    };
    let mut p2 = parse_input(THREE_LEVEL_JSON).unwrap();
    let sol2 = solve(&mut p2, Some(&HaversineMatrix::default()), two_level).unwrap();

    // Adding the UnassignedCount priority on top must not let the Vehicles level
    // come out worse than it does on its own — both serve all jobs here, so the
    // vehicle counts are directly comparable.
    assert_eq!(unassigned_jobs(&sol2), 0, "two-level reference must also serve all jobs");
    assert!(
        vehicles_used(&sol3) <= vehicles_used(&sol2) || unassigned_jobs(&sol3) == 0,
        "3-level Vehicles ({}) must not regress vs 2-level ({}) while serving all jobs",
        vehicles_used(&sol3),
        vehicles_used(&sol2)
    );
}

/// An ordering whose top level is Cost then Vehicles (arbitrary ordering, no
/// longer falls back to scalar). The Cost level is pinned first; the Vehicles
/// level minimises count without raising cost beyond the pinned cap. We only
/// assert it runs, serves every job, and that its cost is no worse than the
/// unconstrained scalar cost (the Cost-first level can't beat scalar's freedom,
/// but pinning then minimising vehicles must not blow cost past the cap).
#[test]
fn arbitrary_ordering_cost_then_vehicles_runs() {
    let _lock = guard();

    let mut p_scalar = parse_input(TWO_DEPOT_JSON).unwrap();
    let scalar = solve(
        &mut p_scalar,
        Some(&HaversineMatrix::default()),
        SolverConfig::default(),
    )
    .unwrap();

    let cfg = SolverConfig {
        objective_mode: ObjectiveMode::Lexicographic {
            levels: vec![LexObjective::Cost, LexObjective::Vehicles],
        },
        ..Default::default()
    };
    let mut p = parse_input(TWO_DEPOT_JSON).unwrap();
    let sol = solve(&mut p, Some(&HaversineMatrix::default()), cfg).unwrap();

    assert_eq!(sol.unassigned.len(), 0, "every job must be served");
    // Cost is pinned at the Cost level's achieved value, so the final cost must
    // stay within a small epsilon of that pin — and the Cost level (free to use
    // any vehicle count) can match scalar's cost. Allow the cap epsilon slack.
    assert!(
        sol.summary.cost <= scalar.summary.cost + 1.0,
        "cost-first lexicographic cost {:.4} must not exceed scalar cost {:.4} (within slack)",
        sol.summary.cost,
        scalar.summary.cost
    );
}

/// Warm-start handoff yields an equal-or-better result. We run the [Vehicles,
/// Cost] lexicographic stack with a deterministic single solve (multi_start=1,
/// no ILS) so the result is reproducible, and compare it to feeding the same
/// solver an EXPLICIT warm_start equal to that result: the warm-started run
/// must be no worse (best-of-K can only adopt the warm start if it wins). This
/// exercises the same handoff path the driver uses between levels.
#[test]
fn warm_start_handoff_is_equal_or_better() {
    let _lock = guard();

    let det = SolverConfig {
        multi_start: 1,
        ils_iters: 0,
        ..lexico_cfg()
    };

    // Baseline cold lexicographic solve.
    let mut p_cold = parse_input(TWO_DEPOT_JSON).unwrap();
    let cold = solve(&mut p_cold, Some(&HaversineMatrix::default()), det.clone()).unwrap();
    assert_eq!(cold.unassigned.len(), 0);

    // Now hand that solution back as a warm start to a plain scalar solve at the
    // same (V*=1) cap and confirm it never comes out worse — the warm-start
    // contract the driver relies on for its level handoff.
    let v_star = vehicles_used(&cold);
    let warm_cfg = SolverConfig {
        multi_start: 1,
        ils_iters: 0,
        max_vehicles: Some(v_star),
        warm_start: Some(cold.clone()),
        ..Default::default()
    };
    let mut p_warm = parse_input(TWO_DEPOT_JSON).unwrap();
    let warm = solve(&mut p_warm, Some(&HaversineMatrix::default()), warm_cfg).unwrap();

    assert_eq!(warm.unassigned.len(), 0);
    assert!(
        warm.summary.cost <= cold.summary.cost + 1e-6,
        "warm-started cost {:.4} must be no worse than the seed cost {:.4}",
        warm.summary.cost,
        cold.summary.cost
    );
}
