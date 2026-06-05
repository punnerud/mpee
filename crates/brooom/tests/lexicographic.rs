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
            primary: LexObjective::Vehicles,
            secondary: LexObjective::Cost,
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
