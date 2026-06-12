//! Top-level solve orchestration.

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
// Browser-safe clock on wasm; std::time on native.
use web_time::Instant;

use crate::error::{Error, Result};
use crate::granular::Granular;
use crate::insertion::{greedy_insertion, greedy_insertion_seeded};
use crate::local_search::{local_search, local_search_full, local_search_seeded, route_split_pass};
use crate::matrix::{resolve_coords, Matrix, MatrixSource};
use crate::problem::Problem;
use crate::solution::{Solution, TaskRef};

#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// How many local-search passes to run before giving up. Each pass picks
    /// the single best improving move across all operators.
    pub max_local_search_passes: usize,
    /// Granular-neighborhood K (Toth & Vigo). Smaller K → faster but more
    /// local optima. `None` disables granularity (slower, marginally better).
    pub granular_k: Option<usize>,
    /// Number of parallel multi-start attempts. `1` is the deterministic
    /// single-solve path; larger K runs that many seeded variants in
    /// parallel and returns the cheapest. By construction K≥1 is never
    /// worse than K=1.
    pub multi_start: usize,
    /// Iterated local search: after LS converges, perform this many
    /// destroy-and-repair kicks per multi-start variant. Each kick removes
    /// `ils_kick_size` random tasks and reinserts them, then re-runs LS.
    /// Best-ever cost is tracked across kicks. 0 disables ILS.
    pub ils_iters: usize,
    /// Fraction (0.0..1.0) of tasks to remove per ILS kick.
    pub ils_kick_size: f64,
    /// Optional wall-time budget in milliseconds. When set, the ILS loop
    /// stops as soon as elapsed time exceeds this — useful when you want
    /// "give me the best you can in N seconds".
    pub time_limit_ms: Option<u64>,
    /// If set, print summary lines as the search progresses.
    pub verbose: bool,
    /// Pre-built warm-start solution. When provided, seed=0 skips the
    /// insertion phase and drops straight into local search on this
    /// solution. Other multi-start seeds keep their normal starts so
    /// best-of-K still benefits from diversity. The warm-start must
    /// reference the same `problem`'s jobs/vehicles.
    pub warm_start: Option<crate::solution::Solution>,
    /// Run a GPU megakernel polish pass on the multi-start winner before
    /// the final CPU polish. Falls back silently to CPU-only if GPU
    /// initialization fails or if the GPU pass produces a worse / non-
    /// feasible result. Useful for N≥500 where GPU LS iters cost far
    /// less than CPU iters.
    pub use_gpu: bool,
    /// Cap on the number of vehicles/routes used. `None` = unlimited. Enforced
    /// as a (large) solution-level penalty so jobs that don't fit within the
    /// cap land in `unassigned` (and are charged their prize).
    pub max_vehicles: Option<usize>,
    /// If > 0, add a fairness penalty proportional to the spread of the chosen
    /// metric across used routes (balances workload). 0 disables it.
    pub fairness_weight: f64,
    /// Which per-route quantity fairness balances (duration or load).
    pub fairness_metric: crate::global_constraint::FairnessMetric,
    /// HARD balance: cap the spread (max − min) of `fairness_metric` across used
    /// routes. `None` (default) = no hard cap. Unlike `fairness_weight` (a soft
    /// nudge), this is enforced as a HARD global so the search guarantees the
    /// balance. See [`crate::global_constraint::balance_spread_cap`].
    pub balance_spread: Option<i64>,
    /// Client-group cardinality `(min, max)` served per group. `None` (default)
    /// = exactly-one-per-group (the historical behaviour). `Some((k, k))` forces
    /// exactly k; `Some((min, max))` a range — OR-Tools-style k-of-N disjunctions.
    /// Applies to every group; per-group differing cardinality is not yet exposed.
    pub group_cardinality: Option<(u32, u32)>,
    /// Run the native structured-propagation pre-pass before search
    /// ([`crate::propagate::tighten`]): tighten time windows, close precedence,
    /// detect provably-unservable jobs. `true` (default) — it is sound (never
    /// removes a feasible option) and speeds the search; set `false` to A/B it.
    pub propagate: bool,
    /// Late-Acceptance Hill-Climbing history length for the ILS acceptance
    /// criterion (Burke & Bykov 2017, as in PyVRP). The candidate is accepted if
    /// it beats the cost from this many iterations ago, letting the walk escape a
    /// local basin. Default 20.
    pub lahc_history: usize,
    /// Run extra LAHC + TW-granular multi-start variants ALONGSIDE the proven
    /// greedy ones (best-of-all), for small instances (N≤300). Additive, so it
    /// can only improve (greedy variants are always included). Default `true`;
    /// set `false` for cluster sub-solves so the large-N path is byte-identical.
    pub allow_lahc: bool,
    /// Optional weighted-scalarization weights on the global cost components
    /// (travel / span / custom). `None` (or an all-1.0 set) leaves the objective
    /// exactly as today. When set with non-unit weights the solver registers a
    /// global cost re-weighting that multiplies each component before LS runs.
    ///
    /// HONEST CAVEAT: this is weighted scalarization, not true lexicographic
    /// multi-objective search. The LS still minimises one aggregated scalar;
    /// these weights merely shape it. A real phase-1 (minimise vehicle count)
    /// then phase-2 (minimise cost) solver is out of scope here.
    pub objective_weights: Option<crate::global_constraint::ObjectiveWeights>,
    /// How the objective is optimised. `Scalar` (the default) is today's exact
    /// behaviour: one aggregated cost scalar minimised by the metaheuristic.
    /// `Lexicographic` runs the two-phase driver in [`solve_with_matrix`]:
    /// phase 1 minimises the vehicle count, phase 2 re-solves minimising cost
    /// with the phase-1 vehicle count pinned as a hard cap. See [`ObjectiveMode`].
    pub objective_mode: ObjectiveMode,
    /// Penalty-managed soft constraints (PyVRP-style time-warp). When the search
    /// runs in soft mode the ILS may pass through time-window / capacity /
    /// duration-infeasible solutions — each violation is charged at an adaptive
    /// weight instead of being rejected — and the best *hard-feasible* solution
    /// seen is returned. This lets the search cross the infeasible ridges that
    /// trap a feasible-only local search on time-constrained instances.
    ///
    /// `None` (default) = AUTO: on when the problem has time windows, off
    /// otherwise. `Some(true)`/`Some(false)` force it on/off. The returned
    /// solution is always hard-feasible, so the worst case equals the
    /// feasible-only result. Structural constraints (skills, precedence, …) stay
    /// hard regardless. See [`crate::solution::set_soft_penalties`].
    pub soft_search: Option<bool>,
}

/// Selects scalar (default) vs. N-level lexicographic optimisation.
///
/// `Scalar` is byte-identical to the historical solver: a single aggregated
/// cost is minimised and the secondary "use fewer vehicles" preference is only
/// whatever the cost function happens to encode.
///
/// `Lexicographic { levels }` is a real N-LEVEL driver over an ordered stack of
/// [`LexObjective`]s, highest priority first. For each level `i`:
///   * install that level's *bias penalty* global (e.g. a large per-vehicle
///     fixed cost for `Vehicles`) and solve — warm-started from the previous
///     level's solution so there is no cold re-solve;
///   * read back the achieved value `A_i` of that level's measure;
///   * install that level's *HARD cap* global (`measure ≤ A_i`) for every
///     subsequent level, so a lower-priority level can never regress a
///     higher-priority one.
///
/// The final level minimises its own measure with all prior caps pinned. The
/// classic two-level `[Vehicles, Cost]` ordering is just the N=2 case.
///
/// HONEST CAVEAT: each `A_i` is the metaheuristic's *best-found* value, NOT a
/// proven optimum, so this is best-effort lexicographic (the standard practical
/// meaning), not exact. The cap globals also ride on the HARD-magnitude
/// assumption (one unit of any capped measure out-ranks any cost a later level
/// could shave) — see the cap functions in `global_constraint.rs`. Arbitrary
/// orderings are now supported (no longer a fall-back-to-scalar for anything
/// other than `[Vehicles, Cost]`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ObjectiveMode {
    /// Today's behaviour: minimise one aggregated cost scalar.
    #[default]
    Scalar,
    /// N-level lexicographic: minimise `levels[0]` first, then `levels[1]` with
    /// `levels[0]` pinned at its achieved value, and so on. Ordered highest
    /// priority first. An empty stack degrades to a plain scalar solve.
    Lexicographic { levels: Vec<LexObjective> },
}

/// The objectives a lexicographic level can optimise. Every variant supplies a
/// bias-penalty global (drives the measure down while it is the active level),
/// a HARD cap global (pins the achieved value for later levels), and a measure
/// read off the solution. See the per-variant helpers on the enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexObjective {
    /// Number of vehicles/routes used (`vehicles_used`).
    Vehicles,
    /// Number of unassigned single jobs (`unassigned_count`).
    UnassignedCount,
    /// Aggregated travel/operating cost (summed route cost).
    Cost,
    /// Makespan: the longest route duration (max end-start over routes).
    Makespan,
    /// Total distance summed across all routes.
    Distance,
}

/// A level's achieved value, kept in the measure's native units so the matching
/// cap global can pin it exactly. Vehicles/UnassignedCount are integer counts;
/// Cost is the float route-cost sum; Makespan/Distance are integer seconds/metres.
#[derive(Debug, Clone, Copy, PartialEq)]
enum LexMeasure {
    Count(usize),
    Cost(f64),
    Int(i64),
}

impl LexObjective {
    /// The bias-penalty global installed while this level is the active
    /// objective. `penalty` is the HARD-scaled magnitude the driver picks so the
    /// measure dominates lower-priority terms during this level's solve.
    fn bias_penalty(&self, penalty: crate::problem::Cost) -> std::sync::Arc<crate::global_constraint::GlobalConstraintFn> {
        use crate::global_constraint as gc;
        match self {
            LexObjective::Vehicles => gc::vehicle_count_penalty(penalty),
            LexObjective::UnassignedCount => gc::unassigned_count_penalty(penalty),
            LexObjective::Cost => {
                // The base objective already *is* route cost; no extra bias is
                // needed to make the search minimise it. Install a zero global
                // so the level still has a registered (no-op) presence.
                std::sync::Arc::new(|_: &gc::SolutionView| 0.0)
            }
            LexObjective::Makespan => gc::makespan_penalty(penalty),
            LexObjective::Distance => gc::distance_penalty(penalty),
        }
    }

    /// Read this level's achieved measure off `sol`.
    fn measure(&self, sol: &Solution) -> LexMeasure {
        match self {
            LexObjective::Vehicles => {
                LexMeasure::Count(sol.routes.iter().filter(|r| !r.steps.is_empty()).count())
            }
            LexObjective::UnassignedCount => LexMeasure::Count(
                sol.unassigned.iter().filter(|t| matches!(t, TaskRef::Job(_))).count(),
            ),
            LexObjective::Cost => {
                LexMeasure::Cost(sol.routes.iter().map(|r| r.metrics.cost).sum())
            }
            LexObjective::Makespan => LexMeasure::Int(
                sol.routes
                    .iter()
                    .filter(|r| !r.steps.is_empty())
                    .map(|r| r.metrics.end_time - r.metrics.start_time)
                    .max()
                    .unwrap_or(0),
            ),
            LexObjective::Distance => {
                LexMeasure::Int(sol.routes.iter().map(|r| r.metrics.distance).sum())
            }
        }
    }

    /// The HARD cap global pinning `achieved` for every subsequent level. A
    /// small epsilon slack is added so float noise on a re-evaluated previous
    /// solution can't make it look infeasible against its own cap.
    fn cap(&self, achieved: LexMeasure) -> std::sync::Arc<crate::global_constraint::GlobalConstraintFn> {
        use crate::global_constraint as gc;
        match (self, achieved) {
            (LexObjective::Vehicles, LexMeasure::Count(c)) => gc::max_vehicles(c),
            (LexObjective::UnassignedCount, LexMeasure::Count(c)) => gc::unassigned_count_cap(c),
            (LexObjective::Cost, LexMeasure::Cost(c)) => gc::cost_cap(c + 1e-6),
            (LexObjective::Makespan, LexMeasure::Int(c)) => gc::makespan_cap(c),
            (LexObjective::Distance, LexMeasure::Int(c)) => gc::distance_cap(c),
            // Mismatched measure/objective never happens (measure() and cap()
            // are always called with the same objective). Install a no-op rather
            // than panic so a future variant addition fails soft.
            _ => std::sync::Arc::new(|_: &gc::SolutionView| 0.0),
        }
    }
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_local_search_passes: 50,
            granular_k: Some(20),
            // K=8 by default — closes the cost gap to Vroom by running 8
            // seeded variants in parallel and keeping the cheapest. The
            // wall-time cost is acceptable because we have 30-40× headroom
            // vs Vroom on N≥500 anyway.
            multi_start: 8,
            // 30 ILS kicks per variant with 40% destruction. Combined with
            // K=8 multi-start, gives 240 distinct local optima per solve.
            // Closes the cost gap on N≤250 to within ~1%, larger N benefits
            // from `--time-limit-s` to spend more compute on quality.
            ils_iters: 30,
            ils_kick_size: 0.4,
            time_limit_ms: None,
            verbose: false,
            warm_start: None,
            use_gpu: false,
            max_vehicles: None,
            fairness_weight: 0.0,
            fairness_metric: crate::global_constraint::FairnessMetric::Duration,
            balance_spread: None,
            group_cardinality: None,
            propagate: true,
            lahc_history: 20,
            allow_lahc: true,
            objective_weights: None,
            objective_mode: ObjectiveMode::Scalar,
            soft_search: None,
        }
    }
}

/// Solver result bundle. Holding the matrix lets the caller render per-step
/// timing in the output (otherwise step-level distance would be unknown).
#[derive(Debug, Clone)]
pub struct Solved {
    pub matrix: Matrix,
    pub solution: Solution,
}

/// Resolve coordinates → matrix → initial solution → local search.
///
/// `source` is consulted only if the problem does not already carry a matrix
/// for the vehicles' profile. Pass `Some(&HaversineMatrix::default())` for a
/// network-free build.
pub fn solve(
    problem: &mut Problem,
    source: Option<&dyn MatrixSource>,
    config: SolverConfig,
) -> Result<Solution> {
    let s = solve_full(problem, source, config)?;
    Ok(s.solution)
}

/// Same as `solve` but also returns the matrix used.
pub fn solve_full(
    problem: &mut Problem,
    source: Option<&dyn MatrixSource>,
    config: SolverConfig,
) -> Result<Solved> {
    problem.validate()?;
    let matrix = build_matrix(problem, source)?;
    // Drop the (possibly very large) raw `Vec<Vec<i64>>` matrices that came
    // from JSON now that we have the compact i32 runtime form. On 1000-node
    // instances this releases ~16 MB per matrix.
    problem.matrices.clear();
    // Native structured propagation: tighten windows, close precedence, detect
    // provably-unservable jobs. Sound (never removes a feasible option). Resolve
    // `soft` exactly as the solver will, so window-end tightening / infeasibility
    // flagging only fire when those bounds are actually hard.
    if config.propagate {
        let soft = config
            .soft_search
            .unwrap_or_else(|| problem_has_time_windows(problem));
        let infeasible = crate::propagate::tighten(problem, &matrix, soft);
        if config.verbose && !infeasible.is_empty() {
            for inf in &infeasible {
                eprintln!("brooom: propagation — job {} unservable: {}", inf.job_id, inf.reason);
            }
        }
    }
    let solution = solve_with_matrix(problem, &matrix, &config);
    Ok(Solved { matrix, solution })
}

/// Run insertion + local search using a pre-built matrix.
///
/// With `config.multi_start > 1`, runs that many seeded variants in parallel
/// (rayon) and returns the cheapest. Seed 0 is always the deterministic
/// baseline so we can never beat-then-lose; seeds 1..K shuffle pending tasks
/// within priority class to give LS distinct starting points.
pub fn solve_with_matrix(problem: &Problem, matrix: &Matrix, config: &SolverConfig) -> Solution {
    match &config.objective_mode {
        // Default path: byte-identical to the historical single-scalar solver.
        ObjectiveMode::Scalar => solve_scalar(problem, matrix, config),
        ObjectiveMode::Lexicographic { levels } => {
            solve_lexicographic(problem, matrix, config, levels)
        }
    }
}

/// N-level lexicographic driver over an ordered `levels` stack (highest
/// priority first). For each level `i`:
///   1. install level `i`'s *bias penalty* plus the HARD caps of all PRIOR
///      levels, warm-started from the previous level's solution;
///   2. solve, and read back the achieved measure `A_i`;
///   3. add level `i`'s HARD cap (`measure ≤ A_i`) to the pinned-cap set so
///      every later level is forced to respect it.
/// The final level minimises its own measure with every prior cap pinned.
///
/// WARM-START HANDOFF: each level feeds the previous level's `Solution` as
/// `config.warm_start`. The warm-start contract is strictly safe (best-of-K in
/// `solve_scalar_with_extra_global` only adopts it if it wins), so this removes
/// the cold re-solve without ever regressing a level.
///
/// HONEST CAVEAT: each `A_i` is the metaheuristic's best-found value, NOT a
/// proven optimum — best-effort lexicographic (the standard practical meaning),
/// not exact. The cap globals ride on the HARD-magnitude assumption: one unit of
/// a capped measure is assumed to out-rank any cost a lower-priority level could
/// shave (true for all practical instances; see `global_constraint.rs`).
/// Arbitrary orderings are supported — no fall-back-to-scalar.
fn solve_lexicographic(
    problem: &Problem,
    matrix: &Matrix,
    config: &SolverConfig,
    levels: &[LexObjective],
) -> Solution {
    // An empty stack has no objective to optimise — degrade to a plain scalar
    // solve (byte-identical to the default path).
    if levels.is_empty() {
        return solve_scalar(problem, matrix, config);
    }

    // Bias magnitude for the active level. It must dominate any realistic
    // difference in the lower-priority objectives between solutions that differ
    // by one unit of this level's measure. We reuse the global-constraint HARD
    // scale (1e12), which already out-ranks any real route cost / prize.
    let bias = crate::global_constraint::HARD;

    // HARD caps pinned by all already-solved levels (highest priority first).
    let mut pinned_caps: Vec<std::sync::Arc<crate::global_constraint::GlobalConstraintFn>> =
        Vec::new();
    // Carries the previous level's solution into the next level's warm-start.
    let mut prev: Option<Solution> = None;

    for (i, level) in levels.iter().enumerate() {
        let is_last = i + 1 == levels.len();

        // Per-level globals: every prior level's HARD cap, plus this level's bias
        // penalty (a no-op zero global for `Cost`, whose bias *is* the base
        // objective). The cap set guarantees no higher-priority level regresses.
        let mut extra = pinned_caps.clone();
        extra.push(level.bias_penalty(bias));

        // Warm-start from the previous level's solution. Strictly safe: seed 0
        // adopts it only if best-of-K keeps it, so a level can never come back
        // worse than the warm start it was handed.
        let level_cfg = SolverConfig {
            objective_mode: ObjectiveMode::Scalar,
            warm_start: prev.clone().or_else(|| config.warm_start.clone()),
            ..config.clone()
        };

        let sol = solve_scalar_with_extra_global(problem, matrix, &level_cfg, extra);
        let achieved = level.measure(&sol);

        if config.verbose {
            eprintln!(
                "brooom: lexicographic level {} ({:?}) — achieved {:?} (cost={:.2}){}",
                i, level, achieved, sol.summary.cost,
                if is_last { " [final]" } else { "" }
            );
        }

        // Pin this level's achieved value for every subsequent level.
        if !is_last {
            pinned_caps.push(level.cap(achieved));
        }
        prev = Some(sol);
    }

    // `prev` is always `Some` here (levels is non-empty and the loop ran).
    let mut best = prev.expect("non-empty levels solved at least one level");
    best.recompute_summary(problem);
    best
}

/// The historical single-scalar solve. Extracted verbatim from the old
/// `solve_with_matrix` body so the default path is byte-identical.
fn solve_scalar(problem: &Problem, matrix: &Matrix, config: &SolverConfig) -> Solution {
    solve_scalar_with_extra_global(problem, matrix, config, Vec::new())
}

/// True if the problem carries any real **job** time window — the trigger for
/// AUTO soft search. We deliberately key on customer delivery windows, NOT a
/// vehicle's shift window: nearly every vehicle has a finite shift, so triggering
/// on that would auto-soften capacity/duration on ordinary CVRP problems and
/// change long-standing hard-capacity semantics. A job with only the universal
/// window does not count.
fn problem_has_time_windows(problem: &Problem) -> bool {
    problem.jobs.iter().any(|j| {
        j.time_windows.iter().any(|w| w != &crate::problem::TimeWindow::FOREVER)
    })
}

/// Backing implementation for the scalar solve, optionally installing extra
/// global constraints (used by the lexicographic driver to add the active
/// level's bias penalty plus all prior levels' HARD caps). An empty
/// `extra_globals` is exactly the historical path.
fn solve_scalar_with_extra_global(
    problem: &Problem,
    matrix: &Matrix,
    config: &SolverConfig,
    extra_globals: Vec<std::sync::Arc<crate::global_constraint::GlobalConstraintFn>>,
) -> Solution {
    // Drop any stale eval cache from a previous solve. Worker threads each
    // have their own thread-local cache; rayon will warm them as it spawns.
    crate::solution::eval_cache_invalidate();

    // Register solution-level (cross-route) constraints for this solve: a
    // max-vehicles cap, fairness balancing, and exactly-one-per-group whenever
    // any job declares a group. The guard clears them when the solve returns.
    let mut globals: Vec<std::sync::Arc<crate::global_constraint::GlobalConstraintFn>> = Vec::new();
    if let Some(cap) = config.max_vehicles {
        globals.push(crate::global_constraint::max_vehicles(cap));
    }
    if config.fairness_weight > 0.0 {
        globals.push(crate::global_constraint::fairness(config.fairness_weight, config.fairness_metric));
    }
    if let Some(max_spread) = config.balance_spread {
        globals.push(crate::global_constraint::balance_spread_cap(max_spread, config.fairness_metric));
    }
    // Weighted-scalarization objective weights: register only when non-identity,
    // so default solves keep their exact historical objective. Multiplies each
    // global cost component before LS converges. NOT lexicographic — see
    // SolverConfig::objective_weights.
    if let Some(w) = config.objective_weights {
        if !w.is_identity() {
            globals.push(crate::global_constraint::objective_weights(w));
        }
    }
    if problem.jobs.iter().any(|j| j.group.is_some()) {
        globals.push(match config.group_cardinality {
            Some((min, max)) => crate::global_constraint::k_of_n_per_group(min, max),
            None => crate::global_constraint::exactly_one_per_group(),
        });
    }
    // The lexicographic driver injects the active level's bias penalty plus all
    // prior levels' HARD caps here. An empty list (every non-lexicographic
    // solve) leaves the registered globals exactly as today.
    globals.extend(extra_globals);
    let _global_guard = (!globals.is_empty())
        .then(|| crate::global_constraint::GlobalConstraintGuard::install(globals));

    // TW-aware granular neighbourhood (Vidal proximity) when the problem has time
    // windows — temporally-compatible candidate lists help R/RC instances. Falls
    // back to pure-distance for window-less problems (build_tw is byte-identical
    // there), so the CVRP path is unchanged.
    // The small-N quality package (TW-aware granular + LAHC acceptance below) is
    // gated to N≤300: it lifts small-window-constrained instances but shifts the
    // large-N trajectory (where our tuned greedy multi-start wins ~20% vs OR-Tools
    // and beats PyVRP), so above the threshold we keep the proven behaviour.
    // The proven distance-based granular neighbourhood (unchanged for every
    // existing seed — so the additive LAHC variants below can never regress).
    let granular = config.granular_k.map(|k| Granular::build(matrix, k));
    if config.verbose {
        if let Some(g) = &granular {
            eprintln!("brooom: built granular neighborhood K={} (n={})", g.k(), g.n());
        }
    }

    let k = config.multi_start.max(1);
    let ils_iters = config.ils_iters;
    let ils_kick = config.ils_kick_size.max(0.0).min(1.0);

    // Small/mid-N quality boost (gated via `allow_lahc`; cluster sub-solves
    // disable it, so the large-N path stays byte-identical). When on, we run K
    // EXTRA variants that use Late-Acceptance Hill-Climbing acceptance + a
    // TW-aware (Vidal) granular neighbourhood, ALONGSIDE the K greedy variants,
    // and take best-of-all. Because the greedy variants are untouched, the result
    // is ≥ the greedy-only result — additive, provably non-regressing.
    // The N-gate also gates the HGS phase: at 300 it cut HGS off at G&H n=400,
    // where the route-count basin needs Split/SREX (measured R2 +12–14% vs
    // PyVRP with HGS off, brooom stuck at 12 routes vs 19). 500 covers n=400
    // while leaving the protected N=1000 path untouched. Override via
    // BROOOM_LAHC_MAX_N.
    let lahc_max_n: usize = std::env::var("BROOOM_LAHC_MAX_N")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(500);
    let lahc_on =
        config.allow_lahc && problem.jobs.len() <= lahc_max_n && ils_iters > 0 && ils_kick > 0.0;
    let granular_tw = if lahc_on && problem_has_time_windows(problem) {
        config.granular_k.map(|k| Granular::build_tw(matrix, k, problem))
    } else {
        None
    };
    let lahc_variants = if lahc_on { 2 * k } else { k };
    // Population HGS (giant-tour OX crossover + Split). Runs `n_hgs` island
    // populations as EXTRA variants in the same parallel pool, best-of-all ⇒
    // additive. Split reconstructs route partitions from scratch — the global
    // move ILS can't make — so HGS reaches route structures (e.g. r205's 5-route
    // basin) the perturbation search never finds. Gated to the HGS envelope
    // (single-dim capacity, job-only, homogeneous fleet) and small N (via
    // `lahc_on`). Off via BROOOM_NO_HGS.
    let hgs_candidate = lahc_on
        && crate::genetic::hgs_applicable(problem)
        && config.time_limit_ms.is_some() // HGS phase needs a wall-clock budget
        && std::env::var("BROOOM_NO_HGS").is_err();
    // Route-flexibility gate: the giant-tour Split that powers HGS only helps when
    // the optimal route COUNT is flexible — i.e. routes pack many jobs (wide time
    // windows / long horizon). On tight-window instances routes are forced short,
    // Split adds nothing, and HGS just steals budget from the proven ILS (small
    // regression). Greedy jobs-per-route is a scale-invariant proxy that cleanly
    // separates the two (measured: tight R1/RC1 ≈ 4–6, wide R2/RC2 ≈ 17–33).
    // Override the threshold via BROOOM_HGS_MIN_ROUTELEN.
    let hgs_on = if hgs_candidate {
        let g = greedy_insertion(problem, matrix);
        let routes = g.routes.iter().filter(|r| !r.steps.is_empty()).count().max(1);
        let assigned: usize = g.routes.iter().map(|r| r.steps.len()).sum();
        let avg_route_len = assigned as f64 / routes as f64;
        let min_len: f64 = std::env::var("BROOOM_HGS_MIN_ROUTELEN")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(8.0);
        // Second signal: TIME-WINDOW SLACK. The route-length proxy misses
        // instances with mid-size routes but wide windows (real-map CVRPTW:
        // ~6 stops/route, 3 h windows on a 12 h horizon — measured on
        // sf_s11_n80: forcing HGS closed the PyVRP gap +0.65% → +0.06%).
        // Windows ≥ `tw_frac` of the vehicle horizon leave Split/SREX room to
        // re-partition, which is what HGS exploits; tight-window R1/RC1 sit
        // at ~4% and stay on the protected pure-ILS path.
        let tw_frac_min: f64 = std::env::var("BROOOM_HGS_TW_FRAC")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(0.15);
        let horizon = {
            let vw = problem.vehicles[0].time_window();
            ((vw.end - vw.start).max(1)) as f64
        };
        let avg_tw_frac = if problem.jobs.is_empty() {
            1.0
        } else {
            let sum: f64 = problem
                .jobs
                .iter()
                .map(|j| match j.time_windows.first() {
                    Some(w) => ((w.end - w.start) as f64 / horizon).min(1.0),
                    None => 1.0, // no window = fully flexible
                })
                .sum();
            sum / problem.jobs.len() as f64
        };
        let on = avg_route_len >= min_len || avg_tw_frac >= tw_frac_min;
        if config.verbose {
            eprintln!(
                "brooom: HGS gate — greedy avg route len {:.1} (min {:.1}), avg TW width {:.0}% of horizon (min {:.0}%) ⇒ HGS {}",
                avg_route_len, min_len, avg_tw_frac * 100.0, tw_frac_min * 100.0,
                if on { "ON" } else { "off" }
            );
        }
        on
    } else {
        false
    };
    // HGS education passes: bounded (cold LS is the GA's budget — the default
    // 50 would allow only a handful of generations). 4 was the sweet spot when
    // education was slow; with incremental Split + the full fast-LS operator
    // set, 8 measures better (rc208 78504→78007 consistent, r205 slightly
    // better, none worse). Override via BROOOM_HGS_PASSES.
    let hgs_passes = std::env::var("BROOOM_HGS_PASSES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| config.max_local_search_passes.min(8));
    let total_variants = lahc_variants;
    // Budget split: when HGS is on, the ILS multi-start gets the first fraction
    // of the wall clock (all cores), then the HGS phase gets the rest (all
    // cores), SEEDED with the ILS best. Phasing avoids the core starvation that
    // running both pools concurrently caused. Default 55% ILS / 45% HGS;
    // override the ILS share via BROOOM_HGS_SPLIT (e.g. "0.6").
    let now = Instant::now();
    let full_deadline = config
        .time_limit_ms
        .map(|ms| now + std::time::Duration::from_millis(ms));
    // HGS-primary: a short ILS phase produces a seed + best-of safety floor, then
    // HGS gets the bulk of the budget (measured sweet spot ~0.15; HGS reliably
    // matches/beats the ILS on the gated wide-window instances). Override via
    // BROOOM_HGS_SPLIT.
    let ils_frac: f64 = std::env::var("BROOOM_HGS_SPLIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|f: &f64| *f > 0.05 && *f < 0.95)
        .unwrap_or(0.15);
    let deadline = if hgs_on {
        config
            .time_limit_ms
            .map(|ms| now + std::time::Duration::from_millis((ms as f64 * ils_frac) as u64))
    } else {
        full_deadline
    };

    // Penalty-managed soft constraints (OR-Tools soft-bound semantics). AUTO
    // (None) turns it on when the problem has time windows; Some(_) forces it.
    // When on, the whole search runs in penalised space: a stop may be served
    // slightly late, or a vehicle slightly over capacity / duration, charged
    // `λ × violation` instead of being rejected. λ is FIXED and high (≈50× the
    // per-second travel cost, clamped), derived once from a reference solution:
    //   * on a fully-feasible instance no violation ever lowers the penalised
    //     cost, so the search is identical to hard mode (no quality regression);
    //   * on an over-constrained instance, serving a stop a little late costs
    //     `λ × lateness`, which is far below the drop prize, so the solver serves
    //     it rather than leaving it unassigned.
    // (We deliberately do NOT use infeasibility as a search *bridge* to better
    // hard-feasible solutions à la PyVRP — measured on Solomon, that regressed
    // R/RC at our ILS budget; see benchmarks/results.)
    let soft_on = config.soft_search.unwrap_or_else(|| problem_has_time_windows(problem));
    let soft_weights = if soft_on {
        let ref_sol = greedy_insertion(problem, matrix);
        let denom = (ref_sol.summary.travel_time + ref_sol.summary.service_time).max(1) as f64;
        let scale = (ref_sol.summary.cost / denom).max(1e-6);
        // λ high enough that no violation is ever beneficial on a feasible
        // instance (so it stays byte-identical to hard), yet far below the drop
        // prize (DEFAULT_PRIZE ≈ 1e9) so serving a stop late always beats dropping
        // it. The scale (per-second travel cost) × 1000 sits comfortably between.
        let lam = (scale * 1000.0).clamp(10.0, 1.0e5);
        Some(crate::solution::SoftWeights { tw: lam, load: lam, dur: lam })
    } else {
        None
    };
    if config.verbose && soft_on {
        let w = soft_weights.unwrap();
        eprintln!("brooom: penalty-managed soft constraints ON (λ≈{:.1}, time/load/duration)", w.tw);
    }

    // For each of K seeds: insertion → LS → ILS-kick loop. Best across all
    // attempts wins. With K=1 and ils_iters=0 this is the original baseline;
    // any larger setting trades wall time for cost.
    let solve_one = |seed: u64| -> Solution {
        // Construction runs in HARD mode so each seed starts from a clean
        // feasible base (rayon may reuse a worker thread that left soft armed).
        crate::solution::set_soft_penalties(None);
        // Arm the LS wall-clock for this variant: construction (greedy + full
        // LS to convergence) had no deadline checks of its own, and at n≥400 it
        // alone overshoots the ILS window by seconds — both blowing the user's
        // time limit and starving the HGS phase that is budgeted to follow.
        // seed 0 arms AFTER construction so at least one variant is always a
        // complete solution regardless of budget. Cleared before returning
        // (pooled thread).
        if seed != 0 {
            crate::local_search::set_ls_deadline(deadline);
        }
        let t_var = Instant::now();
        // Variants [0, k) are the proven greedy ones (distance granular, greedy
        // acceptance — byte-identical to before). Variants [k, 2k) are the LAHC
        // boost (TW granular + late-acceptance). best-of-all ⇒ never worse.
        let is_lahc = lahc_on && (seed as usize) >= k;
        let gran = if is_lahc {
            granular_tw.as_ref().or(granular.as_ref())
        } else {
            granular.as_ref()
        };
        // Diversify starting solutions across multi-start variants:
        //   seed=0     → deterministic greedy cheapest (baseline)
        //   even seeds → seeded greedy (shuffled within priority)
        //   odd seeds  → Solomon I1 with varied λ (1.0, 1.5, 2.0, 3.0)
        // Solomon I1 produces structurally different starts (favors far-from-
        // depot tasks first), which gives LS access to local optima the
        // greedy variants can't reach. Verified on N=1000 to close ~1-2% gap
        // vs Vroom by complementing the greedy seeds.
        let dbg_phase = std::env::var("BROOOM_ILS_DEBUG").is_ok();
        let mut sol = if seed == 0 {
            // Warm-start (if any) replaces the deterministic greedy baseline
            // for seed=0. Other seeds keep their normal diversifying starts —
            // best-of-K then takes warm-start vs alternatives without ever
            // losing, so warm-start is strictly safe.
            if let Some(ws) = config.warm_start.as_ref() {
                ws.clone()
            } else {
                greedy_insertion(problem, matrix)
            }
        } else {
            greedy_insertion_seeded(problem, matrix, seed)
        };
        let t_constr = t_var.elapsed();
        crate::local_search::set_ls_deadline(deadline); // seed 0 arms here
        local_search(
            problem, matrix, &mut sol,
            config.max_local_search_passes, gran,
        );
        let t_ls = t_var.elapsed();

        // RouteSplit on every seed with a per-seed revert guard: snapshot the
        // cost, try split + re-LS, and keep it only if it strictly improves —
        // otherwise restore. This spreads the split's benefit (key on wide-window
        // R2/RC2 instances, where spreading onto more, shorter routes lowers
        // distance) across the whole multi-start pool while making the earlier
        // +0.6% N=1000 regression impossible (a seed can never end up worse than
        // its no-split result).
        // Small instances: try split on every seed (the split + re-LS is cheap
        // at this size, and the revert guard makes a per-seed regression
        // impossible). Larger instances keep the original seed==7-only behaviour
        // so the extra split + re-LS never eats into the ILS time budget at scale.
        // Deadline guard: at n=400 the construction (greedy + full LS) above can
        // already exhaust the phase budget; the split + re-LS below would then
        // push wall-clock 1.1–2× past the user's time limit (measured 10.9–19.4 s
        // at -l 10 on G&H 400). The split is a quality bonus, never required for
        // validity — skip it when the phase deadline has passed.
        let split_this_seed = (if problem.jobs.len() <= 400 { true } else { seed == 7 })
            && deadline.map_or(true, |d| Instant::now() < d);
        if split_this_seed {
            let before_cost = sol.summary.cost;
            let snapshot = sol.clone();
            route_split_pass(problem, matrix, &mut sol, 10);
            local_search(
                problem, matrix, &mut sol,
                config.max_local_search_passes, gran,
            );
            if sol.summary.cost >= before_cost - 1e-9 {
                sol = snapshot; // split didn't help this seed — revert.
            }
        }
        if dbg_phase {
            eprintln!(
                "variant seed={seed}: start={:.2}s constr={:.2}s ls={:.2}s split-block={:.2}s",
                t_var.duration_since(now).as_secs_f64(),
                t_constr.as_secs_f64(),
                (t_ls - t_constr).as_secs_f64(),
                (t_var.elapsed() - t_ls).as_secs_f64()
            );
        }

        // Arm soft penalties for the ILS only when they can matter: construction
        // left stops unassigned (over-constrained — soft serves them late instead
        // of dropping), the caller forced soft mode, or BROOOM_NO_HARD_ILS=1
        // restores the old always-arm behaviour for A/B. On a fully-assigned
        // solution the fixed high λ makes soft trajectory-identical to hard by
        // design — but soft_is_active() disables every fast LS path, so arming it
        // unconditionally paid the slow path for zero trajectory difference.
        // (Construction above stayed hard either way.)
        let force_soft = config.soft_search == Some(true)
            || std::env::var("BROOOM_NO_HARD_ILS").is_ok();
        let mut soft_armed = false;
        if let Some(w) = soft_weights {
            if force_soft || !sol.unassigned.is_empty() {
                crate::solution::set_soft_penalties(Some(w));
                soft_armed = true;
            }
        }

        // ILS with Late-Acceptance Hill-Climbing (Burke & Bykov 2017), the
        // acceptance criterion PyVRP's ILS uses. Instead of only accepting a
        // kick that beats the global best (greedy — which sticks in a basin, e.g.
        // the wide-window over-consolidation trap), we kick from the CURRENT
        // working solution and accept it if it beats the cost from `L` iterations
        // ago OR the current cost. That lets the walk climb out of a basin (take
        // a temporarily worse plan) and reach cheaper feasible optima the greedy
        // walk can't. `best` is tracked separately and returned.
        if is_lahc {
            let mut best_cost = sol.summary.cost;
            let mut best_sol = sol.clone();
            let mut cur = sol.clone();
            let mut cur_cost = best_cost;
            // History length: short enough to turn over within our budget so the
            // late comparison actually bites (PyVRP uses 300 over millions of
            // iters; at our iteration counts a smaller window is what works).
            let hist_len = config.lahc_history.max(1);
            let mut hist = vec![best_cost; hist_len];
            let mut hidx = 0usize;
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xA5A5));
            // Deep trajectory (PyVRP recipe): small perturbation with a
            // route-OPENING move (lift a cluster onto a spare vehicle) + LS, run
            // to the deadline. LAHC accepts the temporarily-worse opening so the
            // walk can reach a higher-route basin. No time limit ⇒ a fixed cap.
            let max_perturb = (problem.jobs.len() / 10).clamp(1, 25);
            let max_iters = if deadline.is_some() { usize::MAX } else { ils_iters.max(1) * 80 };
            // Seeded re-convergence: `cur` is always LS-converged, so only the
            // perturbed region needs re-probing (see local_search_seeded).
            // BROOOM_NO_SEEDED_LS forces the old full-reset behaviour for A/B.
            let seeded_ls = std::env::var("BROOOM_NO_SEEDED_LS").is_err();
            let mut iters: u64 = 0;
            for _ in 0..max_iters {
                if let Some(d) = deadline {
                    if Instant::now() >= d { break; }
                }
                iters += 1;
                let mut perturbed = cur.clone();
                let n_moves = rng.gen_range(1..=max_perturb);
                let mut touched = perturb_small(&mut perturbed, n_moves, 0.08, &mut rng, problem, matrix);
                if seeded_ls {
                    // Spatial spillover: a task in an UNTOUCHED route may gain an
                    // improving move into the perturbed region (freed capacity /
                    // TW slack), so also unsettle every task that has a touched
                    // location among its granular neighbours. Without this the
                    // seeded restart converges to worse optima on tight-window
                    // instances (measured +0.5% on rc101).
                    if let Some(g) = gran {
                        let touched_locs: std::collections::HashSet<usize> = touched
                            .iter()
                            .filter_map(|t| t.description(problem).location.index)
                            .collect();
                        for r in &perturbed.routes {
                            for &t in &r.steps {
                                if touched.contains(&t) { continue; }
                                if let Some(loc) = t.description(problem).location.index {
                                    if g.neighbors(loc).any(|nb| touched_locs.contains(&nb)) {
                                        touched.insert(t);
                                    }
                                }
                            }
                        }
                    }
                    local_search_seeded(
                        problem, matrix, &mut perturbed,
                        config.max_local_search_passes, gran, &touched,
                    );
                } else {
                    local_search(
                        problem, matrix, &mut perturbed,
                        config.max_local_search_passes, gran,
                    );
                }
                // One-way latch: a kick can strand a task (re-insertion failed
                // under hard mode) — from then on this variant runs soft as
                // before, so the walk can serve it late rather than carry the
                // drop prize.
                if !soft_armed && !perturbed.unassigned.is_empty() {
                    if let Some(w) = soft_weights {
                        crate::solution::set_soft_penalties(Some(w));
                        soft_armed = true;
                    }
                }
                let cc = perturbed.summary.cost;
                // New global best → keep it, and (exhaustive-on-best) give it a
                // full no-don't-look-bit polish, à la PyVRP `exhaustive_on_best`.
                if cc < best_cost - 1e-9 {
                    best_cost = cc;
                    best_sol = perturbed.clone();
                    let mut polished = best_sol.clone();
                    local_search_full(problem, matrix, &mut polished, config.max_local_search_passes, gran);
                    if polished.summary.cost < best_cost - 1e-9 {
                        best_cost = polished.summary.cost;
                        best_sol = polished;
                    }
                }
                // Late-acceptance: accept into the walk if better than the cost
                // `hist_len` iters ago or than the current solution.
                let late = hist[hidx];
                if cc < late || cc < cur_cost {
                    cur = perturbed;
                    cur_cost = cc;
                }
                // Update history only when current improves on the stored value.
                if cur_cost < hist[hidx] {
                    hist[hidx] = cur_cost;
                }
                hidx = (hidx + 1) % hist_len;
            }
            if std::env::var("BROOOM_ILS_DEBUG").is_ok() {
                eprintln!("ILS-LAHC seed={seed}: iters={iters} best={best_cost:.0}");
            }
            sol = best_sol;
        } else if ils_iters > 0 && ils_kick > 0.0 {
            // Large-N path: the proven greedy ILS (kick from best, accept iff it
            // beats the global best). Unchanged from before — protects the
            // large-N win that the LAHC trajectory shift would otherwise dent.
            let mut best_cost = sol.summary.cost;
            let mut best_sol = sol.clone();
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xA5A5));
            for _ in 0..ils_iters {
                if let Some(d) = deadline {
                    if Instant::now() >= d { break; }
                }
                let mut perturbed = best_sol.clone();
                kick(&mut perturbed, ils_kick, &mut rng, problem, matrix);
                local_search(
                    problem, matrix, &mut perturbed,
                    config.max_local_search_passes, granular.as_ref(),
                );
                if !soft_armed && !perturbed.unassigned.is_empty() {
                    if let Some(w) = soft_weights {
                        crate::solution::set_soft_penalties(Some(w));
                        soft_armed = true;
                    }
                }
                if perturbed.summary.cost < best_cost {
                    best_cost = perturbed.summary.cost;
                    best_sol = perturbed;
                }
            }
            sol = best_sol;
        }
        crate::local_search::set_ls_deadline(None);
        sol
    };

    // Run `total_variants` (= k greedy, plus k LAHC variants when `lahc_on`) and
    // keep the cheapest. Greedy variants [0,k) are unchanged, so best-of-all is
    // always ≤ the greedy-only result — the LAHC boost can only help.
    #[allow(unused_mut)]
    let mut best = if total_variants == 1 {
        solve_one(0)
    } else {
        // Parallel multi-start on native; serial on wasm (no rayon).
        #[cfg(feature = "parallel")]
        {
            (0..total_variants as u64)
                .into_par_iter()
                .map(solve_one)
                .min_by(|a, b| {
                    a.summary
                        .cost
                        .partial_cmp(&b.summary.cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("at least one variant")
        }
        #[cfg(not(feature = "parallel"))]
        {
            (1..total_variants as u64).map(solve_one).fold(solve_one(0), |a, b| {
                if b.summary.cost < a.summary.cost { b } else { a }
            })
        }
    };

    // ── HGS phase ──────────────────────────────────────────────────────────
    // Population Hybrid Genetic Search (giant-tour OX crossover + Split) on the
    // remaining wall-clock budget, all cores, SEEDED with the ILS best. Split
    // reconstructs route partitions from scratch — the global move ILS can't make
    // — reaching route structures (e.g. r205's 5-route basin) the perturbation
    // search never finds. The seed is Split+educated (both can only improve it)
    // and kept in the population, so HGS returns ≤ the ILS best ⇒ this phase is
    // non-regressing vs the ILS result it starts from. Soft-search must be OFF so
    // the GA's education stays hard-feasible.
    #[cfg(feature = "parallel")]
    if hgs_on {
        crate::solution::set_soft_penalties(None);
        // Plain DISTANCE granular (not the TW-aware one): the GA's cold education
        // recombines whole orderings, where the distance neighbourhood reaches
        // better optima than the TW-proximity lists (measured: TW granular gave
        // ~+1.5% worse HGS results on R2/RC2).
        let gran_hgs = granular.as_ref();
        let n_islands = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .clamp(2, 16);
        // Reserve a small slice of the budget for the final polish below: the
        // islands' LS is deadline-cut, so the HGS winner arrives UNCONVERGED at
        // larger n and the polish needs real time — unbudgeted, it ran seconds
        // past the user's limit on the main thread (n=400: 14–18 s at -l 10).
        let hgs_deadline = config.time_limit_ms.map(|ms| {
            let reserve = (ms / 20).max(200).min(ms / 2);
            now + std::time::Duration::from_millis(ms - reserve)
        });
        if std::env::var("BROOOM_HGS_DEBUG").is_ok() {
            eprintln!("HGS phase start at {:.2}s", now.elapsed().as_secs_f64());
        }
        if let Some(hgs_sol) = crate::genetic::solve_genetic_parallel(
            problem, matrix, gran_hgs, hgs_passes, 0x00C0_FFEE, hgs_deadline, n_islands,
            std::slice::from_ref(&best),
        ) {
            if hgs_sol.summary.cost + 1e-9 < best.summary.cost {
                if config.verbose {
                    eprintln!(
                        "brooom: HGS phase: {:.2} → {:.2} (Δ={:.2})",
                        best.summary.cost, hgs_sol.summary.cost,
                        best.summary.cost - hgs_sol.summary.cost
                    );
                }
                best = hgs_sol;
            }
        }
    }

    // GPU megakernel polish pass on the multi-start winner. Falls back
    // silently if GPU init fails or no improvement. Only invoked for
    // larger N where cross-route interactions matter — at the per-cluster
    // level after cluster_decompose the routes are too short for batch
    // GPU to find diversity-driven wins. The outer flow in main.rs runs
    // a separate top-level gpu_polish on the merged solution.
    // Custom constraints run only on the CPU evaluator, so skip GPU polishing
    // when any are registered (the megakernel can't call arbitrary closures).
    // Arm soft penalties for the serial polish too, so it never re-rejects a
    // legitimately-served-late route (and can pull a dropped stop in late via the
    // repair pass). On a feasible instance the high λ keeps the polish identical
    // to hard mode — so when everything is assigned (nothing served late, nothing
    // to pull in) we stay hard and keep the fast LS paths, same contract as the
    // ILS arming above. BROOOM_NO_HARD_ILS=1 restores always-arm.
    if let Some(w) = soft_weights {
        let served_late = best.routes.iter().any(|r| {
            r.metrics.tw_excess > 0.0 || r.metrics.load_excess > 0.0 || r.metrics.dur_excess > 0.0
        });
        if config.soft_search == Some(true)
            || std::env::var("BROOOM_NO_HARD_ILS").is_ok()
            || !best.unassigned.is_empty()
            || served_late
        {
            crate::solution::set_soft_penalties(Some(w));
        }
    }

    // GPU polish uses a hard-only megakernel (no soft penalties), so skip it when
    // soft constraints are active — a hard pass would re-reject served-late routes.
    #[cfg(feature = "gpu")]
    if config.use_gpu && !soft_on && best.routes.len() > 0 && matrix.n >= 500
        && !crate::constraint::has_constraints()
        && !crate::global_constraint::has_global()
        && !problem.any_multi_trip()
    {
        let t_gpu = std::time::Instant::now();
        let max_iter = if matrix.n >= 5000 { 2000 } else { 1000 };
        if let Some(gpu_sol) = crate::gpu_polish::gpu_polish(
            problem, matrix, &best, granular.as_ref(), max_iter, config.verbose,
        ) {
            if gpu_sol.summary.cost + 1e-9 < best.summary.cost {
                if config.verbose {
                    eprintln!(
                        "brooom: GPU polish: {:.2} → {:.2} (Δ={:.2}, t={:.2}s)",
                        best.summary.cost, gpu_sol.summary.cost,
                        best.summary.cost - gpu_sol.summary.cost,
                        t_gpu.elapsed().as_secs_f64()
                    );
                }
                best = gpu_sol;
            }
        }
    }

    // Final polishing pass on the multi-start winner: full LS with no
    // don't-look bits — every task reconsidered every pass. Vroom-style.
    // Don't-look-LS converges fast but can prematurely settle tasks that
    // a later move could free up. This finishing pass picks up those
    // missed moves once.
    // The polish runs on the main thread with the FULL deadline armed: it gets
    // the reserve the HGS phase set aside, and is hard-cut at the user's limit
    // (an unconverged polish result is still valid — moves apply atomically).
    crate::local_search::set_ls_deadline(full_deadline);
    let pre_polish_cost = best.summary.cost;
    local_search_full(
        problem, matrix, &mut best,
        config.max_local_search_passes, granular.as_ref(),
    );
    if config.verbose && best.summary.cost + 1e-9 < pre_polish_cost {
        eprintln!(
            "brooom: polish pass: {:.2} → {:.2} (Δ={:.2})",
            pre_polish_cost, best.summary.cost, pre_polish_cost - best.summary.cost
        );
    }

    // R2/RC2 over-consolidation lever: on wide-window instances the cheaper plan
    // spreads onto more, shorter routes, but greedy+LS tends to pack into a few
    // long ones. Pull short interior segments onto fresh vehicles when it lowers
    // cost, then re-LS — keeping the result only if the whole solution improved.
    // Small-N only + revert-guarded, so it can never regress the N≥1000 win (an
    // earlier route-opening attempt cost +0.6% there) or any instance it touches.
    let pre_spread = best.summary.cost;
    spread_polish(&mut best, problem, matrix, granular.as_ref(), &config);
    if config.verbose && best.summary.cost + 1e-9 < pre_spread {
        eprintln!(
            "brooom: spread pass: {:.2} → {:.2} (Δ={:.2})",
            pre_spread, best.summary.cost, pre_spread - best.summary.cost
        );
    }

    // Validity work below (repair) must never be deadline-cut — clear the LS
    // wall-clock armed for the polish above.
    crate::local_search::set_ls_deadline(None);

    // Final guaranteed-assignment pass. The ILS `kick` drops empty routes and
    // only reinserts into surviving ones, so when vehicles outnumber demand a
    // feasible job can get stranded in `unassigned` instead of opening a spare
    // vehicle. Repair that here: place each remaining job in its cheapest
    // feasible slot (existing route or an unused vehicle).
    let pre_repair_unassigned = best.unassigned.len();
    repair_unassigned(&mut best, problem, matrix);
    if config.verbose && best.unassigned.len() < pre_repair_unassigned {
        eprintln!(
            "brooom: repair pass: assigned {} stranded job(s) → unassigned={}",
            pre_repair_unassigned - best.unassigned.len(),
            best.unassigned.len()
        );
    }

    // Prize-collecting: the cost-greedy insertion fills scarce capacity with the
    // cheapest-to-reach jobs, ignoring value. Swap a served low-value job for an
    // unassigned higher-value one whenever that lowers the objective. No-op when
    // every job keeps the default sentinel prize and no explicit disjunction
    // drop penalty is set (an explicit penalty also makes serving worthwhile and
    // so warrants the swap pass).
    if problem.jobs.iter().any(|j| {
        j.prize < crate::problem::DEFAULT_PRIZE || j.disjunction_penalty.is_some()
    }) {
        prize_swap_pass(&mut best, problem, matrix);
    }

    // Hard-enforce the solution-level caps the search only soft-penalized:
    // trim to the vehicle cap and to one served member per client group, moving
    // any surplus to `unassigned`. No-ops unless configured / groups present.
    if let Some(cap) = config.max_vehicles {
        enforce_max_vehicles(&mut best, problem, cap);
    }
    if problem.jobs.iter().any(|j| j.group.is_some()) {
        let max_per_group = config.group_cardinality.map(|(_, mx)| mx as usize).unwrap_or(1);
        enforce_groups(&mut best, problem, matrix, max_per_group);
    }

    if config.verbose {
        eprintln!(
            "brooom: multi_start={} best — routes={} unassigned={} cost={:.2}",
            k,
            best.routes.len(),
            best.unassigned.len(),
            best.summary.cost
        );
    }
    // Disarm soft penalties — the solve is complete and any later evaluation
    // (output rendering, callers) must see hard semantics.
    crate::solution::set_soft_penalties(None);
    best
}

/// Prize-collecting improvement: replace a served job with an unassigned
/// higher-value job when doing so lowers the objective (route-cost delta plus
/// the swapped unassigned costs). "Value" here is `Job::unassigned_cost()` =
/// `prize + disjunction_penalty`, i.e. exactly what the objective charges when a
/// job is dropped, so the swap is consistent with `recompute_summary`. Greedy,
/// applied to convergence; O(unassigned × stops) per round, only when finite
/// prizes or explicit disjunction penalties exist.
fn prize_swap_pass(sol: &mut Solution, problem: &Problem, matrix: &Matrix) {
    loop {
        let mut applied = false;
        'search: for ui in 0..sol.unassigned.len() {
            let u = sol.unassigned[ui];
            if !matches!(u, TaskRef::Job(_)) {
                continue;
            }
            let u_cost = u.description(problem).unassigned_cost();
            for ri in 0..sol.routes.len() {
                for pos in 0..sol.routes[ri].steps.len() {
                    let s = sol.routes[ri].steps[pos];
                    if !matches!(s, TaskRef::Job(_)) {
                        continue;
                    }
                    let s_cost = s.description(problem).unassigned_cost();
                    if s_cost >= u_cost {
                        continue; // only swap in a strictly more valuable job
                    }
                    let mut cand = sol.routes[ri].steps.clone();
                    cand[pos] = u;
                    let veh = &problem.vehicles[sol.routes[ri].vehicle_idx];
                    if let Ok(m) = crate::solution::evaluate_route(problem, matrix, veh, &cand) {
                        // Δobjective = route-cost change + (s now unassigned) − (u no longer unassigned).
                        let delta = (m.cost - sol.routes[ri].metrics.cost) + s_cost - u_cost;
                        if delta < -1e-6 {
                            sol.routes[ri].steps = cand;
                            sol.routes[ri].metrics = m;
                            sol.unassigned[ui] = s;
                            applied = true;
                            break 'search;
                        }
                    }
                }
            }
        }
        if !applied {
            break;
        }
    }
    sol.recompute_summary(problem);
}

/// Hard-enforce a max-vehicles cap: while more than `cap` routes are non-empty,
/// drop the smallest route (fewest stops) and move its tasks to `unassigned`.
fn enforce_max_vehicles(sol: &mut Solution, problem: &Problem, cap: usize) {
    loop {
        let nonempty: Vec<usize> = sol.routes.iter().enumerate()
            .filter(|(_, r)| !r.steps.is_empty())
            .map(|(i, _)| i)
            .collect();
        if nonempty.len() <= cap {
            break;
        }
        let victim = *nonempty
            .iter()
            .min_by_key(|&&i| sol.routes[i].steps.len())
            .unwrap();
        let removed = std::mem::take(&mut sol.routes[victim].steps);
        sol.unassigned.extend(removed);
    }
    sol.routes.retain(|r| !r.steps.is_empty());
    sol.recompute_summary(problem);
}

/// Hard-enforce "exactly one served member per client group": keep the
/// lowest-index served member of each group, evict the rest to `unassigned`.
/// R2/RC2 over-consolidation lever — see the call site. Greedily pulls a short
/// interior segment (1–3 stops) onto a fresh unused vehicle whenever that lowers
/// cost, then re-runs local search; keeps the result only if the whole solution
/// improved (otherwise reverts). Guarded to small instances so it never touches
/// the large-N path. Safe by construction: it can only ever lower `best`'s cost.
fn spread_polish(
    sol: &mut Solution,
    problem: &Problem,
    matrix: &Matrix,
    granular: Option<&Granular>,
    config: &SolverConfig,
) {
    use crate::solution::{evaluate_route, Route, RouteMetrics};
    use std::collections::HashSet;

    // Small instances only — large N keeps the existing, tuned behaviour.
    if problem.jobs.len() > 300 {
        return;
    }
    let snapshot = sol.clone();
    let before = sol.summary.cost;
    let mut moved_any = false;

    loop {
        let used: HashSet<usize> = sol.routes.iter().map(|r| r.vehicle_idx).collect();
        let unused: Vec<usize> = (0..problem.vehicles.len())
            .filter(|v| !used.contains(v))
            .collect();
        if unused.is_empty() {
            break;
        }

        // Best (r1, seg_start, seg_len, v2, rest_metrics, seg_metrics, delta).
        let mut best_move: Option<(usize, usize, usize, usize, RouteMetrics, RouteMetrics, f64)> =
            None;
        for r1 in 0..sol.routes.len() {
            let route = &sol.routes[r1];
            let veh1 = &problem.vehicles[route.vehicle_idx];
            let cur = route.metrics.cost;
            let n = route.steps.len();
            if n < 2 {
                continue;
            }
            let max_seg = 3.min(n - 1);
            for seglen in 1..=max_seg {
                for i in 0..=(n - seglen) {
                    let mut rest: Vec<TaskRef> = Vec::with_capacity(n - seglen);
                    rest.extend_from_slice(&route.steps[..i]);
                    rest.extend_from_slice(&route.steps[i + seglen..]);
                    if rest.is_empty() {
                        continue; // emptying a whole route is consolidation, not spread
                    }
                    let m1 = match evaluate_route(problem, matrix, veh1, &rest) {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    let seg = &route.steps[i..i + seglen];
                    for &v2 in &unused {
                        let veh2 = &problem.vehicles[v2];
                        if !seg.iter().all(|t| veh2.has_skills(t.skills(problem))) {
                            continue;
                        }
                        if !seg.iter().all(|t| t.description(problem).allows_vehicle(veh2.id)) {
                            continue;
                        }
                        let m2 = match evaluate_route(problem, matrix, veh2, seg) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };
                        let delta = (m1.cost + m2.cost) - cur;
                        if delta < -1e-9 && best_move.as_ref().map_or(true, |b| delta < b.6) {
                            best_move = Some((r1, i, seglen, v2, m1, m2, delta));
                        }
                    }
                }
            }
        }

        match best_move {
            Some((r1, i, seglen, v2, m1, m2, _)) => {
                let seg: Vec<TaskRef> = sol.routes[r1].steps[i..i + seglen].to_vec();
                let mut rest: Vec<TaskRef> = Vec::with_capacity(sol.routes[r1].steps.len() - seglen);
                rest.extend_from_slice(&sol.routes[r1].steps[..i]);
                rest.extend_from_slice(&sol.routes[r1].steps[i + seglen..]);
                sol.routes[r1].steps = rest;
                sol.routes[r1].metrics = m1;
                sol.routes.push(Route { vehicle_idx: v2, steps: seg, metrics: m2 });
                moved_any = true;
            }
            None => break,
        }
    }

    if moved_any {
        // Let LS re-optimise around the freshly-opened routes, then keep the
        // result only if the whole solution actually got cheaper.
        local_search(problem, matrix, sol, config.max_local_search_passes, granular);
        sol.recompute_summary(problem);
        if sol.summary.cost >= before - 1e-9 {
            *sol = snapshot;
        }
    }
}

fn enforce_groups(sol: &mut Solution, problem: &Problem, matrix: &Matrix, max_per_group: usize) {
    use std::collections::{HashMap, HashSet};
    let keep = max_per_group.max(1);
    let mut served_by_group: HashMap<u32, Vec<usize>> = HashMap::new();
    for r in &sol.routes {
        for s in &r.steps {
            if let TaskRef::Job(ji) = s {
                if let Some(g) = problem.jobs[*ji].group {
                    served_by_group.entry(g).or_default().push(*ji);
                }
            }
        }
    }
    // Hard-trim any group with more than `keep` served members down to `keep`
    // (the search's k-of-N global already steers toward the target; this is the
    // final guarantee of the upper bound). The lower bound can't be enforced by
    // trimming, so it relies on the global penalty during search.
    let mut to_remove: HashSet<usize> = HashSet::new();
    for (_g, mut js) in served_by_group {
        if js.len() > keep {
            js.sort_unstable();
            for &ji in js.iter().skip(keep) {
                to_remove.insert(ji);
            }
        }
    }
    if to_remove.is_empty() {
        return;
    }
    for r in &mut sol.routes {
        let before = r.steps.len();
        r.steps.retain(|s| !matches!(s, TaskRef::Job(ji) if to_remove.contains(ji)));
        if r.steps.len() != before {
            let veh = &problem.vehicles[r.vehicle_idx];
            r.metrics = crate::solution::evaluate_route(problem, matrix, veh, &r.steps)
                .unwrap_or_default();
        }
    }
    for ji in to_remove {
        sol.unassigned.push(TaskRef::Job(ji));
    }
    sol.routes.retain(|r| !r.steps.is_empty());
    sol.recompute_summary(problem);
}

/// Final guaranteed-assignment pass: greedily place any still-unassigned
/// single jobs into the cheapest *feasible* slot — across existing routes
/// AND any unused vehicle — using `evaluate_route`, so capacity, skills,
/// time windows and max-distance/-time are all still honoured. It only ever
/// ADDS assignments, so it cannot make a feasible plan worse; it just stops
/// the ILS from stranding jobs in spare vehicles. A placement that would
/// need an unreachable (sentinel-distance) leg is rejected, so a job with no
/// real path stays unassigned rather than getting an absurd 100 000-km route.
fn repair_unassigned(sol: &mut Solution, problem: &Problem, matrix: &Matrix) {
    use crate::solution::{eval_cache_invalidate, evaluate_route, Route, RouteMetrics};
    // 100 000 km — far beyond any real road leg; flags the sentinel value
    // mpee uses for "no path" without rejecting legitimately long routes.
    const UNREACHABLE_M: i64 = 100_000_000;

    if sol.unassigned.is_empty() {
        return;
    }
    // The solve just filled the per-thread evaluate_route cache; bump the
    // epoch so this pass recomputes feasibility from scratch (avoids any
    // stale/collided cache entry stranding a placeable job).
    eval_cache_invalidate();
    enum Slot {
        Existing(usize, usize),
        NewVehicle(usize),
        /// Replace a route's steps wholesale (used to open a new multi-trip leg).
        Replace(usize, Vec<TaskRef>),
    }

    // Multi-pass: a job whose *solo* round-trip is infeasible (e.g. a one-way
    // snap makes depot→job→depot unreachable) can still be inserted mid-route
    // once another job has opened a vehicle — so retry the leftovers until a
    // full pass places nothing new.
    loop {
    let pending = std::mem::take(&mut sol.unassigned);
    let mut leftovers: Vec<TaskRef> = Vec::new();
    let mut placed_any = false;

    for task in pending {
        // Shipment halves must be inserted as a pickup→delivery pair; leave
        // those to the main solver. This repair only handles standalone jobs.
        if matches!(task, TaskRef::Pickup(_) | TaskRef::Delivery(_)) {
            leftovers.push(task);
            continue;
        }
        let req = task.skills(problem);
        let mut best: Option<(Slot, f64, RouteMetrics)> = None;

        // (a) every position in every existing route.
        for ri in 0..sol.routes.len() {
            let veh = &problem.vehicles[sol.routes[ri].vehicle_idx];
            if !veh.has_skills(req) {
                continue;
            }
            let old = sol.routes[ri].metrics.cost;
            for pos in 0..=sol.routes[ri].steps.len() {
                let mut cand = sol.routes[ri].steps.clone();
                cand.insert(pos, task);
                if let Ok(m) = evaluate_route(problem, matrix, veh, &cand) {
                    if m.distance >= UNREACHABLE_M {
                        continue;
                    }
                    let delta = m.cost - old;
                    if best.as_ref().map_or(true, |b| delta < b.1) {
                        best = Some((Slot::Existing(ri, pos), delta, m));
                    }
                }
            }
            // Multi-trip: open a new trip for this vehicle if it has trips left.
            // A reload resets the load, so a job that won't fit the current trip
            // can ride a fresh one.
            if veh.is_multi_trip() {
                let reloads = sol.routes[ri].steps.iter().filter(|s| s.is_reload()).count();
                if reloads + 1 < veh.max_trips {
                    let mut cand = sol.routes[ri].steps.clone();
                    cand.push(TaskRef::Reload);
                    cand.push(task);
                    if let Ok(m) = evaluate_route(problem, matrix, veh, &cand) {
                        if m.distance < UNREACHABLE_M {
                            let delta = m.cost - old;
                            if best.as_ref().map_or(true, |b| delta < b.1) {
                                best = Some((Slot::Replace(ri, cand), delta, m));
                            }
                        }
                    }
                }
            }
        }
        // (b) one fresh route per currently-unused vehicle.
        let used: std::collections::HashSet<usize> =
            sol.routes.iter().map(|r| r.vehicle_idx).collect();
        for vi in 0..problem.vehicles.len() {
            if used.contains(&vi) {
                continue;
            }
            let veh = &problem.vehicles[vi];
            if !veh.has_skills(req) {
                continue;
            }
            if let Ok(m) = evaluate_route(problem, matrix, veh, &[task]) {
                if m.distance >= UNREACHABLE_M {
                    continue;
                }
                if best.as_ref().map_or(true, |b| m.cost < b.1) {
                    best = Some((Slot::NewVehicle(vi), m.cost, m));
                }
            }
        }

        match best {
            Some((Slot::Existing(ri, pos), _, m)) => {
                sol.routes[ri].steps.insert(pos, task);
                sol.routes[ri].metrics = m;
                placed_any = true;
            }
            Some((Slot::NewVehicle(vi), _, m)) => {
                sol.routes.push(Route { vehicle_idx: vi, steps: vec![task], metrics: m });
                placed_any = true;
            }
            Some((Slot::Replace(ri, steps), _, m)) => {
                sol.routes[ri].steps = steps;
                sol.routes[ri].metrics = m;
                placed_any = true;
            }
            None => leftovers.push(task),
        }
    }

    sol.unassigned = leftovers;
    if !placed_any || sol.unassigned.is_empty() {
        break;
    }
    }
    sol.recompute_summary(problem);
}

/// Build a matrix from whatever the problem provides.
///
/// Order of preference:
/// 1. A matrix already in `problem.matrices[profile]` for the first vehicle's profile.
/// 2. The provided `source` applied to the resolved coordinate list.
pub fn build_matrix(problem: &mut Problem, source: Option<&dyn MatrixSource>) -> Result<Matrix> {
    let profile = problem
        .vehicles
        .first()
        .map(|v| v.profile.clone())
        .unwrap_or_else(|| "car".to_string());
    if let Some(p) = problem.matrices.get(&profile) {
        return Matrix::from_provided(p);
    }
    let coords = resolve_coords(problem);
    if coords.is_empty() {
        return Err(Error::Invalid(
            "no coordinates found and no matrix provided".into(),
        ));
    }
    let src = source.ok_or_else(|| {
        Error::Invalid(
            "problem has no matrix and no MatrixSource was supplied (try haversine)".into(),
        )
    })?;
    src.build(&coords)
}

/// ILS perturbation: remove `frac` of all assigned tasks at random, then
/// reinsert each via cheapest-feasible insertion (probe + full eval). Tasks
/// that can't be placed feasibly land in `unassigned`. LS then drives the
/// result to a (hopefully different) local optimum.
///
/// Random *placement* would be cheaper but can leave the solution
/// infeasible; LS then can't recover because its delta-cost reasoning
/// assumes a feasible base. Feasible-cheapest is the safe choice.
///
/// `pub` so the population-polish in `crate::population` can reuse the
/// same destroy-and-repair logic.
/// Small perturbation for a deep ILS trajectory (PyVRP-style), WITH a
/// route-opening move. Applies `n_moves` random steps; each is either a single
/// relocation OR (with probability `open_p`, when a spare vehicle exists) a
/// **route opening**: lift a small contiguous cluster (1–3 stops) onto a fresh
/// unused vehicle. Opening a cluster (not a lone stop) gives the new route enough
/// substance that local search won't trivially re-absorb it — the move that lets
/// the search reach a higher-route basin (our diagnosed R2/RC2 gap). Greedy LS
/// alone can't take this (it's temporarily worse); pair it with LAHC acceptance.
/// Returns the set of tasks whose routes were modified (every task in a route
/// the perturbation rewrote). The ILS loop feeds this to `local_search_seeded`
/// so re-convergence only re-probes the perturbed region — the same
/// route-level invalidation rule the LS itself applies after a move.
pub fn perturb_small<R: rand::Rng>(
    sol: &mut Solution,
    n_moves: usize,
    open_p: f64,
    rng: &mut R,
    problem: &Problem,
    matrix: &Matrix,
) -> std::collections::HashSet<crate::solution::TaskRef> {
    use crate::solution::{evaluate_route, Route, RouteMetrics, TaskRef};
    let mut touched: std::collections::HashSet<TaskRef> = std::collections::HashSet::new();
    for _ in 0..n_moves {
        let nonempty: Vec<usize> = (0..sol.routes.len())
            .filter(|&r| !sol.routes[r].steps.is_empty())
            .collect();
        if nonempty.is_empty() {
            return touched;
        }
        let r1 = nonempty[rng.gen_range(0..nonempty.len())];
        let len1 = sol.routes[r1].steps.len();
        let i = rng.gen_range(0..len1);

        // Spare vehicles (recomputed each move, since opening consumes one).
        let used: std::collections::HashSet<usize> =
            sol.routes.iter().map(|r| r.vehicle_idx).collect();
        let unused: Vec<usize> = (0..problem.vehicles.len())
            .filter(|v| !used.contains(v))
            .collect();

        if !unused.is_empty() && len1 >= 2 && rng.gen_bool(open_p) {
            // Route-opening: move a contiguous cluster [i, i+seg) to a new vehicle.
            let seg_len = rng.gen_range(1..=3.min(len1 - 1));
            let start = i.min(len1 - seg_len);
            let seg: Vec<TaskRef> = sol.routes[r1].steps[start..start + seg_len].to_vec();
            if !seg.iter().all(|t| matches!(t, TaskRef::Job(_))) {
                continue; // keep shipment halves where they are
            }
            let v2 = unused[rng.gen_range(0..unused.len())];
            let veh2 = &problem.vehicles[v2];
            if !seg.iter().all(|t| veh2.has_skills(t.skills(problem))) {
                continue;
            }
            let mut s1: Vec<TaskRef> = sol.routes[r1].steps.clone();
            s1.drain(start..start + seg_len);
            let veh1 = &problem.vehicles[sol.routes[r1].vehicle_idx];
            let m1 = if s1.is_empty() {
                Ok(RouteMetrics::default())
            } else {
                evaluate_route(problem, matrix, veh1, &s1)
            };
            if let (Ok(m1), Ok(m2)) = (m1, evaluate_route(problem, matrix, veh2, &seg)) {
                touched.extend(s1.iter().copied());
                touched.extend(seg.iter().copied());
                sol.routes[r1].steps = s1;
                sol.routes[r1].metrics = m1;
                sol.routes.push(Route { vehicle_idx: v2, steps: seg, metrics: m2 });
            }
            continue;
        }

        // Plain single-stop relocation to a random existing route.
        let task = sol.routes[r1].steps[i];
        if !matches!(task, TaskRef::Job(_)) {
            continue;
        }
        let r2 = rng.gen_range(0..sol.routes.len());
        let mut s1: Vec<TaskRef> = sol.routes[r1].steps.clone();
        s1.remove(i);
        if r1 == r2 {
            let pos = rng.gen_range(0..=s1.len());
            let mut s = s1;
            s.insert(pos, task);
            let veh = &problem.vehicles[sol.routes[r1].vehicle_idx];
            if let Ok(m) = evaluate_route(problem, matrix, veh, &s) {
                touched.extend(s.iter().copied());
                sol.routes[r1].steps = s;
                sol.routes[r1].metrics = m;
            }
        } else {
            let mut s2: Vec<TaskRef> = sol.routes[r2].steps.clone();
            let pos = rng.gen_range(0..=s2.len());
            s2.insert(pos, task);
            let veh1 = &problem.vehicles[sol.routes[r1].vehicle_idx];
            let veh2 = &problem.vehicles[sol.routes[r2].vehicle_idx];
            if let (Ok(m1), Ok(m2)) = (
                evaluate_route(problem, matrix, veh1, &s1),
                evaluate_route(problem, matrix, veh2, &s2),
            ) {
                touched.extend(s1.iter().copied());
                touched.extend(s2.iter().copied());
                sol.routes[r1].steps = s1;
                sol.routes[r1].metrics = m1;
                sol.routes[r2].steps = s2;
                sol.routes[r2].metrics = m2;
            }
        }
    }
    sol.routes.retain(|r| !r.steps.is_empty());
    sol.recompute_summary(problem);
    touched
}

pub fn kick<R: rand::Rng>(
    sol: &mut Solution,
    frac: f64,
    rng: &mut R,
    problem: &Problem,
    matrix: &Matrix,
) {
    use crate::eval::{precompute, try_insert_single};
    use crate::solution::evaluate_route;

    let mut assigned: Vec<(usize, usize)> = Vec::new();
    for (r, route) in sol.routes.iter().enumerate() {
        for i in 0..route.steps.len() {
            assigned.push((r, i));
        }
    }
    if assigned.is_empty() { return; }

    let n_kick = ((assigned.len() as f64) * frac).round() as usize;
    if n_kick == 0 { return; }
    assigned.shuffle(rng);
    assigned.truncate(n_kick);

    // Sort high→low step index so removals don't shift earlier ones.
    assigned.sort_by(|a, b| (b.0, b.1).cmp(&(a.0, a.1)));

    let mut pulled: Vec<TaskRef> = Vec::with_capacity(n_kick);
    for (r, i) in &assigned {
        if let Some(route) = sol.routes.get_mut(*r) {
            if *i < route.steps.len() {
                pulled.push(route.steps.remove(*i));
            }
        }
    }

    // Refresh metrics on every route (loads/timing changed).
    for r in 0..sol.routes.len() {
        let veh = &problem.vehicles[sol.routes[r].vehicle_idx];
        if let Ok(m) = evaluate_route(problem, matrix, veh, &sol.routes[r].steps) {
            sol.routes[r].metrics = m;
        }
    }
    sol.routes.retain(|r| !r.steps.is_empty());

    // Build precomp for each surviving route.
    let mut precomps: Vec<Option<crate::eval::RoutePrecomp>> = (0..sol.routes.len())
        .map(|r| precompute(problem, matrix, &problem.vehicles[sol.routes[r].vehicle_idx], r, &sol.routes[r].steps))
        .collect();

    pulled.shuffle(rng);

    // Regret-3 reinsertion: for each pulled task, track best, 2nd-best, AND
    // 3rd-best insertion across all (route, pos). Regret = (2nd + 3rd) − 2·best.
    // This generalizes regret-2 (which only used 2nd-best). Capturing the
    // 3rd-best alternative gives a better signal about how "trapped" a task
    // is — if both 2nd and 3rd are far from best, the task is highly
    // committed to its top choice and should be placed first.
    //
    // Empirically: regret-3 ≥ regret-2 ≥ regret-1 (= cheapest-feasible) on
    // CVRPTW (Potvin & Rousseau 1993).
    while !pulled.is_empty() {
        let mut scored: Vec<(f64, usize, usize, usize)> = Vec::new();
        for (idx, &t) in pulled.iter().enumerate() {
            let mut best: Option<(usize, usize, f64)> = None;
            let mut second: f64 = f64::INFINITY;
            let mut third: f64 = f64::INFINITY;
            for r in 0..sol.routes.len() {
                let Some(pre) = precomps[r].as_ref() else { continue; };
                let veh = &problem.vehicles[sol.routes[r].vehicle_idx];
                if !veh.has_skills(t.skills(problem)) { continue; }
                let positions = sol.routes[r].steps.len() + 2;
                for pos in 1..positions {
                    if let Some(d) = try_insert_single(pre, problem, matrix, veh, pos, t) {
                        match best {
                            None => best = Some((r, pos, d)),
                            Some((_, _, bd)) if d < bd => {
                                third = second;
                                second = bd;
                                best = Some((r, pos, d));
                            }
                            Some(_) if d < second => {
                                third = second;
                                second = d;
                            }
                            Some(_) if d < third => third = d,
                            _ => {}
                        }
                    }
                }
            }
            if let Some((r, pos, b)) = best {
                // Regret-3: (2nd + 3rd) − 2·best. Falls back to regret-2 when
                // 3rd is infinity (only ≤2 feasible insertions exist).
                let r2 = if second.is_finite() { second - b } else { 1e6 };
                let r3 = if third.is_finite() { third - b } else { r2 };
                let regret = r2 + r3;
                scored.push((regret, idx, r, pos));
            }
        }

        if scored.is_empty() {
            // No task is feasibly insertable anywhere — drop remaining.
            for t in pulled.drain(..) { sol.unassigned.push(t); }
            break;
        }

        // Pick highest-regret task.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let (_, idx, r, pos) = scored[0];
        let t = pulled.remove(idx);
        let mut cand = sol.routes[r].steps.clone();
        cand.insert(pos - 1, t);
        let veh = &problem.vehicles[sol.routes[r].vehicle_idx];
        if let Ok(m) = evaluate_route(problem, matrix, veh, &cand) {
            sol.routes[r].steps = cand;
            sol.routes[r].metrics = m;
            precomps[r] = precompute(problem, matrix, veh, r, &sol.routes[r].steps);
        } else {
            sol.unassigned.push(t);
        }
    }

    sol.recompute_summary(problem);
}

