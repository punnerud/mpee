//! Solution representation and route evaluation.
//!
//! A `Solution` is a set of `Route`s plus a list of unassigned tasks. Each
//! route is a sequence of `TaskRef`s served by one vehicle. Route timing,
//! load, cost, and feasibility are computed on demand from the problem.

use serde::{Deserialize, Serialize};

use crate::matrix::Matrix;
use crate::problem::{Cost, Job, JobKind, Problem, Time, TimeWindow, Vehicle};

/// One visit in a route. `Job(i)` references `problem.jobs[i]`; the pickup/
/// delivery variants reference `problem.shipments[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskRef {
    Job(usize),
    Pickup(usize),
    Delivery(usize),
    /// Return-to-depot reload marker for a multi-trip vehicle. Carries no job;
    /// `description` yields a neutral sentinel so step-iterating heuristics
    /// never panic, while the authoritative walkers (`evaluate_route` / io)
    /// handle it against the vehicle's depot.
    Reload,
}

/// Neutral job returned for `TaskRef::Reload` — no location, no demand, no
/// skills — so heuristics that read a step's job don't panic. The reload's real
/// effect (depot leg + load reset) is applied by the route evaluator.
fn reload_sentinel() -> &'static Job {
    static S: std::sync::OnceLock<Job> = std::sync::OnceLock::new();
    S.get_or_init(|| Job {
        id: 0,
        location: crate::problem::Location { coord: None, index: None },
        kind: Default::default(),
        service: 0,
        setup: 0,
        release: 0,
        delivery: vec![],
        pickup: vec![],
        skills: vec![],
        allowed_vehicles: None,
        priority: 0,
        time_windows: vec![],
        prize: crate::problem::DEFAULT_PRIZE,
        disjunction_penalty: None,
        group: None,
        description: None,
    })
}

impl TaskRef {
    pub fn description<'a>(&self, p: &'a Problem) -> &'a Job {
        match self {
            TaskRef::Job(i) => &p.jobs[*i],
            TaskRef::Pickup(i) => &p.shipments[*i].pickup,
            TaskRef::Delivery(i) => &p.shipments[*i].delivery,
            TaskRef::Reload => reload_sentinel(),
        }
    }
    /// Whether this step is a depot reload (multi-trip boundary).
    pub fn is_reload(&self) -> bool {
        matches!(self, TaskRef::Reload)
    }
    pub fn kind(&self) -> JobKind {
        match self {
            TaskRef::Job(_) | TaskRef::Reload => JobKind::Single,
            TaskRef::Pickup(_) => JobKind::Pickup,
            TaskRef::Delivery(_) => JobKind::Delivery,
        }
    }
    pub fn skills<'a>(&self, p: &'a Problem) -> &'a [u32] {
        match self {
            TaskRef::Job(i) => &p.jobs[*i].skills,
            TaskRef::Reload => &[],
            TaskRef::Pickup(i) | TaskRef::Delivery(i) => {
                let s = &p.shipments[*i];
                if !s.skills.is_empty() { &s.skills } else { &s.pickup.skills }
            }
        }
    }
    pub fn priority(&self, p: &Problem) -> u8 {
        match self {
            TaskRef::Job(i) => p.jobs[*i].priority,
            TaskRef::Reload => 0,
            TaskRef::Pickup(i) | TaskRef::Delivery(i) => p.shipments[*i].priority,
        }
    }
}

/// Computed metrics for a single route. All times are in seconds.
///
/// `cost` is the single aggregated scalar the local search minimises. The
/// `cost_*` fields below decompose that scalar into named components purely for
/// reporting / DSL access — they always sum to `cost` (modulo the global
/// objective-weight multiplier the solver applies *outside* `evaluate_route`):
///   `cost == cost_travel + cost_span + cost_custom`.
///
/// NOTE: within a single solve pass this is *weighted scalarization*, not true
/// lexicographic optimisation — the LS loop minimises one number; the
/// components only let you shape and inspect that number.
///
/// For a genuine count-then-cost ordering, set
/// `SolverConfig::objective_mode = ObjectiveMode::Lexicographic { levels:
/// vec![LexObjective::Vehicles, LexObjective::Cost] }`: that runs an N-level
/// driver (level 0 minimises vehicle count V*, level 1 minimises cost with
/// `max_vehicles(V*)` pinned as a hard cap), warm-started level-to-level. The
/// stack accepts any ordering of {Vehicles, UnassignedCount, Cost, Makespan,
/// Distance}. BEST-EFFORT caveat: each level's achieved value is the
/// metaheuristic's best-found, not a proven optimum.
#[derive(Debug, Clone, Copy, Default)]
pub struct RouteMetrics {
    pub start_time: Time,
    pub end_time: Time,
    pub travel_time: Time,
    pub service_time: Time,
    pub waiting_time: Time,
    pub setup_time: Time,
    pub distance: i64,
    /// Aggregated scalar minimised by local search (the sum of the components).
    pub cost: Cost,
    /// Travel/time/distance + fixed component (the historical cost basis).
    pub cost_travel: Cost,
    /// Route-span component (`span_cost × route span`). 0 unless `span_cost` set.
    pub cost_span: Cost,
    /// Custom per-route constraint penalty (from `constraint::apply`). 0 unless
    /// a custom/DSL constraint added one.
    pub cost_custom: Cost,
    /// Number of driver breaks actually scheduled on this route.
    pub break_count: u32,
    /// Total seconds spent on driver breaks on this route.
    pub break_duration: Time,
    /// Soft-mode time-window lateness on this route (seconds of time warp).
    /// Always `0.0` outside penalty-managed soft mode (see [`set_soft_penalties`]).
    pub tw_excess: f64,
    /// Soft-mode peak capacity overload on this route (load units). `0.0` in hard mode.
    pub load_excess: f64,
    /// Soft-mode duration/travel-time overrun on this route (seconds). `0.0` in hard mode.
    pub dur_excess: f64,
}

impl RouteMetrics {
    /// Total raw soft-constraint violation. `0.0` ⇒ this route is hard-feasible.
    #[inline]
    pub fn violation(&self) -> f64 {
        self.tw_excess + self.load_excess + self.dur_excess
    }
}

/// One route in the final solution.
#[derive(Debug, Clone)]
pub struct Route {
    pub vehicle_idx: usize,
    pub steps: Vec<TaskRef>,
    pub metrics: RouteMetrics,
}

#[derive(Debug, Clone, Default)]
pub struct Summary {
    pub cost: Cost,
    pub routes: usize,
    pub unassigned: usize,
    pub travel_time: Time,
    pub service_time: Time,
    pub waiting_time: Time,
    pub distance: i64,
}

#[derive(Debug, Clone, Default)]
pub struct Solution {
    pub routes: Vec<Route>,
    pub unassigned: Vec<TaskRef>,
    pub summary: Summary,
}

/// Output step in Vroom-style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepKind {
    Start,
    Job,
    Pickup,
    Delivery,
    Break,
    Reload,
    End,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    #[serde(rename = "type")]
    pub kind: StepKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location_index: Option<usize>,
    pub arrival: Time,
    #[serde(default)]
    pub service: Time,
    #[serde(default)]
    pub waiting_time: Time,
    #[serde(default)]
    pub setup: Time,
    pub load: Vec<i64>,
    pub distance: i64,
    /// Soft time-window violation at this stop: how late (seconds) service
    /// started past the job's last time-window end. 0 = on time. Non-zero only
    /// when soft constraints let the search accept a late stop (so callers can
    /// surface "served, but N s late" instead of just dropping the job).
    #[serde(default, skip_serializing_if = "is_zero_time")]
    pub lateness: Time,
}

/// serde helper: omit zero-valued lateness/time-warp so a clean (on-time)
/// solution serialises exactly as before.
pub fn is_zero_time(t: &Time) -> bool {
    *t == 0
}

/// serde helper: omit zero counts.
pub fn is_zero_usize(n: &usize) -> bool {
    *n == 0
}

// Per-thread scratch space reused by `evaluate_route` so the hot path
// doesn't allocate. `load` is grown to the problem's capacity dimension on
// first use and left there.
thread_local! {
    static SCRATCH_LOAD: std::cell::RefCell<Vec<i64>> = std::cell::RefCell::new(Vec::new());
}

// Per-thread LRU cache for evaluate_route. Keyed by (epoch, vehicle.id,
// step-hash). The epoch is a global atomic that the solver bumps at the
// start of each solve — so even rayon worker threads with persistent
// thread-locals correctly invalidate when a new (problem, matrix) arrives.
const ROUTE_EVAL_CACHE_CAP: usize = 4096;

static EVAL_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
    static EVAL_CACHE: std::cell::RefCell<EvalCache> = std::cell::RefCell::new(EvalCache::new(ROUTE_EVAL_CACHE_CAP));
}

struct EvalCache {
    cap: usize,
    epoch: u64,
    map: std::collections::HashMap<(usize, u64, u64), Result<RouteMetrics, &'static str>>,
}

impl EvalCache {
    fn new(cap: usize) -> Self {
        Self { cap, epoch: 0, map: std::collections::HashMap::with_capacity(cap) }
    }
}

/// Bump the global epoch. Existing thread-local caches will treat their
/// entries as stale on the next lookup. Must be called at the start of
/// every fresh solve so tests / sequential solves don't see each other's
/// cached metrics.
pub fn eval_cache_invalidate() {
    EVAL_EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// Hash a step sequence stable-enough for cache keying. A collision here
/// would only return a wrong-but-consistent metric for one Solution; the
/// outer LS bookkeeping invalidates routes after every accepted move.
#[inline]
fn hash_steps(steps: &[TaskRef]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    steps.hash(&mut h);
    h.finish()
}

/// Walk a route forward and either compute its metrics or report the first
/// constraint violation seen.
///
/// Hot path notes:
///   - The `load` vector is pulled from a thread-local scratch buffer to
///     avoid per-call heap allocation. We size it once to `dim` and reuse.
///   - All capacity accesses are direct slice reads with `.get(i).unwrap_or(&0)`
///     — no `pad()` allocations.
///   - Pickup-before-delivery precedence is tracked with a small `Vec<bool>`
///     keyed by shipment index; for instances without shipments this skips
///     the check entirely.
///   - Results are memoized in a per-thread LRU. Repeated evaluations of
///     the same (vehicle_idx, step-sequence) return the cached metrics.
pub fn evaluate_route(
    problem: &Problem,
    matrix: &Matrix,
    vehicle: &Vehicle,
    steps: &[TaskRef],
) -> Result<RouteMetrics, &'static str> {
    let cur_epoch = EVAL_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
    // Include problem pointer so concurrent solves (e.g. parallel tests) on
    // distinct Problem instances don't share cache entries when their
    // vehicle.id + step hash happen to collide.
    let prob_id = problem as *const Problem as usize;
    // Fold the soft generation into the step hash so a soft-penalised result is
    // never served to a hard query (or across a λ update). `0` in hard mode ⇒
    // the key — and behaviour — is byte-identical to before.
    let key = (prob_id, vehicle.id, hash_steps(steps) ^ soft_generation());

    // Lookup. Reset cache if epoch has bumped since we last touched it.
    if let Some(hit) = EVAL_CACHE.with(|cell| {
        let mut c = cell.borrow_mut();
        if c.epoch != cur_epoch {
            c.map.clear();
            c.epoch = cur_epoch;
            None
        } else {
            c.map.get(&key).copied()
        }
    }) {
        return hit;
    }

    let result = SCRATCH_LOAD.with(|cell| {
        let mut buf = cell.borrow_mut();
        evaluate_route_with_buf(problem, matrix, vehicle, steps, &mut buf)
    });
    EVAL_CACHE.with(|cell| {
        let mut c = cell.borrow_mut();
        if c.map.len() >= c.cap {
            c.map.clear();
        }
        c.map.insert(key, result);
    });
    result
}

/// A matrix leg at or beyond this value means there was no path between the
/// two points (the routing engine's "unreachable" sentinel is ~2.1e9). Such a
/// leg makes the route infeasible — it is never a real, merely-long one
/// (no road leg approaches 100 000 km / ~3 years of travel time).
const UNREACHABLE_LEG: i64 = 100_000_000;

/// A plain job is a *backhaul* (collect-only) stop when it has a pickup amount
/// and no delivery. Vroom-style routing requires every linehaul (delivery)
/// stop to be served before any backhaul on the same route.
#[inline]
fn is_backhaul(job: &Job) -> bool {
    !job.pickup.is_empty() && job.delivery.is_empty()
}

#[inline]
fn evaluate_route_with_buf(
    problem: &Problem,
    matrix: &Matrix,
    vehicle: &Vehicle,
    steps: &[TaskRef],
    load: &mut Vec<i64>,
) -> Result<RouteMetrics, &'static str> {
    let dim = problem.capacity_dim().max(vehicle.capacity.len()).max(1);

    // Resize-and-zero the load scratch.
    if load.len() < dim {
        load.resize(dim, 0);
    }
    for v in load[..dim].iter_mut() { *v = 0; }

    // Penalty-managed soft constraints (off by default → all three excess
    // accumulators stay 0 and every check below behaves as a hard reject).
    let soft = soft_active();
    let sw = if soft { soft_weights() } else { SoftWeights { tw: 0.0, load: 0.0, dur: 0.0 } };
    let mut tw_late: i64 = 0; // seconds of time-window lateness (time warp)
    let mut load_excess: i64 = 0; // peak units of capacity overload
    let mut dur_excess: i64 = 0; // seconds over shift / max travel time

    // Initial load = sum of the FIRST trip's single-job deliveries (shipments
    // are picked up en-route). For multi-trip routes the load is reset to the
    // next trip's deliveries at each `Reload`, so only sum up to the first one.
    for s in steps {
        if let TaskRef::Reload = s { break; }
        if let TaskRef::Job(_) = s {
            let j = s.description(problem);
            for (i, &v) in j.delivery.iter().enumerate() {
                if i < dim { load[i] += v; }
            }
        }
    }
    // Capacity check at depot start.
    {
        let mut over = 0i64;
        for (i, &cap_i) in vehicle.capacity.iter().enumerate() {
            if i < dim && load[i] > cap_i { over += load[i] - cap_i; }
        }
        if over > 0 {
            if soft { load_excess = load_excess.max(over); }
            else { return Err("capacity exceeded at route start"); }
        }
    }

    // Pickup-before-delivery precedence: only allocate the bitset if we see
    // any shipment task. For pure-CVRP instances this stays empty → free.
    let mut pickups_seen: Vec<bool> = Vec::new();
    for s in steps {
        if !vehicle.has_skills(s.skills(problem)) {
            return Err("vehicle missing required skill");
        }
        if !s.description(problem).allows_vehicle(vehicle.id) {
            return Err("job not allowed on this vehicle");
        }
        match s {
            TaskRef::Pickup(i) => {
                if pickups_seen.len() <= *i { pickups_seen.resize(*i + 1, false); }
                pickups_seen[*i] = true;
            }
            TaskRef::Delivery(i) => {
                if !pickups_seen.get(*i).copied().unwrap_or(false) {
                    return Err("delivery before pickup");
                }
            }
            TaskRef::Job(_) | TaskRef::Reload => {}
        }
    }

    // First-class precedence: for each (a, b) where both job ids appear on THIS
    // route, a must precede b. Guarded by a non-empty list, so pure-CVRP routes
    // pay nothing. O(pairs × route) — precedence lists are small in practice.
    if !problem.precedence.is_empty() {
        let pos_of = |job_id: u64| -> Option<usize> {
            steps.iter().position(|s| matches!(s, TaskRef::Job(_)) && s.description(problem).id == job_id)
        };
        for &(a, b) in &problem.precedence {
            if let (Some(pa), Some(pb)) = (pos_of(a), pos_of(b)) {
                if pa > pb {
                    return Err("precedence violated (a after b on the same route)");
                }
            }
        }
    }

    let vw = vehicle.time_window();
    let speed = vehicle.speed_factor.max(0.01);

    let start_idx = vehicle
        .start
        .as_ref()
        .and_then(|l| l.index)
        .or_else(|| vehicle.end.as_ref().and_then(|l| l.index));
    let end_idx = vehicle
        .end
        .as_ref()
        .and_then(|l| l.index)
        .or(start_idx);

    let mut t: Time = vw.start;
    let mut prev_idx: Option<usize> = start_idx;
    // Custom-dimension arcs (P5): collected only when a dimension is registered
    // (single relaxed atomic load otherwise → free, behaviour byte-identical).
    // Each entry is the arc (from_loc, to_loc) with the arrival time at `to`, fed
    // to `dimension::accumulate` after the walk. The leading start-depot position
    // has no incoming arc (it is position 0 in the cumul vector).
    let track_dims = crate::dimension::has_dimensions();
    let mut dim_arcs: Vec<crate::dimension::Arc2> = Vec::new();
    let mut travel_time: Time = 0;
    let mut service_time: Time = 0;
    let mut waiting_time: Time = 0;
    let mut setup_time: Time = 0;
    let mut distance: i64 = 0;
    let mut tasks_count: usize = 0;
    // Backhaul ordering: once a collect-only stop is served, no further
    // delivery stop may follow. Free for pure-CVRP routes (just a bool).
    let mut seen_backhaul = false;
    // Driver breaks are taken in input order, greedily at the first open
    // window. `break_idx` is the next break still to schedule.
    let breaks = &vehicle.breaks;
    let mut break_idx = 0usize;
    // Track how many breaks were scheduled and their total duration so finished
    // routes can expose them (used by the DSL `route.has_break` etc.).
    let mut break_count: u32 = 0;
    let mut break_duration: Time = 0;
    // Multi-trip: shipments may not be carried across a depot reload.
    let mut open_shipments: i32 = 0;

    for (k, s) in steps.iter().enumerate() {
        // Multi-trip reload: close the current trip back at the depot, reset the
        // load to the next trip's deliveries, and depart the depot again. Time,
        // travel and distance accumulate across the whole shift.
        if let TaskRef::Reload = s {
            if open_shipments != 0 {
                return Err("reload while a shipment is still on board");
            }
            if let (Some(p), Some(d)) = (prev_idx, start_idx) {
                let raw = matrix.duration(p, d);
                if raw as i64 >= UNREACHABLE_LEG {
                    return Err("unreachable leg (no road to depot for reload)");
                }
                let dur = ((raw as f64) * speed).round() as i64;
                t += dur;
                travel_time += dur;
                distance += matrix.distance(p, d);
            }
            // Reset load to the next trip's single-job deliveries.
            for v in load[..dim].iter_mut() { *v = 0; }
            for ns in &steps[k + 1..] {
                if let TaskRef::Reload = ns { break; }
                if let TaskRef::Job(_) = ns {
                    let nj = ns.description(problem);
                    for (i, &v) in nj.delivery.iter().enumerate() {
                        if i < dim { load[i] += v; }
                    }
                }
            }
            {
                let mut over = 0i64;
                for (i, &cap_i) in vehicle.capacity.iter().enumerate() {
                    if i < dim && load[i] > cap_i { over += load[i] - cap_i; }
                }
                if over > 0 {
                    if soft { load_excess = load_excess.max(over); }
                    else { return Err("capacity exceeded after reload"); }
                }
            }
            seen_backhaul = false;
            prev_idx = start_idx;
            continue;
        }

        let job = s.description(problem);
        let here = job.location.index.ok_or("job location missing matrix index")?;

        match s {
            TaskRef::Pickup(_) => open_shipments += 1,
            TaskRef::Delivery(_) => open_shipments -= 1,
            _ => {}
        }

        if let TaskRef::Job(_) = s {
            if is_backhaul(job) {
                seen_backhaul = true;
            } else if !job.delivery.is_empty() && seen_backhaul {
                return Err("linehaul after backhaul");
            }
        }

        let mut arc_dur: i64 = 0;
        let mut arc_dist: i64 = 0;
        if let Some(p) = prev_idx {
            let raw = matrix.duration(p, here);
            // A sentinel-valued leg means "no road between these points"; such
            // a route is infeasible, never a real (if long) one.
            if raw as i64 >= UNREACHABLE_LEG {
                return Err("unreachable leg (no road between stops)");
            }
            let dur = ((raw as f64) * speed).round() as i64;
            arc_dur = dur;
            arc_dist = matrix.distance(p, here);
            t += dur;
            travel_time += dur;
            distance += arc_dist;
        }

        // Custom-dimension arc: from the previous node to `here`, with `t` as the
        // physical arrival time (after travel, before setup/service). Recorded
        // for every step so the cumul vector has one entry per route position
        // (start depot at index 0, then one per visited step), matching how the
        // DSL indexes `route.<dim>[k]`. Skipped when there is no incoming node.
        // Distance/duration are the physical arc cost, threaded into `ArcCtx`.
        if track_dims {
            if let Some(p) = prev_idx {
                dim_arcs.push(crate::dimension::Arc2 {
                    from: p, to: here, arrival: t, distance: arc_dist, duration: arc_dur,
                });
            } else {
                // No start location: treat the first stop as a self-arc so the
                // cumul still has a position-1 entry (callback decides the delta).
                dim_arcs.push(crate::dimension::Arc2 {
                    from: here, to: here, arrival: t, distance: 0, duration: 0,
                });
            }
        }

        let do_setup = match prev_idx {
            Some(p) => p != here && job.setup > 0,
            None => job.setup > 0,
        };
        if do_setup {
            t += job.setup;
            setup_time += job.setup;
        }

        // Release time: service may not begin before `release`; wait if early.
        // Default 0 ⇒ this branch never fires (no behavior change).
        if t < job.release {
            waiting_time += job.release - t;
            t = job.release;
        }

        let chosen_tw = match pick_time_window(&job.time_windows, t) {
            Some(w) => w,
            // Past every window: hard mode rejects; soft mode targets the latest
            // window and charges the lateness past its end as time warp.
            None if soft => latest_window(&job.time_windows),
            None => return Err("time window missed"),
        };
        if t < chosen_tw.start {
            waiting_time += chosen_tw.start - t;
            t = chosen_tw.start;
        }
        if t > chosen_tw.end {
            if soft { tw_late += t - chosen_tw.end; }
            else { return Err("arrived after time window end"); }
        }

        // Apply load change in place — no allocations.
        match s {
            TaskRef::Job(_) => {
                for (i, &v) in job.delivery.iter().enumerate() {
                    if i < dim { load[i] -= v; }
                }
                for (i, &v) in job.pickup.iter().enumerate() {
                    if i < dim { load[i] += v; }
                }
            }
            TaskRef::Pickup(i) => {
                let s = &problem.shipments[*i];
                let amt = if !s.amount.is_empty() { &s.amount } else { &s.pickup.pickup };
                for (i, &v) in amt.iter().enumerate() {
                    if i < dim { load[i] += v; }
                }
            }
            TaskRef::Delivery(i) => {
                let s = &problem.shipments[*i];
                let amt = if !s.amount.is_empty() { &s.amount } else { &s.pickup.pickup };
                for (i, &v) in amt.iter().enumerate() {
                    if i < dim { load[i] -= v; }
                }
            }
            TaskRef::Reload => {} // handled at the top of the loop
        }
        for i in 0..dim {
            if load[i] < 0 { return Err("negative load (over-delivery)"); }
        }
        {
            let mut over = 0i64;
            for (i, &cap_i) in vehicle.capacity.iter().enumerate() {
                if i < dim && load[i] > cap_i { over += load[i] - cap_i; }
            }
            if over > 0 {
                if soft { load_excess = load_excess.max(over); }
                else { return Err("capacity exceeded mid-route"); }
            }
        }

        t += job.service;
        service_time += job.service;

        // Take any due breaks whose chosen window is already open at `t`.
        // Break time pushes the timeline (and thus end_time / vehicle window /
        // later job windows) but is not travel — so `travel_time` is untouched.
        while break_idx < breaks.len() {
            let br = &breaks[break_idx];
            let tw = pick_time_window(&br.time_windows, t).ok_or("break time window missed")?;
            if t < tw.start { break; }
            t += br.service;
            break_count += 1;
            break_duration += br.service;
            break_idx += 1;
        }

        prev_idx = Some(here);
        tasks_count += 1;
        if let Some(max) = vehicle.max_tasks {
            if tasks_count > max { return Err("max_tasks exceeded"); }
        }
    }

    // Final leg back to depot.
    if let (Some(p), Some(e)) = (prev_idx, end_idx) {
        let raw = matrix.duration(p, e);
        if raw as i64 >= UNREACHABLE_LEG {
            return Err("unreachable leg (no road back to depot)");
        }
        let dur = ((raw as f64) * speed).round() as i64;
        let dist = matrix.distance(p, e);
        t += dur;
        travel_time += dur;
        distance += dist;
        // Final custom-dimension arc back to the end depot.
        if track_dims {
            dim_arcs.push(crate::dimension::Arc2 {
                from: p, to: e, arrival: t, distance: dist, duration: dur,
            });
        }
    }

    // Any breaks not yet taken must still fit before the vehicle's day ends;
    // wait for the window to open if needed, else the route is infeasible.
    while break_idx < breaks.len() {
        let br = &breaks[break_idx];
        let tw = pick_time_window(&br.time_windows, t).ok_or("break time window missed")?;
        if t < tw.start {
            waiting_time += tw.start - t;
            t = tw.start;
        }
        if t > tw.end {
            return Err("break time window missed");
        }
        t += br.service;
        break_count += 1;
        break_duration += br.service;
        break_idx += 1;
    }

    if t > vw.end {
        if soft { dur_excess += t - vw.end; }
        else { return Err("route ends after vehicle time window"); }
    }
    if let Some(max) = vehicle.max_travel_time {
        if travel_time > max {
            if soft { dur_excess += travel_time - max; }
            else { return Err("max_travel_time exceeded"); }
        }
    }
    // max_distance stays HARD even in soft mode (we soften time + load + duration,
    // not distance — a road-distance cap is a physical, not a temporal, limit).
    if let Some(max) = vehicle.max_distance {
        if distance > max {
            return Err("max_distance exceeded");
        }
    }

    let route_dur = t - vw.start;

    // --- Weighted-scalarization cost shaping --------------------------------
    // The components below are summed into a single `cost` scalar; the LS loop
    // is UNCHANGED — it still minimises that one number. This is NOT true
    // lexicographic multi-objective optimisation (which would need a two-phase
    // count-then-cost search); the weights merely shape the scalar.
    //
    // Defaults reproduce the historical cost exactly:
    //   time_weight     = 1.0  → keeps the `travel_time × per_hour/3600` term
    //   distance_weight = 0.0  → distance does not enter the cost
    //   span_cost       = 0.0  → no span term
    let cost_travel = vehicle.fixed
        + (travel_time as f64) * (vehicle.per_hour / 3600.0).max(0.0) * vehicle.time_weight
        + (distance as f64) * vehicle.distance_weight
        // tiny tiebreaker per service second so longer service still costs:
        + (service_time as f64) * 1e-6;
    // Span cost: charged per second of total shift span (end − start), which
    // includes waiting and breaks, not just travel. 0 by default.
    let cost_span = (route_dur as f64) * vehicle.span_cost.max(0.0);

    let mut metrics = RouteMetrics {
        start_time: vw.start,
        end_time: t,
        travel_time,
        service_time,
        waiting_time,
        setup_time,
        distance,
        cost: cost_travel + cost_span,
        cost_travel,
        cost_span,
        cost_custom: 0.0,
        break_count,
        break_duration,
        tw_excess: 0.0,
        load_excess: 0.0,
        dur_excess: 0.0,
    };

    // Penalty-managed soft constraints: charge accumulated time warp, load
    // excess and duration excess into the route cost (so penalised LS prefers
    // feasible routes but may pass through infeasible ones), and record each
    // violation so the search can tell a feasible route from a penalised one and
    // adapt each weight independently. No-op when soft mode is off (all three
    // accumulators are 0).
    if soft {
        let pen = sw.tw * tw_late as f64
            + sw.load * load_excess as f64
            + sw.dur * dur_excess as f64;
        if pen != 0.0 {
            metrics.cost_custom += pen;
            metrics.cost += pen;
        }
        metrics.tw_excess = tw_late as f64;
        metrics.load_excess = load_excess as f64;
        metrics.dur_excess = dur_excess as f64;
    }

    // Custom dimensions (P5): accumulate every registered dimension along this
    // route's arcs. Free when none are registered (`track_dims` is the relaxed
    // atomic load taken once at the top of the walk). A registered hard min/max
    // cumul bound is honoured HERE, at full route evaluation, so an out-of-bounds
    // route is always rejected and never committed — this remains the authority.
    // Spike `res` ALSO mirrors the `max` bound of a *monotone* dimension into the
    // O(1) insertion probe (eval.rs) so a breaching insertion is pruned early;
    // non-monotone/unbounded dimensions (and the `min` bound) still rely solely on
    // this full-eval check (see dimension.rs caveats).
    let dim_cumuls = if track_dims {
        crate::dimension::accumulate(&dim_arcs)
    } else {
        crate::dimension::DimensionCumuls::default()
    };
    if dim_cumuls.bound_violated {
        return Err("custom dimension bound exceeded");
    }
    // Soft cumul bounds (Phase 1): a cumul outside its soft band but within the
    // hard band adds a penalty to the route cost instead of being a hard reject —
    // mirrors OR-Tools SetCumulVarSoft{Upper,Lower}Bound. Zero unless a soft band
    // is configured AND breached, so this is a no-op for every existing dimension.
    if dim_cumuls.soft_penalty != 0.0 {
        metrics.cost_custom += dim_cumuls.soft_penalty;
        metrics.cost += dim_cumuls.soft_penalty;
    }

    // User-supplied custom constraints (code, from Rust or Python). The flag
    // check is a single relaxed atomic load — free when none are registered.
    if crate::constraint::has_constraints() {
        let view = crate::constraint::RouteView {
            problem,
            vehicle,
            steps,
            metrics: &metrics,
            dim_cumuls: &dim_cumuls,
        };
        match crate::constraint::apply(&view) {
            Ok(penalty) => {
                metrics.cost_custom += penalty;
                metrics.cost += penalty;
            }
            Err(e) => return Err(e),
        }
    }

    Ok(metrics)
}

/// Recompute the custom-dimension cumuls (P5) for a finished route, so route
/// summaries / reports can read per-stop dimension values without re-running the
/// whole evaluator's cost machinery. Returns an empty [`DimensionCumuls`] when no
/// dimension is registered (the common case → no work). This walks the same arc
/// sequence the evaluator threads through `dimension::accumulate`, so it matches
/// the values the DSL saw during search.
pub fn dimension_cumuls_for_route(
    problem: &Problem,
    matrix: &Matrix,
    vehicle: &Vehicle,
    steps: &[TaskRef],
) -> crate::dimension::DimensionCumuls {
    if !crate::dimension::has_dimensions() {
        return crate::dimension::DimensionCumuls::default();
    }
    let vw = vehicle.time_window();
    let speed = vehicle.speed_factor.max(0.01);
    let start_idx = vehicle.start.as_ref().and_then(|l| l.index)
        .or_else(|| vehicle.end.as_ref().and_then(|l| l.index));
    let end_idx = vehicle.end.as_ref().and_then(|l| l.index).or(start_idx);

    let mut t: Time = vw.start;
    let mut prev_idx: Option<usize> = start_idx;
    let mut arcs: Vec<crate::dimension::Arc2> = Vec::with_capacity(steps.len() + 1);
    for s in steps {
        if let TaskRef::Reload = s {
            // Reload legs are travel back to the depot; mirror the evaluator's
            // continuous accumulation (no per-trip reset for custom dims — see
            // dimension.rs caveats) by simply continuing through the depot.
            if let (Some(p), Some(d)) = (prev_idx, start_idx) {
                let dur = ((matrix.duration(p, d) as f64) * speed).round() as i64;
                let dist = matrix.distance(p, d);
                t += dur;
                arcs.push(crate::dimension::Arc2 {
                    from: p, to: d, arrival: t, distance: dist, duration: dur,
                });
                prev_idx = start_idx;
            }
            continue;
        }
        let here = match s.description(problem).location.index {
            Some(i) => i,
            None => continue,
        };
        let arrival;
        if let Some(p) = prev_idx {
            let dur = ((matrix.duration(p, here) as f64) * speed).round() as i64;
            let dist = matrix.distance(p, here);
            t += dur;
            arrival = t;
            arcs.push(crate::dimension::Arc2 {
                from: p, to: here, arrival, distance: dist, duration: dur,
            });
        } else {
            arrival = t;
            arcs.push(crate::dimension::Arc2 {
                from: here, to: here, arrival, distance: 0, duration: 0,
            });
        }
        // Advance the clock through this stop's service so downstream arrival
        // times stay in step with the evaluator (release/TW waits are not
        // modelled here — this is the report path, kept simple and deterministic).
        let job = s.description(problem);
        t += job.service;
        prev_idx = Some(here);
    }
    if let (Some(p), Some(e)) = (prev_idx, end_idx) {
        let dur = ((matrix.duration(p, e) as f64) * speed).round() as i64;
        let dist = matrix.distance(p, e);
        t += dur;
        arcs.push(crate::dimension::Arc2 {
            from: p, to: e, arrival: t, distance: dist, duration: dur,
        });
    }
    crate::dimension::accumulate(&arcs)
}

/// Return the first time window in `tws` whose end is ≥ `arrival`. If `tws`
/// is empty, returns the universal window.
pub fn pick_time_window(tws: &[TimeWindow], arrival: Time) -> Option<TimeWindow> {
    if tws.is_empty() {
        return Some(TimeWindow::FOREVER);
    }
    tws.iter().copied().find(|w| arrival <= w.end)
}

/// The window with the latest end. Soft mode targets it when an arrival is past
/// every window; the lateness past its end is then charged as time warp. Caller
/// ensures `tws` is non-empty (empty windows already pick `FOREVER`).
#[inline]
fn latest_window(tws: &[TimeWindow]) -> TimeWindow {
    tws.iter().copied().max_by_key(|w| w.end).unwrap_or(TimeWindow::FOREVER)
}

// ---- penalty-managed soft constraints (PyVRP-style time-warp) --------------
// When active on the current thread, `evaluate_route` does NOT hard-reject a
// route that misses a time window, exceeds capacity, or overruns its duration —
// instead it accumulates the violation magnitude and charges it at the supplied
// per-unit weight into `cost_custom` (so the existing accept/select machinery
// optimises in penalised space, exactly like PyVRP's time-warp/load penalties).
// State is THREAD-LOCAL: each rayon multi-start seed runs on its own worker
// thread, so per-seed penalty adaptation never clobbers a sibling seed. Outside
// soft mode every field is its default and the hot path is byte-identical.
//
// Structural constraints stay HARD even in soft mode (as in PyVRP): skills,
// vehicle allowlist, pickup-before-delivery, linehaul-before-backhaul,
// unreachable legs, reload-while-loaded, negative load, max_tasks, max_distance,
// driver-break windows, and custom hard dimension/constraint bounds.

/// Per-unit soft-penalty weights (cost per second of time warp, per unit of load
/// excess, per second of duration excess).
#[derive(Clone, Copy, Debug)]
pub struct SoftWeights {
    pub tw: f64,
    pub load: f64,
    pub dur: f64,
}

thread_local! {
    static SOFT_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static SOFT_TW: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    static SOFT_LOAD: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    static SOFT_DUR: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    /// Bumped on every weight change so the route-eval cache never serves a
    /// soft-penalised result to a hard query (or across a λ update). `0` ⇒ hard
    /// mode, which keeps the cache key — and thus behaviour — byte-identical.
    static SOFT_GEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Enable penalty-managed soft constraints on the current thread with the given
/// weights, or pass `None` to restore hard constraints. Call at the start of a
/// per-seed search and clear (`None`) before any final hard re-evaluation.
pub fn set_soft_penalties(weights: Option<SoftWeights>) {
    match weights {
        Some(w) => {
            SOFT_TW.with(|c| c.set(w.tw));
            SOFT_LOAD.with(|c| c.set(w.load));
            SOFT_DUR.with(|c| c.set(w.dur));
            SOFT_ACTIVE.with(|c| c.set(true));
            // Advance to a fresh, non-zero generation so cached hard/old-λ
            // entries are not reused. Odd step keeps successive gens distinct.
            SOFT_GEN.with(|c| c.set(c.get().wrapping_add(0x9E37_79B9_7F4A_7C15)));
        }
        None => {
            SOFT_ACTIVE.with(|c| c.set(false));
            SOFT_GEN.with(|c| c.set(0));
        }
    }
}

#[inline]
fn soft_active() -> bool {
    SOFT_ACTIVE.with(|c| c.get())
}

/// Public view of whether penalty-managed soft mode is armed on this thread.
/// The fast O(1) cost-delta LS path is only exact in hard mode (soft penalties
/// fold time-warp/overload into the cost, which the arc-sum model doesn't see).
#[inline]
pub fn soft_is_active() -> bool {
    soft_active()
}

#[inline]
fn soft_weights() -> SoftWeights {
    SoftWeights {
        tw: SOFT_TW.with(|c| c.get()),
        load: SOFT_LOAD.with(|c| c.get()),
        dur: SOFT_DUR.with(|c| c.get()),
    }
}

#[inline]
fn soft_generation() -> u64 {
    SOFT_GEN.with(|c| c.get())
}

impl Solution {
    pub fn recompute_summary(&mut self, problem: &Problem) {
        let mut s = Summary { routes: self.routes.len(), unassigned: self.unassigned.len(), ..Default::default() };
        for r in &self.routes {
            s.cost += r.metrics.cost;
            s.travel_time += r.metrics.travel_time;
            s.service_time += r.metrics.service_time;
            s.waiting_time += r.metrics.waiting_time;
            s.distance += r.metrics.distance;
        }
        // Prize-collecting: charge each unassigned task its prize. A job's prize
        // defaults to a large sentinel (problem::DEFAULT_PRIZE), so this matches
        // the historical flat `count * 1e9` unless a finite prize was set. Only
        // single jobs are optional; shipment halves keep the sentinel.
        for t in &self.unassigned {
            s.cost += t.description(problem).prize;
        }
        // Disjunctions (OR-Tools `AddDisjunction` semantics): on top of `prize`,
        // charge each unassigned job its explicit `disjunction_penalty` if set.
        // `prize` answers "what is serving worth"; `disjunction_penalty` answers
        // "what does *dropping* cost" — keeping them separate lets local search
        // trade a known drop penalty against routing cost. Unset penalties are 0,
        // so this loop is a no-op for inputs that don't use the feature. Only
        // single jobs carry one; shipment halves keep the sentinel job.
        for t in &self.unassigned {
            if let Some(p) = t.description(problem).disjunction_penalty {
                s.cost += p;
            }
        }
        // Solution-level (cross-route) penalty term, behind a lock-free fast path.
        if crate::global_constraint::has_global() {
            let view = crate::global_constraint::SolutionView {
                problem,
                routes: &self.routes,
                unassigned: &self.unassigned,
            };
            s.cost += crate::global_constraint::apply(&view);
        }
        self.summary = s;
    }
}
