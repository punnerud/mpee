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
        priority: 0,
        time_windows: vec![],
        prize: crate::problem::DEFAULT_PRIZE,
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
#[derive(Debug, Clone, Copy, Default)]
pub struct RouteMetrics {
    pub start_time: Time,
    pub end_time: Time,
    pub travel_time: Time,
    pub service_time: Time,
    pub waiting_time: Time,
    pub setup_time: Time,
    pub distance: i64,
    pub cost: Cost,
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
    let key = (prob_id, vehicle.id, hash_steps(steps));

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
    for (i, &cap_i) in vehicle.capacity.iter().enumerate() {
        if i < dim && load[i] > cap_i {
            return Err("capacity exceeded at route start");
        }
    }

    // Pickup-before-delivery precedence: only allocate the bitset if we see
    // any shipment task. For pure-CVRP instances this stays empty → free.
    let mut pickups_seen: Vec<bool> = Vec::new();
    for s in steps {
        if !vehicle.has_skills(s.skills(problem)) {
            return Err("vehicle missing required skill");
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
            for (i, &cap_i) in vehicle.capacity.iter().enumerate() {
                if i < dim && load[i] > cap_i {
                    return Err("capacity exceeded after reload");
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

        if let Some(p) = prev_idx {
            let raw = matrix.duration(p, here);
            // A sentinel-valued leg means "no road between these points"; such
            // a route is infeasible, never a real (if long) one.
            if raw as i64 >= UNREACHABLE_LEG {
                return Err("unreachable leg (no road between stops)");
            }
            let dur = ((raw as f64) * speed).round() as i64;
            t += dur;
            travel_time += dur;
            distance += matrix.distance(p, here);
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

        let chosen_tw = pick_time_window(&job.time_windows, t).ok_or("time window missed")?;
        if t < chosen_tw.start {
            waiting_time += chosen_tw.start - t;
            t = chosen_tw.start;
        }
        if t > chosen_tw.end {
            return Err("arrived after time window end");
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
        for (i, &cap_i) in vehicle.capacity.iter().enumerate() {
            if i < dim && load[i] > cap_i {
                return Err("capacity exceeded mid-route");
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
        t += dur;
        travel_time += dur;
        distance += matrix.distance(p, e);
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
        break_idx += 1;
    }

    if t > vw.end {
        return Err("route ends after vehicle time window");
    }
    if let Some(max) = vehicle.max_travel_time {
        if travel_time > max {
            return Err("max_travel_time exceeded");
        }
    }
    if let Some(max) = vehicle.max_distance {
        if distance > max {
            return Err("max_distance exceeded");
        }
    }

    let route_dur = t - vw.start;
    let cost = vehicle.fixed
        + (travel_time as f64) * (vehicle.per_hour / 3600.0).max(0.0)
        // tiny tiebreaker per service second so longer service still costs:
        + (service_time as f64) * 1e-6;
    let _ = route_dur;

    let mut metrics = RouteMetrics {
        start_time: vw.start,
        end_time: t,
        travel_time,
        service_time,
        waiting_time,
        setup_time,
        distance,
        cost,
    };

    // User-supplied custom constraints (code, from Rust or Python). The flag
    // check is a single relaxed atomic load — free when none are registered.
    if crate::constraint::has_constraints() {
        let view = crate::constraint::RouteView { problem, vehicle, steps, metrics: &metrics };
        match crate::constraint::apply(&view) {
            Ok(penalty) => metrics.cost += penalty,
            Err(e) => return Err(e),
        }
    }

    Ok(metrics)
}

/// Return the first time window in `tws` whose end is ≥ `arrival`. If `tws`
/// is empty, returns the universal window.
pub fn pick_time_window(tws: &[TimeWindow], arrival: Time) -> Option<TimeWindow> {
    if tws.is_empty() {
        return Some(TimeWindow::FOREVER);
    }
    tws.iter().copied().find(|w| arrival <= w.end)
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
