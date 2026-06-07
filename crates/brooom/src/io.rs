//! Vroom-compatible JSON input/output.
//!
//! Vroom's wire format uses bare arrays for locations and time windows
//! (`[lon, lat]`, `[start, end]`), whereas our internal types are structured.
//! These types do the translation in both directions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;
use crate::problem::{
    Capacity, Cost, Job, JobKindOpt, Location, Problem, ProvidedMatrix, Shipment, SkillSet, Time,
    TimeWindow, Vehicle,
};
use crate::solution::{evaluate_route, Route, Solution, Step, StepKind, TaskRef};

// =========================================================================
// INPUT
// =========================================================================

/// A coordinate that deserializes from any of the common spellings, so
/// hand-written JSON is forgiving:
///   * `[lon, lat]`              — VROOM's native bare array
///   * `{"lon": .., "lat": ..}`  — explicit keys (unambiguous; recommended)
///   * `{"coord": [lon, lat]}`   — our internal struct form
/// Always serializes back to the VROOM `[lon, lat]` array.
#[derive(Debug, Clone, Copy)]
pub struct Coord {
    pub lon: f64,
    pub lat: f64,
}

impl<'de> Deserialize<'de> for Coord {
    fn deserialize<D>(d: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Arr([f64; 2]),
            Keys { lon: f64, lat: f64 },
            Wrapped { coord: [f64; 2] },
        }
        Ok(match Raw::deserialize(d)? {
            Raw::Arr([lon, lat]) => Coord { lon, lat },
            Raw::Keys { lon, lat } => Coord { lon, lat },
            Raw::Wrapped { coord: [lon, lat] } => Coord { lon, lat },
        })
    }
}

impl Serialize for Coord {
    fn serialize<S>(&self, s: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        [self.lon, self.lat].serialize(s)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VroomInput {
    #[serde(default)]
    pub jobs: Vec<JobIn>,
    #[serde(default)]
    pub shipments: Vec<ShipmentIn>,
    pub vehicles: Vec<VehicleIn>,
    #[serde(default)]
    pub matrices: HashMap<String, ProvidedMatrix>,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    /// First-class precedence pairs `[a, b]` (job id a before job id b on the same
    /// route). Maps to [`Problem::precedence`].
    #[serde(default)]
    pub precedence: Vec<(u64, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobIn {
    pub id: u64,
    #[serde(default)]
    pub location: Option<Coord>,
    #[serde(default)]
    pub location_index: Option<usize>,
    #[serde(default)]
    pub service: Time,
    #[serde(default)]
    pub setup: Time,
    #[serde(default)]
    pub release: Time,
    #[serde(default)]
    pub delivery: Capacity,
    #[serde(default)]
    pub pickup: Capacity,
    #[serde(default)]
    pub skills: SkillSet,
    /// Vehicle allowlist: if present, only these vehicle ids may serve the job.
    #[serde(default)]
    pub allowed_vehicles: Option<Vec<u64>>,
    #[serde(default)]
    pub priority: u8,
    #[serde(default)]
    pub time_windows: Vec<[Time; 2]>,
    #[serde(default = "default_prize")]
    pub prize: Cost,
    /// Explicit per-node drop penalty (OR-Tools `AddDisjunction`). `None`/omitted
    /// means freely droppable at no extra cost; see `Job::disjunction_penalty`.
    #[serde(default)]
    pub disjunction_penalty: Option<Cost>,
    #[serde(default)]
    pub group: Option<u32>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipmentIn {
    pub pickup: JobIn,
    pub delivery: JobIn,
    #[serde(default)]
    pub amount: Capacity,
    #[serde(default)]
    pub skills: SkillSet,
    #[serde(default)]
    pub priority: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleIn {
    pub id: u64,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub start: Option<Coord>,
    #[serde(default)]
    pub start_index: Option<usize>,
    #[serde(default)]
    pub end: Option<Coord>,
    #[serde(default)]
    pub end_index: Option<usize>,
    #[serde(default)]
    pub capacity: Capacity,
    #[serde(default)]
    pub skills: SkillSet,
    #[serde(default)]
    pub time_window: Option<[Time; 2]>,
    #[serde(default)]
    pub speed_factor: Option<f64>,
    #[serde(default)]
    pub max_tasks: Option<usize>,
    #[serde(default)]
    pub max_travel_time: Option<Time>,
    #[serde(default)]
    pub max_distance: Option<i64>,
    #[serde(default)]
    pub fixed: Option<Cost>,
    #[serde(default)]
    pub per_hour: Option<Cost>,
    /// Cost per second of route span. Absent ⇒ 0.0 (no span cost), reproducing
    /// today's behaviour. See `problem::Vehicle::span_cost`.
    #[serde(default)]
    pub span_cost: Option<Cost>,
    /// Weight on the per-distance travel cost. Absent ⇒ 0.0.
    #[serde(default)]
    pub distance_weight: Option<f64>,
    /// Weight on the per-hour travel-time cost. Absent ⇒ 1.0 (unchanged).
    #[serde(default)]
    pub time_weight: Option<f64>,
    #[serde(default)]
    pub breaks: Vec<BreakIn>,
    #[serde(default = "default_max_trips_in")]
    pub max_trips: usize,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakIn {
    pub id: u64,
    #[serde(default)]
    pub service: Time,
    #[serde(default)]
    pub time_windows: Vec<[Time; 2]>,
    #[serde(default)]
    pub description: Option<String>,
}

fn loc_from(coord: Option<Coord>, index: Option<usize>) -> Option<Location> {
    if coord.is_none() && index.is_none() {
        None
    } else {
        Some(Location { coord: coord.map(|c| [c.lon, c.lat]), index })
    }
}

fn tw_from(arr: [Time; 2]) -> TimeWindow {
    TimeWindow { start: arr[0], end: arr[1] }
}

fn default_prize() -> Cost { crate::problem::DEFAULT_PRIZE }

fn job_from(j: &JobIn) -> Job {
    Job {
        id: j.id,
        location: loc_from(j.location, j.location_index)
            .unwrap_or_else(|| Location { coord: None, index: None }),
        kind: JobKindOpt::Single,
        service: j.service,
        setup: j.setup,
        release: j.release,
        delivery: j.delivery.clone(),
        pickup: j.pickup.clone(),
        skills: j.skills.clone(),
        allowed_vehicles: j.allowed_vehicles.clone(),
        priority: j.priority,
        time_windows: j.time_windows.iter().copied().map(tw_from).collect(),
        prize: j.prize,
        disjunction_penalty: j.disjunction_penalty,
        group: j.group,
        description: j.description.clone(),
    }
}

fn vehicle_from(v: &VehicleIn) -> Vehicle {
    Vehicle {
        id: v.id,
        start: loc_from(v.start, v.start_index),
        end: loc_from(v.end, v.end_index),
        capacity: v.capacity.clone(),
        skills: v.skills.clone(),
        time_window: v.time_window.map(tw_from),
        speed_factor: v.speed_factor.unwrap_or(1.0),
        max_tasks: v.max_tasks,
        max_travel_time: v.max_travel_time,
        max_distance: v.max_distance,
        fixed: v.fixed.unwrap_or(0.0),
        per_hour: v.per_hour.unwrap_or(3600.0),
        span_cost: v.span_cost.unwrap_or(0.0),
        distance_weight: v.distance_weight.unwrap_or(0.0),
        time_weight: v.time_weight.unwrap_or(1.0),
        profile: v.profile.clone().unwrap_or_else(|| "car".to_string()),
        breaks: v.breaks.iter().map(|b| crate::problem::Break {
            id: b.id,
            service: b.service,
            time_windows: b.time_windows.iter().copied().map(tw_from).collect(),
            description: b.description.clone(),
        }).collect(),
        max_trips: v.max_trips,
        description: v.description.clone(),
    }
}

fn default_max_trips_in() -> usize { 1 }

impl From<VroomInput> for Problem {
    fn from(v: VroomInput) -> Self {
        Problem {
            jobs: v.jobs.iter().map(job_from).collect(),
            shipments: v.shipments.iter().map(|s| Shipment {
                pickup: job_from(&s.pickup),
                delivery: job_from(&s.delivery),
                amount: s.amount.clone(),
                skills: s.skills.clone(),
                priority: s.priority,
            }).collect(),
            vehicles: v.vehicles.iter().map(vehicle_from).collect(),
            matrices: v.matrices,
            precedence: v.precedence.clone(),
            description: None,
        }
    }
}

pub fn parse_input(json: &str) -> Result<Problem> {
    let v: VroomInput = serde_json::from_str(json)?;
    Ok(v.into())
}

/// Streaming parser. For inputs containing big matrices, this avoids loading
/// the entire file into a `String` first and roughly halves peak RSS.
pub fn parse_input_reader<R: std::io::Read>(reader: R) -> Result<Problem> {
    let v: VroomInput = serde_json::from_reader(reader)?;
    Ok(v.into())
}

/// Parse both the problem AND its `options` object (objective + dimensions).
/// The `options` key is parsed into a [`crate::options::SolverOptions`]; absent
/// `options` yields the default (scalar objective, no dimensions), so the
/// returned problem is byte-identical to [`parse_input`]'s.
pub fn parse_input_with_options(json: &str) -> Result<(Problem, crate::options::SolverOptions)> {
    let v: VroomInput = serde_json::from_str(json)?;
    let opts = crate::options::SolverOptions::from_value(v.options.as_ref())?;
    Ok((v.into(), opts))
}

/// Streaming counterpart of [`parse_input_with_options`].
pub fn parse_input_reader_with_options<R: std::io::Read>(
    reader: R,
) -> Result<(Problem, crate::options::SolverOptions)> {
    let v: VroomInput = serde_json::from_reader(reader)?;
    let opts = crate::options::SolverOptions::from_value(v.options.as_ref())?;
    Ok((v.into(), opts))
}

// =========================================================================
// OUTPUT
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VroomOutput {
    /// 0 == optimal-ish, 1 == infeasible, 2 == error.
    pub code: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub summary: SummaryOut,
    pub unassigned: Vec<UnassignedOut>,
    pub routes: Vec<RouteOut>,
    /// Jobs served but past their time window (soft-constraint degradation).
    /// Empty on a clean on-time plan. Lets a dispatcher see exactly who is late
    /// and by how much instead of getting a bare "infeasible" — or a plan that
    /// silently hides the violation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub late: Vec<LateOut>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryOut {
    pub cost: f64,
    pub routes: usize,
    pub unassigned: usize,
    pub setup: Time,
    pub service: Time,
    pub duration: Time,
    pub waiting_time: Time,
    pub distance: i64,
    /// Total soft time-window violation across all stops (seconds). 0 when the
    /// plan is fully on-time; positive when soft constraints accepted lateness.
    #[serde(default, skip_serializing_if = "crate::solution::is_zero_time")]
    pub time_warp: Time,
    /// Number of stops served after their window closed.
    #[serde(default, skip_serializing_if = "crate::solution::is_zero_usize")]
    pub late_jobs: usize,
    /// Worst single lateness (seconds).
    #[serde(default, skip_serializing_if = "crate::solution::is_zero_time")]
    pub max_lateness: Time,
}

/// One late stop: which job, when it was reached, when it was due, and by how
/// much it overran.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateOut {
    pub id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location_index: Option<usize>,
    /// Time service actually started.
    pub arrival: Time,
    /// Latest time-window end the job had.
    pub due: Time,
    /// `arrival - due` (seconds late).
    pub lateness: Time,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnassignedOut {
    pub id: u64,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<[f64; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteOut {
    pub vehicle: u64,
    pub cost: f64,
    pub setup: Time,
    pub service: Time,
    pub duration: Time,
    pub waiting_time: Time,
    pub distance: i64,
    /// Total soft lateness on this route (seconds); 0 when on-time.
    #[serde(default, skip_serializing_if = "crate::solution::is_zero_time")]
    pub time_warp: Time,
    pub steps: Vec<Step>,
}

/// Build a Vroom-compatible output from an internal Solution. This re-walks
/// each route to emit per-step arrival/load timing.
///
/// `matrix` is optional. Without it, per-step `arrival`/`distance` fall back
/// to the route-level totals (which are still correct in the summary).
pub fn to_output(
    problem: &Problem,
    sol: &Solution,
    matrix: Option<&crate::matrix::Matrix>,
) -> VroomOutput {
    let mut routes_out: Vec<RouteOut> = Vec::with_capacity(sol.routes.len());
    for r in &sol.routes {
        routes_out.push(route_steps(problem, r, matrix));
    }

    // Collect soft time-window violations across all routes so a caller can act
    // on "served, but late" instead of a hidden penalty.
    let mut late: Vec<LateOut> = Vec::new();
    for r in &routes_out {
        for s in &r.steps {
            if s.lateness > 0 {
                late.push(LateOut {
                    id: s.job_id.unwrap_or(0),
                    location_index: s.location_index,
                    arrival: s.arrival,
                    due: s.arrival - s.lateness,
                    lateness: s.lateness,
                });
            }
        }
    }
    let time_warp: Time = late.iter().map(|l| l.lateness).sum();
    let max_lateness: Time = late.iter().map(|l| l.lateness).max().unwrap_or(0);

    let summary = SummaryOut {
        cost: sol.summary.cost,
        routes: sol.summary.routes,
        unassigned: sol.summary.unassigned,
        setup: routes_out.iter().map(|r| r.setup).sum(),
        service: sol.summary.service_time,
        duration: sol.summary.travel_time,
        waiting_time: sol.summary.waiting_time,
        distance: sol.summary.distance,
        time_warp,
        late_jobs: late.len(),
        max_lateness,
    };

    let unassigned = sol.unassigned.iter().map(|t| {
        let j = t.description(problem);
        UnassignedOut {
            id: j.id,
            kind: match t {
                TaskRef::Job(_) => "job".to_string(),
                TaskRef::Pickup(_) => "pickup".to_string(),
                TaskRef::Delivery(_) => "delivery".to_string(),
                TaskRef::Reload => "reload".to_string(),
            },
            location_index: j.location.index,
            location: j.location.coord,
        }
    }).collect();

    VroomOutput {
        code: if sol.unassigned.is_empty() { 0 } else { 1 },
        error: None,
        summary,
        unassigned,
        routes: routes_out,
        late,
    }
}

fn route_steps(
    problem: &Problem,
    r: &Route,
    explicit_matrix: Option<&crate::matrix::Matrix>,
) -> RouteOut {
    let veh = &problem.vehicles[r.vehicle_idx];
    let dim = problem.capacity_dim().max(veh.capacity.len());
    let speed = veh.speed_factor.max(0.01);
    let vw = veh.time_window();

    // Recompute load profile: leave the depot loaded with the FIRST trip's
    // deliveries (reset at each reload below for multi-trip routes).
    let mut load = vec![0i64; dim];
    for s in &r.steps {
        if let TaskRef::Reload = s { break; }
        let j = s.description(problem);
        for k in 0..dim {
            load[k] += *j.delivery.get(k).unwrap_or(&0);
        }
    }

    let mut steps = Vec::with_capacity(r.steps.len() + 2);
    let start_idx = veh.start.as_ref().and_then(|l| l.index)
        .or_else(|| veh.end.as_ref().and_then(|l| l.index));
    let end_idx = veh.end.as_ref().and_then(|l| l.index).or(start_idx);

    let mut t: Time = vw.start;
    let mut prev: Option<usize> = start_idx;
    let mut total_travel: Time = 0;
    let mut total_service: Time = 0;
    let mut total_setup: Time = 0;
    let mut total_wait: Time = 0;
    let mut total_dist: i64 = 0;
    let mut total_lateness: Time = 0;
    // Mirror the break scheduling in `evaluate_route` so emitted `break` steps
    // line up with the timings the solver actually used.
    let breaks = &veh.breaks;
    let mut break_idx = 0usize;

    steps.push(Step {
        kind: StepKind::Start,
        job_id: None,
        location_index: start_idx,
        arrival: t,
        service: 0,
        waiting_time: 0,
        setup: 0,
        load: load.clone(),
        distance: 0,
        lateness: 0,
    });

    // Prefer an explicit matrix (passed in from solve_full); fall back to one
    // embedded in the problem; otherwise per-step legs report 0 (route-level
    // totals are still right because they were computed during solve).
    let owned_matrix = problem
        .matrices
        .get(&veh.profile)
        .and_then(|p| crate::matrix::Matrix::from_provided(p).ok());
    let matrix: Option<&crate::matrix::Matrix> = explicit_matrix.or(owned_matrix.as_ref());

    for (k, s) in r.steps.iter().enumerate() {
        // Multi-trip reload: travel back to the depot, reset the load to the
        // next trip's deliveries, emit a `reload` step, and depart again.
        if let TaskRef::Reload = s {
            if let (Some(p), Some(d)) = (prev, start_idx) {
                if let Some(mx) = matrix {
                    let dur = ((mx.duration(p, d) as f64) * speed).round() as i64;
                    t += dur;
                    total_travel += dur;
                    total_dist += mx.distance(p, d);
                }
            }
            for v in load.iter_mut() { *v = 0; }
            for ns in &r.steps[k + 1..] {
                if let TaskRef::Reload = ns { break; }
                if let TaskRef::Job(_) = ns {
                    let nj = ns.description(problem);
                    for kk in 0..dim { load[kk] += *nj.delivery.get(kk).unwrap_or(&0); }
                }
            }
            steps.push(Step {
                kind: StepKind::Reload,
                job_id: None,
                location_index: start_idx,
                arrival: t,
                service: 0,
                waiting_time: 0,
                setup: 0,
                load: load.clone(),
                distance: total_dist,
                lateness: 0,
            });
            prev = start_idx;
            continue;
        }

        let j = s.description(problem);
        let here = j.location.index.unwrap_or(0);

        let (dur, dist) = if let (Some(p), Some(mx)) = (prev, matrix) {
            let raw_dur = mx.duration(p, here);
            let scaled = ((raw_dur as f64) * speed).round() as i64;
            (scaled, mx.distance(p, here))
        } else {
            (0i64, 0i64)
        };
        t += dur;
        total_travel += dur;
        total_dist += dist;

        let do_setup = match prev {
            Some(p) => p != here && j.setup > 0,
            None => j.setup > 0,
        };
        let setup = if do_setup { j.setup } else { 0 };
        t += setup;
        total_setup += setup;

        // Release-time wait (mirror of the evaluator) before window selection.
        let rel_wait = if t < j.release { j.release - t } else { 0 };
        t += rel_wait;
        total_wait += rel_wait;

        let chosen = crate::solution::pick_time_window(&j.time_windows, t)
            .unwrap_or(TimeWindow::FOREVER);
        let waiting = if t < chosen.start { chosen.start - t } else { 0 };
        t += waiting;
        total_wait += waiting;

        // Apply load delta for the step output.
        match s {
            TaskRef::Job(_) => {
                for k in 0..dim {
                    load[k] -= *j.delivery.get(k).unwrap_or(&0);
                    load[k] += *j.pickup.get(k).unwrap_or(&0);
                }
            }
            TaskRef::Pickup(i) => {
                let amt = if !problem.shipments[*i].amount.is_empty() {
                    &problem.shipments[*i].amount
                } else {
                    &problem.shipments[*i].pickup.pickup
                };
                for k in 0..dim { load[k] += *amt.get(k).unwrap_or(&0); }
            }
            TaskRef::Delivery(i) => {
                let amt = if !problem.shipments[*i].amount.is_empty() {
                    &problem.shipments[*i].amount
                } else {
                    &problem.shipments[*i].pickup.pickup
                };
                for k in 0..dim { load[k] -= *amt.get(k).unwrap_or(&0); }
            }
            TaskRef::Reload => {} // handled above
        }

        let kind = match s {
            TaskRef::Job(_) => StepKind::Job,
            TaskRef::Pickup(_) => StepKind::Pickup,
            TaskRef::Delivery(_) => StepKind::Delivery,
            TaskRef::Reload => StepKind::Reload,
        };

        // Soft lateness: service started at `t`; if that is past the job's
        // latest window end, the soft search accepted a late stop. Surface it.
        let due = j.time_windows.iter().map(|w| w.end).max();
        let lateness = match due {
            Some(e) if t > e => t - e,
            _ => 0,
        };
        total_lateness += lateness;

        steps.push(Step {
            kind,
            job_id: Some(j.id),
            location_index: Some(here),
            arrival: t,
            service: j.service,
            waiting_time: waiting,
            setup,
            load: load.clone(),
            distance: total_dist,
            lateness,
        });

        t += j.service;
        total_service += j.service;

        // Emit any breaks whose window is already open at `t` (same greedy
        // placement as the evaluator). A break carries the current load and
        // has no location.
        while break_idx < breaks.len() {
            let br = &breaks[break_idx];
            let tw = crate::solution::pick_time_window(&br.time_windows, t)
                .unwrap_or(TimeWindow::FOREVER);
            if t < tw.start { break; }
            steps.push(Step {
                kind: StepKind::Break,
                job_id: Some(br.id),
                location_index: None,
                arrival: t,
                service: br.service,
                waiting_time: 0,
                setup: 0,
                load: load.clone(),
                distance: total_dist,
                lateness: 0,
            });
            t += br.service;
            break_idx += 1;
        }

        prev = Some(here);
    }

    // Final leg back to depot, if any.
    if let (Some(p), Some(e)) = (prev, end_idx) {
        if let Some(mx) = matrix {
            let raw = mx.duration(p, e);
            let dur = ((raw as f64) * speed).round() as i64;
            t += dur;
            total_travel += dur;
            total_dist += mx.distance(p, e);
        }
    }

    // Any breaks not yet placed must still be taken before the day ends
    // (waiting for the window to open if needed) — mirrors the evaluator.
    while break_idx < breaks.len() {
        let br = &breaks[break_idx];
        let tw = match crate::solution::pick_time_window(&br.time_windows, t) {
            Some(w) => w,
            None => break,
        };
        let waiting = if t < tw.start { tw.start - t } else { 0 };
        t += waiting;
        total_wait += waiting;
        steps.push(Step {
            kind: StepKind::Break,
            job_id: Some(br.id),
            location_index: None,
            arrival: t,
            service: br.service,
            waiting_time: waiting,
            setup: 0,
            load: load.clone(),
            distance: total_dist,
            lateness: 0,
        });
        t += br.service;
        break_idx += 1;
    }

    steps.push(Step {
        kind: StepKind::End,
        job_id: None,
        location_index: end_idx,
        arrival: t,
        service: 0,
        waiting_time: 0,
        setup: 0,
        load: load.clone(),
        distance: total_dist,
        lateness: 0,
    });

    RouteOut {
        vehicle: veh.id,
        cost: r.metrics.cost,
        setup: total_setup,
        service: total_service,
        duration: total_travel,
        waiting_time: total_wait,
        distance: total_dist,
        time_warp: total_lateness,
        steps,
    }
}

/// Verify a full route's evaluation against the recorded metrics. Useful in
/// tests after applying local-search moves.
pub fn verify_solution(problem: &Problem, matrix: &crate::matrix::Matrix, sol: &Solution) -> Result<()> {
    use crate::error::Error;
    for r in &sol.routes {
        let veh = &problem.vehicles[r.vehicle_idx];
        let m = evaluate_route(problem, matrix, veh, &r.steps)
            .map_err(|e| Error::Infeasible(format!("route {} infeasible: {e}", veh.id)))?;
        if (m.cost - r.metrics.cost).abs() > 1e-3 {
            return Err(Error::Infeasible(format!(
                "route {} cost mismatch: expected {} stored {}",
                veh.id, m.cost, r.metrics.cost
            )));
        }
    }
    Ok(())
}
