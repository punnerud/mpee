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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobIn {
    pub id: u64,
    #[serde(default)]
    pub location: Option<[f64; 2]>,
    #[serde(default)]
    pub location_index: Option<usize>,
    #[serde(default)]
    pub service: Time,
    #[serde(default)]
    pub setup: Time,
    #[serde(default)]
    pub delivery: Capacity,
    #[serde(default)]
    pub pickup: Capacity,
    #[serde(default)]
    pub skills: SkillSet,
    #[serde(default)]
    pub priority: u8,
    #[serde(default)]
    pub time_windows: Vec<[Time; 2]>,
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
    pub start: Option<[f64; 2]>,
    #[serde(default)]
    pub start_index: Option<usize>,
    #[serde(default)]
    pub end: Option<[f64; 2]>,
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
    #[serde(default)]
    pub description: Option<String>,
}

fn loc_from(coord: Option<[f64; 2]>, index: Option<usize>) -> Option<Location> {
    if coord.is_none() && index.is_none() {
        None
    } else {
        Some(Location { coord, index })
    }
}

fn tw_from(arr: [Time; 2]) -> TimeWindow {
    TimeWindow { start: arr[0], end: arr[1] }
}

fn job_from(j: &JobIn) -> Job {
    Job {
        id: j.id,
        location: loc_from(j.location, j.location_index)
            .unwrap_or_else(|| Location { coord: None, index: None }),
        kind: JobKindOpt::Single,
        service: j.service,
        setup: j.setup,
        delivery: j.delivery.clone(),
        pickup: j.pickup.clone(),
        skills: j.skills.clone(),
        priority: j.priority,
        time_windows: j.time_windows.iter().copied().map(tw_from).collect(),
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
        profile: v.profile.clone().unwrap_or_else(|| "car".to_string()),
        description: v.description.clone(),
    }
}

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

    let summary = SummaryOut {
        cost: sol.summary.cost,
        routes: sol.summary.routes,
        unassigned: sol.summary.unassigned,
        setup: routes_out.iter().map(|r| r.setup).sum(),
        service: sol.summary.service_time,
        duration: sol.summary.travel_time,
        waiting_time: sol.summary.waiting_time,
        distance: sol.summary.distance,
    };

    let unassigned = sol.unassigned.iter().map(|t| {
        let j = t.description(problem);
        UnassignedOut {
            id: j.id,
            kind: match t {
                TaskRef::Job(_) => "job".to_string(),
                TaskRef::Pickup(_) => "pickup".to_string(),
                TaskRef::Delivery(_) => "delivery".to_string(),
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

    // Recompute load profile: leave depot loaded with all deliveries.
    let mut load = vec![0i64; dim];
    for s in &r.steps {
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
    });

    // Prefer an explicit matrix (passed in from solve_full); fall back to one
    // embedded in the problem; otherwise per-step legs report 0 (route-level
    // totals are still right because they were computed during solve).
    let owned_matrix = problem
        .matrices
        .get(&veh.profile)
        .and_then(|p| crate::matrix::Matrix::from_provided(p).ok());
    let matrix: Option<&crate::matrix::Matrix> = explicit_matrix.or(owned_matrix.as_ref());

    for s in &r.steps {
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
        }

        let kind = match s {
            TaskRef::Job(_) => StepKind::Job,
            TaskRef::Pickup(_) => StepKind::Pickup,
            TaskRef::Delivery(_) => StepKind::Delivery,
        };

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
        });

        t += j.service;
        total_service += j.service;
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
    });

    RouteOut {
        vehicle: veh.id,
        cost: r.metrics.cost,
        setup: total_setup,
        service: total_service,
        duration: total_travel,
        waiting_time: total_wait,
        distance: total_dist,
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
