//! Parse a Vroom-style solution JSON into a brooom `Solution`.
//!
//! Used by `--warm-start <path>`: lets the solver bypass insertion and
//! drop straight into local search on a pre-built solution. The file
//! must reference the same `problem`'s job ids and vehicle ids; we
//! look them up by `location_index` and `vehicle` respectively.
//!
//! Unassigned tasks (jobs in `problem` that are not present in any
//! input route) are placed in `Solution::unassigned`. LS will not
//! reinsert them automatically — caller responsibility to ensure the
//! warm-start covers all tasks if full coverage is required.

use std::path::Path;

use serde::Deserialize;

use crate::error::Error;
use crate::matrix::Matrix;
use crate::problem::Problem;
use crate::solution::{evaluate_route, Route, Solution, TaskRef};

/// Loose Vroom-output schema — only the fields we read.
#[derive(Debug, Deserialize)]
struct InputRoute {
    /// Vehicle id matching one in `problem.vehicles[].id`.
    vehicle: u64,
    steps: Vec<InputStep>,
}

#[derive(Debug, Deserialize)]
struct InputStep {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    location_index: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct InputDoc {
    routes: Vec<InputRoute>,
}

/// Build a Solution from a Vroom-style solution file. Routes are
/// re-evaluated against the (problem, matrix) pair, so the metrics
/// reflect the current matrix even if the input came from a different
/// solver.
pub fn load_warm_start(
    problem: &Problem,
    matrix: &Matrix,
    path: impl AsRef<Path>,
) -> Result<Solution, Error> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| Error::Other(format!("warm-start: open: {e}")))?;
    let doc: InputDoc =
        serde_json::from_slice(&bytes).map_err(|e| Error::Other(format!("warm-start: parse: {e}")))?;

    // Lookup: location_index → job_idx. Multiple jobs at the same location
    // are unusual but possible; we take the first match.
    let loc_to_job: std::collections::HashMap<usize, usize> = problem
        .jobs
        .iter()
        .enumerate()
        .filter_map(|(i, j)| j.location.index.map(|li| (li, i)))
        .collect();

    // Lookup: location_index → (shipment_idx, is_pickup), so a warm-start can
    // carry pickup&delivery shipments (e.g. from the CP-SAT bridge). A location
    // that is both a job and a shipment half is resolved as a job (above) first.
    let mut loc_to_ship: std::collections::HashMap<usize, (usize, bool)> =
        std::collections::HashMap::new();
    for (i, s) in problem.shipments.iter().enumerate() {
        if let Some(li) = s.pickup.location.index {
            loc_to_ship.entry(li).or_insert((i, true));
        }
        if let Some(li) = s.delivery.location.index {
            loc_to_ship.entry(li).or_insert((i, false));
        }
    }

    let veh_id_to_idx: std::collections::HashMap<u64, usize> = problem
        .vehicles
        .iter()
        .enumerate()
        .map(|(i, v)| (v.id, i))
        .collect();

    let mut routes: Vec<Route> = Vec::with_capacity(doc.routes.len());
    let mut visited_jobs = std::collections::HashSet::new();

    for r in &doc.routes {
        let veh_idx = *veh_id_to_idx
            .get(&r.vehicle)
            .ok_or_else(|| Error::Other(format!("warm-start: unknown vehicle id {}", r.vehicle)))?;
        let vehicle = &problem.vehicles[veh_idx];

        let mut steps: Vec<TaskRef> = Vec::new();
        for s in &r.steps {
            if s.kind == "job" {
                let li = s.location_index.ok_or_else(|| {
                    Error::Other("warm-start: job step missing location_index".into())
                })?;
                if let Some(&job_idx) = loc_to_job.get(&li) {
                    steps.push(TaskRef::Job(job_idx));
                    visited_jobs.insert(job_idx);
                } else if let Some(&(si, is_pickup)) = loc_to_ship.get(&li) {
                    steps.push(if is_pickup {
                        TaskRef::Pickup(si)
                    } else {
                        TaskRef::Delivery(si)
                    });
                } else {
                    return Err(Error::Other(format!(
                        "warm-start: no job or shipment half at location {li}"
                    )));
                }
            }
            // Ignore start/end/break/etc — re-derived by evaluate_route.
        }

        if steps.is_empty() {
            continue; // Empty route; skip.
        }

        let metrics = evaluate_route(problem, matrix, vehicle, &steps)
            .map_err(|e| Error::Other(format!("warm-start: route {} infeasible: {e}", r.vehicle)))?;

        routes.push(Route { vehicle_idx: veh_idx, steps, metrics });
    }

    // Unassigned = jobs not in any route.
    let unassigned: Vec<TaskRef> = (0..problem.jobs.len())
        .filter(|i| !visited_jobs.contains(i))
        .map(TaskRef::Job)
        .collect();

    let mut sol = Solution {
        routes,
        unassigned,
        summary: Default::default(),
    };
    sol.recompute_summary(problem);
    Ok(sol)
}
