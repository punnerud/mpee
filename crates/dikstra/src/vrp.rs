//! Vehicle Routing Problem solver (CVRPTW: Capacity + Time Windows).
//!
//! Solves a single-depot or multi-depot VRP variant with the following
//! constraints:
//!
//!   * **Capacity**: sum of `Job::demand` along a route ≤ `Vehicle::capacity`.
//!   * **Time windows**: each job has `[tw_start, tw_end]`; arrival time at
//!     the job must be ≤ `tw_end`. If we arrive before `tw_start`, the
//!     vehicle waits (no charge).
//!   * **Service durations**: each job adds `service_time` to the schedule.
//!   * **Vehicle shifts**: each vehicle has its own start time + horizon.
//!
//! Algorithm:
//!
//!   1. **Greedy parallel insertion** (constructive). Repeatedly find the
//!      cheapest feasible (vehicle, position, job) triple and insert it.
//!      Stops when no insertion is feasible.
//!   2. **Local search** (improving): relocate (move one job within or
//!      between routes) and 2-opt (reverse a sub-segment). First-improvement
//!      acceptance, restart sweep on improvement, stop on full no-improving
//!      pass.
//!
//! The travel-time matrix is supplied externally (e.g. from `ch::matrix`).
//! Indices into the matrix are: `0` = depot/start virtual node, `1..` =
//! each job's location, in the order given. (Multi-depot extends this by
//! giving each vehicle its own start/end node.)

#[derive(Debug, Clone)]
pub struct Job {
    pub id: u32,
    /// Index into the matrix (≥ 1 — index 0 reserved for the first depot).
    pub matrix_idx: u32,
    pub demand: f32,
    pub tw_start: f32,
    pub tw_end: f32,
    pub service_time: f32,
}

#[derive(Debug, Clone)]
pub struct Vehicle {
    pub id: u32,
    /// Matrix index for this vehicle's start location.
    pub start_idx: u32,
    /// Matrix index for the end location (often == start; depot return).
    pub end_idx: u32,
    pub capacity: f32,
    pub shift_start: f32,
    pub shift_end: f32,
}

#[derive(Debug, Clone)]
pub struct Problem<'a> {
    pub jobs: Vec<Job>,
    pub vehicles: Vec<Vehicle>,
    /// Row-major: `matrix[i * size + j]` = travel time (seconds) i → j.
    pub matrix: &'a [f32],
    pub matrix_size: usize,
}

#[derive(Debug, Clone)]
pub struct Route {
    pub vehicle_id: u32,
    /// Indexes into `Problem::jobs`.
    pub jobs: Vec<usize>,
    /// Cumulative arrival times at each job (matches `jobs.len()`).
    pub arrival: Vec<f32>,
    /// Total in-transit time (excludes wait + service). For pure routing
    /// cost it's the most useful single number.
    pub travel_time: f32,
    /// Total demand carried.
    pub load: f32,
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub routes: Vec<Route>,
    pub unassigned: Vec<usize>,
    pub total_travel_time: f32,
}

#[inline]
fn t(p: &Problem, i: u32, j: u32) -> f32 {
    p.matrix[i as usize * p.matrix_size + j as usize]
}

/// Forward time-feasibility scan: simulates a candidate route and, if
/// feasible, returns its (arrival_times, total_travel_time, total_load).
fn simulate(
    p: &Problem,
    vehicle: &Vehicle,
    seq: &[usize],
) -> Option<(Vec<f32>, f32, f32)> {
    let mut load = 0.0;
    let mut arrival = Vec::with_capacity(seq.len());
    let mut now = vehicle.shift_start;
    let mut prev_idx = vehicle.start_idx;
    let mut travel = 0.0;
    for &job_idx in seq {
        let j = &p.jobs[job_idx];
        let tt = t(p, prev_idx, j.matrix_idx);
        travel += tt;
        let arr = now + tt;
        if arr > j.tw_end {
            return None;
        }
        let start_service = arr.max(j.tw_start);
        load += j.demand;
        if load > vehicle.capacity {
            return None;
        }
        arrival.push(arr);
        now = start_service + j.service_time;
        prev_idx = j.matrix_idx;
    }
    let return_tt = t(p, prev_idx, vehicle.end_idx);
    travel += return_tt;
    if now + return_tt > vehicle.shift_end {
        return None;
    }
    Some((arrival, travel, load))
}

/// Build an initial solution by greedy parallel insertion: pick the
/// (vehicle, position, job) triple that minimises the marginal travel-time
/// increase, insert, repeat until no feasible insertion remains.
fn greedy_insert(p: &Problem) -> Solution {
    let mut routes: Vec<Vec<usize>> = vec![Vec::new(); p.vehicles.len()];
    let mut current: Vec<Option<(Vec<f32>, f32, f32)>> = (0..p.vehicles.len())
        .map(|i| simulate(p, &p.vehicles[i], &[]))
        .collect();
    let mut assigned = vec![false; p.jobs.len()];

    loop {
        let mut best: Option<(usize, usize, usize, f32, Vec<f32>, f32, f32)> = None;
        // (vehicle_idx, job_idx, position, delta_cost, new_arrival, new_travel, new_load)
        for (v_idx, vehicle) in p.vehicles.iter().enumerate() {
            let cur_travel = current[v_idx].as_ref().map(|(_, t, _)| *t).unwrap_or(0.0);
            let route = &routes[v_idx];
            for j_idx in 0..p.jobs.len() {
                if assigned[j_idx] {
                    continue;
                }
                for pos in 0..=route.len() {
                    let mut trial = route.clone();
                    trial.insert(pos, j_idx);
                    if let Some((arr, travel, load)) = simulate(p, vehicle, &trial) {
                        let delta = travel - cur_travel;
                        if best
                            .as_ref()
                            .map_or(true, |(_, _, _, d, ..)| delta < *d)
                        {
                            best = Some((v_idx, j_idx, pos, delta, arr, travel, load));
                        }
                    }
                }
            }
        }
        match best {
            Some((v_idx, j_idx, pos, _delta, arr, travel, load)) => {
                routes[v_idx].insert(pos, j_idx);
                current[v_idx] = Some((arr, travel, load));
                assigned[j_idx] = true;
            }
            None => break,
        }
    }

    materialise(p, routes, assigned)
}

fn materialise(p: &Problem, routes: Vec<Vec<usize>>, assigned: Vec<bool>) -> Solution {
    let mut out_routes = Vec::new();
    let mut total = 0.0;
    for (v_idx, jobs) in routes.into_iter().enumerate() {
        if jobs.is_empty() {
            continue;
        }
        let v = &p.vehicles[v_idx];
        if let Some((arrival, travel, load)) = simulate(p, v, &jobs) {
            total += travel;
            out_routes.push(Route {
                vehicle_id: v.id,
                jobs,
                arrival,
                travel_time: travel,
                load,
            });
        }
    }
    let unassigned: Vec<usize> = (0..p.jobs.len()).filter(|&i| !assigned[i]).collect();
    Solution {
        routes: out_routes,
        unassigned,
        total_travel_time: total,
    }
}

/// Local-search improvement: relocate (move one job within/between routes)
/// + intra-route 2-opt. Repeats until a full sweep finds no improving move
/// or `max_iterations` is reached.
fn improve(p: &Problem, sol: &mut Solution, max_iterations: usize) {
    let mut iter = 0;
    loop {
        iter += 1;
        if iter > max_iterations {
            break;
        }
        if !try_relocate(p, sol) && !try_two_opt(p, sol) {
            break;
        }
    }
}

/// Try moving a single job from one position to another (in same or
/// different route). Returns true if any improving move was applied.
fn try_relocate(p: &Problem, sol: &mut Solution) -> bool {
    let mut improved = false;
    'outer: for src_route in 0..sol.routes.len() {
        for src_pos in 0..sol.routes[src_route].jobs.len() {
            // Cost of removing the job from src_route.
            let mut try_route = sol.routes[src_route].jobs.clone();
            let job_idx = try_route.remove(src_pos);
            let src_vehicle = vehicle_for(p, sol.routes[src_route].vehicle_id);
            let src_after = match simulate(p, src_vehicle, &try_route) {
                Some(s) => s,
                None => continue, // Can't even rebuild without this job? Shouldn't happen.
            };
            let src_old_travel = sol.routes[src_route].travel_time;
            let src_save = src_old_travel - src_after.1;

            // Try inserting into every route at every position.
            for dst_route in 0..sol.routes.len() {
                let dst_vehicle = vehicle_for(p, sol.routes[dst_route].vehicle_id);
                let dst_jobs_base = if dst_route == src_route {
                    &try_route
                } else {
                    &sol.routes[dst_route].jobs
                };
                let dst_old_travel = if dst_route == src_route {
                    src_after.1
                } else {
                    sol.routes[dst_route].travel_time
                };
                for dst_pos in 0..=dst_jobs_base.len() {
                    if dst_route == src_route && dst_pos == src_pos {
                        continue; // No-op.
                    }
                    let mut trial = dst_jobs_base.clone();
                    trial.insert(dst_pos, job_idx);
                    if let Some((arr, travel, load)) = simulate(p, dst_vehicle, &trial) {
                        let dst_increase = travel - dst_old_travel;
                        // Net change to total cost.
                        let net = if dst_route == src_route {
                            travel - src_old_travel
                        } else {
                            -src_save + dst_increase
                        };
                        if net < -1e-3 {
                            // Apply.
                            sol.total_travel_time += net;
                            if dst_route == src_route {
                                sol.routes[src_route].jobs = trial;
                                sol.routes[src_route].travel_time = travel;
                                sol.routes[src_route].arrival = arr;
                                sol.routes[src_route].load = load;
                            } else {
                                sol.routes[src_route].jobs = try_route.clone();
                                sol.routes[src_route].travel_time = src_after.1;
                                sol.routes[src_route].arrival = src_after.0.clone();
                                sol.routes[src_route].load = src_after.2;
                                sol.routes[dst_route].jobs = trial;
                                sol.routes[dst_route].travel_time = travel;
                                sol.routes[dst_route].arrival = arr;
                                sol.routes[dst_route].load = load;
                            }
                            improved = true;
                            continue 'outer; // Restart sweep.
                        }
                    }
                }
            }
        }
    }
    improved
}

/// Intra-route 2-opt: reverse `[i..=j]` if it improves the route.
fn try_two_opt(p: &Problem, sol: &mut Solution) -> bool {
    let mut improved = false;
    'outer: for r_idx in 0..sol.routes.len() {
        let v = vehicle_for(p, sol.routes[r_idx].vehicle_id);
        let n = sol.routes[r_idx].jobs.len();
        if n < 3 {
            continue;
        }
        for i in 0..n - 1 {
            for j in (i + 1)..n {
                let mut trial = sol.routes[r_idx].jobs.clone();
                trial[i..=j].reverse();
                if let Some((arr, travel, load)) = simulate(p, v, &trial) {
                    if travel + 1e-3 < sol.routes[r_idx].travel_time {
                        sol.total_travel_time += travel - sol.routes[r_idx].travel_time;
                        sol.routes[r_idx].jobs = trial;
                        sol.routes[r_idx].travel_time = travel;
                        sol.routes[r_idx].arrival = arr;
                        sol.routes[r_idx].load = load;
                        improved = true;
                        continue 'outer;
                    }
                }
            }
        }
    }
    improved
}

fn vehicle_for<'a>(p: &'a Problem<'_>, vehicle_id: u32) -> &'a Vehicle {
    p.vehicles
        .iter()
        .find(|v| v.id == vehicle_id)
        .expect("vehicle not in problem")
}

/// End-to-end: build initial solution + run local search.
pub fn solve(p: &Problem, max_iterations: usize) -> Solution {
    let mut sol = greedy_insert(p);
    improve(p, &mut sol, max_iterations);
    sol
}
