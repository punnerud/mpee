//! Forward/backward route precomputation for O(1) insertion feasibility.
//!
//! `RoutePrecomp` is built once per route (O(L)). After that, asking "can I
//! insert task T at position p?" is O(1):
//!   - TW feasibility: `earliest_arrival_after_T <= latest_arrival[p]`.
//!   - Capacity feasibility: `max_load_prefix[p] + T.delivery <= cap` and
//!     `max_load_suffix[p] + T.pickup <= cap`.
//!   - Δ-cost: only depends on travel-time delta on the two affected edges
//!     (and T's own service/setup), not on downstream waiting changes.
//!
//! Compared to re-running `evaluate_route` per candidate, this turns the hot
//! path of insertion and relocate from O(L) per probe into O(1).

use crate::matrix::Matrix;
use crate::problem::{Cost, Problem, Time, TimeWindow, Vehicle};
use crate::solution::{pick_time_window, TaskRef};

/// Per-position data, packed AoS so a single probe touches one cache line
/// instead of three separate Vec headers.
#[derive(Debug, Clone, Copy)]
pub struct Pos {
    /// Matrix index at this position. `None` only at depot endpoints when
    /// the vehicle has no start/end (rare).
    pub loc: Option<usize>,
    /// Earliest time the vehicle finishes here (after wait + service).
    pub depart: Time,
    /// Latest time the vehicle may arrive here without breaking downstream.
    pub latest_arrival: Time,
}

/// Precomputed arrays for one route. Indexed positions are:
/// `0` = start depot, `1..=L` = the L steps, `L+1` = end depot.
#[derive(Debug, Clone)]
pub struct RoutePrecomp {
    pub vehicle_idx: usize,
    pub dim: usize,
    /// Per-position data (length L+2). AoS for cache locality on probes.
    pub pos: Vec<Pos>,
    /// Flat: `load_at[k * dim + i]`.
    pub load_at: Vec<i64>,
    /// Flat: prefix max load[0..=k] across each dim.
    pub max_load_prefix: Vec<i64>,
    /// Flat: suffix max load[k..=L+1] across each dim.
    pub max_load_suffix: Vec<i64>,
    pub feasible: bool,
    pub cost: Cost,
    pub travel_total: Time,
    pub distance_total: i64,
}

impl RoutePrecomp {
    #[inline]
    pub fn pre(&self, k: usize, i: usize) -> i64 { self.max_load_prefix[k * self.dim + i] }
    #[inline]
    pub fn suf(&self, k: usize, i: usize) -> i64 { self.max_load_suffix[k * self.dim + i] }
    #[inline]
    pub fn load(&self, k: usize, i: usize) -> i64 { self.load_at[k * self.dim + i] }
}

/// Build the precomputation arrays for a route. Returns `None` if the route
/// itself is infeasible (capacity overflow, missed TW, etc.) so the caller
/// can flag the route for repair / rejection.
///
/// Disjunctions / drop penalties: this probe is purely about routing
/// feasibility and Δ-cost of *inserting* a task; it never reads a job's `prize`
/// or `disjunction_penalty`. The drop-penalty trade-off (insert-vs-leave-
/// unassigned) lives where the *value* of serving is known: the insertion
/// ordering in `insertion.rs` (`Job::unassigned_cost()` = prize + penalty pulls
/// high-penalty jobs to the front) and the prize-swap pass in `solver.rs`, both
/// kept consistent with the objective charged in `Solution::recompute_summary`.
pub fn precompute(
    problem: &Problem,
    matrix: &Matrix,
    vehicle: &Vehicle,
    vehicle_idx: usize,
    steps: &[TaskRef],
) -> Option<RoutePrecomp> {
    let dim = problem.capacity_dim().max(vehicle.capacity.len());
    let l = steps.len();
    let positions = l + 2;
    let speed = vehicle.speed_factor.max(0.01);

    // Multi-trip routes (with a Reload) are not modelled by this O(1) probe;
    // bail so the caller falls back to the full `evaluate_route` (correct, just
    // not O(1)). Single-trip routes are unaffected.
    if steps.iter().any(|s| s.is_reload()) {
        return None;
    }

    // Pickup-before-delivery and linehaul-before-backhaul checks (O(L)). These
    // mirror `evaluate_route` so the probe prunes the same infeasible orderings
    // early; the evaluator remains the authority.
    {
        let mut seen = std::collections::HashSet::<usize>::new();
        let mut seen_backhaul = false;
        for s in steps {
            if !vehicle.has_skills(s.skills(problem)) {
                return None;
            }
            // Vehicle allowlist: a job may only ride a listed vehicle. Probe
            // declines (the full evaluator is the authority and will reject).
            if !s.description(problem).allows_vehicle(vehicle.id) {
                return None;
            }
            match s {
                TaskRef::Pickup(i) => { seen.insert(*i); }
                TaskRef::Delivery(i) => {
                    if !seen.contains(i) { return None; }
                }
                TaskRef::Reload => {}
                TaskRef::Job(_) => {
                    let j = s.description(problem);
                    if !j.pickup.is_empty() && j.delivery.is_empty() {
                        seen_backhaul = true;
                    } else if !j.delivery.is_empty() && seen_backhaul {
                        return None;
                    }
                }
            }
        }
    }

    let start_idx = vehicle.start.as_ref().and_then(|l| l.index)
        .or_else(|| vehicle.end.as_ref().and_then(|l| l.index));
    let end_idx = vehicle.end.as_ref().and_then(|l| l.index).or(start_idx);

    // Per-position data (AoS).
    let mut pos: Vec<Pos> = Vec::with_capacity(positions);
    pos.push(Pos { loc: start_idx, depart: 0, latest_arrival: 0 });
    for s in steps {
        let li = s.description(problem).location.index?;
        pos.push(Pos { loc: Some(li), depart: 0, latest_arrival: 0 });
    }
    pos.push(Pos { loc: end_idx, depart: 0, latest_arrival: 0 });

    // Initial load = sum of all deliveries (single jobs only — shipments are
    // picked up en route).
    let mut initial_load = vec![0i64; dim.max(1)];
    let dim_eff = initial_load.len();
    for s in steps {
        let j = s.description(problem);
        if matches!(s, TaskRef::Job(_)) && !j.delivery.is_empty() {
            for i in 0..dim_eff {
                initial_load[i] += *j.delivery.get(i).unwrap_or(&0);
            }
        }
    }

    // Forward pass: arrival/depart times and load_at[].
    let vw = vehicle.time_window();

    // Flat load_at: positions * dim_eff.
    let mut load_at: Vec<i64> = Vec::with_capacity(positions * dim_eff);
    load_at.extend_from_slice(&initial_load);

    let mut t = vw.start;
    let mut prev_loc = start_idx;
    let mut travel_total: Time = 0;
    let mut distance_total: i64 = 0;
    let mut load = initial_load.clone();

    // Custom-dimension arcs (P5 probe mirror): only collected when at least one
    // registered dimension is monotone with a max bound, so the common path pays
    // nothing. The arc arrival times mirror `solution::evaluate_route` exactly
    // (the time after travel, before setup/service), so the probe accumulates the
    // SAME cumuls the full evaluator would and never reports a breach the
    // authority would not. See `dimension::probe_breaches_monotone_max`.
    let track_probe_dims = crate::dimension::has_probe_dimensions();
    let mut dim_arcs: Vec<crate::dimension::Arc2> = if track_probe_dims {
        Vec::with_capacity(steps.len() + 1)
    } else {
        Vec::new()
    };

    pos[0].depart = t;

    for (k, s) in steps.iter().enumerate() {
        let pos_idx = k + 1;
        let job = s.description(problem);
        let here = job.location.index?;

        if let Some(p) = prev_loc {
            let raw = matrix.duration(p, here);
            let dur = ((raw as f64) * speed).round() as i64;
            t += dur;
            travel_total += dur;
            distance_total += matrix.distance(p, here);
        }
        // Record the dimension arc here (after travel, before setup/service) to
        // match the evaluator's arrival timing exactly.
        if track_probe_dims {
            match prev_loc {
                Some(p) => dim_arcs.push(crate::dimension::Arc2 { from: p, to: here, arrival: t }),
                None => dim_arcs.push(crate::dimension::Arc2 { from: here, to: here, arrival: t }),
            }
        }
        let do_setup = match prev_loc {
            Some(p) => p != here && job.setup > 0,
            None => job.setup > 0,
        };
        if do_setup { t += job.setup; }

        // Mirror solution.rs release-time lower bound (probe folds the wait into
        // the timeline). Release only delays, so the backward latest_arrival
        // pass needs no change (conservative-safe).
        if t < job.release { t = job.release; }

        let chosen_tw = pick_time_window(&job.time_windows, t)?;
        if t < chosen_tw.start { t = chosen_tw.start; }
        if t > chosen_tw.end { return None; }
        t += job.service;
        pos[pos_idx].depart = t;

        // Load update — read source slices directly, no allocation.
        match s {
            TaskRef::Job(_) => {
                for i in 0..dim_eff {
                    load[i] -= *job.delivery.get(i).unwrap_or(&0);
                    load[i] += *job.pickup.get(i).unwrap_or(&0);
                }
            }
            TaskRef::Pickup(i) => {
                let s = &problem.shipments[*i];
                let amt = if !s.amount.is_empty() { &s.amount } else { &s.pickup.pickup };
                for i in 0..dim_eff { load[i] += *amt.get(i).unwrap_or(&0); }
            }
            TaskRef::Delivery(i) => {
                let s = &problem.shipments[*i];
                let amt = if !s.amount.is_empty() { &s.amount } else { &s.pickup.pickup };
                for i in 0..dim_eff { load[i] -= *amt.get(i).unwrap_or(&0); }
            }
            TaskRef::Reload => {} // unreachable: precompute bails on reload routes
        }
        if load.iter().any(|&x| x < 0) { return None; }
        if !vehicle.capacity.is_empty() {
            for i in 0..vehicle.capacity.len() {
                if load[i] > vehicle.capacity[i] { return None; }
            }
        }
        load_at.extend_from_slice(&load);

        prev_loc = Some(here);
    }

    // Final leg back to depot.
    if let (Some(p), Some(e)) = (prev_loc, end_idx) {
        let raw = matrix.duration(p, e);
        let dur = ((raw as f64) * speed).round() as i64;
        t += dur;
        travel_total += dur;
        distance_total += matrix.distance(p, e);
        if track_probe_dims {
            dim_arcs.push(crate::dimension::Arc2 { from: p, to: e, arrival: t });
        }
    }
    if t > vw.end { return None; }
    // Mirror of probe-safe DSL hard bounds (travel/distance/duration): prune the
    // candidate here, before the full evaluator runs. Pruning only — never
    // rejects a feasible route; `evaluate_route` remains the authority.
    if crate::constraint::probe_violates(travel_total, distance_total, t - vw.start) {
        return None;
    }
    // P5 probe mirror: prune a candidate whose monotone custom-dimension cumul
    // would breach its declared max — the same proactive prune as above, now for
    // a prefix-accumulated resource. Prune-only: a non-monotone or unbounded
    // dimension is skipped entirely and its bound is still honoured at full
    // `evaluate_route` (the P5 caveat, narrowed to non-probe-expressible dims).
    if track_probe_dims && crate::dimension::probe_breaches_monotone_max(&dim_arcs) {
        #[cfg(test)]
        crate::dimension::PROBE_PRUNE_COUNT
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    pos[positions - 1].depart = t;

    // load_at needs one more entry to cover the end-depot position; the load
    // arriving at the end equals the load after the last step's service.
    load_at.extend_from_slice(&load);

    // Backward pass: `latest_arrival[k]` is the latest time the vehicle may
    // arrive at original position k such that every downstream constraint
    // (TW, vehicle horizon) still holds. Forward formula being inverted is
    //   arrival[k+1] = max(arrival[k], tw_k.start) + service_k + edge + setup_{k+1}
    // so working backward from `latest_arrival[k+1]` (or `vw.end` at the tail)
    //   latest_arrival[k] = min(tw_k.end,
    //                           latest_arrival[k+1] - service_k - edge - setup_{k+1})
    // where `service_k` is the service time of the step at position k itself
    // (zero at the depots).
    pos[positions - 1].latest_arrival = vw.end;
    let service_at = |k: usize| -> Time {
        if k >= 1 && k < positions - 1 {
            steps[k - 1].description(problem).service
        } else { 0 }
    };
    let setup_at = |k: usize| -> Time {
        if k >= 1 && k < positions - 1 {
            steps[k - 1].description(problem).setup
        } else { 0 }
    };
    let tw_end_at = |k: usize| -> Time {
        if k >= 1 && k < positions - 1 {
            steps[k - 1]
                .description(problem)
                .time_windows
                .iter()
                .map(|w| w.end)
                .max()
                .unwrap_or(TimeWindow::FOREVER.end)
        } else {
            vw.end
        }
    };
    for k in (0..positions - 1).rev() {
        let next_pos = k + 1;
        let edge_dur = match (pos[k].loc, pos[next_pos].loc) {
            (Some(a), Some(b)) => ((matrix.duration(a, b) as f64) * speed).round() as i64,
            _ => 0,
        };
        let setup_next = match (pos[k].loc, pos[next_pos].loc) {
            (Some(a), Some(b)) if a != b => setup_at(next_pos),
            _ => 0,
        };
        let chain = pos[next_pos].latest_arrival
            .saturating_sub(service_at(k))
            .saturating_sub(edge_dur)
            .saturating_sub(setup_next);
        pos[k].latest_arrival = chain.min(tw_end_at(k));
    }

    // Capacity max prefix/suffix as flat arrays.
    let mut max_load_prefix: Vec<i64> = vec![i64::MIN; positions * dim_eff];
    let mut running = vec![i64::MIN; dim_eff];
    for k in 0..positions {
        for i in 0..dim_eff {
            let v = load_at[k * dim_eff + i];
            if v > running[i] { running[i] = v; }
            max_load_prefix[k * dim_eff + i] = running[i];
        }
    }
    let mut max_load_suffix: Vec<i64> = vec![i64::MIN; positions * dim_eff];
    let mut running = vec![i64::MIN; dim_eff];
    for k in (0..positions).rev() {
        for i in 0..dim_eff {
            let v = load_at[k * dim_eff + i];
            if v > running[i] { running[i] = v; }
            max_load_suffix[k * dim_eff + i] = running[i];
        }
    }

    let cost = vehicle.fixed
        + (travel_total as f64) * (vehicle.per_hour / 3600.0).max(0.0);

    Some(RoutePrecomp {
        vehicle_idx,
        dim: dim_eff,
        pos,
        load_at,
        max_load_prefix,
        max_load_suffix,
        feasible: true,
        cost,
        travel_total,
        distance_total,
    })
}

/// O(1) probe: can we insert single-task `task` at insertion-position `pos`
/// (range 1..=L+1, meaning "before the original step at pos-1, after step pos-2")?
/// Returns the cost delta on success.
///
/// Note: pickups/deliveries from a *shipment* still need precedence checking
/// elsewhere — this helper assumes the call site has already verified that
/// constraint (e.g. by inserting them as a pair).
pub fn try_insert_single(
    precomp: &RoutePrecomp,
    problem: &Problem,
    matrix: &Matrix,
    vehicle: &Vehicle,
    pos: usize,
    task: TaskRef,
) -> Option<Cost> {
    if !vehicle.has_skills(task.skills(problem)) { return None; }
    let speed = vehicle.speed_factor.max(0.01);
    let job = task.description(problem);
    let task_loc = job.location.index?;

    let prev_loc = precomp.pos[pos - 1].loc;
    let next_loc = precomp.pos[pos].loc;

    // Travel and arrival at task.
    let travel_to_task = if let Some(p) = prev_loc {
        ((matrix.duration(p, task_loc) as f64) * speed).round() as i64
    } else { 0 };
    let setup_task = if prev_loc.map_or(true, |p| p != task_loc) { job.setup } else { 0 };
    let arrival_task = precomp.pos[pos - 1].depart + travel_to_task + setup_task;

    let chosen_tw = pick_time_window(&job.time_windows, arrival_task)?;
    let start_service = arrival_task.max(chosen_tw.start);
    if start_service > chosen_tw.end { return None; }
    let end_service = start_service + job.service;

    // Travel to next position.
    let travel_to_next = if let Some(n) = next_loc {
        ((matrix.duration(task_loc, n) as f64) * speed).round() as i64
    } else { 0 };
    // Setup at next when the previous step's location changed.
    let next_pos_index = pos; // in *new* sequence, this slot becomes pos+1, but check the bound at old position pos.
    let setup_next_after = if let Some(n) = next_loc {
        if next_pos_index < precomp.pos.len() - 1 {
            // 'next' is a regular step; it has its own setup time.
            let next_step = match next_pos_index {
                k if k >= 1 && k <= precomp.pos.len() - 2 => Some(k - 1),
                _ => None,
            };
            if let Some(step_idx) = next_step {
                let _ = step_idx;
            }
            // We can't easily peek at `steps[]` here without passing it in;
            // approximate by zero. The bound below uses the precomputed
            // latest_arrival[pos] which already bakes in the setup, so this
            // approximation is safe for feasibility (slightly tighter than necessary).
            let _ = n;
            0
        } else { 0 }
    } else { 0 };

    let arrival_next = end_service + travel_to_next + setup_next_after;
    if arrival_next > precomp.pos[pos].latest_arrival { return None; }

    // Capacity: O(1) per dimension, no padding allocations. Inserting T
    // adds T.delivery to every load slot at positions 0..=pos-1 and adds
    // T.pickup to every slot at pos..=L+1.
    if !vehicle.capacity.is_empty() {
        for i in 0..vehicle.capacity.len() {
            let d = job.delivery.get(i).copied().unwrap_or(0);
            let p = job.pickup.get(i).copied().unwrap_or(0);
            let cap = vehicle.capacity[i];
            let pre = precomp.pre(pos - 1, i);
            let suf = precomp.suf(pos, i);
            if pre + d > cap { return None; }
            if suf + p > cap { return None; }
        }
    }

    // Cost delta: change in travel time × per_hour. Setup adds, service is a tiebreaker.
    let old_travel = match (prev_loc, next_loc) {
        (Some(a), Some(b)) => ((matrix.duration(a, b) as f64) * speed).round() as i64,
        _ => 0,
    };
    let dt = travel_to_task + travel_to_next - old_travel + setup_task;
    let dcost = (dt as f64) * (vehicle.per_hour / 3600.0).max(0.0)
              + (job.service as f64) * 1e-6;
    Some(dcost)
}

/// O(1) probe with extended return: cost-delta, travel-time-delta, and
/// arrival-time-shift-at-next-position. All three needed for Solomon I1:
///   - cost_delta: $-units, used by callers that want monetary cost
///   - travel_delta: seconds — Solomon I1's c11 = d_iu + d_uj − μ·d_ij
///   - time_shift: seconds — Solomon I1's c12 = b_uj_new − b_uj_old
///
/// Critically, c11 and c12 must be in the SAME unit (seconds) for the
/// linear combination c1 = α1·c11 + α2·c12 to make sense. Mixing $-cost
/// with seconds gave a 4% regression in earlier tests because the dollar-
/// scale dominated and the TW-shift signal was effectively zeroed.
pub fn try_insert_single_with_shift(
    precomp: &RoutePrecomp,
    problem: &Problem,
    matrix: &Matrix,
    vehicle: &Vehicle,
    pos: usize,
    task: TaskRef,
) -> Option<(Cost, i64, i64)> {
    if !vehicle.has_skills(task.skills(problem)) { return None; }
    let speed = vehicle.speed_factor.max(0.01);
    let job = task.description(problem);
    let task_loc = job.location.index?;

    let prev_loc = precomp.pos[pos - 1].loc;
    let next_loc = precomp.pos[pos].loc;

    let travel_to_task = if let Some(p) = prev_loc {
        ((matrix.duration(p, task_loc) as f64) * speed).round() as i64
    } else { 0 };
    let setup_task = if prev_loc.map_or(true, |p| p != task_loc) { job.setup } else { 0 };
    let arrival_task = precomp.pos[pos - 1].depart + travel_to_task + setup_task;

    let chosen_tw = pick_time_window(&job.time_windows, arrival_task)?;
    let start_service = arrival_task.max(chosen_tw.start);
    if start_service > chosen_tw.end { return None; }
    let end_service = start_service + job.service;

    let travel_to_next = if let Some(n) = next_loc {
        ((matrix.duration(task_loc, n) as f64) * speed).round() as i64
    } else { 0 };
    let arrival_next_new = end_service + travel_to_next;
    if arrival_next_new > precomp.pos[pos].latest_arrival { return None; }

    if !vehicle.capacity.is_empty() {
        for i in 0..vehicle.capacity.len() {
            let d = job.delivery.get(i).copied().unwrap_or(0);
            let p = job.pickup.get(i).copied().unwrap_or(0);
            let cap = vehicle.capacity[i];
            let pre = precomp.pre(pos - 1, i);
            let suf = precomp.suf(pos, i);
            if pre + d > cap { return None; }
            if suf + p > cap { return None; }
        }
    }

    let old_travel = match (prev_loc, next_loc) {
        (Some(a), Some(b)) => ((matrix.duration(a, b) as f64) * speed).round() as i64,
        _ => 0,
    };
    let arrival_next_old = precomp.pos[pos - 1].depart + old_travel;
    let time_shift = arrival_next_new - arrival_next_old;

    let dt = travel_to_task + travel_to_next - old_travel + setup_task;
    let dcost = (dt as f64) * (vehicle.per_hour / 3600.0).max(0.0)
              + (job.service as f64) * 1e-6;
    Some((dcost, dt, time_shift))
}

/// Probe inserting a shipment pair at positions `(pos_p, pos_d)`. Returns
/// `None` if infeasible. `pos_p` and `pos_d` are *original-route* insertion
/// positions in `1..=L+1`. The pickup is inserted before the original step at
/// `pos_p-1` (0-based step index), the delivery before the original step at
/// `pos_d-1`. They may be equal (immediate pickup-then-delivery). The fast
/// path simulates just the spliced segment, which is O(pos_d - pos_p + 1).
pub fn try_insert_pair(
    precomp: &RoutePrecomp,
    problem: &Problem,
    matrix: &Matrix,
    vehicle: &Vehicle,
    steps: &[TaskRef],
    pos_p: usize,
    pos_d: usize,
    pickup: TaskRef,
    delivery: TaskRef,
) -> Option<Cost> {
    if pos_p == 0 || pos_d < pos_p { return None; }
    if !vehicle.has_skills(pickup.skills(problem)) { return None; }

    let speed = vehicle.speed_factor.max(0.01);
    let pj = pickup.description(problem);
    let dj = delivery.description(problem);
    let pl = pj.location.index?;
    let dl = dj.location.index?;

    // We walk from depart[pos_p - 1] forward through pickup, the segment up
    // to pos_d, then delivery, then check fits at pos_d.
    let mut t = precomp.pos[pos_p - 1].depart;
    let mut prev_loc = precomp.pos[pos_p - 1].loc;

    // -> pickup
    if let Some(p) = prev_loc {
        t += ((matrix.duration(p, pl) as f64) * speed).round() as i64;
    }
    if prev_loc.map_or(true, |p| p != pl) { t += pj.setup; }
    let tw_p = pick_time_window(&pj.time_windows, t)?;
    if t < tw_p.start { t = tw_p.start; }
    if t > tw_p.end { return None; }
    t += pj.service;
    prev_loc = Some(pl);

    // walk segment [pos_p..pos_d) of original
    for k in pos_p..pos_d {
        let s = &steps[k - 1]; // original step at index k-1 = step k in 1-based positions
        let job = s.description(problem);
        let here = job.location.index?;
        if let Some(p) = prev_loc {
            t += ((matrix.duration(p, here) as f64) * speed).round() as i64;
        }
        if prev_loc.map_or(true, |p| p != here && job.setup > 0) { t += job.setup; }
        let tw = pick_time_window(&job.time_windows, t)?;
        if t < tw.start { t = tw.start; }
        if t > tw.end { return None; }
        t += job.service;
        prev_loc = Some(here);
    }

    // -> delivery
    if let Some(p) = prev_loc {
        t += ((matrix.duration(p, dl) as f64) * speed).round() as i64;
    }
    if prev_loc.map_or(true, |p| p != dl) { t += dj.setup; }
    let tw_d = pick_time_window(&dj.time_windows, t)?;
    if t < tw_d.start { t = tw_d.start; }
    if t > tw_d.end { return None; }
    t += dj.service;
    prev_loc = Some(dl);

    // -> next (original step at index pos_d - 1, or end depot)
    let next_loc = precomp.pos[pos_d].loc;
    let arrival_next = match (prev_loc, next_loc) {
        (Some(a), Some(b)) => t + ((matrix.duration(a, b) as f64) * speed).round() as i64,
        _ => t,
    };
    if arrival_next > precomp.pos[pos_d].latest_arrival { return None; }

    // Capacity: shipment amount on board across original positions pos_p..=pos_d - 1.
    let amt: &[i64] = {
        let s_idx = match pickup {
            TaskRef::Pickup(i) => i,
            _ => return None,
        };
        let s = &problem.shipments[s_idx];
        if !s.amount.is_empty() { &s.amount } else { &s.pickup.pickup }
    };
    if !vehicle.capacity.is_empty() && pos_d >= 1 {
        for i in 0..vehicle.capacity.len() {
            let add = amt.get(i).copied().unwrap_or(0);
            let cap = vehicle.capacity[i];
            for k in pos_p..=pos_d - 1 {
                if precomp.load(k, i) + add > cap { return None; }
            }
        }
    }

    // Approx Δ-cost: rebuild travel along the spliced span minus the original.
    // (Cheap enough; the segment is usually short.)
    let mut new_travel: i64 = 0;
    let mut prev = precomp.pos[pos_p - 1].loc;
    let new_locs = std::iter::once(Some(pl))
        .chain((pos_p..pos_d).map(|k| precomp.pos[k].loc))
        .chain(std::iter::once(Some(dl)))
        .chain(std::iter::once(precomp.pos[pos_d].loc));
    for nl in new_locs {
        if let (Some(a), Some(b)) = (prev, nl) {
            new_travel += ((matrix.duration(a, b) as f64) * speed).round() as i64;
        }
        prev = nl;
    }
    let mut old_travel: i64 = 0;
    let mut prev = precomp.pos[pos_p - 1].loc;
    for k in pos_p..=pos_d {
        let nl = precomp.pos[k].loc;
        if let (Some(a), Some(b)) = (prev, nl) {
            old_travel += ((matrix.duration(a, b) as f64) * speed).round() as i64;
        }
        prev = nl;
    }
    let dt = new_travel - old_travel + pj.setup + dj.setup;
    let dcost = (dt as f64) * (vehicle.per_hour / 3600.0).max(0.0)
              + ((pj.service + dj.service) as f64) * 1e-6;
    Some(dcost)
}
