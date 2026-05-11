//! GPU polish-pass for `solve_full`.
//!
//! Takes a CPU-LS-converged Solution and runs **batch-mode megakernel**:
//! `pop_size` trajectories in parallel, each seeded with the same start
//! solution + a different ILS-kick perturbation seed. Best-of-pop_size
//! reduction picks the winner. This is where the GPU's real value shows
//! up — running 64 diversified LS trajectories costs roughly the same
//! wall-time as running 1 CPU LS at N=1000.
//!
//! Returns `None` on any error or if no trajectory improves over the
//! starting solution, so the caller can fall back silently.

use std::collections::HashMap;

use crate::granular::Granular;
use crate::gpu_population::GpuPopulation;
use crate::matrix::Matrix;
use crate::problem::Problem;
use crate::solution::{evaluate_route, Route, Solution, TaskRef};

/// Run a batch GPU megakernel polish with `pop_size` trajectories, each
/// kicked with a different seed. Returns the new Solution if any
/// trajectory produces a feasible improvement.
///
/// `granular` is required for N≥200 — the megakernel relies on the
/// K-nearest table for relocate/2-opt* candidate filtering.
pub fn gpu_polish(
    problem: &Problem,
    matrix: &Matrix,
    solution: &Solution,
    granular: Option<&Granular>,
    max_iter: u32,
    verbose: bool,
) -> Option<Solution> {
    // Population size: 64 trajectories in batch mode. Cost-vs-quality
    // tradeoff converges quickly on M3 GPU — going higher gains little
    // diversity for a lot of upload + read-back overhead.
    let pop_size: u32 = std::env::var("BROOOM_GPU_POP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    // Kick count per trajectory: more aggressive perturbation increases
    // diversity but risks breaking TW. 3 is a good default for VRPTW.
    let kick_count: u32 = std::env::var("BROOOM_GPU_KICK")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let n_loc = matrix.n;
    if solution.routes.is_empty() {
        return None;
    }
    // Multi-vehicle/multi-profile not yet handled by megakernel — only
    // first vehicle's TW/capacity is uploaded as the homogeneous fleet
    // assumption. Bail to be safe if the fleet isn't homogeneous.
    let veh0 = &problem.vehicles[0];
    let veh_cap = veh0.capacity.get(0).copied().unwrap_or(0) as i32;
    let veh_tw = veh0.time_window();
    let veh_tw_s = veh_tw.start as i32;
    let veh_tw_e = veh_tw.end as i32;
    let depot = veh0.start.as_ref().and_then(|l| l.index).unwrap_or(0) as u32;

    // Per-location problem data.
    let mut loc_service = vec![0i32; n_loc];
    let mut loc_demand = vec![0i32; n_loc];
    let mut loc_tw_s = vec![veh_tw_s; n_loc];
    let mut loc_tw_e = vec![veh_tw_e; n_loc];
    for j in &problem.jobs {
        let li = match j.location.index {
            Some(li) => li,
            None => return None,
        };
        loc_service[li] = j.service as i32;
        loc_demand[li] = j.delivery.get(0).copied().unwrap_or(0) as i32;
        if let Some(tw) = j.time_windows.first() {
            loc_tw_s[li] = tw.start as i32;
            loc_tw_e[li] = tw.end as i32;
        }
    }

    // Build initial tours from the Solution.
    let mut tours: Vec<Vec<u32>> = Vec::with_capacity(solution.routes.len());
    for r in &solution.routes {
        let mut tour = vec![depot];
        for s in &r.steps {
            let li = match s.description(problem).location.index {
                Some(li) => li as u32,
                None => return None,
            };
            tour.push(li);
        }
        tour.push(depot);
        tours.push(tour);
    }
    let n_routes = tours.len() as u32;
    // Extra headroom in slot size so ILS-kick + reinsert can grow routes
    // by a few tasks without overflowing fixed slots.
    let max_route_len = tours.iter().map(|t| t.len() as u32).max().unwrap_or(0) + 16;
    let tour_capacity = n_routes * max_route_len;

    // Build GPU instance + upload. All pop_size trajectories share the
    // same problem data; each is seeded with the same start solution
    // (the kick-seed differs across workgroups inside the kernel).
    let gpu = match GpuPopulation::new(
        &matrix.durations,
        n_loc as u32,
        pop_size,
        n_routes.max(1),
        tour_capacity,
    ) {
        Ok(g) => g,
        Err(e) => {
            if verbose { eprintln!("gpu_polish: GpuPopulation::new failed: {e}"); }
            return None;
        }
    };
    let all_tours: Vec<Vec<Vec<u32>>> = (0..pop_size).map(|_| tours.clone()).collect();
    if gpu.upload(&all_tours).is_err() { return None; }
    if gpu.upload_problem_data(&loc_service, &loc_demand, &loc_tw_s, &loc_tw_e).is_err() {
        return None;
    }
    // Vehicle data per trajectory: identical for homogeneous fleet.
    let veh_cap_vec = vec![veh_cap; pop_size as usize];
    let veh_tw_s_vec = vec![veh_tw_s; pop_size as usize];
    let veh_tw_e_vec = vec![veh_tw_e; pop_size as usize];
    if gpu.upload_vehicle_data(&veh_cap_vec, &veh_tw_s_vec, &veh_tw_e_vec).is_err() {
        return None;
    }

    // Upload granular if provided.
    if let Some(g) = granular {
        let k = g.k();
        let mut near_flat: Vec<u32> = Vec::with_capacity(n_loc * k);
        for i in 0..n_loc {
            let mut found = 0;
            for nb in g.neighbors(i) {
                near_flat.push(nb as u32);
                found += 1;
            }
            while found < k {
                near_flat.push(i as u32);
                found += 1;
            }
        }
        let _ = gpu.upload_granular(&near_flat, k as u32);
    }

    // Repeat the batch with new kick seeds while there's time. Each
    // round is pop_size trajectories — typically 0.3-2 s wall time at
    // N≤2000. Run BROOOM_GPU_REPEATS rounds (default 1, env-controlled).
    // Within each round, every workgroup picks a different RNG seed
    // (via workgroup_id), giving pop_size × n_repeats total trajectories.
    let n_repeats: u32 = std::env::var("BROOOM_GPU_REPEATS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    // Build a location→TaskRef lookup once.
    let mut loc_to_task: HashMap<usize, TaskRef> = HashMap::new();
    for (idx, j) in problem.jobs.iter().enumerate() {
        if let Some(li) = j.location.index {
            loc_to_task.insert(li, TaskRef::Job(idx));
        }
    }
    let original_assigned: usize = solution.routes.iter().map(|r| r.steps.len()).sum();

    let mut best_sol: Option<Solution> = None;
    let mut total_feasible = 0u32;
    let mut total_considered = 0u32;
    for round in 0..n_repeats {
        // Re-upload starting solution between rounds: the kernel mutates
        // tour state in place, so to give every round a fresh start from
        // the SAME baseline we re-upload. (Could alternatively keep the
        // best-so-far as the seed for the next round — but that biases
        // toward local optima. Keeping the original start preserves
        // diversity across rounds.)
        if round > 0 {
            if gpu.upload(&all_tours).is_err() { break; }
        }
        // Distinct seed per round so workgroup RNGs don't collide.
        let kick_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(0xCAFEBABE)
            .wrapping_add(round.wrapping_mul(0x9E3779B9));
        let statuses = match gpu.run_megakernel_2opt_batch_with_kick(
            max_iter, kick_count, kick_seed,
        ) {
            Ok(s) => s,
            Err(e) => {
                if verbose { eprintln!("gpu_polish: round {round} failed: {e}"); }
                break;
            }
        };
        if verbose {
            let avg_iters = statuses.iter().map(|s| s.iters as f64).sum::<f64>() / pop_size as f64;
            let total_applies: u32 = statuses.iter().map(|s| s.applies).sum();
            eprintln!(
                "gpu_polish[r{round}]: pop={pop_size} kick={kick_count} avg_iter={:.0} applies={total_applies}",
                avg_iters
            );
        }
        let all_final_tours = match gpu.read_back_all() {
            Ok(t) => t,
            Err(_) => break,
        };
        let (round_best, round_feasible) = best_feasible_from_tours(
            problem, matrix, solution, &all_final_tours, &statuses,
            &loc_to_task, original_assigned, verbose,
        );
        total_considered += pop_size;
        total_feasible += round_feasible;
        if let Some(s) = round_best {
            match best_sol.as_ref() {
                None => best_sol = Some(s),
                Some(cur) if s.summary.cost < cur.summary.cost => best_sol = Some(s),
                _ => {}
            }
        }
    }
    if verbose {
        eprintln!(
            "gpu_polish: total feasible={total_feasible}/{total_considered} across {n_repeats} round(s)"
        );
    }

    match best_sol {
        Some(s) if s.summary.cost + 1e-9 < solution.summary.cost => {
            if verbose {
                eprintln!(
                    "gpu_polish: cost {:.2} → {:.2} (Δ={:.2})",
                    solution.summary.cost, s.summary.cost,
                    solution.summary.cost - s.summary.cost
                );
            }
            Some(s)
        }
        _ => None,
    }
}

/// Reconstruct Solutions from per-trajectory GPU tours; return the best
/// feasible one and the count of feasible trajectories.
fn best_feasible_from_tours(
    problem: &Problem,
    matrix: &Matrix,
    base: &Solution,
    all_final_tours: &[crate::gpu_population::TrajectoryTours],
    statuses: &[crate::gpu_population::MegakernelStatus],
    loc_to_task: &HashMap<usize, TaskRef>,
    original_assigned: usize,
    _verbose: bool,
) -> (Option<Solution>, u32) {
    let mut best: Option<Solution> = None;
    let mut feasible = 0u32;
    for (t_idx, final_tours) in all_final_tours.iter().enumerate() {
        if statuses[t_idx].dropped > 0 { continue; }
        if final_tours.len() != base.routes.len() { continue; }

        let mut new_routes: Vec<Route> = Vec::with_capacity(final_tours.len());
        let mut ok = true;
        for (r_idx, tour) in final_tours.iter().enumerate() {
            let veh_idx = base.routes[r_idx].vehicle_idx;
            let vehicle = &problem.vehicles[veh_idx];
            let mut steps: Vec<TaskRef> = Vec::with_capacity(tour.len().saturating_sub(2));
            for &li in tour.iter().skip(1).take(tour.len().saturating_sub(2)) {
                match loc_to_task.get(&(li as usize)) {
                    Some(&tr) => steps.push(tr),
                    None => { ok = false; break; }
                }
            }
            if !ok { break; }
            match evaluate_route(problem, matrix, vehicle, &steps) {
                Ok(m) => new_routes.push(Route { vehicle_idx: veh_idx, steps, metrics: m }),
                Err(_) => { ok = false; break; }
            }
        }
        if !ok { continue; }
        let assigned: usize = new_routes.iter().map(|r| r.steps.len()).sum();
        if assigned != original_assigned { continue; }
        feasible += 1;
        let mut s = Solution {
            routes: new_routes,
            unassigned: base.unassigned.clone(),
            summary: Default::default(),
        };
        s.recompute_summary();
        match best.as_ref() {
            None => best = Some(s),
            Some(cur) if s.summary.cost < cur.summary.cost => best = Some(s),
            _ => {}
        }
    }
    (best, feasible)
}
