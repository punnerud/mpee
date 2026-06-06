//! Cluster-first, route-second VRP decomposition.
//!
//! Splits the customers into K geometric/temporal clusters via K-medoids
//! on the distance matrix, allocates a fair share of vehicles per cluster,
//! solves each sub-problem in parallel via the existing pipeline, then
//! concatenates the routes. For large N, expected speedup is O(K) on the
//! N²-bound LS phase plus O(K) parallel multi-start.
//!
//! Trade-off: cross-cluster moves are forbidden, so a customer that
//! "should" have been in another cluster's vehicle stays misallocated.
//! Followed by an optional polish pass on the assembled solution to
//! rescue cross-cluster improvements.
//!
//! ## Why K-medoids and not K-means
//!
//! K-medoids works on a distance matrix directly — no coordinates needed.
//! For Solomon/Gehring-Homberger instances we have the matrix in hand;
//! coordinates are a derived view. Medoids are also more robust to TW
//! outliers than means.

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::HashMap;

use crate::matrix::Matrix;
use crate::problem::{Job, Problem, Vehicle};
use crate::solution::{Route, Solution, TaskRef};
use crate::solver::{solve_with_matrix, SolverConfig};

/// Result of a clustering pass: each job is assigned to a cluster index.
pub struct ClusterAssignment {
    /// `cluster_of[job_idx]` is the cluster index ∈ [0, k).
    pub cluster_of: Vec<usize>,
    pub k: usize,
    /// Medoid location index per cluster.
    pub medoids: Vec<usize>,
}

/// K-medoids over the distance matrix. Customers are the rows we cluster
/// (depot at index 0 is excluded). Initial medoids are furthest-point
/// samples to spread coverage. Lloyd-style assignment + medoid update.
pub fn kmedoids(matrix: &Matrix, customer_locs: &[usize], k: usize, max_iter: usize) -> ClusterAssignment {
    assert!(k >= 1);
    let n = customer_locs.len();
    if n == 0 {
        return ClusterAssignment { cluster_of: Vec::new(), k, medoids: Vec::new() };
    }
    let k = k.min(n);

    // Initial medoid selection: farthest-point sampling.
    let mut medoids: Vec<usize> = Vec::with_capacity(k);
    medoids.push(customer_locs[0]);
    while medoids.len() < k {
        // Pick point with maximum min-dist to current medoids.
        let mut best = 0usize;
        let mut best_d = i64::MIN;
        for &li in customer_locs {
            if medoids.contains(&li) { continue; }
            let mut min_d = i64::MAX;
            for &m in &medoids {
                let d = matrix.duration(li, m);
                if d < min_d { min_d = d; }
            }
            if min_d > best_d {
                best_d = min_d;
                best = li;
            }
        }
        medoids.push(best);
    }

    let mut cluster_of: Vec<usize> = vec![0; n];
    for _ in 0..max_iter {
        // Assignment step: each customer joins nearest medoid's cluster.
        let mut changed = false;
        for (i, &li) in customer_locs.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = i64::MAX;
            for (c, &m) in medoids.iter().enumerate() {
                let d = matrix.duration(li, m);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if cluster_of[i] != best {
                cluster_of[i] = best;
                changed = true;
            }
        }

        // Update step: new medoid = member with min total distance to others
        // in same cluster. O(|C|²) per cluster — fine for K=N/50 buckets.
        let mut new_medoids = vec![0usize; k];
        for c in 0..k {
            let members: Vec<usize> = customer_locs
                .iter()
                .enumerate()
                .filter(|(i, _)| cluster_of[*i] == c)
                .map(|(_, &li)| li)
                .collect();
            if members.is_empty() {
                new_medoids[c] = medoids[c]; // empty cluster — keep old medoid
                continue;
            }
            // For each candidate, sum dist to all other members.
            let mut best = members[0];
            let mut best_total = i64::MAX;
            for &cand in &members {
                let total: i64 = members.iter().map(|&o| matrix.duration(cand, o) as i64).sum();
                if total < best_total {
                    best_total = total;
                    best = cand;
                }
            }
            new_medoids[c] = best;
        }
        if new_medoids == medoids && !changed {
            break;
        }
        medoids = new_medoids;
    }

    ClusterAssignment { cluster_of, k, medoids }
}

/// Build a sub-`Problem` containing only the given jobs and a fair share of
/// vehicles. The shared `Matrix` is unchanged — sub-problems index it via
/// the original location indices.
fn subproblem(parent: &Problem, jobs: &[Job], vehicles: &[Vehicle]) -> Problem {
    Problem {
        jobs: jobs.to_vec(),
        vehicles: vehicles.to_vec(),
        shipments: Vec::new(),
        matrices: HashMap::new(),
        precedence: parent.precedence.clone(),
        description: parent.description.clone(),
    }
}

/// Solve via cluster-first decomposition. Caller decides `k` (commonly
/// `ceil(N / 50)`); 1 disables decomposition (falls back to flat solve).
pub fn solve_decomposed(
    problem: &Problem,
    matrix: &Matrix,
    config: &SolverConfig,
    k: usize,
) -> Solution {
    if k <= 1 {
        return solve_with_matrix(problem, matrix, config);
    }

    // Customer-side location indices (skip the depot at 0 if it's the only
    // shared start; we take all jobs' locations).
    let customer_locs: Vec<usize> = problem
        .jobs
        .iter()
        .filter_map(|j| j.location.index)
        .collect();
    if customer_locs.is_empty() {
        return solve_with_matrix(problem, matrix, config);
    }

    let assn = kmedoids(matrix, &customer_locs, k, 20);

    // Group jobs by cluster.
    let mut jobs_by_cluster: Vec<Vec<Job>> = vec![Vec::new(); assn.k];
    for (j, &c) in problem.jobs.iter().zip(assn.cluster_of.iter()) {
        jobs_by_cluster[c].push(j.clone());
    }

    // Allocate vehicles proportional to cluster demand (simple share).
    // Always give each cluster at least one vehicle.
    let n_veh = problem.vehicles.len();
    let mut veh_by_cluster: Vec<Vec<Vehicle>> = vec![Vec::new(); assn.k];
    let total_jobs: usize = problem.jobs.len();
    let mut allocated = 0usize;
    for c in 0..assn.k {
        let cluster_jobs = jobs_by_cluster[c].len();
        if cluster_jobs == 0 {
            continue;
        }
        let share = ((cluster_jobs as f64 / total_jobs as f64) * n_veh as f64).round() as usize;
        let share = share.max(1);
        let take = share.min(n_veh.saturating_sub(allocated));
        for v in &problem.vehicles[allocated..allocated + take] {
            // Re-id within the sub-problem so warm_start / output stays clean.
            let mut vc = v.clone();
            vc.id = veh_by_cluster[c].len() as u64; // local id; rewritten on assemble
            veh_by_cluster[c].push(vc);
        }
        allocated += take;
    }
    // If any cluster has no vehicles (shouldn't happen with n_veh ≥ k),
    // give it the last available.
    for c in 0..assn.k {
        if jobs_by_cluster[c].is_empty() { continue; }
        if veh_by_cluster[c].is_empty() && allocated < n_veh {
            let mut vc = problem.vehicles[allocated].clone();
            vc.id = 0;
            veh_by_cluster[c].push(vc);
            allocated += 1;
        }
    }

    // Solve clusters in parallel (native) or serially (wasm, no rayon).
    let solve_cluster = |c: usize| -> Solution {
        let sub = subproblem(problem, &jobs_by_cluster[c], &veh_by_cluster[c]);
        solve_with_matrix(&sub, matrix, config)
    };
    #[cfg(feature = "parallel")]
    let sub_solutions: Vec<Solution> = (0..assn.k).into_par_iter().map(solve_cluster).collect();
    #[cfg(not(feature = "parallel"))]
    let sub_solutions: Vec<Solution> = (0..assn.k).map(solve_cluster).collect();

    // Reassemble. Each sub-route's TaskRef::Job(idx) is local to its
    // sub-problem; re-map to parent indices.
    let mut routes: Vec<Route> = Vec::new();
    let mut unassigned: Vec<TaskRef> = Vec::new();
    let mut veh_offset = 0usize;
    for (c, sol) in sub_solutions.iter().enumerate() {
        // Map local job idx → parent job idx by id lookup.
        let local_to_parent: HashMap<u64, usize> = jobs_by_cluster[c]
            .iter()
            .map(|j| j.id)
            .enumerate()
            .map(|(local_idx, id)| {
                let parent_idx = problem
                    .jobs
                    .iter()
                    .position(|pj| pj.id == id)
                    .unwrap_or(local_idx);
                (id, parent_idx)
            })
            .collect();

        for r in &sol.routes {
            let parent_steps: Vec<TaskRef> = r
                .steps
                .iter()
                .filter_map(|s| match s {
                    TaskRef::Job(local) => {
                        let local_id = jobs_by_cluster[c].get(*local)?.id;
                        local_to_parent.get(&local_id).copied().map(TaskRef::Job)
                    }
                    other => Some(*other), // shipments unsupported here
                })
                .collect();
            let parent_veh_idx = veh_offset + r.vehicle_idx;
            // Re-evaluate against parent problem so metrics align.
            let metrics = crate::solution::evaluate_route(
                problem,
                matrix,
                &problem.vehicles[parent_veh_idx],
                &parent_steps,
            )
            .unwrap_or_default();
            routes.push(Route {
                vehicle_idx: parent_veh_idx,
                steps: parent_steps,
                metrics,
            });
        }
        for u in &sol.unassigned {
            if let TaskRef::Job(local) = *u {
                if let Some(j) = jobs_by_cluster[c].get(local) {
                    if let Some(&pidx) = local_to_parent.get(&j.id) {
                        unassigned.push(TaskRef::Job(pidx));
                    }
                }
            }
        }
        veh_offset += veh_by_cluster[c].len();
    }

    let mut combined = Solution {
        routes,
        unassigned,
        summary: Default::default(),
    };
    combined.recompute_summary(problem);

    // Polish pass: cross-cluster moves were forbidden during the parallel
    // sub-solves; recover them now with a full-LS pass on the assembled
    // solution. Without this we typically lose 0.5-2% vs flat-solve, but
    // post-polish the gap closes to <0.5% on most instances.
    //
    // When --gpu is enabled, this CPU polish is followed by a top-level
    // GPU batch polish in main.rs. We alternate: CPU polish (finds large
    // cross-cluster moves) → GPU batch polish (diversification kicks +
    // GPU LS). The alternation can be repeated by callers via the
    // top-level loop until the time budget exhausts.
    let pre_polish_cost = combined.summary.cost;
    let granular = config.granular_k.map(|k| crate::granular::Granular::build(matrix, k));
    crate::local_search::local_search_full(
        problem, matrix, &mut combined,
        config.max_local_search_passes, granular.as_ref(),
    );
    if config.verbose && combined.summary.cost + 1e-9 < pre_polish_cost {
        eprintln!(
            "brooom: cluster-decompose polish: {:.2} → {:.2} (Δ={:.2})",
            pre_polish_cost, combined.summary.cost, pre_polish_cost - combined.summary.cost
        );
    }

    combined
}

/// Run a single CPU local-search polish pass on a Solution. Useful for
/// alternating GPU and CPU polish in the top-level loop.
pub fn polish_cpu_full(
    problem: &crate::problem::Problem,
    matrix: &crate::matrix::Matrix,
    solution: &mut Solution,
    config: &SolverConfig,
) {
    let granular = config.granular_k.map(|k| crate::granular::Granular::build(matrix, k));
    crate::local_search::local_search_full(
        problem, matrix, solution,
        config.max_local_search_passes, granular.as_ref(),
    );
}
