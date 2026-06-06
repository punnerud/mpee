//! Native structured constraint propagation — a deductive pre-pass that runs
//! once before search.
//!
//! Local search is *evaluative* (score a formed route); it never *infers* bounds
//! before a move is proposed. This module adds the missing deductive step for the
//! structured VRP constraints, the way PyVRP / OR-Tools / LKH all preprocess:
//!
//!   1. **Temporal tightening** — narrow each job's time window to the interval
//!      it could physically be served in (earliest arrival from any eligible
//!      depot; latest start that still returns within some eligible shift).
//!   2. **Precedence transitive closure** — close `a→b, b→c ⇒ a→c` so the
//!      route-walk enforces implied orders; detect ordering cycles.
//!   3. **Provable infeasibility** — flag a job no vehicle can ever serve
//!      (skills exclude all / demand exceeds every capacity / window unreachable).
//!
//! This is NOT general constraint programming. Arbitrary DSL/code constraints are
//! still black boxes; their *general* propagation stays with the CP-SAT bridge
//! (see `docs/cpsat-boundary.md`). This pass covers the structured cases that
//! cover the vast majority of real VRPs.
//!
//! **Soundness is absolute: never tighten away a feasible option.** Every bound
//! below is a provable physical/structural limit. Bounds that depend on a HARD
//! shift/window are skipped under penalty-managed soft mode (where late service
//! and shift overrun are allowed), so the two features never fight.

use crate::matrix::Matrix;
use crate::problem::{Problem, TimeWindow};

/// A job the pass proved can never be served by any vehicle, with a human reason.
/// Reported (not silently dropped) — the search still runs; this just explains
/// up front why a job will end up unassigned, instead of after a full search.
#[derive(Debug, Clone)]
pub struct Infeasible {
    pub job_id: u64,
    pub reason: String,
}

/// Speed-scaled, rounded travel duration between two matrix indices — matches the
/// rounding in `solution::evaluate_route` so a tightened bound is consistent with
/// the evaluator that later checks it.
#[inline]
fn travel(matrix: &Matrix, from: usize, to: usize, speed_factor: f64) -> i64 {
    let speed = speed_factor.max(0.01);
    ((matrix.duration(from, to) as f64) * speed).round() as i64
}

/// Run the propagation pre-pass, mutating `problem` in place (tighter windows,
/// closed precedence). Returns the provably-unservable jobs. Idempotent: running
/// it again on an already-tightened problem changes nothing.
///
/// `soft` mirrors the solver's penalty-managed mode. When `true`, only the always-
/// sound steps run (precedence closure + the earliest-arrival floor); window-end
/// tightening and infeasibility flagging are skipped, since soft mode may serve a
/// stop late or overrun a shift rather than drop it.
pub fn tighten(problem: &mut Problem, matrix: &Matrix, soft: bool) -> Vec<Infeasible> {
    let mut infeasible: Vec<Infeasible> = Vec::new();

    // ---- precedence transitive closure + cycle detection (always sound) ------
    propagate_precedence(problem, &mut infeasible);

    // ---- per-job temporal tightening + infeasibility -------------------------
    // Precompute eligible-vehicle data once.
    let n_jobs = problem.jobs.len();
    for ji in 0..n_jobs {
        let (job_idx, job_id, job_service, job_release) = {
            let j = &problem.jobs[ji];
            (j.location.index, j.id, j.service, j.release)
        };

        // Earliest arrival from any eligible depot, and latest start that still
        // returns within some eligible shift. min/max over eligible vehicles.
        let mut earliest: Option<i64> = None;
        let mut latest: Option<i64> = None;
        let mut any_eligible = false;
        let mut cap_fits_somewhere = problem.jobs[ji].delivery.is_empty()
            && problem.jobs[ji].pickup.is_empty();

        for v in &problem.vehicles {
            // Eligibility: skills + allowlist.
            if !v.has_skills(&problem.jobs[ji].skills) {
                continue;
            }
            if !problem.jobs[ji].allows_vehicle(v.id) {
                continue;
            }
            any_eligible = true;

            // Capacity: does this vehicle's capacity cover the job's demand?
            if !cap_fits_somewhere {
                let fits = demand_fits(&problem.jobs[ji].delivery, &v.capacity)
                    && demand_fits(&problem.jobs[ji].pickup, &v.capacity);
                if fits {
                    cap_fits_somewhere = true;
                }
            }

            // Temporal bounds need a job index, a vehicle start (for earliest)
            // and end (for latest); skip a vehicle that lacks them.
            let vw = v.time_window();
            let start_idx = v.start.as_ref().and_then(|l| l.index)
                .or_else(|| v.end.as_ref().and_then(|l| l.index));
            let end_idx = v.end.as_ref().and_then(|l| l.index).or(start_idx);
            if let (Some(jidx), Some(si)) = (job_idx, start_idx) {
                let e = vw.start + travel(matrix, si, jidx, v.speed_factor);
                earliest = Some(earliest.map_or(e, |cur: i64| cur.min(e)));
            }
            if let (Some(jidx), Some(ei)) = (job_idx, end_idx) {
                // Latest START so the vehicle can serve, then return by shift end.
                let l = vw.end - travel(matrix, jidx, ei, v.speed_factor) - job_service;
                latest = Some(latest.map_or(l, |cur: i64| cur.max(l)));
            }
        }

        // Provable infeasibility (hard mode only — soft serves late, not drop).
        if !soft {
            if !any_eligible {
                infeasible.push(Infeasible {
                    job_id,
                    reason: "no vehicle has the required skills / is on the allowlist".into(),
                });
                continue;
            }
            if !cap_fits_somewhere {
                infeasible.push(Infeasible {
                    job_id,
                    reason: "demand exceeds every vehicle's capacity".into(),
                });
                continue;
            }
        }

        // Earliest floor: service cannot begin before max(release, earliest).
        let floor = match earliest {
            Some(e) => e.max(job_release),
            None => job_release,
        };

        // Apply tightening to the job's windows.
        let job = &mut problem.jobs[ji];
        if job.time_windows.is_empty() {
            // No declared window: do NOT synthesise one — that would change the
            // search trajectory for a job the user left temporally unconstrained
            // (sound, but it must not alter the solution). Only *report* the rare
            // case where even a windowless job can't be reached and returned
            // within any shift (floor > latest); the report has no side effect.
            if !soft {
                if let Some(l) = latest {
                    if floor > l {
                        infeasible.push(Infeasible {
                            job_id,
                            reason: format!(
                                "cannot be reached and returned within any shift: \
                                 earliest start {floor} > latest start {l}"
                            ),
                        });
                    }
                }
            }
            continue;
        }

        // Tighten each declared window; drop those that become empty.
        let mut kept: Vec<TimeWindow> = Vec::with_capacity(job.time_windows.len());
        for w in &job.time_windows {
            let start = w.start.max(floor);
            // Window-end tightening is only sound when the shift/window is hard.
            let end = if soft { w.end } else { latest.map_or(w.end, |l| w.end.min(l)) };
            if start <= end {
                kept.push(TimeWindow { start, end });
            }
        }
        if kept.is_empty() && !soft {
            infeasible.push(Infeasible {
                job_id,
                reason: "no time window is reachable within any vehicle's shift".into(),
            });
            // Leave the original windows so the solver still treats it normally
            // (it will land in unassigned); we only *report* the proof.
        } else if !kept.is_empty() {
            job.time_windows = kept;
        }
    }

    infeasible
}

/// True if every dimension of `demand` fits within `capacity` (missing capacity
/// dims are treated as 0, matching the evaluator).
#[inline]
fn demand_fits(demand: &[i64], capacity: &[i64]) -> bool {
    demand
        .iter()
        .enumerate()
        .all(|(d, &need)| need <= capacity.get(d).copied().unwrap_or(0))
}

/// Transitive closure of `problem.precedence` (a→b, b→c ⇒ a→c) so the route-walk
/// enforces implied orders without the user spelling them out. Reports a cycle
/// (a→…→a) — those jobs can never share a route — without dropping anything.
fn propagate_precedence(problem: &mut Problem, infeasible: &mut Vec<Infeasible>) {
    if problem.precedence.is_empty() {
        return;
    }
    use std::collections::{BTreeSet, HashMap, HashSet};

    // Adjacency over job ids that actually appear in precedence pairs.
    let mut adj: HashMap<u64, BTreeSet<u64>> = HashMap::new();
    let mut nodes: BTreeSet<u64> = BTreeSet::new();
    for &(a, b) in &problem.precedence {
        adj.entry(a).or_default().insert(b);
        nodes.insert(a);
        nodes.insert(b);
    }

    // Closure via DFS reachability from each node (job graphs are small).
    let mut closure: HashSet<(u64, u64)> = HashSet::new();
    for &src in &nodes {
        let mut stack: Vec<u64> = adj.get(&src).into_iter().flatten().copied().collect();
        let mut seen: HashSet<u64> = HashSet::new();
        while let Some(x) = stack.pop() {
            if !seen.insert(x) {
                continue;
            }
            closure.insert((src, x));
            if x == src {
                infeasible.push(Infeasible {
                    job_id: src,
                    reason: "precedence cycle — these jobs cannot share a route".into(),
                });
            }
            if let Some(next) = adj.get(&x) {
                stack.extend(next.iter().copied());
            }
        }
    }

    // Merge closure edges back into precedence (dedup, stable order).
    let mut set: BTreeSet<(u64, u64)> = problem.precedence.iter().copied().collect();
    for &(a, b) in &closure {
        if a != b {
            set.insert((a, b));
        }
    }
    problem.precedence = set.into_iter().collect();
}
