//! Correctness of the O(1) slack feasibility prefilter (src/slack.rs).
//!
//! 1. **Verdict equivalence** (the load-bearing property): on random
//!    instances inside the exact envelope — jobs only, ≤1 time window, no
//!    setup/release/breaks, no skills/allowlists, haversine matrix (no
//!    unreachable legs) — `RouteSlack::replace_seg_ok` must agree EXACTLY
//!    with `evaluate_route`'s verdict for removals, insertions, exchanges
//!    and reversals. A false negative here would silently skip feasible
//!    moves in the fast LS; a false positive would only cost one wasted
//!    confirm evaluation.
//! 2. **Gating**: features the slack math does not model exactly must
//!    disable `slack_eligible`.
//! 3. **End-to-end smoke**: full local_search with the prefilter on/off
//!    (BROOOM_NO_SLACK_LS) returns feasible solutions of matching quality.
//!    Bit-identity is NOT asserted: the swap* fast path ranks pairs by exact
//!    arc-math deltas, which can differ from evaluation-summed deltas by an
//!    ulp and reorder near-ties — the quality guards (GH/Solomon benches)
//!    arbitrate quality.

use brooom::granular::Granular;
use brooom::insertion::greedy_insertion_seeded;
use brooom::io::parse_input;
use brooom::local_search::local_search;
use brooom::matrix::HaversineMatrix;
use brooom::slack::{slack_eligible, RouteSlack};
use brooom::solution::{eval_cache_invalidate, evaluate_route, TaskRef};
use brooom::solver::build_matrix;
use std::path::Path;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn rnd(s: &mut u64, lo: i64, hi: i64) -> i64 {
    lo + (xorshift(s) % ((hi - lo + 1) as u64)) as i64
}

/// Random jobs-only instance inside the exact envelope.
fn gen_instance(seed: u64) -> String {
    let mut s = seed.max(1);
    let n = rnd(&mut s, 5, 25);
    let nv = rnd(&mut s, 2, 4);
    let dim = if rnd(&mut s, 0, 3) == 0 { 2 } else { 1 };
    let horizon = 40_000i64;
    let speed = match rnd(&mut s, 0, 2) {
        0 => "1.0",
        1 => "1.3",
        _ => "0.77",
    };
    let mut jobs = Vec::new();
    for id in 1..=n {
        let lon = 10.0 + rnd(&mut s, 0, 2000) as f64 / 10_000.0;
        let lat = 60.0 + rnd(&mut s, 0, 2000) as f64 / 10_000.0;
        let service = rnd(&mut s, 0, 600);
        // delivery always non-empty (zeros allowed) so no job is a backhaul.
        let del: Vec<i64> = (0..dim).map(|_| rnd(&mut s, 0, 8)).collect();
        let pick: Vec<i64> = if rnd(&mut s, 0, 2) == 0 {
            (0..dim).map(|_| rnd(&mut s, 0, 5)).collect()
        } else {
            vec![0; dim as usize]
        };
        let tw = match rnd(&mut s, 0, 3) {
            0 => String::from("[]"), // no window (FOREVER)
            1 => {
                // wide
                let a = rnd(&mut s, 0, horizon / 2);
                format!("[[{}, {}]]", a, a + rnd(&mut s, 5_000, 20_000))
            }
            _ => {
                // tight
                let a = rnd(&mut s, 0, horizon - 2_000);
                format!("[[{}, {}]]", a, a + rnd(&mut s, 300, 2_000))
            }
        };
        jobs.push(format!(
            r#"{{"id": {id}, "location": [{lon:.5}, {lat:.5}], "service": {service},
                "delivery": {del:?}, "pickup": {pick:?}, "time_windows": {tw}}}"#
        ));
    }
    let mut vehicles = Vec::new();
    for v in 0..nv {
        let cap: Vec<i64> = (0..dim).map(|_| rnd(&mut s, 15, 60)).collect();
        let vw_end = rnd(&mut s, horizon / 2, horizon);
        vehicles.push(format!(
            r#"{{"id": {v}, "start": [10.10, 60.10], "end": [10.10, 60.10],
                 "capacity": {cap:?}, "time_window": [0, {vw_end}], "speed_factor": {speed}}}"#
        ));
    }
    format!(
        r#"{{"vehicles": [{}], "jobs": [{}]}}"#,
        vehicles.join(","),
        jobs.join(",")
    )
}

/// Insert `seg` at `[i, k)`'s place in `steps` (replacing that span).
fn splice(steps: &[TaskRef], i: usize, k: usize, seg: &[TaskRef]) -> Vec<TaskRef> {
    let mut out = Vec::with_capacity(steps.len() - (k - i) + seg.len());
    out.extend_from_slice(&steps[..i]);
    out.extend_from_slice(seg);
    out.extend_from_slice(&steps[k..]);
    out
}

fn verdict_equivalence_on_random_instances() {
    let mut checked = 0u64;
    let mut rejected = 0u64;
    for seed in 1..=200u64 {
        let json = gen_instance(seed * 7919);
        let mut problem = parse_input(&json).unwrap();
        let matrix = build_matrix(&mut problem, Some(&HaversineMatrix::default())).unwrap();
        assert!(slack_eligible(&problem), "generated instance must be eligible (seed {seed})");
        eval_cache_invalidate();
        let sol = greedy_insertion_seeded(&problem, &matrix, seed);

        // Pool of foreign tasks for insertion/exchange candidates.
        let all_tasks: Vec<(usize, TaskRef)> = sol
            .routes
            .iter()
            .enumerate()
            .flat_map(|(r, route)| route.steps.iter().map(move |&t| (r, t)))
            .collect();

        for (r, route) in sol.routes.iter().enumerate() {
            let veh = &problem.vehicles[route.vehicle_idx];
            let n = route.steps.len();
            if n == 0 {
                continue;
            }
            let slack = RouteSlack::build(&problem, &matrix, veh, &route.steps)
                .unwrap_or_else(|| panic!("slack build failed on feasible route (seed {seed})"));

            let mut check = |i: usize, k: usize, seg: &[TaskRef]| {
                let cand = splice(&route.steps, i, k, seg);
                if cand.is_empty() {
                    return; // whole-route removal: deliberately always "pass"
                }
                let verdict = slack.replace_seg_ok(&problem, &matrix, veh, i, k, seg);
                let truth = evaluate_route(&problem, &matrix, veh, &cand).is_ok();
                assert_eq!(
                    verdict, truth,
                    "verdict mismatch seed={seed} route={r} i={i} k={k} seg={seg:?}"
                );
                checked += 1;
                if !verdict {
                    rejected += 1;
                }
            };

            // Removals of length 1..=3.
            for i in 0..n {
                for l in 1..=3usize.min(n - i) {
                    check(i, i + l, &[]);
                }
            }
            // Insertions of foreign tasks (single and pairs, fwd + rev).
            for (fr, t) in all_tasks.iter().take(8) {
                if *fr == r {
                    continue;
                }
                for j in 0..=n {
                    check(j, j, &[*t]);
                }
            }
            for w in all_tasks.windows(2).take(6) {
                let ((ra, a), (rb, b)) = (w[0], w[1]);
                if ra == r || rb == r {
                    continue;
                }
                for j in [0usize, n / 2, n] {
                    check(j, j, &[a, b]);
                    check(j, j, &[b, a]);
                }
            }
            // Single-task exchanges.
            for (fr, t) in all_tasks.iter().take(8) {
                if *fr == r {
                    continue;
                }
                for i in 0..n {
                    check(i, i + 1, &[*t]);
                }
            }
            // Intra-route reversals up to length 6.
            for a in 0..n {
                for b in (a + 1)..n.min(a + 6) {
                    let mut rev: Vec<TaskRef> = route.steps[a..=b].to_vec();
                    rev.reverse();
                    check(a, b + 1, &rev);
                }
            }
        }
    }
    eprintln!("slack verdicts checked: {checked} ({rejected} rejections) — all exact");
    assert!(checked > 30_000, "expected substantial coverage, got {checked}");
    assert!(rejected > 500, "expected real rejections to exercise both sides, got {rejected}");
}

fn gating() {
    let base = |extra_job: &str, extra_veh: &str| {
        format!(
            r#"{{"vehicles": [{{"id": 0, "start": [10.0, 60.0], "capacity": [10]{extra_veh}}}],
                 "jobs": [{{"id": 1, "location": [10.1, 60.0], "delivery": [1]{extra_job}}}]}}"#
        )
    };
    let eligible = |json: &str| {
        let problem = parse_input(json).unwrap();
        slack_eligible(&problem)
    };
    assert!(eligible(&base("", "")), "clean instance must be eligible");
    assert!(
        !eligible(&base(r#", "time_windows": [[0, 100], [200, 300]]"#, "")),
        "multi-TW must gate"
    );
    assert!(!eligible(&base(r#", "setup": 60"#, "")), "setup must gate");
    assert!(!eligible(&base(r#", "release": 60"#, "")), "release must gate");
    assert!(
        !eligible(&base("", r#", "breaks": [{"id": 1, "service": 600}]"#)),
        "breaks must gate"
    );
    assert!(!eligible(&base("", r#", "max_tasks": 5"#)), "max_tasks must gate");
    assert!(
        !eligible(&base("", r#", "max_travel_time": 1000"#)),
        "max_travel_time must gate"
    );
    let shipment = r#"{"vehicles": [{"id": 0, "start": [10.0, 60.0], "capacity": [10]}],
        "shipments": [{"pickup": {"id": 1, "location": [10.1, 60.0]},
                        "delivery": {"id": 2, "location": [10.2, 60.0]}, "amount": [1]}]}"#;
    assert!(!eligible(shipment), "shipments must gate");
}

fn smoke_on_off() {
    let load = |name: &str| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("benchmarks/instances_solomon")
            .join(format!("{name}.json"));
        let mut p = parse_input(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let m = build_matrix(&mut p, None).unwrap();
        (p, m)
    };
    for name in ["r101", "r201", "rc201", "c101"] {
        let mut costs = [0.0f64; 2];
        for (idx, slack_on) in [(0usize, true), (1usize, false)] {
            if slack_on {
                std::env::remove_var("BROOOM_NO_SLACK_LS");
            } else {
                std::env::set_var("BROOOM_NO_SLACK_LS", "1");
            }
            let (problem, matrix) = load(name);
            let granular = Granular::build(&matrix, 20);
            eval_cache_invalidate();
            let mut s = greedy_insertion_seeded(&problem, &matrix, 1);
            local_search(&problem, &matrix, &mut s, 50, Some(&granular));
            assert!(s.unassigned.is_empty(), "{name}: unassigned with slack={slack_on}");
            costs[idx] = s.summary.cost;
        }
        std::env::remove_var("BROOOM_NO_SLACK_LS");
        let rel = (costs[0] - costs[1]).abs() / costs[1].max(1.0);
        eprintln!("{name}: slack-on {:.1} slack-off {:.1} (rel diff {:.2e})", costs[0], costs[1], rel);
        assert!(
            rel < 0.02,
            "{name}: slack on/off quality diverged: {} vs {}",
            costs[0],
            costs[1]
        );
    }
}

#[test]
fn slack_correctness() {
    // One #[test] running all parts sequentially: the smoke part toggles a
    // process-global env var and must not race the other parts.
    verdict_equivalence_on_random_instances();
    gating();
    smoke_on_off();
}
