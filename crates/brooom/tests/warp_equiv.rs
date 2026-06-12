//! Correctness of the O(1) time-warp soft-delta machinery (src/warp.rs).
//!
//! 1. **Violation exactness** (the load-bearing property): on random
//!    instances inside the warp envelope — jobs only, ≤1 time window, single
//!    capacity dimension, no setup/release/breaks, haversine matrix —
//!    `RouteWarp::viol` and `replace_seg_viol` must agree EXACTLY
//!    (component-wise, i64) with a straight-line clamp-semantics simulator,
//!    on feasible AND infeasible step sequences (the soft LS works on
//!    violating routes; that is its purpose).
//! 2. **evaluate_route consistency**: under `SoftMode::Warp` the evaluator's
//!    soft accumulators equal the simulator's, and the penalty folded into
//!    `cost` is exactly `w · viol` — so a fast operator that ranks by
//!    arc-delta + warp penalty-delta ranks by the TRUE delta and the
//!    rank-then-confirm-first contract carries over with no mis-ranking.
//! 3. **Hard cross-check**: zero violations ⟺ hard `evaluate_route` accepts
//!    ⟺ `RouteSlack::build` succeeds; and on feasible routes the slack
//!    verdict `replace_seg_ok` equals `replace_seg_viol == 0` — wiring the
//!    new warp math to the proven slack arrays.
//! 4. **Lower bound**: warp violations never exceed the public carry-forward
//!    soft mode's (clamping only makes downstream arrivals earlier) — the
//!    property a future carry-forward prune would rely on.

use brooom::io::parse_input;
use brooom::matrix::HaversineMatrix;
use brooom::problem::{Problem, Vehicle};
use brooom::slack::RouteSlack;
use brooom::solution::{
    eval_cache_invalidate, evaluate_route, set_soft_penalties, set_soft_penalties_mode, SoftMode,
    SoftWeights, TaskRef,
};
use brooom::solver::build_matrix;
use brooom::warp::{warp_eligible, RouteWarp, Viol};

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn rnd(s: &mut u64, lo: i64, hi: i64) -> i64 {
    lo + (xorshift(s) % ((hi - lo + 1) as u64)) as i64
}

/// Random jobs-only single-dimension instance inside the warp envelope.
/// Tighter vehicle windows than slack_equiv's generator so a healthy share
/// of derangements actually violate.
fn gen_instance(seed: u64) -> String {
    let mut s = seed.max(1);
    let n = rnd(&mut s, 5, 25);
    let nv = rnd(&mut s, 2, 4);
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
        let del = rnd(&mut s, 0, 8);
        let pick = if rnd(&mut s, 0, 2) == 0 { rnd(&mut s, 0, 5) } else { 0 };
        let tw = match rnd(&mut s, 0, 3) {
            0 => String::from("[]"),
            1 => {
                let a = rnd(&mut s, 0, horizon / 2);
                format!("[[{}, {}]]", a, a + rnd(&mut s, 5_000, 20_000))
            }
            _ => {
                let a = rnd(&mut s, 0, horizon - 2_000);
                format!("[[{}, {}]]", a, a + rnd(&mut s, 300, 2_000))
            }
        };
        jobs.push(format!(
            r#"{{"id": {id}, "location": [{lon:.5}, {lat:.5}], "service": {service},
                "delivery": [{del}], "pickup": [{pick}], "time_windows": {tw}}}"#
        ));
    }
    let mut vehicles = Vec::new();
    for v in 0..nv {
        let cap = rnd(&mut s, 10, 40);
        let vw_end = rnd(&mut s, horizon / 3, horizon);
        vehicles.push(format!(
            r#"{{"id": {v}, "start": [10.10, 60.10], "end": [10.10, 60.10],
                 "capacity": [{cap}], "time_window": [0, {vw_end}], "speed_factor": {speed}}}"#
        ));
    }
    format!(
        r#"{{"vehicles": [{}], "jobs": [{}]}}"#,
        vehicles.join(","),
        jobs.join(",")
    )
}

/// Reference clamp-semantics simulator: the warp `Viol` of `steps` on `veh`,
/// mirroring `evaluate_route` under `SoftMode::Warp` line by line (depart at
/// vw.start, wait to w.start, clamp + charge past w.end, service after the
/// window check, load checked after each stop's delta, peak excess, end
/// overrun past vw.end).
fn simulate(problem: &Problem, matrix: &brooom::matrix::Matrix, veh: &Vehicle, steps: &[TaskRef]) -> Viol {
    let speed = veh.speed_factor.max(0.01);
    let vw = veh.time_window();
    let dur = |a: usize, b: usize| -> i64 { ((matrix.duration(a, b) as f64) * speed).round() as i64 };
    let start_idx = veh
        .start
        .as_ref()
        .and_then(|l| l.index)
        .or_else(|| veh.end.as_ref().and_then(|l| l.index));
    let end_idx = veh.end.as_ref().and_then(|l| l.index).or(start_idx);
    let cap = veh.capacity.first().copied().unwrap_or(i64::MAX / 4);

    let mut load: i64 = steps
        .iter()
        .map(|s| s.description(problem).delivery.first().copied().unwrap_or(0))
        .sum();
    let mut excess = (load - cap).max(0);
    let mut warp = 0i64;
    let mut t = vw.start;
    let mut prev = start_idx;
    for &s in steps {
        let job = s.description(problem);
        let loc = job.location.index.unwrap();
        if let Some(p) = prev {
            t += dur(p, loc);
        }
        let w = job
            .time_windows
            .first()
            .copied()
            .unwrap_or(brooom::problem::TimeWindow::FOREVER);
        if t < w.start {
            t = w.start;
        }
        if t > w.end {
            warp += t - w.end;
            t = w.end;
        }
        load -= job.delivery.first().copied().unwrap_or(0);
        load += job.pickup.first().copied().unwrap_or(0);
        excess = excess.max(load - cap);
        t += job.service;
        prev = Some(loc);
    }
    if let (Some(p), Some(e)) = (prev, end_idx) {
        t += dur(p, e);
    }
    Viol { tw: warp, load: excess.max(0), dur: (t - vw.end).max(0) }
}

/// Insert `seg` at `[i, k)`'s place in `steps` (replacing that span).
fn splice(steps: &[TaskRef], i: usize, k: usize, seg: &[TaskRef]) -> Vec<TaskRef> {
    let mut out = Vec::with_capacity(steps.len() - (k - i) + seg.len());
    out.extend_from_slice(&steps[..i]);
    out.extend_from_slice(seg);
    out.extend_from_slice(&steps[k..]);
    out
}

/// Random multi-route step lists over the instance's jobs, DERANGED so a
/// large share of routes are hard-infeasible (random order, random route
/// sizes) — RouteWarp must build and judge all of them.
fn random_routes(problem: &Problem, seed: u64) -> Vec<(usize, Vec<TaskRef>)> {
    let mut s = seed.max(1);
    let mut tasks: Vec<TaskRef> = (0..problem.jobs.len()).map(TaskRef::Job).collect();
    // Fisher–Yates with the xorshift stream.
    for i in (1..tasks.len()).rev() {
        let j = (xorshift(&mut s) % (i as u64 + 1)) as usize;
        tasks.swap(i, j);
    }
    let nv = problem.vehicles.len();
    let mut routes: Vec<(usize, Vec<TaskRef>)> = (0..nv).map(|v| (v, Vec::new())).collect();
    for t in tasks {
        let v = (xorshift(&mut s) % nv as u64) as usize;
        routes[v].1.push(t);
    }
    routes.retain(|(_, r)| !r.is_empty());
    routes
}

fn viol_exactness_and_consistency() {
    let mut checked = 0u64;
    let mut violating = 0u64;
    let w = SoftWeights { tw: 7.0, load: 11.0, dur: 13.0 };
    for seed in 1..=200u64 {
        let json = gen_instance(seed * 9973);
        let mut problem = parse_input(&json).unwrap();
        let matrix = build_matrix(&mut problem, Some(&HaversineMatrix::default())).unwrap();
        assert!(warp_eligible(&problem), "generated instance must be warp-eligible (seed {seed})");
        let routes = random_routes(&problem, seed * 31);

        // Pool of foreign tasks for insertion/exchange candidates.
        let all_tasks: Vec<(usize, TaskRef)> = routes
            .iter()
            .enumerate()
            .flat_map(|(r, (_, steps))| steps.iter().map(move |&t| (r, t)))
            .collect();

        for (r, (veh_idx, steps)) in routes.iter().enumerate() {
            let veh = &problem.vehicles[*veh_idx];
            let n = steps.len();
            let warp = RouteWarp::build(&problem, &matrix, veh, steps)
                .unwrap_or_else(|| panic!("warp build failed (seed {seed} route {r})"));

            // (1) Base violations match the simulator.
            let base = simulate(&problem, &matrix, veh, steps);
            assert_eq!(warp.viol(), base, "base viol mismatch seed={seed} route={r}");
            if !base.is_zero() {
                violating += 1;
            }

            // (2) evaluate_route in Warp mode agrees, and its penalty is w·viol.
            eval_cache_invalidate();
            set_soft_penalties_mode(Some(w), SoftMode::Warp);
            let soft_m = evaluate_route(&problem, &matrix, veh, steps)
                .expect("warp-soft evaluation must accept any TW/load violation");
            set_soft_penalties_mode(None, SoftMode::Warp);
            eval_cache_invalidate();
            assert_eq!(soft_m.tw_excess as i64, base.tw, "tw_excess seed={seed} route={r}");
            assert_eq!(soft_m.load_excess as i64, base.load, "load_excess seed={seed} route={r}");
            assert_eq!(soft_m.dur_excess as i64, base.dur, "dur_excess seed={seed} route={r}");
            let hard = evaluate_route(&problem, &matrix, veh, steps);
            // (3) zero viol ⟺ hard accepts; feasible ⇒ slack builds (the
            // converse does not hold: RouteSlack::build only fails on a
            // forward job-window miss, not on load/end-window violations).
            assert_eq!(base.is_zero(), hard.is_ok(), "viol-zero/hard mismatch seed={seed} route={r}");
            if base.is_zero() {
                assert!(
                    RouteSlack::build(&problem, &matrix, veh, steps).is_some(),
                    "feasible route must build slack (seed {seed} route {r})"
                );
            }
            if let Ok(hm) = &hard {
                let pen = soft_m.cost - hm.cost;
                assert!(pen.abs() < 1e-6, "penalty on feasible route: {pen} (seed {seed})");
            } else {
                let expected = w.tw * base.tw as f64 + w.load * base.load as f64 + w.dur * base.dur as f64;
                assert!(
                    (soft_m.cost_custom - expected).abs() < 1e-6,
                    "penalty mismatch seed={seed} route={r}: {} vs {expected}",
                    soft_m.cost_custom
                );
            }
            // (4) warp viol ≤ carry-forward viol, component-wise on tw
            // (dur/load are identical accumulators in both modes only when no
            // clamping happens upstream, so only tw is a guaranteed bound —
            // check the weighted total instead, which clamping cannot raise).
            eval_cache_invalidate();
            set_soft_penalties(Some(w));
            let cf = evaluate_route(&problem, &matrix, veh, steps)
                .expect("carry-forward soft evaluation must accept");
            set_soft_penalties(None);
            eval_cache_invalidate();
            assert!(
                base.tw <= cf.tw_excess as i64,
                "warp tw exceeds carry-forward seed={seed} route={r}"
            );

            // (5) replace_seg_viol == simulator on spliced routes.
            // Slack cross-check (6) only on a FEASIBLE base route — that is
            // replace_seg_ok's contract (slack arrays on a violating route
            // are not meaningful).
            let slack = if base.is_zero() {
                RouteSlack::build(&problem, &matrix, veh, steps)
            } else {
                None
            };
            let mut check = |i: usize, k: usize, seg: &[TaskRef]| {
                let cand = splice(steps, i, k, seg);
                if cand.is_empty() {
                    return;
                }
                let got = warp
                    .replace_seg_viol(&problem, &matrix, i, k, seg)
                    .expect("structurally judgeable");
                let want = simulate(&problem, &matrix, veh, &cand);
                assert_eq!(got, want, "replace viol mismatch seed={seed} route={r} i={i} k={k}");
                // (6) on feasible routes, slack's verdict == (viol == 0).
                if let Some(sl) = &slack {
                    let ok = sl.replace_seg_ok(&problem, &matrix, veh, i, k, seg);
                    assert_eq!(
                        ok,
                        got.is_zero(),
                        "slack/warp verdict mismatch seed={seed} route={r} i={i} k={k}"
                    );
                }
                checked += 1;
            };
            // Removals of length 1..=3.
            for i in 0..n {
                for l in 1..=3usize.min(n - i) {
                    check(i, i + l, &[]);
                }
            }
            // Insertions of foreign tasks (singles and pairs).
            for (fr, t) in all_tasks.iter().take(8) {
                if *fr == r {
                    continue;
                }
                for j in 0..=n {
                    check(j, j, &[*t]);
                }
            }
            for w2 in all_tasks.windows(2).take(6) {
                let ((ra, a), (rb, b)) = (w2[0], w2[1]);
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
                    let mut rev: Vec<TaskRef> = steps[a..=b].to_vec();
                    rev.reverse();
                    check(a, b + 1, &rev);
                }
            }
        }
    }
    eprintln!("warp viols checked: {checked} (violating base routes: {violating})");
    assert!(checked > 30_000, "expected substantial coverage, got {checked}");
    assert!(violating > 100, "expected real violating routes to exercise warp, got {violating}");
}

fn gating() {
    let base = |extra_job: &str, extra_veh: &str, cap: &str| {
        format!(
            r#"{{"vehicles": [{{"id": 0, "start": [10.0, 60.0], "capacity": {cap}{extra_veh}}}],
                 "jobs": [{{"id": 1, "location": [10.1, 60.0], "delivery": [1]{extra_job}}}]}}"#
        )
    };
    let eligible = |json: &str| warp_eligible(&parse_input(json).unwrap());
    assert!(eligible(&base("", "", "[10]")), "clean 1-dim instance must be eligible");
    assert!(
        !eligible(&base(r#", "time_windows": [[0, 100], [200, 300]]"#, "", "[10]")),
        "multi-TW must gate"
    );
    assert!(!eligible(&base(r#", "setup": 60"#, "", "[10]")), "setup must gate");
    assert!(!eligible(&base(r#", "release": 60"#, "", "[10]")), "release must gate");
    assert!(
        !eligible(&base("", r#", "breaks": [{"id": 1, "service": 600}]"#, "[10]")),
        "breaks must gate"
    );
    assert!(!eligible(&base("", "", "[10, 5]")), "multi-dimension capacity must gate");
    std::env::set_var("BROOOM_NO_WARP_LS", "1");
    assert!(!eligible(&base("", "", "[10]")), "kill switch must gate");
    std::env::remove_var("BROOOM_NO_WARP_LS");
}

#[test]
fn warp_correctness() {
    // One #[test] running both parts sequentially: gating toggles a
    // process-global env var and must not race the property part.
    viol_exactness_and_consistency();
    gating();
}
