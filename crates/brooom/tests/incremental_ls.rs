//! Equivalence + speedup of the fast O(1)-cost-delta local-search path.
//!
//! The fast path (relocate / 2-opt / swap*) computes each move's COST delta in
//! O(1) edge math — exact whenever the objective is the plain arc-based one
//! (span_cost==0, no custom dims, no soft mode) — and confirms FEASIBILITY
//! lazily via the authoritative `evaluate_route`. It must therefore produce the
//! SAME result as the full-evaluate path, only faster.
//!
//! Run: cargo test --release -p brooom --test incremental_ls -- --ignored --nocapture
use std::path::Path;
use brooom::io::parse_input;
use brooom::solver::build_matrix;

fn load(name: &str) -> (brooom::Problem, brooom::Matrix) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/instances_solomon").join(format!("{name}.json"));
    let mut p = parse_input(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let m = build_matrix(&mut p, None).unwrap();
    (p, m)
}

fn run(name: &str, seed: u64, fast: bool) -> (f64, usize, f64) {
    use brooom::granular::Granular;
    use brooom::insertion::greedy_insertion_seeded;
    use brooom::local_search::local_search;
    use brooom::solution::eval_cache_invalidate;
    use web_time::Instant;
    if fast { std::env::remove_var("BROOOM_NO_FAST_LS"); }
    else { std::env::set_var("BROOOM_NO_FAST_LS", "1"); }
    // This test asserts fast == slow; the level-B swap* heuristic is
    // deliberately non-exact (argmin-only candidates), so ensure the opt-in
    // flag is unset. Quality lives in tests/ls_heuristic_switches.rs.
    std::env::remove_var("BROOOM_SWAPSTAR_TOP3CAND");
    let (problem, matrix) = load(name);
    let granular = Granular::build(&matrix, 20);
    eval_cache_invalidate();
    let mut s = greedy_insertion_seeded(&problem, &matrix, seed);
    let t = Instant::now();
    local_search(&problem, &matrix, &mut s, 50, Some(&granular));
    let ms = t.elapsed().as_secs_f64()*1000.0;
    let routes = s.routes.iter().filter(|r| !r.steps.is_empty()).count();
    (s.summary.cost, routes, ms)
}

#[test]
#[ignore]
fn fast_equiv_and_speed() {
    let mut maxreldiff = 0.0f64;
    for name in ["r211","r205","rc208","r101","rc101","c101","r201","rc201"] {
        for seed in [1u64,2,3] {
            let (cf,rf,mf) = run(name, seed, true);
            let (cs,_rs,ms) = run(name, seed, false);
            let reld = (cf-cs).abs()/cs.max(1.0);
            maxreldiff = maxreldiff.max(reld);
            let flag = if reld > 1e-6 { "  *** COST MISMATCH ***" } else { "" };
            eprintln!("{name} s{seed}: fast {cf:.0}({rf}r,{mf:.1}ms) slow {cs:.0}({ms:.1}ms)  speedup={:.1}x{flag}", ms/mf.max(1e-9));
        }
    }
    eprintln!("max relative cost diff fast-vs-slow = {:.2e}", maxreldiff);
    assert!(maxreldiff < 1e-6, "fast LS diverged from slow LS");
}
