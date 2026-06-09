//! Isolated validation of the Split (Prins/Vidal) procedure.
//!
//! The warm-start hold test proved brooom's LS *holds* PyVRP's optima but our
//! search can't *reach* them. Split is the missing mechanism: given a good
//! customer ORDERING it reconstructs the cost-optimal route partition. This test
//! feeds PyVRP's own r205 visiting order (saved as a warm-start) through Split
//! and checks it recovers PyVRP's solution — proving Split works before the GA
//! is built around it.
//!
//! Prereq: /tmp/warm_r205.json (PyVRP's r205 solution, Vroom-style) — generated
//! by the bench helper. Skips gracefully if absent.
//!
//! Run: cargo test --release -p brooom --test genetic_split -- --nocapture

use std::path::Path;

use brooom::genetic::{hgs_applicable, solution_to_giant_tour, split};
use brooom::io::parse_input;
use brooom::solver::build_matrix;
use brooom::warm_start::load_warm_start;

fn load(name: &str) -> (brooom::Problem, brooom::Matrix) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks/instances_solomon")
        .join(format!("{name}.json"));
    let json = std::fs::read_to_string(&path).expect("read instance");
    let mut problem = parse_input(&json).expect("parse");
    let matrix = build_matrix(&mut problem, None).expect("matrix");
    brooom::solution::eval_cache_invalidate();
    (problem, matrix)
}

#[test]
fn split_recovers_pyvrp_r205_ordering() {
    let warm_path = "/tmp/warm_r205.json";
    if !Path::new(warm_path).exists() {
        eprintln!("SKIP: {warm_path} not present");
        return;
    }
    let (problem, matrix) = load("r205");
    assert!(hgs_applicable(&problem), "r205 should be in the HGS envelope");

    let warm = load_warm_start(&problem, &matrix, warm_path).expect("warm-start");
    let warm_cost = warm.summary.cost;
    let warm_routes = warm.routes.iter().filter(|r| !r.steps.is_empty()).count();
    eprintln!("PyVRP warm-start: cost={warm_cost:.0} routes={warm_routes}");

    let tour = solution_to_giant_tour(&warm, &problem);
    assert_eq!(tour.len(), problem.jobs.len(), "giant tour covers every job");

    let split_sol = split(&tour, &problem, &matrix).expect("split feasible");
    let split_cost = split_sol.summary.cost;
    let split_routes = split_sol.routes.iter().filter(|r| !r.steps.is_empty()).count();
    eprintln!(
        "Split of PyVRP ordering: cost={split_cost:.0} routes={split_routes} (PyVRP ref 95415/5)"
    );

    // Split re-partitions PyVRP's *ordering* optimally, so it can only match or
    // beat PyVRP's own partition of that order. Allow a hair for rounding.
    assert!(
        split_cost <= warm_cost + 1e-6,
        "Split ({split_cost:.0}) must not be worse than PyVRP's own partition ({warm_cost:.0})"
    );
    // Every job placed.
    let placed: usize = split_sol.routes.iter().map(|r| r.steps.len()).sum();
    assert_eq!(placed, problem.jobs.len(), "all jobs placed by Split");
    assert!(split_sol.unassigned.is_empty(), "no unassigned after Split");
}

/// The incremental Split engine must produce the SAME partition cost as the
/// reference (full-evaluator) engine on every ordering it sees. Random tours +
/// greedy tours across the Solomon families exercise waiting, tight windows,
/// capacity limits and infeasible orderings.
#[test]
fn fast_split_matches_reference() {
    use brooom::genetic::split_reference;
    use brooom::insertion::greedy_insertion_seeded;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    for inst in ["r101", "c101", "rc101", "r201", "c201", "rc201", "r205", "r211", "rc208"] {
        let (problem, matrix) = load(inst);
        if !hgs_applicable(&problem) {
            continue;
        }
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xBEEF);
        let mut tours: Vec<Vec<usize>> = Vec::new();
        // Greedy-derived orderings (realistic) …
        for seed in 0..3u64 {
            let s = greedy_insertion_seeded(&problem, &matrix, seed);
            tours.push(solution_to_giant_tour(&s, &problem));
        }
        // … and pure-random permutations (adversarial: mostly infeasible segments).
        for _ in 0..5 {
            let mut t: Vec<usize> = (0..problem.jobs.len()).collect();
            t.shuffle(&mut rng);
            tours.push(t);
        }
        for (ti, tour) in tours.iter().enumerate() {
            let fast = split(tour, &problem, &matrix);
            let slow = split_reference(tour, &problem, &matrix);
            match (&fast, &slow) {
                (Some(f), Some(s)) => {
                    assert_eq!(
                        f.summary.cost.to_bits(),
                        s.summary.cost.to_bits(),
                        "{inst} tour {ti}: fast {} vs reference {}",
                        f.summary.cost,
                        s.summary.cost
                    );
                    assert_eq!(f.routes.len(), s.routes.len(), "{inst} tour {ti}: route count");
                }
                (None, None) => {}
                _ => panic!(
                    "{inst} tour {ti}: feasibility verdict differs (fast={:?} reference={:?})",
                    fast.is_some(),
                    slow.is_some()
                ),
            }
        }
    }
}

#[test]
#[ignore]
fn hgs_quality_vs_pyvrp() {
    use brooom::genetic::solve_genetic;
    use brooom::granular::Granular;
    use web_time::{Duration, Instant};
    let refs = [("r205", 95415.0), ("r211", 75523.0), ("rc208", 77892.0),
                ("r207", 79789.0), ("rc206", 105460.0)];
    for (name, py) in refs {
        let (problem, matrix) = load(name);
        let granular = Granular::build(&matrix, 30);
        brooom::solution::eval_cache_invalidate();
        let t0 = Instant::now();
        let deadline = Some(t0 + Duration::from_millis(9000));
        let sol = solve_genetic(&problem, &matrix, Some(&granular), 30, 42, deadline, &[])
            .expect("hgs solves");
        let routes = sol.routes.iter().filter(|r| !r.steps.is_empty()).count();
        let c = sol.summary.cost;
        eprintln!("HGS {name}: cost={c:.0} routes={routes} gap={:+.2}% t={:.1}s",
            (c - py) / py * 100.0, t0.elapsed().as_secs_f64());
        assert!(sol.unassigned.is_empty());
    }
}

#[test]
#[ignore]
fn hgs_microbench() {
    use brooom::genetic::{split, solution_to_giant_tour};
    use brooom::granular::Granular;
    use brooom::insertion::greedy_insertion_seeded;
    use brooom::local_search::local_search;
    use web_time::Instant;
    let (problem, matrix) = load("r211");
    let granular = Granular::build(&matrix, 30);
    eprintln!("n_jobs={}", problem.jobs.len());
    // greedy
    let t = Instant::now();
    let mut s = greedy_insertion_seeded(&problem, &matrix, 1);
    eprintln!("greedy: {:.1}ms", t.elapsed().as_secs_f64()*1000.0);
    // split
    let tour = solution_to_giant_tour(&s, &problem);
    let t = Instant::now();
    for _ in 0..10 { let _ = split(&tour, &problem, &matrix); }
    eprintln!("split x10: {:.1}ms ({:.1}ms each)", t.elapsed().as_secs_f64()*1000.0, t.elapsed().as_secs_f64()*100.0);
    // cold local_search
    let t = Instant::now();
    local_search(&problem, &matrix, &mut s, 30, Some(&granular));
    eprintln!("cold LS: {:.1}ms", t.elapsed().as_secs_f64()*1000.0);
}

#[test]
#[ignore]
fn hgs_microbench2() {
    use brooom::genetic::{split, solution_to_giant_tour};
    use brooom::granular::Granular;
    use brooom::insertion::greedy_insertion_seeded;
    use brooom::local_search::local_search;
    use web_time::Instant;
    let (problem, matrix) = load("r211");
    let s0 = greedy_insertion_seeded(&problem, &matrix, 1);
    let tour = solution_to_giant_tour(&s0, &problem);
    for k in [8usize, 12, 20] {
        let granular = Granular::build(&matrix, k);
        for passes in [1usize, 2, 4] {
            let mut s = split(&tour, &problem, &matrix).unwrap();
            let t = Instant::now();
            local_search(&problem, &matrix, &mut s, passes, Some(&granular));
            eprintln!("K={k} passes={passes}: {:.1}ms cost={:.0}", t.elapsed().as_secs_f64()*1000.0, s.summary.cost);
        }
    }
}

#[test]
#[ignore]
fn hgs_parallel_vs_pyvrp() {
    use brooom::genetic::solve_genetic_parallel;
    use brooom::granular::Granular;
    use web_time::{Duration, Instant};
    let refs = [("r205", 95415.0), ("r211", 75523.0), ("rc208", 77892.0),
                ("r207", 79789.0), ("rc206", 105460.0)];
    let seeds = [42u64, 7, 123];
    for (name, py) in refs {
        let (problem, matrix) = load(name);
        let granular = Granular::build(&matrix, 20);
        let mut gaps: Vec<f64> = Vec::new();
        for &seed in &seeds {
            brooom::solution::eval_cache_invalidate();
            let t0 = Instant::now();
            let deadline = Some(t0 + Duration::from_millis(10000));
            // 11 islands ≈ logical cores; medium education (passes=4).
            let sol = solve_genetic_parallel(&problem, &matrix, Some(&granular), 4, seed, deadline, 11, &[])
                .expect("hgs solves");
            let routes = sol.routes.iter().filter(|r| !r.steps.is_empty()).count();
            let c = sol.summary.cost;
            let gap = (c - py) / py * 100.0;
            gaps.push(gap);
            eprintln!("HGS-par {name} seed{seed}: cost={c:.0} routes={routes} gap={gap:+.2}%");
        }
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let worst = gaps.iter().cloned().fold(f64::MIN, f64::max);
        eprintln!("  >>> {name}: mean={mean:+.2}% worst={worst:+.2}%");
    }
}

#[test]
#[ignore]
fn greedy_route_len() {
    use brooom::insertion::greedy_insertion;
    for inst in ["r101","rc101","c101","r201","rc201","r205","r211","rc208","r207","rc206","c201"] {
        let (problem, matrix) = match std::panic::catch_unwind(|| load(inst)) { Ok(v)=>v, Err(_)=>continue };
        let s = greedy_insertion(&problem, &matrix);
        let routes = s.routes.iter().filter(|r| !r.steps.is_empty()).count();
        let jobs: usize = s.routes.iter().map(|r| r.steps.len()).sum();
        eprintln!("{inst:6} greedy routes={routes} jobs/route={:.1}", jobs as f64/routes.max(1) as f64);
    }
}
