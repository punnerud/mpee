//! Trajectory identity of the O(K) enumeration machinery (posidx.rs).
//!
//! PosIndex-locate, the inverted granular relocate enumeration, swap*'s
//! persistent top-3 cache and the r1-side gap trick are all designed to be
//! BIT-IDENTICAL to the paths they replace — same candidate sets, in the same
//! order, with the same float values. So local search from the same start
//! must produce the exact same solution with each switch toggled off, and
//! with the paranoia cross-check (BROOOM_CHECK_INV) armed.
//!
//! Run: cargo test --release -p brooom --test posidx_equiv
use std::path::Path;

use brooom::granular::Granular;
use brooom::insertion::greedy_insertion_seeded;
use brooom::io::parse_input;
use brooom::local_search::local_search;
use brooom::solution::eval_cache_invalidate;
use brooom::solver::build_matrix;

const SWITCHES: [&str; 6] = [
    "BROOOM_NO_POSIDX",
    "BROOOM_NO_RELOC_INV",
    "BROOOM_NO_SWAPSTAR_TOPCACHE",
    "BROOOM_NO_SWAPSTAR_R1GAP",
    "BROOOM_NO_PAIR_INV",
    "BROOOM_CHECK_INV",
];

fn clear_switches() {
    for s in SWITCHES {
        std::env::remove_var(s);
    }
}

/// Greedy + LS under the given switch (all others cleared); returns the full
/// route fingerprint (steps as debug strings) and the cost.
fn run(name: &str, seed: u64, switch: Option<&str>) -> (Vec<String>, f64) {
    clear_switches();
    if let Some(s) = switch {
        std::env::set_var(s, "1");
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks/instances_solomon")
        .join(format!("{name}.json"));
    let mut p = parse_input(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let m = build_matrix(&mut p, None).unwrap();
    let granular = Granular::build(&m, 20);
    eval_cache_invalidate();
    let mut s = greedy_insertion_seeded(&p, &m, seed);
    local_search(&p, &m, &mut s, 50, Some(&granular));
    let fp = s.routes.iter().map(|r| format!("{:?}", r.steps)).collect();
    (fp, s.summary.cost)
}

#[test]
fn switch_identity() {
    // Env mutation is process-global: this test must own the switches for
    // its duration, so all configurations run inside the one test fn.
    for name in ["r101", "rc101", "c101", "r201", "rc208"] {
        for seed in [1u64, 2, 3] {
            let (base_fp, base_cost) = run(name, seed, None);
            for switch in SWITCHES {
                let (fp, cost) = run(name, seed, Some(switch));
                assert_eq!(
                    base_cost.to_bits(),
                    cost.to_bits(),
                    "{name} s{seed}: cost diverged under {switch}: {base_cost} vs {cost}"
                );
                assert_eq!(
                    base_fp, fp,
                    "{name} s{seed}: route set diverged under {switch}"
                );
            }
        }
    }
    clear_switches();
}
