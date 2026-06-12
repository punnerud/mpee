//! Sanity for the deliberately trajectory-CHANGING LS heuristics from
//! session 10 (both opt-in A/B arms, off by default): Vidal-style SWAP*
//! level B (BROOOM_SWAPSTAR_TOP3CAND=1) and the relocate extreme-position
//! cut (BROOOM_RELOC_EXTREMES=marked|off).
//!
//! These are NOT identity-preserving (that's tests/posidx_equiv.rs's
//! contract for the O(K) machinery) — quality is judged by benchmarks. Here
//! we pin down: every configuration still produces a full feasible solution
//! of comparable cost, the default configuration is bit-identical to leaving
//! the flags unset, and the paranoia cross-checks stay green under the new
//! gating.
//!
//! Run: cargo test --release -p brooom --test ls_heuristic_switches
use std::path::Path;

use brooom::granular::Granular;
use brooom::insertion::greedy_insertion_seeded;
use brooom::io::parse_input;
use brooom::local_search::local_search;
use brooom::solution::eval_cache_invalidate;
use brooom::solver::build_matrix;

const FLAGS: [&str; 4] = [
    "BROOOM_SWAPSTAR_TOP3CAND",
    "BROOOM_RELOC_EXTREMES",
    "BROOOM_CHECK_INV",
    "BROOOM_NO_POSIDX",
];

fn clear_flags() {
    for f in FLAGS {
        std::env::remove_var(f);
    }
}

/// Greedy + LS under the given (flag, value) pairs; returns the route
/// fingerprint, the cost and the unassigned count.
fn run(name: &str, seed: u64, set: &[(&str, &str)]) -> (Vec<String>, f64, usize) {
    clear_flags();
    for (k, v) in set {
        std::env::set_var(k, v);
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
    (fp, s.summary.cost, s.unassigned.len())
}

#[test]
fn heuristic_switches() {
    // Env mutation is process-global: all configurations in one test fn.
    for name in ["r101", "rc101", "c101", "r201", "rc208"] {
        for seed in [1u64, 2, 3] {
            let (base_fp, base_cost, base_uns) = run(name, seed, &[]);
            assert_eq!(base_uns, 0, "{name} s{seed}: baseline left unassigned");

            // Explicit "full" must be bit-identical to leaving the flag unset.
            let (fp, cost, _) = run(name, seed, &[("BROOOM_RELOC_EXTREMES", "full")]);
            assert_eq!(base_cost.to_bits(), cost.to_bits(), "{name} s{seed}: full != default");
            assert_eq!(base_fp, fp, "{name} s{seed}: full != default (routes)");

            // Every heuristic configuration: full solution, loose cost guard.
            let configs: &[&[(&str, &str)]] = &[
                &[("BROOOM_SWAPSTAR_TOP3CAND", "1")],
                &[("BROOOM_RELOC_EXTREMES", "marked")],
                &[("BROOOM_RELOC_EXTREMES", "off")],
                &[
                    ("BROOOM_SWAPSTAR_TOP3CAND", "1"),
                    ("BROOOM_RELOC_EXTREMES", "marked"),
                ],
                // Paranoia cross-checks under the new gating: the relocate
                // scan/inverted comparison must agree per extremes-mode, and
                // the r1 gap-trick assert still runs with level B on.
                &[("BROOOM_SWAPSTAR_TOP3CAND", "1"), ("BROOOM_CHECK_INV", "1")],
                &[("BROOOM_RELOC_EXTREMES", "marked"), ("BROOOM_CHECK_INV", "1")],
                &[("BROOOM_RELOC_EXTREMES", "off"), ("BROOOM_CHECK_INV", "1")],
            ];
            for set in configs {
                let (_, cost, uns) = run(name, seed, set);
                assert_eq!(uns, 0, "{name} s{seed} {set:?}: unassigned tasks");
                let rel = (cost - base_cost).abs() / base_cost.max(1.0);
                // Sanity bound, not a quality judgment (benchmarks own that):
                // single LS descents on R2 instances land in local optima up
                // to ~7 % apart when the candidate sets differ.
                assert!(
                    rel < 0.10,
                    "{name} s{seed} {set:?}: cost diverged {rel:.4} ({cost} vs {base_cost})"
                );
            }
        }
    }
    clear_flags();
}
