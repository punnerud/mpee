//! Small correctness test for CH: build CH on random sparse graphs and
//! verify that ch::query matches full Dijkstra on 100 random (s,d) pairs.

use sssp_bench::ch;
use sssp_bench::dijkstra::dijkstra_binary;
use sssp_bench::graph::{gen_random_sparse, CsrGraph, Rng};

fn test_correctness(g: &CsrGraph, n_pairs: usize, label: &str) -> bool {
    println!(
        "\n=== {label}: n={}, m={}, avg_deg={:.2} ===",
        g.n,
        g.m(),
        g.m() as f32 / g.n.max(1) as f32
    );
    let t = std::time::Instant::now();
    let h = ch::build(g);
    println!("  ch::build: {:.2} s", t.elapsed().as_secs_f64());

    let mut rng = Rng(2026_05_09);
    let mut bad = 0usize;
    for i in 0..n_pairs {
        let s = rng.range(g.n as u32);
        let d = rng.range(g.n as u32);
        if g.head[s as usize + 1] - g.head[s as usize] == 0 {
            continue;
        }
        let ref_dist = dijkstra_binary(g, s);
        let r = ref_dist[d as usize];
        // h.perm maps CSR-IDs → CH-internal (rank-ordered) IDs.
        let q = ch::query(&h, h.perm[s as usize], h.perm[d as usize])
            .unwrap_or(f32::INFINITY);
        let ok = if r.is_infinite() || q.is_infinite() {
            r == q
        } else {
            (r - q).abs() <= 1e-3 * (1.0 + r.abs())
        };
        if !ok {
            if bad < 5 {
                println!("    DIFF: pair {} (s={s} d={d}): ref={r} ch={q}", i);
            }
            bad += 1;
        }
    }
    println!("  korrekthet: {} / {} OK", n_pairs - bad, n_pairs);
    bad == 0
}

fn main() {
    let mut all_ok = true;
    // Liten sparse graf
    let g = gen_random_sparse(50, 4, 42);
    all_ok &= test_correctness(&g, 50, "tiny sparse n=50");

    let g = gen_random_sparse(500, 6, 13);
    all_ok &= test_correctness(&g, 100, "sparse n=500 deg=6");

    let g = gen_random_sparse(2000, 8, 99);
    all_ok &= test_correctness(&g, 100, "sparse n=2000 deg=8");

    if all_ok {
        println!("\n✓ ALLE TEST OK");
    } else {
        println!("\n✗ FEIL DETEKTERT");
        std::process::exit(1);
    }
}
