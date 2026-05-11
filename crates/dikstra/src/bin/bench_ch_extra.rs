//! Test CH on graph types other than road networks: RMAT (scale-free),
//! grid (regular geometric), Rubik (regular Cayley graph), and SNAP
//! (real-world social network).
//!
//! CH is designed for hierarchical graphs (like roads). On scale-free
//! networks with hubs, the shortcut count tends to explode. We document
//! that here.

use std::time::Instant;

use sssp_bench::ch;
use sssp_bench::dijkstra::dijkstra_binary;
use sssp_bench::graph::{gen_grid, gen_random_sparse, CsrGraph, Rng};
use sssp_bench::rubik::build_pocket_cube_graph;
use sssp_bench::synth::gen_rmat;

fn run_ch_bench(label: &str, g: &CsrGraph, n_pairs: usize) {
    let n = g.n;
    let m = g.m();
    println!("\n=== {label} ===");
    println!("Graph: n={n}, m={m}, avg_deg={:.2}", m as f32 / n.max(1) as f32);

    let t = Instant::now();
    let h = ch::build(g);
    let build_secs = t.elapsed().as_secs_f64();
    let m_aug = h.graph_fwd.m();
    println!(
        "CH build: {:.2} s, {} aug edges (×{:.2}), {} shortcuts",
        build_secs,
        m_aug,
        m_aug as f64 / m.max(1) as f64,
        m_aug.saturating_sub(m)
    );

    // Random query pairs
    let mut rng = Rng(20260509);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(n_pairs);
    while pairs.len() < n_pairs {
        let s = rng.range(n as u32);
        let d = rng.range(n as u32);
        if g.head[s as usize + 1] - g.head[s as usize] == 0 {
            continue;
        }
        if g.head[d as usize + 1] - g.head[d as usize] == 0 {
            continue;
        }
        pairs.push((s, d));
    }

    // CH queries (h.perm: CSR-id → CH-internal id; identity for SSSPCH1A)
    let t = Instant::now();
    let mut ch_results: Vec<f32> = Vec::with_capacity(n_pairs);
    for &(s, d) in &pairs {
        let si = h.perm[s as usize];
        let di = h.perm[d as usize];
        ch_results.push(ch::query(&h, si, di).unwrap_or(f32::INFINITY));
    }
    let ch_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Reference: full Dijkstra on first 10 pairs
    let n_ref = n_pairs.min(10);
    let t = Instant::now();
    let mut ref_results: Vec<f32> = Vec::with_capacity(n_ref);
    for &(s, d) in &pairs[..n_ref] {
        let dist = dijkstra_binary(g, s);
        ref_results.push(dist[d as usize]);
    }
    let ref_ms = t.elapsed().as_secs_f64() * 1000.0;

    let mut bad = 0;
    for i in 0..n_ref {
        let r = ref_results[i];
        let c = ch_results[i];
        let ok = if r.is_infinite() || c.is_infinite() {
            r == c
        } else {
            (r - c).abs() <= 1e-3 * (1.0 + r.abs())
        };
        if !ok {
            if bad < 3 {
                println!("    DIFF pair {i}: ref={r} ch={c}");
            }
            bad += 1;
        }
    }

    let speedup = (ref_ms / n_ref as f64) / (ch_ms / n_pairs as f64).max(0.0001);
    println!(
        "ch::query:        {:>9.3} ms/query ({:.0} ms total for {n_pairs} queries)",
        ch_ms / n_pairs as f64,
        ch_ms
    );
    println!(
        "full Dijkstra:    {:>9.1} ms/query ({:.0} ms total for {n_ref} queries)",
        ref_ms / n_ref as f64,
        ref_ms
    );
    println!(
        "speedup ch vs full: {:.1}× (correctness: {}/{} OK)",
        speedup,
        n_ref - bad,
        n_ref
    );
}

fn main() {
    println!("============ CH on non-road graphs ============");
    println!("CH is designed for hierarchical road networks.");
    println!("Here we measure how it degenerates on other graph types.\n");

    // Small RMAT — likely many shortcuts (CH degenerates on scale-free)
    {
        let g = gen_rmat(12, 8, 42);
        run_ch_bench("RMAT scale=12 (n=4k) [scale-free]", &g, 100);
    }
    {
        let g = gen_rmat(14, 8, 42);
        run_ch_bench("RMAT scale=14 (n=16k) [scale-free]", &g, 100);
    }

    // Grid — regular geometric, should work decently
    {
        let g = gen_grid(200, 7);
        run_ch_bench("Grid 200×200 (n=40k) [regular]", &g, 100);
    }

    // Sparse random
    {
        let g = gen_random_sparse(50_000, 4, 42);
        run_ch_bench("Sparse random n=50k deg=4 [random]", &g, 100);
    }

    // Rubik — small, very regular Cayley graph
    {
        let (g, _depth) = build_pocket_cube_graph();
        run_ch_bench("Rubik 2×2×2 (n=5040) [Cayley]", &g, 100);
    }

    // SNAP — DISABLED: CH degenerates badly on scale-free. Documented separately.
    println!("\n(SNAP omitted — CH degenerates on scale-free graphs;");
    println!(" see RMAT results above for the same effect.)");
}
