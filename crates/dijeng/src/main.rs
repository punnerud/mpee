use sssp_bench::auto::sssp_auto;
use sssp_bench::delta_step::delta_stepping;
use sssp_bench::dijeng::{dijeng_4ary, dijeng_binary, INF};
use sssp_bench::duan::duan_inspired;
use sssp_bench::graph::{
    gen_grid, gen_path, gen_power_law, gen_random_exp_weights, gen_random_sparse, time_it,
    CsrGraph,
};

fn verify(reference: &[f32], other: &[f32], label: &str) -> bool {
    if reference.len() != other.len() {
        println!("  ! {label}: length mismatch");
        return false;
    }
    let eps = 1e-4_f32;
    let mut bad = 0usize;
    let mut first_bad = None;
    for i in 0..reference.len() {
        let a = reference[i];
        let b = other[i];
        let ok = if a == INF || b == INF {
            a == b
        } else {
            (a - b).abs() <= eps * (1.0 + a.abs())
        };
        if !ok {
            if first_bad.is_none() {
                first_bad = Some((i, a, b));
            }
            bad += 1;
        }
    }
    if bad == 0 {
        println!("  OK {label}: matches reference");
        true
    } else {
        let (i, a, b) = first_bad.unwrap();
        println!("  FAIL {label}: {bad} mismatches (first at v={i}: ref={a}, got={b})");
        false
    }
}

fn estimate_max_degree(g: &CsrGraph) -> usize {
    let mut m = 0;
    for u in 0..g.n {
        let d = (g.head[u + 1] - g.head[u]) as usize;
        if d > m {
            m = d;
        }
    }
    m
}

fn estimate_avg_weight(g: &CsrGraph) -> f32 {
    if g.edge_w.is_empty() {
        return 1.0;
    }
    let s: f64 = g.edge_w.iter().map(|&w| w as f64).sum();
    (s / g.edge_w.len() as f64) as f32
}

fn run_suite(label: &str, g: CsrGraph, src: u32) {
    let n = g.n;
    let m = g.m();
    let max_deg = estimate_max_degree(&g);
    let avg_w = estimate_avg_weight(&g);
    println!();
    println!("=== {label} ===");
    println!("  n = {n}, m = {m}, max_deg = {max_deg}, avg_w = {avg_w:.3}");

    let avg_deg = (m as f32 / n as f32).max(1.0);
    // Skaler delta etter gjennomsnittlig kantvekt: delta ≈ avg_w / avg_deg
    // er en standard heuristikk som overlever ikke-uniforme vekter.
    let delta = (avg_w / avg_deg).max(1e-4);
    println!("  delta = {delta:.4} (avg_w / avg_deg)");

    let bw = 4.0 * delta;

    let bench = |label: &str, f: &dyn Fn() -> Vec<f32>| -> (Vec<f32>, f64) {
        let mut best = f64::INFINITY;
        let mut last = Vec::new();
        for _ in 0..3 {
            let (r, t) = time_it(label, f);
            if t < best {
                best = t;
            }
            last = r;
        }
        println!("  {:<28} best: {:>8.3} ms", label, best * 1000.0);
        (last, best)
    };

    let (ref_dist, t_bin) = bench("dijeng_binary", &|| dijeng_binary(&g, src));
    let (d_4ary, t_4ary) = bench("dijeng_4ary", &|| dijeng_4ary(&g, src));
    let (d_dstep, t_dstep) = bench("delta_stepping", &|| delta_stepping(&g, src, delta));
    let (d_duan, t_duan) = bench("duan_inspired", &|| duan_inspired(&g, src, bw));
    let (d_auto, t_auto) = bench("sssp_auto (hybrid)", &|| sssp_auto(&g, src));

    println!();
    verify(&ref_dist, &d_4ary, "dijeng_4ary");
    verify(&ref_dist, &d_dstep, "delta_stepping");
    verify(&ref_dist, &d_duan, "duan_inspired");
    verify(&ref_dist, &d_auto, "sssp_auto");

    println!();
    println!("  Speedup vs binary heap Dijeng:");
    println!("    4-ary heap     : {:>5.2}x", t_bin / t_4ary);
    println!("    delta-stepping : {:>5.2}x", t_bin / t_dstep);
    println!("    duan-inspirert : {:>5.2}x", t_bin / t_duan);
    println!("    sssp_auto      : {:>5.2}x", t_bin / t_auto);
}

fn main() {
    println!("SSSP-benchmark - Dijeng-varianter vs delta-stepping vs Duan-inspirert");
    println!("(beste-av-3 wall-clock, samme seed paa tvers av algoritmer)");

    // Liten korrekthetstest forst.
    {
        let g = gen_random_sparse(2_000, 6, 1);
        let r = dijeng_binary(&g, 0);
        let a = dijeng_4ary(&g, 0);
        let avg_deg = (g.m() as f32 / g.n as f32).max(1.0);
        let delta = 1.0 / avg_deg;
        let s = delta_stepping(&g, 0, delta);
        let d = duan_inspired(&g, 0, 4.0 * delta);
        println!("\n[korrekthetstest n=2000, deg=6]");
        verify(&r, &a, "dijeng_4ary");
        verify(&r, &s, "delta_stepping");
        verify(&r, &d, "duan_inspired");
    }

    // Scale ladder: same graph type, increasing n, shows how the algorithms
    // scale.
    run_suite("Sparse random n=10k, avg_deg=8", gen_random_sparse(10_000, 8, 42), 0);
    run_suite("Sparse random n=100k, avg_deg=8", gen_random_sparse(100_000, 8, 42), 0);
    run_suite("Sparse random n=1M, avg_deg=8", gen_random_sparse(1_000_000, 8, 42), 0);
    run_suite("Sparse random n=4M, avg_deg=8", gen_random_sparse(4_000_000, 8, 42), 0);

    // Density ladder: how does it change as deg grows?
    run_suite("Sparse random n=200k, avg_deg=4", gen_random_sparse(200_000, 4, 42), 0);
    run_suite("Sparse random n=200k, avg_deg=16", gen_random_sparse(200_000, 16, 42), 0);
    run_suite("Sparse random n=200k, avg_deg=64", gen_random_sparse(200_000, 64, 42), 0);

    // Geometrisk: 2D-grid (alle naboer, korte avstander).
    run_suite("Grid 500x500 (n=250k)", gen_grid(500, 7), 0);
    run_suite("Grid 1000x1000 (n=1M)", gen_grid(1000, 7), 0);
    run_suite("Grid 2000x2000 (n=4M)", gen_grid(2000, 7), 0);

    // Eksponensielle vekter: utfordrer bucket-bredder.
    run_suite(
        "Exp-weights n=500k, avg_deg=8 (mean=1.0)",
        gen_random_exp_weights(500_000, 8, 1.0, 42),
        0,
    );
    run_suite(
        "Exp-weights n=500k, avg_deg=8 (mean=10.0)",
        gen_random_exp_weights(500_000, 8, 10.0, 42),
        0,
    );

    // Power-law: hubs, ujevne grader.
    run_suite("Power-law n=200k, m_edges=4", gen_power_law(200_000, 4, 13), 0);
    run_suite("Power-law n=500k, m_edges=8", gen_power_law(500_000, 8, 13), 0);

    // Verstefall for bucket-algoritmer: en sti.
    run_suite("Path graph n=100k", gen_path(100_000, 11), 0);
    run_suite("Path graph n=500k", gen_path(500_000, 11), 0);
}
