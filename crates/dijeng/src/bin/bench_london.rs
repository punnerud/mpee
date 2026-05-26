//! Benchmark SSSP algorithms on the Greater London routing graph.
//!
//! Each SSSP from a single source gives the shortest distance to all ~1M
//! nodes, i.e. ~1M (from, to) routes. We run from K random sources and report
//! mean + spread.

use std::time::Instant;

use sssp_bench::auto::sssp_auto;
use sssp_bench::delta_step::delta_stepping;
use sssp_bench::dijeng::{dijeng_4ary, dijeng_binary, INF};
use sssp_bench::duan::duan_inspired;
use sssp_bench::graph::{CsrGraph, Rng};
use sssp_bench::osm::load_with_cache;
use sssp_bench::osm_profile::Profile;

fn verify(reference: &[f32], other: &[f32], label: &str) -> bool {
    if reference.len() != other.len() {
        println!("    ! {label}: length mismatch");
        return false;
    }
    let eps = 1e-3_f32; // a bit looser than synthetic since weights are metres (larger numbers)
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
        true
    } else {
        let (i, a, b) = first_bad.unwrap();
        println!(
            "    FAIL {label}: {bad} mismatches (first v={i}: ref={a}, got={b})"
        );
        false
    }
}

fn estimate_avg_weight(g: &CsrGraph) -> f32 {
    if g.edge_w.is_empty() {
        return 1.0;
    }
    // Sample 100k edges for speed.
    let stride = (g.edge_w.len() / 100_000).max(1);
    let mut s = 0.0f64;
    let mut c = 0u64;
    let mut i = 0usize;
    while i < g.edge_w.len() {
        s += g.edge_w[i] as f64;
        c += 1;
        i += stride;
    }
    (s / c as f64) as f32
}

fn count_reachable(dist: &[f32]) -> usize {
    dist.iter().filter(|&&d| d.is_finite()).count()
}

#[derive(Default, Clone)]
struct TimingStats {
    times_ms: Vec<f64>,
}
impl TimingStats {
    fn add(&mut self, t_ms: f64) {
        self.times_ms.push(t_ms);
    }
    fn mean(&self) -> f64 {
        let s: f64 = self.times_ms.iter().sum();
        s / self.times_ms.len() as f64
    }
    fn min(&self) -> f64 {
        *self
            .times_ms
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
    fn max(&self) -> f64 {
        *self
            .times_ms
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
}

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/greater-london.osm.pbf".to_string());
    let cache_path = format!("{}.csr", &path);

    println!("=== London routing-graf benchmark ===");
    let total_load = Instant::now();
    let (g, _coords, _edge_dist) = load_with_cache(&path, &cache_path, Profile::Car)?;
    println!(
        "[osm] total load-tid: {:.0} ms",
        total_load.elapsed().as_secs_f64() * 1000.0
    );

    let avg_deg = g.m() as f32 / g.n.max(1) as f32;
    let avg_w = estimate_avg_weight(&g);
    println!(
        "  graph: n = {}, m = {}, avg_deg = {:.2}, avg_w = {:.2} m",
        g.n,
        g.m(),
        avg_deg,
        avg_w
    );

    let delta = (avg_w / avg_deg).max(1e-3);
    let bw = 4.0 * delta;
    println!("  delta = {delta:.2} m, bucket_width(duan) = {bw:.2} m");

    // We want to run from K random sources. But on a realistic OSM graph
    // many nodes are not in the largest component — pick only nodes with
    // degree ≥ 1.
    let mut candidates: Vec<u32> = Vec::new();
    for u in 0..g.n {
        if g.head[u + 1] - g.head[u] > 0 {
            candidates.push(u as u32);
        }
    }
    if candidates.is_empty() {
        eprintln!("no vertices with edges — aborting");
        std::process::exit(1);
    }
    println!("  {} vertices have at least one edge (source candidates)", candidates.len());

    let k_sources = 5usize;
    let mut rng = Rng(20260509);
    let mut sources: Vec<u32> = Vec::with_capacity(k_sources);
    for _ in 0..k_sources {
        let pick = rng.range(candidates.len() as u32) as usize;
        sources.push(candidates[pick]);
    }
    println!("  sources: {:?}", sources);
    println!();

    // For each algorithm, accumulate timings across the K sources.
    let mut t_bin = TimingStats::default();
    let mut t_4ary = TimingStats::default();
    let mut t_dstep = TimingStats::default();
    let mut t_duan = TimingStats::default();
    let mut t_auto = TimingStats::default();

    let mut total_correct = 0usize;
    let mut total_checks = 0usize;

    for (idx, &src) in sources.iter().enumerate() {
        println!("[source {}: vertex {}]", idx + 1, src);

        let t = Instant::now();
        let d_bin = dijeng_binary(&g, src);
        let dt = t.elapsed().as_secs_f64() * 1000.0;
        t_bin.add(dt);
        let reachable = count_reachable(&d_bin);
        println!(
            "  dijeng_binary: {:>7.1} ms   ({} of {} nodes reached)",
            dt, reachable, g.n
        );

        let t = Instant::now();
        let d_4 = dijeng_4ary(&g, src);
        let dt = t.elapsed().as_secs_f64() * 1000.0;
        t_4ary.add(dt);
        println!("  dijeng_4ary:   {:>7.1} ms", dt);
        total_checks += 1;
        if verify(&d_bin, &d_4, "4ary") {
            total_correct += 1;
        }

        let t = Instant::now();
        let d_ds = delta_stepping(&g, src, delta);
        let dt = t.elapsed().as_secs_f64() * 1000.0;
        t_dstep.add(dt);
        println!("  delta_stepping:  {:>7.1} ms", dt);
        total_checks += 1;
        if verify(&d_bin, &d_ds, "dstep") {
            total_correct += 1;
        }

        let t = Instant::now();
        let d_du = duan_inspired(&g, src, bw);
        let dt = t.elapsed().as_secs_f64() * 1000.0;
        t_duan.add(dt);
        println!("  duan_inspired:   {:>7.1} ms", dt);
        total_checks += 1;
        if verify(&d_bin, &d_du, "duan") {
            total_correct += 1;
        }

        let t = Instant::now();
        let d_au = sssp_auto(&g, src);
        let dt = t.elapsed().as_secs_f64() * 1000.0;
        t_auto.add(dt);
        println!("  sssp_auto:       {:>7.1} ms", dt);
        total_checks += 1;
        if verify(&d_bin, &d_au, "auto") {
            total_correct += 1;
        }

        println!();
    }

    println!("============ sammendrag (over {} kilder) ============", k_sources);
    println!(
        "  {:<22} mean       min       max",
        "algoritme"
    );
    let print_row = |name: &str, s: &TimingStats| {
        println!(
            "  {:<22} {:>7.1} ms {:>7.1} ms {:>7.1} ms",
            name,
            s.mean(),
            s.min(),
            s.max()
        );
    };
    print_row("dijeng_binary", &t_bin);
    print_row("dijeng_4ary", &t_4ary);
    print_row("delta_stepping", &t_dstep);
    print_row("duan_inspired", &t_duan);
    print_row("sssp_auto", &t_auto);

    println!();
    println!(
        "  speedup (mean) vs dijeng_binary:  4ary={:.2}x  dstep={:.2}x  duan={:.2}x  auto={:.2}x",
        t_bin.mean() / t_4ary.mean(),
        t_bin.mean() / t_dstep.mean(),
        t_bin.mean() / t_duan.mean(),
        t_bin.mean() / t_auto.mean()
    );
    println!();
    println!(
        "  korrekthet: {} / {} OK",
        total_correct, total_checks
    );
    println!(
        "  total ekvivalent ruter: {} kilder × {} noder = {} (s,t)-par",
        k_sources,
        g.n,
        k_sources * g.n
    );

    Ok(())
}
