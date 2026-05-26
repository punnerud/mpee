//! Build CH on the London road network and benchmark p2p queries against
//! our bidirectional Dijeng and OSRM (if running on localhost:5002).

use std::time::Instant;

use sssp_bench::bidir::{bidir_dijeng, transpose};
use sssp_bench::cache_ch;
use sssp_bench::cache_pp;
use sssp_bench::ch;
use sssp_bench::dijeng::dijeng_binary;
use sssp_bench::graph::Rng;

fn main() -> std::io::Result<()> {
    // Allow specifying dataset and profile: `bench_ch london [car|motorcycle|bicycle|foot]`.
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "london".to_string());
    let profile_name = std::env::args().nth(2).unwrap_or_else(|| "car".to_string());
    let profile = match sssp_bench::osm_profile::Profile::from_name(&profile_name) {
        Some(p) => p,
        None => {
            eprintln!("unknown profile '{profile_name}' — try car/motorcycle/bicycle/foot");
            std::process::exit(1);
        }
    };
    let suffix = if profile == sssp_bench::osm_profile::Profile::Car {
        String::new()
    } else {
        format!(".{}", profile.name())
    };
    let (pbf_name, csr_path_str, pp_path_str, ch_path_str) = match dataset.as_str() {
        "london" => (
            "Greater London",
            format!("data/greater-london.osm.pbf{suffix}.csr"),
            format!("data/greater-london.osm.pbf{suffix}.pp"),
            format!("data/greater-london.osm.pbf{suffix}.ch"),
        ),
        "england" => (
            "England",
            format!("data/england.osm.pbf{suffix}.csr"),
            format!("data/england.osm.pbf{suffix}.pp"),
            format!("data/england.osm.pbf{suffix}.ch"),
        ),
        // Any other value is treated as a custom PBF base path (same fallback
        // as bench_pp), so `bench_ch path/to/region.osm.pbf` works for ANY
        // region — not just the two built-in shortcuts.
        custom => (
            custom,
            format!("{custom}{suffix}.csr"),
            format!("{custom}{suffix}.pp"),
            format!("{custom}{suffix}.ch"),
        ),
    };
    let csr_path = csr_path_str.as_str();
    let pp_path = pp_path_str.as_str();
    let ch_path = ch_path_str.as_str();
    println!("[bench_ch] profile = {}", profile.name());
    println!("Dataset: {pbf_name}");
    let _ = csr_path;
    let pp = match cache_pp::load_mmap(pp_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("pp-cache missing ({e}). Run bench_pp on this PBF first.");
            std::process::exit(1);
        }
    };

    let g = &pp.graph;
    let n = g.n;
    println!("=== CH on {pbf_name} ===");
    println!("Graph: n={}, m={}, avg_deg={:.2}", n, g.m(), g.m() as f32 / n.max(1) as f32);

    // Try mmap-load CH cache first (instant) — else build and save.
    let ch_cache = ch_path;
    let (h, build_secs) = if std::path::Path::new(ch_cache).exists() {
        let t = Instant::now();
        match cache_ch::load_mmap(ch_cache) {
            Ok(h) => {
                let load_ms = t.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "[ch-cache] mmap-loaded {} nodes + {} edges in {:.2} ms",
                    h.graph_fwd.n,
                    h.graph_fwd.m(),
                    load_ms
                );
                (h, 0.0_f64)
            }
            Err(e) => {
                println!("[ch-cache] corrupt ({e}) — rebuilding");
                println!("\nBuilding CH (this is the expensive step)...");
                let t = Instant::now();
                let h = ch::build_with_dist(g, &pp.edge_dist[..]);
                let secs = t.elapsed().as_secs_f64();
                println!("CH build total: {:.1} s", secs);
                let _ = cache_ch::save(ch_cache, &h);
                (h, secs)
            }
        }
    } else {
        println!("\nBuilding CH (this is the expensive step)...");
        let t = Instant::now();
        let h = ch::build_with_dist(g, &pp.edge_dist[..]);
        let secs = t.elapsed().as_secs_f64();
        println!("CH build total: {:.1} s", secs);
        let t_save = Instant::now();
        match cache_ch::save(ch_cache, &h) {
            Ok(_) => println!(
                "[ch-cache] saved to {} ({:.0} ms)",
                ch_cache,
                t_save.elapsed().as_secs_f64() * 1000.0
            ),
            Err(e) => println!("[ch-cache] could not save: {e}"),
        }
        (h, secs)
    };
    println!(
        "Augmented edges: {} (vs original {})",
        h.graph_fwd.m(),
        g.m()
    );

    // Reverse for our bidir baseline
    let g_bwd = transpose(g);

    // Generate random (s, d) pairs
    let n_queries = 1000usize;
    let mut rng = Rng(20260509);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(n_queries);
    while pairs.len() < n_queries {
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

    // ---- CH query ----
    // For SSSPCH1B (rank-ordered) caches, h.perm maps CSR-IDs → CH-internal IDs.
    // For SSSPCH1A caches it's the identity, so the same call works for both.
    let t = Instant::now();
    let mut ch_results: Vec<f32> = Vec::with_capacity(n_queries);
    for &(s, d) in &pairs {
        let si = h.perm[s as usize];
        let di = h.perm[d as usize];
        ch_results.push(ch::query(&h, si, di).unwrap_or(f32::INFINITY));
    }
    let ch_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "\nch::query:           {:>9.1} ms total ({:.4} ms/query)",
        ch_ms,
        ch_ms / n_queries as f64
    );

    // ---- Our bidir Dijeng ----
    let t = Instant::now();
    let mut bidir_results: Vec<f32> = Vec::with_capacity(n_queries);
    for &(s, d) in &pairs {
        bidir_results.push(bidir_dijeng(g, &g_bwd, s, d).unwrap_or(f32::INFINITY));
    }
    let bidir_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "bidir_dijeng:      {:>9.1} ms total ({:.3} ms/query)",
        bidir_ms,
        bidir_ms / n_queries as f64
    );

    // ---- Reference: full Dijeng on first 30 pairs ----
    let n_ref = 30usize;
    let t = Instant::now();
    let mut ref_results: Vec<f32> = Vec::with_capacity(n_ref);
    for &(s, d) in &pairs[..n_ref] {
        let dist = dijeng_binary(g, s);
        ref_results.push(dist[d as usize]);
    }
    let ref_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "full Dijeng:       {:>9.0} ms for {} queries ({:.1} ms/query)",
        ref_ms, n_ref, ref_ms / n_ref as f64
    );

    // ---- Correctness ----
    let mut bad_ch = 0;
    let mut bad_bidir = 0;
    for i in 0..n_ref {
        let r = ref_results[i];
        let c = ch_results[i];
        let b = bidir_results[i];
        let ok_ch = if r.is_infinite() || c.is_infinite() { r == c } else { (r - c).abs() <= 1e-3 * (1.0 + r.abs()) };
        let ok_b  = if r.is_infinite() || b.is_infinite() { r == b } else { (r - b).abs() <= 1e-3 * (1.0 + r.abs()) };
        if !ok_ch {
            if bad_ch < 3 {
                println!("    DIFF ch:    pair {i}: ref={r} ch={c}");
            }
            bad_ch += 1;
        }
        if !ok_b {
            bad_bidir += 1;
        }
    }
    println!("\nCorrectness vs full Dijeng (first {n_ref} pairs):");
    println!("  ch::query:       {}/{} OK", n_ref - bad_ch, n_ref);
    println!("  bidir_dijeng:  {}/{} OK", n_ref - bad_bidir, n_ref);

    // ---- Speedup vs full SSSP per query ----
    let per_full = ref_ms / n_ref as f64;
    let per_bidir = bidir_ms / n_queries as f64;
    let per_ch = ch_ms / n_queries as f64;
    println!("\nSpeedup vs full Dijeng (per query):");
    println!("  bidir:  {:.1}× ({:.2} ms vs {:.1} ms)", per_full / per_bidir, per_bidir, per_full);
    println!("  ch:     {:.1}× ({:.3} ms vs {:.1} ms)", per_full / per_ch, per_ch, per_full);

    // Try OSRM if available
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(5))
        .build();
    let test = format!(
        "http://localhost:5002/route/v1/driving/{},{};{},{}?overview=false",
        pp.coords[0].1, pp.coords[0].0,
        pp.coords[1].1, pp.coords[1].0
    );
    if agent.get(&test).call().is_ok() {
        println!("\nOSRM detected on localhost:5002 — running 1000 queries...");
        let t = Instant::now();
        let mut osrm_results: Vec<f32> = Vec::with_capacity(n_queries);
        let mut osrm_failed = 0;
        for &(s, d) in &pairs {
            let (lat_s, lon_s) = pp.coords[s as usize];
            let (lat_d, lon_d) = pp.coords[d as usize];
            let url = format!(
                "http://localhost:5002/route/v1/driving/{},{};{},{}?overview=false&steps=false",
                lon_s, lat_s, lon_d, lat_d
            );
            match agent.get(&url).call() {
                Ok(resp) => {
                    let body = resp.into_string().unwrap_or_default();
                    if let Some(idx) = body.find("\"distance\":") {
                        let rest = &body[idx + 11..];
                        let end = rest.find(['}', ',']).unwrap_or(rest.len());
                        osrm_results.push(rest[..end].parse().unwrap_or(f32::NAN));
                    } else {
                        osrm_failed += 1;
                        osrm_results.push(f32::INFINITY);
                    }
                }
                Err(_) => {
                    osrm_failed += 1;
                    osrm_results.push(f32::INFINITY);
                }
            }
        }
        let osrm_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "OSRM CH (HTTP):     {:>9.1} ms total ({:.3} ms/query)  [{} failed]",
            osrm_ms,
            osrm_ms / n_queries as f64,
            osrm_failed
        );
        println!(
            "  ch vs OSRM:  {:.2}× (our {:.3} ms vs OSRM {:.3} ms)",
            (osrm_ms / n_queries as f64) / per_ch,
            per_ch,
            osrm_ms / n_queries as f64
        );
    } else {
        println!("\n(OSRM not running on localhost:5002 — skip OSRM comparison)");
    }

    println!("\n=== Summary ===");
    println!("Preprocessing: {:.1} s (vs OSRM ~37 s)", build_secs);
    println!("Per-query:");
    println!("  full Dijeng:  {:.1} ms", per_full);
    println!("  bidir:          {:.2} ms", per_bidir);
    println!("  ch::query:      {:.4} ms", per_ch);

    Ok(())
}
