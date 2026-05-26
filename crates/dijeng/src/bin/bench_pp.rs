//! Benchmark of the preprocessing pipeline: BFS reorder, light/heavy partition,
//! transpose, and bidirectional Dijeng. Caches the full bundle to a single
//! mmap-friendly file for instant cold start.

use std::time::Instant;

use dijeng::auto::sssp_auto;
use dijeng::bidir::bidir_dijeng;
use dijeng::cache_pp;
use dijeng::delta_step::{delta_stepping, delta_stepping_partitioned};
use dijeng::dijeng::{dijeng_4ary, dijeng_binary};
use dijeng::graph::{CsrGraph, Rng};
use dijeng::osm::load_with_cache;
use dijeng::osm_profile::Profile;
use dijeng::preprocess::{preprocess, remap_src};

fn estimate_avg_weight(g: &CsrGraph) -> f32 {
    if g.edge_w.is_empty() {
        return 1.0;
    }
    let stride = (g.edge_w.len() / 4096).max(1);
    let mut s = 0.0f64;
    let mut c = 0u64;
    let mut i = 0;
    while i < g.edge_w.len() {
        s += g.edge_w[i] as f64;
        c += 1;
        i += stride;
    }
    (s / c as f64) as f32
}

fn main() -> std::io::Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "london".to_string());
    let profile_name = std::env::args().nth(2).unwrap_or_else(|| "car".to_string());
    let profile = match Profile::from_name(&profile_name) {
        Some(p) => p,
        None => {
            eprintln!("unknown profile '{profile_name}' - try car/motorcycle/bicycle/foot");
            std::process::exit(1);
        }
    };
    let pbf_path: String = match dataset.as_str() {
        "london" => "data/greater-london.osm.pbf".to_string(),
        "england" => "data/england.osm.pbf".to_string(),
        custom => custom.to_string(),
    };
    let pbf = pbf_path.as_str();
    // Profile-suffixed caches except for "car", which keeps the unsuffixed
    // names for backward compatibility with previously generated files.
    let suffix = if profile == Profile::Car {
        String::new()
    } else {
        format!(".{}", profile.name())
    };
    let csr_cache_string = format!("{pbf}{suffix}.csr");
    let pp_cache_string = format!("{pbf}{suffix}.pp");
    let csr_cache = csr_cache_string.as_str();
    let pp_cache = pp_cache_string.as_str();

    println!("=== {pbf} preprocessing pipeline ===");

    // ---- Try loading the preprocessed cache first ----
    let pp = if std::path::Path::new(pp_cache).exists() {
        let t = Instant::now();
        match cache_pp::load_mmap(pp_cache) {
            Ok(pp) => {
                println!(
                    "[pp-cache] mmap hit in {:.2} ms - n={}, m={}, delta={:.3}",
                    t.elapsed().as_secs_f64() * 1000.0,
                    pp.graph.n,
                    pp.graph.m(),
                    pp.delta
                );
                Some(pp)
            }
            Err(e) => {
                println!("[pp-cache] corrupt ({e}) - rebuilding");
                None
            }
        }
    } else {
        println!("[pp-cache] missing - building from scratch");
        None
    };

    let pp = match pp {
        Some(pp) => pp,
        None => {
            // 1. Load the raw map from .csr cache (or parse the pbf).
            let (g, coords, edge_dist) = load_with_cache(pbf, csr_cache, profile)?;
            let avg_deg = g.m() as f32 / g.n.max(1) as f32;
            let avg_w = estimate_avg_weight(&g);
            let delta = (avg_w / avg_deg).max(1e-4);
            println!("  raw CSR: n = {}, m = {}, delta = {:.3}", g.n, g.m(), delta);

            let t = Instant::now();
            let preprocessed = preprocess(&g, Some(delta), edge_dist.as_slice());
            println!(
                "  preprocess (reorder + partition): {:.2} s",
                t.elapsed().as_secs_f64()
            );

            let t = Instant::now();
            let (reverse, rev_edge_dist) = dijeng::bidir::transpose_with_dist(
                &preprocessed.graph,
                preprocessed.edge_dist.as_slice(),
            );
            println!(
                "  transpose:                          {:.2} s",
                t.elapsed().as_secs_f64()
            );

            let mut new_coords = vec![(0.0f32, 0.0f32); g.n];
            for old in 0..g.n {
                let new = preprocessed.new_id[old] as usize;
                new_coords[new] = coords[old];
            }

            let t = Instant::now();
            cache_pp::save(
                pp_cache,
                &preprocessed.graph,
                &reverse,
                &preprocessed.light_count,
                &preprocessed.new_id,
                &new_coords,
                delta,
                preprocessed.edge_dist.as_slice(),
                &rev_edge_dist,
            )?;
            println!(
                "  pp-cache saved:                     {:.0} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );

            cache_pp::PpFull {
                graph: preprocessed.graph,
                reverse,
                edge_dist: preprocessed.edge_dist,
                rev_edge_dist: rev_edge_dist.into(),
                light_count: preprocessed.light_count,
                new_id: preprocessed.new_id,
                coords: dijeng::buffer::Buffer::from(new_coords),
                delta,
            }
        }
    };

    let g = &pp.graph;
    let n = g.n;
    let avg_w = estimate_avg_weight(g);
    let avg_deg = g.m() as f32 / n.max(1) as f32;
    println!(
        "  graph: n={}, m={}, avg_deg={:.2}, avg_w={:.2}, delta={:.2}",
        n,
        g.m(),
        avg_deg,
        avg_w,
        pp.delta
    );

    // ----- SSSP bench on the reordered graph -----
    let mut rng = Rng(20260509);
    let src_old = rng.range(n as u32);
    let src = remap_src_local(&pp.new_id[..], src_old);

    println!();
    println!("=== SSSP from src (reordered idx={src}) ===");

    let t = Instant::now();
    let r = dijeng_binary(g, src);
    println!(
        "  dijeng_binary:           {:>8.1} ms",
        t.elapsed().as_secs_f64() * 1000.0
    );

    let t = Instant::now();
    let _ = dijeng_4ary(g, src);
    println!(
        "  dijeng_4ary:             {:>8.1} ms",
        t.elapsed().as_secs_f64() * 1000.0
    );

    let t = Instant::now();
    let _ = delta_stepping(g, src, pp.delta);
    println!(
        "  delta_stepping (plain):    {:>8.1} ms",
        t.elapsed().as_secs_f64() * 1000.0
    );

    let t = Instant::now();
    let pp_result = delta_stepping_partitioned(g, &pp.light_count, src, pp.delta);
    println!(
        "  delta_stepping_pp (light/heavy split): {:>8.1} ms",
        t.elapsed().as_secs_f64() * 1000.0
    );

    let t = Instant::now();
    let _ = sssp_auto(g, src);
    println!(
        "  sssp_auto:                 {:>8.1} ms",
        t.elapsed().as_secs_f64() * 1000.0
    );

    // Verify pp result against reference
    let mut bad = 0;
    for i in 0..n {
        let a = r[i];
        let b = pp_result[i];
        let ok = if a.is_infinite() || b.is_infinite() {
            a == b
        } else {
            (a - b).abs() <= 1e-3 * (1.0 + a.abs())
        };
        if !ok {
            bad += 1;
        }
    }
    println!(
        "  correctness (delta_stepping_pp vs binary): {}",
        if bad == 0 {
            "OK".to_string()
        } else {
            format!("FAIL ({bad})")
        }
    );

    // ----- Bidirectional Dijeng: 1000 random (src, dst) pairs -----
    println!();
    println!("=== Bidirectional Dijeng: 1000 (src,dst) pairs ===");
    let n_queries = 1000usize;
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let s = rng.range(n as u32);
        let d = rng.range(n as u32);
        pairs.push((s, d));
    }

    // Reference: full SSSP from each src and read off dst - only run 100 of them
    // since each is ~50ms.
    let n_ref = 100usize;
    let mut ref_dists: Vec<f32> = Vec::with_capacity(n_ref);
    let t = Instant::now();
    for &(s, d) in &pairs[..n_ref] {
        let dist = dijeng_binary(g, s);
        ref_dists.push(dist[d as usize]);
    }
    let ref_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  reference (100 SSSP):      {:>8.1} ms ({:.2} ms/query)",
        ref_ms,
        ref_ms / n_ref as f64
    );

    // Bidir on the same 100, compare correctness
    let mut bidir_dists: Vec<Option<f32>> = Vec::with_capacity(n_queries);
    let t = Instant::now();
    for &(s, d) in &pairs {
        let r = bidir_dijeng(g, &pp.reverse, s, d);
        bidir_dists.push(r);
    }
    let bidir_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  bidir (1000 queries):      {:>8.1} ms ({:.3} ms/query)",
        bidir_ms,
        bidir_ms / n_queries as f64
    );

    let mut ok = 0;
    let mut bad = 0;
    for i in 0..n_ref {
        let a = ref_dists[i];
        let b = bidir_dists[i].unwrap_or(f32::INFINITY);
        let m = if a.is_infinite() || b.is_infinite() {
            a == b
        } else {
            (a - b).abs() <= 1e-3 * (1.0 + a.abs())
        };
        if m {
            ok += 1;
        } else {
            bad += 1;
            if bad <= 3 {
                println!("    DIFF: pair {}: ref={a} bidir={b}", i);
            }
        }
    }
    println!("  correctness (first 100): {ok}/100 OK");

    println!();
    println!("=== Summary ===");
    println!(
        "  bidir speedup vs full SSSP per (s,t): {:.0}x ({:.3} ms vs {:.1} ms)",
        (ref_ms / n_ref as f64) / (bidir_ms / n_queries as f64),
        bidir_ms / n_queries as f64,
        ref_ms / n_ref as f64
    );

    Ok(())
}

fn remap_src_local(new_id: &[u32], old: u32) -> u32 {
    new_id[old as usize]
}

#[allow(dead_code)]
fn _use_remap_src(pp: &cache_pp::PpFull, src: u32) -> u32 {
    let pp_simple = dijeng::preprocess::Preprocessed {
        graph: CsrGraph {
            n: pp.graph.n,
            head: pp.graph.head.clone(),
            edge_to: pp.graph.edge_to.clone(),
            edge_w: pp.graph.edge_w.clone(),
        },
        edge_dist: pp.edge_dist.clone(),
        light_count: pp.light_count.clone(),
        new_id: pp.new_id.clone(),
        delta_used: pp.delta,
    };
    remap_src(&pp_simple, src)
}
