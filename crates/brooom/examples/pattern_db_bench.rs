//! Benchmark in-memory KNN over the sub-route pattern DB.
//!
//! Loads `benchmarks/gh_canonical/subroutes_n200_k4.jsonl` (≈10K patterns
//! × 4-d signature) and measures load + per-query latency.
//!
//! Run with:
//!   cargo run --release --example pattern_db_bench

use std::time::Instant;

use brooom::pattern_db::PatternDb;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "benchmarks/gh_canonical/subroutes_n200_k4.jsonl";

    let t0 = Instant::now();
    let db = PatternDb::load_jsonl(path)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Loaded {} patterns × {}-d signature in {:.1} ms",
        db.len(),
        db.sig_dim,
        load_ms
    );

    // Stress KD-tree path by replicating the corpus 6× (~60K patterns).
    // This crosses KD_TREE_THRESHOLD = 50_000 so the KD-tree code runs.
    println!("\nReplicating corpus 6× for KD-tree stress test (~60K patterns)...");
    let big_path = "benchmarks/gh_canonical/subroutes_n200_k4_big.jsonl";
    if !std::path::Path::new(big_path).exists() {
        let src = std::fs::read_to_string(path)?;
        let mut all = String::new();
        for _ in 0..6 {
            all.push_str(&src);
        }
        std::fs::write(big_path, all)?;
    }
    let t0 = Instant::now();
    let db_big = PatternDb::load_jsonl(big_path)?;
    let load_big = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  loaded {} patterns in {:.1} ms (KD-tree built)", db_big.len(), load_big);
    let q = db_big.patterns[100].signature.clone();
    let n_queries = 1000;
    let t0 = Instant::now();
    let mut total = 0usize;
    for _ in 0..n_queries {
        total += db_big.knn(&q, 5).len();
    }
    let kd_us_per_query = t0.elapsed().as_secs_f64() * 1e6 / n_queries as f64;
    println!("  KD-tree (n={}, k=5): {:.1} µs/query  total {} neighbors", db_big.len(), kd_us_per_query, total);

    if db.is_empty() {
        return Ok(());
    }

    // Build a query from the first pattern's signature for sanity check
    // (should return itself with distance 0).
    let q = db.patterns[0].signature.clone();
    let nn = db.knn(&q, 3);
    println!("\nSanity check — query = pattern[0] signature:");
    for (i, (d, p)) in nn.iter().enumerate() {
        println!(
            "  rank {} d={:.4}  inst={} v={} cust={:?}",
            i, d, p.instance, p.vehicle, p.customers
        );
    }

    // Benchmark batch queries.
    let n_queries = 1000;
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| db.patterns[i % db.len()].signature.clone())
        .collect();

    let t0 = Instant::now();
    let mut total_neighbors = 0usize;
    for q in &queries {
        let nn = db.knn(q, 5);
        total_neighbors += nn.len();
    }
    let elapsed = t0.elapsed();
    let per_query_us = elapsed.as_secs_f64() * 1e6 / n_queries as f64;

    println!(
        "\n{} queries (k=5) in {:.1} ms — {:.1} µs/query, {} total neighbors returned",
        n_queries,
        elapsed.as_secs_f64() * 1000.0,
        per_query_us,
        total_neighbors
    );

    // Try a "novel" query — perturb pattern[100]'s signature slightly.
    let mut q = db.patterns[100].signature.clone();
    for v in q.iter_mut() {
        *v += 0.05;
    }
    let nn = db.knn(&q, 5);
    println!("\nNovel query — pattern[100] perturbed by 0.05 in each dim:");
    for (i, (d, p)) in nn.iter().enumerate() {
        println!(
            "  rank {} d={:.4}  inst={} v={} cust={:?}",
            i, d, p.instance, p.vehicle, p.customers
        );
    }

    Ok(())
}
