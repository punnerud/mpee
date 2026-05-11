//! Benchmark knn_matrix on London at the requested N=50K, K=160.

use std::time::Instant;

use sssp_bench::cache_pp;
use sssp_bench::graph::Rng;
use sssp_bench::knn::knn_matrix;

fn main() -> std::io::Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "london".to_string());
    let n_customers: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "50000".to_string())
        .parse()
        .unwrap_or(50_000);
    let k: usize = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "160".to_string())
        .parse()
        .unwrap_or(160);

    let pp_path = match dataset.as_str() {
        "london" => "data/greater-london.osm.pbf.pp",
        "england" => "data/england.osm.pbf.pp",
        other => {
            eprintln!("unknown dataset {other}");
            std::process::exit(1);
        }
    };
    let pp = cache_pp::load_mmap(pp_path).expect("pp-cache missing — run bench_pp first");
    let g = &pp.graph;
    println!(
        "[bench_knn] dataset={dataset} graph_n={} n_customers={} k={} threads={}",
        g.n,
        n_customers,
        k,
        rayon::current_num_threads()
    );

    let mut candidates: Vec<u32> = (0..g.n as u32)
        .filter(|&u| g.head[u as usize + 1] - g.head[u as usize] > 0)
        .collect();
    println!("reachable vertices: {}", candidates.len());
    assert!(
        n_customers <= candidates.len(),
        "asked for more customers than reachable vertices"
    );

    let mut rng = Rng(20260511);
    let mut customers: Vec<u32> = Vec::with_capacity(n_customers);
    // Fisher-Yates shuffle prefix to pick unique customer nodes.
    for i in 0..n_customers {
        let j = i + (rng.range((candidates.len() - i) as u32) as usize);
        candidates.swap(i, j);
        customers.push(candidates[i]);
    }

    let t = Instant::now();
    let rows = knn_matrix(g, &customers, k, Some(pp.edge_dist.as_slice()));
    let secs = t.elapsed().as_secs_f64();

    let total: usize = rows.iter().map(|r| r.len()).sum();
    let full_rows = rows.iter().filter(|r| r.len() == k).count();
    let avg_max_dur: f64 = rows
        .iter()
        .filter(|r| r.len() == k)
        .map(|r| r[k - 1].1 as f64)
        .sum::<f64>()
        / full_rows.max(1) as f64;
    let avg_max_dist: f64 = rows
        .iter()
        .filter(|r| r.len() == k)
        .map(|r| r[k - 1].2 as f64)
        .sum::<f64>()
        / full_rows.max(1) as f64;

    let bytes = total * 12 + n_customers * 24; // entries + Vec headers
    println!(
        "\n── knn_matrix complete in {:.2} s ──",
        secs
    );
    println!(
        "  per-src mean: {:.2} µs   throughput: {:.1} src/s",
        secs * 1e6 / n_customers as f64,
        n_customers as f64 / secs
    );
    println!(
        "  output: {:.1} MB ({} total neighbour entries, {} full K rows / {})",
        bytes as f64 / 1024.0 / 1024.0,
        total,
        full_rows,
        n_customers
    );
    println!(
        "  K-th-nearest avg: {:.0} s dur, {:.0} m dist",
        avg_max_dur, avg_max_dist
    );

    Ok(())
}
