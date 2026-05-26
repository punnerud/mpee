//! Measure single-pair CH query latency in nanoseconds, across:
//!   * cold start (first 100 calls)
//!   * warm sequential steady-state (next N calls)
//!   * parallel throughput (rayon, all cores)
//!   * with vs without path unpacking
//!
//! Output is what the consumer (a VRP solver) needs to choose between:
//!   - <100 ns/pair → drop matrix entirely, on-demand single calls
//!   - 100 ns–10 µs/pair → K-NN-only matrix
//!   - >10 µs/pair → batch precompute (current MMM)

use std::time::Instant;

use rayon::prelude::*;
use dijeng::cache_ch;
use dijeng::ch::{self, PathScratch};
use dijeng::graph::Rng;

fn main() -> std::io::Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "london".to_string());
    let n_queries: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "200000".to_string())
        .parse()
        .unwrap_or(200_000);

    let ch_path = match dataset.as_str() {
        "london" => "data/greater-london.osm.pbf.ch",
        "england" => "data/england.osm.pbf.ch",
        other => {
            eprintln!("unknown dataset {other}");
            std::process::exit(1);
        }
    };
    let h = cache_ch::load_mmap(ch_path).expect("ch-cache missing — run bench_ch first");
    let n = h.graph_fwd.n;
    println!(
        "[bench_latency] dataset={dataset} graph_n={n} threads={}",
        rayon::current_num_threads()
    );

    // Pick random reachable pairs once.
    let mut candidates: Vec<u32> = (0..n as u32)
        .filter(|&u| h.graph_fwd.head[u as usize + 1] - h.graph_fwd.head[u as usize] > 0)
        .collect();
    candidates.shrink_to_fit();
    println!("reachable vertices: {}", candidates.len());

    let mut rng = Rng(20260511);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(n_queries);
    while pairs.len() < n_queries {
        let s = candidates[rng.range(candidates.len() as u32) as usize];
        let d = candidates[rng.range(candidates.len() as u32) as usize];
        pairs.push((s, d));
    }

    // ───── Cold: time first 100 calls separately ─────
    let cold_n = 100usize.min(n_queries);
    let t = Instant::now();
    let mut sink = 0.0f32;
    for &(s, d) in &pairs[..cold_n] {
        sink += ch::query(&h, s, d).unwrap_or(0.0);
    }
    let cold_ns = t.elapsed().as_nanos() as f64;
    println!(
        "\n── cold (first {cold_n} calls) ──\n  total: {:.3} ms   mean: {:.0} ns/call",
        cold_ns / 1e6,
        cold_ns / cold_n as f64
    );

    // ───── Warm sequential, ALLOC-PER-CALL (current ch::query) ─────
    // Hammer some pairs first to ensure code+cache are hot, then time.
    for &(s, d) in &pairs[..1024.min(n_queries)] {
        sink += ch::query(&h, s, d).unwrap_or(0.0);
    }
    let t = Instant::now();
    let mut warm_finite = 0usize;
    for &(s, d) in &pairs {
        let v = ch::query(&h, s, d).unwrap_or(f32::INFINITY);
        if v.is_finite() {
            warm_finite += 1;
            sink += v;
        }
    }
    let warm_ns = t.elapsed().as_nanos() as f64;
    let warm_per = warm_ns / n_queries as f64;
    let calls_per_s = |ns: f64| 1.0e9 / ns;
    println!(
        "\n── warm sequential, alloc-per-call ({n_queries} calls, 1 thread) ──\n  \
         total: {:.2} s   mean: {:.0} ns/call ({:.2} µs)   throughput: {:.1} k calls/s   \
         finite: {}/{}",
        warm_ns / 1e9,
        warm_per,
        warm_per / 1000.0,
        calls_per_s(warm_per) / 1000.0,
        warm_finite,
        n_queries
    );

    // ───── Warm sequential, REUSED SCRATCH (proper steady-state) ─────
    // Uses query_with_path_into but ignores the path — same compute work as
    // a path-aware query, but the dist arrays are reused so we measure pure
    // CH compute, not the allocator. This is the number the solver should
    // base its architecture decision on.
    let mut scratch = PathScratch::new(n);
    // Warm-up.
    for &(s, d) in &pairs[..1024.min(n_queries)] {
        let _ = ch::query_with_path_into(&h, s, d, &mut scratch);
    }
    let t = Instant::now();
    let mut reused_finite = 0usize;
    for &(s, d) in &pairs {
        if ch::query_with_path_into(&h, s, d, &mut scratch).is_some() {
            reused_finite += 1;
        }
    }
    let reused_ns = t.elapsed().as_nanos() as f64;
    let reused_per = reused_ns / n_queries as f64;
    println!(
        "\n── warm sequential, REUSED SCRATCH ({n_queries} calls, 1 thread) ──\n  \
         total: {:.2} s   mean: {:.0} ns/call ({:.2} µs)   throughput: {:.1} k calls/s   \
         finite: {}/{}",
        reused_ns / 1e9,
        reused_per,
        reused_per / 1000.0,
        calls_per_s(reused_per) / 1000.0,
        reused_finite,
        n_queries
    );

    // ───── Parallel, alloc-per-call ─────
    let t = Instant::now();
    let par_finite: usize = pairs
        .par_iter()
        .map(|&(s, d)| {
            let v = ch::query(&h, s, d).unwrap_or(f32::INFINITY);
            if v.is_finite() { 1usize } else { 0 }
        })
        .sum();
    let par_ns = t.elapsed().as_nanos() as f64;
    let par_per = par_ns / n_queries as f64;
    println!(
        "\n── parallel, alloc-per-call ({n_queries} calls, {} threads) ──\n  \
         total: {:.2} s   effective: {:.0} ns/call ({:.2} µs)   throughput: {:.1} k calls/s",
        rayon::current_num_threads(),
        par_ns / 1e9,
        par_per,
        par_per / 1000.0,
        calls_per_s(par_per) / 1000.0,
    );
    let _ = par_finite;

    // ───── Parallel, REUSED SCRATCH per worker ─────
    let t = Instant::now();
    let par_reused_finite: usize = pairs
        .par_iter()
        .fold(
            || (PathScratch::new(n), 0usize),
            |(mut scratch, mut acc), &(s, d)| {
                if ch::query_with_path_into(&h, s, d, &mut scratch).is_some() {
                    acc += 1;
                }
                (scratch, acc)
            },
        )
        .map(|(_, a)| a)
        .sum();
    let par_r_ns = t.elapsed().as_nanos() as f64;
    let par_r_per = par_r_ns / n_queries as f64;
    println!(
        "\n── parallel, REUSED SCRATCH ({n_queries} calls, {} threads) ──\n  \
         total: {:.2} s   effective: {:.0} ns/call ({:.2} µs)   throughput: {:.1} k calls/s",
        rayon::current_num_threads(),
        par_r_ns / 1e9,
        par_r_per,
        par_r_per / 1000.0,
        calls_per_s(par_r_per) / 1000.0,
    );
    let _ = par_reused_finite;

    // ───── Path-unpacking with full path materialised ─────
    let path_n = 50_000.min(n_queries);
    let t = Instant::now();
    let mut path_finite = 0usize;
    let mut path_len_sum = 0usize;
    for &(s, d) in &pairs[..path_n] {
        if let Some(_dist) = ch::query_with_path_into(&h, s, d, &mut scratch) {
            path_finite += 1;
            path_len_sum += scratch.path.len();
        }
    }
    let path_ns = t.elapsed().as_nanos() as f64;
    let path_per = path_ns / path_n as f64;
    println!(
        "\n── path-unpacking ({path_n} calls, 1 thread, reused scratch) ──\n  \
         total: {:.2} s   mean: {:.0} ns/call ({:.2} µs)   throughput: {:.1} k calls/s   \
         finite: {}/{}   avg path nodes: {:.0}",
        path_ns / 1e9,
        path_per,
        path_per / 1000.0,
        calls_per_s(path_per) / 1000.0,
        path_finite,
        path_n,
        if path_finite > 0 {
            path_len_sum as f64 / path_finite as f64
        } else {
            0.0
        }
    );

    // ───── MMM single-pair: ch::matrix_with_dist(&[a], &[b]) ─────
    // Demonstrate that the MMM function is the wrong API for single pairs —
    // it does a bucket-allocation pass over n graph nodes per call. Reusing
    // scratch is not even possible (current API allocates internally each
    // call).
    let mmm_n = 5_000.min(n_queries);
    let t = Instant::now();
    let mut mmm_finite = 0usize;
    for &(s, d) in &pairs[..mmm_n] {
        let (dur, _dist) = ch::matrix_with_dist(&h, &[s], &[d]);
        if dur[0].is_finite() {
            mmm_finite += 1;
        }
    }
    let mmm_ns = t.elapsed().as_nanos() as f64;
    let mmm_per = mmm_ns / mmm_n as f64;
    println!(
        "\n── MMM single-pair, matrix_with_dist(&[a], &[b]) ({mmm_n} calls, 1 thread) ──\n  \
         total: {:.2} s   mean: {:.0} ns/call ({:.2} µs)   throughput: {:.1} k calls/s   \
         finite: {}/{}",
        mmm_ns / 1e9,
        mmm_per,
        mmm_per / 1000.0,
        calls_per_s(mmm_per) / 1000.0,
        mmm_finite,
        mmm_n
    );

    println!("\nsink: {sink:.3} (anti-DCE)");
    Ok(())
}
