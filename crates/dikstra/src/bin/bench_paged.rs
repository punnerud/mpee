//! Memory-budgeted CH queries — three production-readiness pieces in one demo:
//!
//!   1. Parallel queries via `rayon` (PagedMmap is now `Sync`)
//!   2. Rank-ordered cache layout (`SSSPCH1B`): hot data clusters at low
//!      file offsets, so a tight LRU naturally keeps it resident
//!   3. Pinning the top-K-rank vertex bytes so they never enter the LRU
//!
//! Run with a dataset name: `bench_paged london`, `bench_paged england`,
//! or pass an absolute path to a `.ch` cache.

use std::sync::Arc;
use std::time::Instant;

use memmap2::Mmap;
use rayon::prelude::*;
use sssp_bench::cache_ch;
use sssp_bench::ch;
use sssp_bench::graph::Rng;
use sssp_bench::paged::{ChLayout, PagedMmap, TouchBuf, is_rank_ordered};

fn main() -> std::io::Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "london".to_string());
    let ch_path = match dataset.as_str() {
        "london" => "data/greater-london.osm.pbf.ch",
        "england" => "data/england.osm.pbf.ch",
        other => other,
    };

    println!("=== Memory-budgeted CH demonstration ({dataset}) ===");

    let ch = cache_ch::load_mmap(ch_path)?;
    let f = std::fs::File::open(ch_path)?;
    let mmap = Arc::new(unsafe { Mmap::map(&f)? });
    let layout = ChLayout::from_cache_file(&mmap)?;
    let cache_file_size = mmap.len();
    let rank_ordered = is_rank_ordered(&mmap);
    println!(
        "CH cache: {} MB on disk, n={}, m_aug={}, layout={}",
        cache_file_size / (1024 * 1024),
        layout.n,
        layout.m,
        if rank_ordered {
            "rank-ordered (SSSPCH1B)"
        } else {
            "legacy (SSSPCH1A) — rebuild for layout benefit"
        }
    );

    let n_queries = 500usize;
    let mut rng = Rng(20260509);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(n_queries);
    while pairs.len() < n_queries {
        let s = rng.range(layout.n as u32);
        let d = rng.range(layout.n as u32);
        if ch.graph_fwd.head[s as usize + 1] - ch.graph_fwd.head[s as usize] == 0 {
            continue;
        }
        if ch.graph_fwd.head[d as usize + 1] - ch.graph_fwd.head[d as usize] == 0 {
            continue;
        }
        pairs.push((s, d));
    }
    println!("Generated {} (src, dst) pairs.\n", n_queries);

    let n_threads = rayon::current_num_threads();

    // ---- Warmup: fault hot pages into OS cache ----
    let t = Instant::now();
    for &(s, d) in &pairs {
        let _ = ch::query(&ch, s, d);
    }
    let warmup_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{:<28}  {:>9.1} ms total  ({:.4} ms/query)  [cold-cache warmup]",
        "warmup", warmup_ms, warmup_ms / n_queries as f64
    );

    // ---- Baseline: serial, untracked, warm OS cache ----
    let t = Instant::now();
    let _: f64 = pairs.iter().map(|&(s, d)| ch::query(&ch, s, d).map(|v| v as f64).unwrap_or(0.0)).sum();
    let unlimited_serial = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{:<28}  {:>9.1} ms total  ({:.4} ms/query)  [warm baseline, serial]",
        "unlimited (serial)",
        unlimited_serial,
        unlimited_serial / n_queries as f64
    );

    // ---- Baseline parallel: untracked, warm OS cache, rayon ----
    let t = Instant::now();
    let _: f64 = pairs
        .par_iter()
        .map(|&(s, d)| ch::query(&ch, s, d).map(|v| v as f64).unwrap_or(0.0))
        .sum();
    let unlimited_par = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{:<28}  {:>9.1} ms total  ({:.4} ms/query)  [warm baseline, par {} threads]",
        "unlimited (parallel)",
        unlimited_par,
        unlimited_par / n_queries as f64,
        n_threads
    );

    println!();
    let budgets_mb: &[usize] = &[1000, 500, 200, 100, 50, 20];

    // ---- Tracked, serial (for comparison with previous results) ----
    println!("-- serial, no pinning --");
    for &budget_mb in budgets_mb {
        let pm = PagedMmap::new(mmap.clone(), budget_mb * 1024 * 1024);
        let mut buf = TouchBuf::new();
        let t = Instant::now();
        let _: f64 = pairs
            .iter()
            .map(|&(s, d)| {
                ch::query_paged_buf(&ch, &layout, &pm, &mut buf, s, d)
                    .map(|v| v as f64)
                    .unwrap_or(0.0)
            })
            .sum();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let stats = pm.stats();
        println!(
            "{:<28}  {:>9.1} ms total  ({:.4} ms/query)  resident={} MB  loaded={}  evicted={}",
            format!("budget {} MB", budget_mb),
            ms,
            ms / n_queries as f64,
            stats.warm_bytes / (1024 * 1024),
            stats.n_pages_loaded,
            stats.n_pages_evicted
        );
    }

    // ---- Tracked, parallel ----
    println!("\n-- parallel ({} threads), no pinning --", n_threads);
    for &budget_mb in budgets_mb {
        let pm = PagedMmap::new(mmap.clone(), budget_mb * 1024 * 1024);
        let t = Instant::now();
        let total: f64 = pairs
            .par_iter()
            .fold(
                || (TouchBuf::new(), 0.0_f64),
                |(mut buf, sum), &(s, d)| {
                    let v = ch::query_paged_buf(&ch, &layout, &pm, &mut buf, s, d)
                        .map(|x| x as f64)
                        .unwrap_or(0.0);
                    (buf, sum + v)
                },
            )
            .map(|(_, sum)| sum)
            .sum();
        let _ = total;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let stats = pm.stats();
        println!(
            "{:<28}  {:>9.1} ms total  ({:.4} ms/query)  resident={} MB  loaded={}  evicted={}",
            format!("budget {} MB", budget_mb),
            ms,
            ms / n_queries as f64,
            stats.warm_bytes / (1024 * 1024),
            stats.n_pages_loaded,
            stats.n_pages_evicted
        );
    }

    // ---- Tracked, parallel, with pinning of top-K-rank pages ----
    if rank_ordered {
        let top_k = (layout.n / 200).max(1024); // ~0.5% of vertices
        println!(
            "\n-- parallel ({} threads), pinning top {} ranks (~0.5%) --",
            n_threads, top_k
        );
        for &budget_mb in budgets_mb {
            let pm = PagedMmap::new(mmap.clone(), budget_mb * 1024 * 1024);
            pin_top_rank(&pm, &layout, &ch, top_k);
            let pin_stats = pm.stats();
            let t = Instant::now();
            let total: f64 = pairs
                .par_iter()
                .fold(
                    || (TouchBuf::new(), 0.0_f64),
                    |(mut buf, sum), &(s, d)| {
                        let v = ch::query_paged_buf(&ch, &layout, &pm, &mut buf, s, d)
                            .map(|x| x as f64)
                            .unwrap_or(0.0);
                        (buf, sum + v)
                    },
                )
                .map(|(_, sum)| sum)
                .sum();
            let _ = total;
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let stats = pm.stats();
            println!(
                "{:<28}  {:>9.1} ms total  ({:.4} ms/query)  pinned={} MB  resident={} MB  loaded={}  evicted={}",
                format!("budget {} MB + pin", budget_mb),
                ms,
                ms / n_queries as f64,
                pin_stats.pinned_bytes / (1024 * 1024),
                stats.warm_bytes / (1024 * 1024),
                stats.n_pages_loaded,
                stats.n_pages_evicted
            );
        }
    } else {
        println!("\n(skipping pin-test: cache is legacy SSSPCH1A — rebuild for rank-ordered layout)");
    }

    println!("\nNote: 'budget X MB' tells the LRU how much page residency to keep.");
    println!("When the budget is binding, cold pages are released via madvise(MADV_DONTNEED).");
    println!("On macOS, MADV_DONTNEED is advisory; under real RAM pressure it would be honored.");
    println!("Pinned pages are excluded from the budget and never evicted.");

    Ok(())
}

/// Pin the byte ranges that hold data for the top-`k` highest-rank vertices.
/// Assumes a rank-ordered cache (SSSPCH1B): vertex IDs 0..k are the topmost.
fn pin_top_rank(
    pm: &PagedMmap,
    layout: &ChLayout,
    ch: &ch::ContractionHierarchy,
    k: usize,
) {
    let k = k.min(layout.n);
    // CSR head arrays: pin entries 0..=k (one extra for the offset boundary).
    pm.pin_range(layout.head_fwd_off, (k + 1) * 4);
    pm.pin_range(layout.head_bwd_off, (k + 1) * 4);
    // up_count and rank arrays: pin entries 0..k.
    pm.pin_range(layout.up_count_fwd_off, k * 4);
    pm.pin_range(layout.up_count_bwd_off, k * 4);
    pm.pin_range(layout.rank_off, k * 4);
    // Edge arrays: pin contiguous prefix that holds edges of vertices 0..k.
    let m_top_fwd = ch.graph_fwd.head[k] as usize;
    let m_top_bwd = ch.graph_bwd.head[k] as usize;
    pm.pin_range(layout.edge_to_fwd_off, m_top_fwd * 4);
    pm.pin_range(layout.edge_w_fwd_off, m_top_fwd * 4);
    pm.pin_range(layout.via_fwd_off, m_top_fwd * 4);
    pm.pin_range(layout.edge_to_bwd_off, m_top_bwd * 4);
    pm.pin_range(layout.edge_w_bwd_off, m_top_bwd * 4);
    pm.pin_range(layout.via_bwd_off, m_top_bwd * 4);
}
