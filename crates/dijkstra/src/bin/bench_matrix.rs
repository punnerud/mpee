//! Pure-compute matrix scaling benchmark, no HTTP. Tests `ch::matrix_with_dist`
//! (in-memory) and `ch::matrix_with_dist_chunked` (bounded RAM) at
//! progressively larger sizes.
//!
//! Usage:
//!   bench_matrix [london|england] [profile] [size1,size2,...] [chunk] [order] [bin_path] [dtype]
//!
//! Defaults: sizes=2000,5000,10000,20000  chunk=0 (auto: chunked ≥30k)
//!           order=random  bin_path=  (empty: no binary output)  dtype=f32
//!
//! `order=ff` enables farthest-first traversal on the picked srcs, so the
//! streamed chunks emit rows in geometrically diverse order. The total
//! compute time is unchanged — only row order differs — so the reported
//! statistics must still match exactly between order=random and order=ff.
//!
//! `bin_path` writes Variant A row-streamed binary output to that path.
//! `dtype` is one of: f32 (default), u16_s0 (1-unit scale, ≤65 km/s cap),
//! u16_s1 (10-unit scale, ≤655 km/s cap), u32_s0.

use std::time::Instant;

use std::fs::File;
use std::io::BufWriter;

use sssp_bench::binary_table::{BinaryTableWriter, CellDtype, WriterConfig};
use sssp_bench::budget::{fmt_bytes, plan_for_budget_with_n_src, MatrixBudget};
use sssp_bench::cache_ch;
use sssp_bench::cache_pp;
use sssp_bench::ch;
use sssp_bench::farthest_first::farthest_first_order;
use sssp_bench::graph::Rng;

fn main() -> std::io::Result<()> {
    let dataset = std::env::args().nth(1).unwrap_or_else(|| "london".to_string());
    let profile_name = std::env::args().nth(2).unwrap_or_else(|| "car".to_string());
    let profile = sssp_bench::osm_profile::Profile::from_name(&profile_name)
        .expect("unknown profile");
    let sizes: Vec<usize> = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "2000,5000,10000,20000".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let chunk_arg: usize = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "0".to_string())
        .parse()
        .unwrap_or(0);
    let order_arg = std::env::args().nth(5).unwrap_or_else(|| "random".to_string());
    let use_ff = order_arg == "ff";
    let bin_path = std::env::args().nth(6).unwrap_or_default();
    let dtype_arg = std::env::args().nth(7).unwrap_or_else(|| "f32".to_string());
    let (bin_dtype, bin_scale_exp) = match dtype_arg.as_str() {
        "f32" => (CellDtype::F32, 0),
        "u16_s0" => (CellDtype::U16, 0),
        "u16_s1" => (CellDtype::U16, 1),
        "u16_sn1" => (CellDtype::U16, -1),
        "u32_s0" => (CellDtype::U32, 0),
        other => panic!("unknown dtype '{other}' — try f32 / u16_s0 / u16_s1 / u32_s0"),
    };
    let flags_arg = std::env::args().nth(8).unwrap_or_default();
    let mut bin_pad_64 = false;
    let mut bin_crc32 = false;
    for f in flags_arg.split(',') {
        match f.trim() {
            "pad64" => bin_pad_64 = true,
            "crc32" => bin_crc32 = true,
            "" => {}
            other => eprintln!("unknown flag '{other}' — try pad64 / crc32"),
        }
    }
    // 9th arg: hard memory cap in MB. 0 = no cap (use existing chunk_arg /
    // auto). When set, overrides `chunk` and runs the matrix inside a
    // sized rayon thread pool so the budget is actually enforced.
    let budget_mb: u64 = std::env::args()
        .nth(9)
        .unwrap_or_else(|| "0".to_string())
        .parse()
        .unwrap_or(0);

    let suffix = if profile == sssp_bench::osm_profile::Profile::Car {
        String::new()
    } else {
        format!(".{}", profile.name())
    };
    let (pp_path, ch_path) = match dataset.as_str() {
        "london" => (
            format!("data/greater-london.osm.pbf{suffix}.pp"),
            format!("data/greater-london.osm.pbf{suffix}.ch"),
        ),
        "england" => (
            format!("data/england.osm.pbf{suffix}.pp"),
            format!("data/england.osm.pbf{suffix}.ch"),
        ),
        other => {
            eprintln!("Unknown dataset '{other}'.");
            std::process::exit(1);
        }
    };

    println!("[bench_matrix] dataset={dataset} profile={}", profile.name());
    let pp = cache_pp::load_mmap(&pp_path).expect("pp-cache missing — run bench_pp first");
    let h = cache_ch::load_mmap(&ch_path).expect("ch-cache missing — run bench_ch first");
    let n = h.graph_fwd.n;
    println!(
        "Graph n={}, m_aug={}, threads={}",
        n,
        h.graph_fwd.m(),
        rayon::current_num_threads()
    );

    // Pre-pick a pool of nodes-with-edges so we can sample without retrying.
    let mut candidates: Vec<u32> = Vec::new();
    for u in 0..pp.graph.n {
        if pp.graph.head[u + 1] - pp.graph.head[u] > 0 {
            candidates.push(u as u32);
        }
    }
    println!("Candidate vertices (deg>=1): {}", candidates.len());

    for &k in &sizes {
        if k > candidates.len() {
            println!("[skip] {k} > candidate pool {}", candidates.len());
            continue;
        }
        // Random k srcs and k dsts (CH-internal IDs).
        let mut rng = Rng(20260509 ^ k as u64);
        let mut srcs_csr: Vec<u32> = Vec::with_capacity(k);
        let mut srcs = Vec::with_capacity(k);
        let mut dsts = Vec::with_capacity(k);
        for _ in 0..k {
            let s_csr = candidates[rng.range(candidates.len() as u32) as usize];
            let d_csr = candidates[rng.range(candidates.len() as u32) as usize];
            srcs_csr.push(s_csr);
            srcs.push(h.perm[s_csr as usize]);
            dsts.push(h.perm[d_csr as usize]);
        }

        // Optional farthest-first ordering on srcs (rows). Dsts stay in their
        // original order. Compute time is unchanged; downstream streaming
        // consumers receive geometrically diverse rows first.
        // `original_of_stream_pos[i]` maps the i-th streamed row to its
        // pre-reorder input index, so the binary writer can record the
        // caller's index in each row header.
        let original_of_stream_pos: Vec<u32> = if use_ff {
            let coords_for_srcs: Vec<(f32, f32)> =
                srcs_csr.iter().map(|&csr| pp.coords[csr as usize]).collect();
            let t_ff = Instant::now();
            let order = farthest_first_order(&coords_for_srcs);
            let ff_ms = t_ff.elapsed().as_secs_f64() * 1000.0;
            let reordered: Vec<u32> = order.iter().map(|&i| srcs[i as usize]).collect();
            srcs = reordered;
            println!("  [ff-order] {k} srcs reordered in {:.0} ms", ff_ms);
            order
        } else {
            (0..k as u32).collect()
        };

        // Pick mode:
        //   budget_mb > 0 → run the planner, pick (threads, chunk) for the cap
        //   chunk_arg > 0 → manual chunk (default thread pool)
        //   else → auto: in-memory below 30k, chunked at-or-above
        // For peak-RAM estimation: matrix_with_dist_chunked always allocates
        // f32 internally (8 bytes per cell, dual-channel). The binary
        // writer's on-disk dtype doesn't affect compute-time memory.
        let bytes_per_cell = 8;
        let _ = bin_dtype;
        let mut plan_threads: Option<usize> = None;
        let chunk = if budget_mb > 0 {
            let budget = MatrixBudget {
                max_bytes: budget_mb * 1024 * 1024,
                graph_n: h.graph_fwd.n as u32,
                bytes_per_output_cell: bytes_per_cell,
            };
            let plan = plan_for_budget_with_n_src(&budget, k as u32, k as u32);
            println!(
                "  [budget] cap={} MB → plan: threads={} chunk={} (est peak {}: \
                 thread={}, chunk={}, bucket={}, overhead={})",
                budget_mb,
                plan.n_threads,
                plan.chunk_size,
                fmt_bytes(plan.estimated_peak_bytes),
                fmt_bytes(plan.breakdown.thread_state),
                fmt_bytes(plan.breakdown.chunk_state),
                fmt_bytes(plan.breakdown.bucket_state),
                fmt_bytes(plan.breakdown.working_overhead),
            );
            if plan.estimated_peak_bytes > budget.max_bytes {
                eprintln!(
                    "  [budget] WARNING: planner could not fit; running smallest viable config"
                );
            }
            plan_threads = Some(plan.n_threads);
            plan.chunk_size
        } else if chunk_arg > 0 {
            chunk_arg
        } else if k >= 30000 {
            (k / 10).max(2000) // 10 batches by default
        } else {
            0
        };

        let t = Instant::now();
        let total_cells = (k * k) as f64;
        let (label, finite_dur, sum_dur, sum_dist) = if chunk == 0 {
            let (dur, dist) = ch::matrix_with_dist(&h, &srcs, &dsts);
            let mut finite_dur = 0usize;
            let mut sum_dur = 0.0_f64;
            let mut sum_dist = 0.0_f64;
            for i in 0..(k * k) {
                if dur[i].is_finite() {
                    finite_dur += 1;
                    sum_dur += dur[i] as f64;
                    sum_dist += dist[i] as f64;
                }
            }
            ("in-mem", finite_dur, sum_dur, sum_dist)
        } else {
            // Chunked: stream blocks via callback, accumulate stats; never
            // hold the full output. Peak RAM per batch = chunk × n_dst × 8 B.
            let mut finite_dur = 0usize;
            let mut sum_dur = 0.0_f64;
            let mut sum_dist = 0.0_f64;

            // Optional binary writer: open file, emit header, stream rows.
            let mut bin_writer: Option<BinaryTableWriter<BufWriter<File>>> = if bin_path.is_empty()
            {
                None
            } else {
                let f = File::create(&bin_path)?;
                let cfg = WriterConfig {
                    n_src: k as u32,
                    n_dst: k as u32,
                    dual_channel: true,
                    cell_dtype: bin_dtype,
                    scale_exp: bin_scale_exp,
                    pad_64: bin_pad_64,
                    crc32_footer: bin_crc32,
                };
                Some(BinaryTableWriter::new(BufWriter::with_capacity(1 << 20, f), cfg)?)
            };

            let run_chunked = |finite_dur: &mut usize,
                               sum_dur: &mut f64,
                               sum_dist: &mut f64,
                               bin_writer: &mut Option<BinaryTableWriter<BufWriter<File>>>| {
                ch::matrix_with_dist_chunked(
                    &h,
                    &srcs,
                    &dsts,
                    chunk,
                    |s_start, s_end, dur, dist| {
                        let n_dst = k;
                        for s_local in 0..(s_end - s_start) {
                            let s_global = s_start + s_local;
                            let row_off = s_local * n_dst;
                            let dur_row = &dur[row_off..row_off + n_dst];
                            let dist_row = &dist[row_off..row_off + n_dst];
                            for j in 0..n_dst {
                                if dur_row[j].is_finite() {
                                    *finite_dur += 1;
                                    *sum_dur += dur_row[j] as f64;
                                    *sum_dist += dist_row[j] as f64;
                                }
                            }
                            if let Some(w) = bin_writer.as_mut() {
                                let original_idx = original_of_stream_pos[s_global];
                                w.write_row(original_idx, dur_row, Some(dist_row))
                                    .expect("binary write");
                            }
                        }
                    },
                );
            };
            // If a thread-cap came out of the planner, run inside a sized
            // rayon pool — this is how we actually enforce the budget.
            if let Some(t) = plan_threads {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .expect("build rayon pool");
                pool.install(|| {
                    run_chunked(&mut finite_dur, &mut sum_dur, &mut sum_dist, &mut bin_writer);
                });
            } else {
                run_chunked(&mut finite_dur, &mut sum_dur, &mut sum_dist, &mut bin_writer);
            }
            if let Some(w) = bin_writer.take() {
                let crc = w.finish()?;
                println!("  [bin] wrote {} (crc={:08x})", bin_path, crc);
            }
            ("chunked", finite_dur, sum_dur, sum_dist)
        };
        let secs = t.elapsed().as_secs_f64();
        let avg_dur = sum_dur / finite_dur.max(1) as f64;
        let avg_dist = sum_dist / finite_dur.max(1) as f64;
        let mb = (k * k * 8) as f64 / 1024.0 / 1024.0;
        println!(
            "{k}x{k} [{label}, chunk={chunk}]: {:.2} s ({:.0} cells/s) — out {:.0} MB, finite {:.1}%, avg dur={:.0}s dist={:.0}m",
            secs,
            total_cells / secs,
            mb,
            100.0 * finite_dur as f64 / total_cells,
            avg_dur,
            avg_dist
        );
    }

    Ok(())
}
