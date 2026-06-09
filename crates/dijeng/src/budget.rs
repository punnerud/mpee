//! Memory budget planner for the chunked many-to-many matrix engine.
//!
//! Use case: the caller (a GUI app, a solver running alongside us, an iOS
//! container) has a hard cap on how much memory we may use. Given that cap
//! plus the CH/graph size and the matrix shape, pick the largest
//! `(n_threads, chunk_size)` combination whose estimated peak fits the
//! budget and maximises throughput.
//!
//! The peak-memory model has three parts:
//!
//! ```text
//!   peak = thread_state          // T × 2 × n_graph × 4 B  (Dijeng arrays)
//!        + chunk_state           // K × n_dst × cell_size  (per-batch output)
//!        + bucket_state          // K × ~24 KB             (per-src bucket entries)
//!        + working_overhead      // ~10 MB                 (heaps, touched, scratch)
//! ```
//!
//! The throughput heuristic prefers more threads (linear speedup up to
//! ~8 cores) and bigger chunks (sub-linear gain past ~5000 due to bucket
//! scan dominance). The planner walks a candidate ladder of thread counts
//! and picks the best fit.
//!
//! API:
//! ```ignore
//! let budget = MatrixBudget {
//!     max_bytes: 1 << 30,                // 1 GiB
//!     graph_n: ch.graph_fwd.n as u32,
//!     bytes_per_output_cell: 8,          // dual f32
//! };
//! let plan = plan_for_budget(&budget, dsts.len() as u32);
//! let pool = rayon::ThreadPoolBuilder::new()
//!     .num_threads(plan.n_threads).build()?;
//! pool.install(|| {
//!     ch::matrix_with_dist_chunked(&ch, &srcs, &dsts, plan.chunk_size, on_chunk);
//! });
//! ```

/// Bytes available to the matrix engine, plus the inputs needed to estimate
/// per-component cost.
#[derive(Debug, Clone, Copy)]
pub struct MatrixBudget {
    /// Hard cap (peak resident bytes ascribed to the matrix engine).
    pub max_bytes: u64,
    /// Number of graph nodes (`ch.graph_fwd.n`).
    pub graph_n: u32,
    /// Bytes per cell in the streamed output. For a dual-channel f32 layout
    /// that's `4 × 2 = 8`; dual-channel u16 is `2 × 2 = 4`; single-channel
    /// u16 is `2`.
    pub bytes_per_output_cell: usize,
}

/// Output of [`plan_for_budget`]: a `(n_threads, chunk_size)` pair plus the
/// model's estimate of the peak it will reach.
#[derive(Debug, Clone, Copy)]
pub struct AutoChunkPlan {
    pub n_threads: usize,
    pub chunk_size: usize,
    pub estimated_peak_bytes: u64,
    /// Component breakdown to make the choice debuggable.
    pub breakdown: PeakBreakdown,
}

#[derive(Debug, Clone, Copy)]
pub struct PeakBreakdown {
    pub thread_state: u64,
    pub chunk_state: u64,
    pub bucket_state: u64,
    pub working_overhead: u64,
}

impl PeakBreakdown {
    pub fn total(&self) -> u64 {
        self.thread_state + self.chunk_state + self.bucket_state + self.working_overhead
    }
}

/// Bytes the working set holds on top of the per-thread/per-chunk costs:
/// rayon work-stealing queues, allocator caching, transient double-allocation
/// between batches, per-thread fold accumulators, binary writer BufWriter,
/// and a safety margin. Empirically the matrix engine carries ~30 MB of slack
/// at all times on top of the modelled state.
const WORKING_OVERHEAD: u64 = 30 * 1024 * 1024;

/// Average bucket entries per src across a CH backward search — empirical
/// for road-network CHs (London-scale). Used only for budget estimation.
const AVG_BUCKET_ENTRIES_PER_SRC: u64 = 1500;
const BUCKET_ENTRY_BYTES: u64 = 16; // (u32 src_idx, f32 dur, f32 dist) padded

/// Multiplier applied to the per-batch chunk + bucket cost when checking
/// whether a plan fits the budget. Accounts for transient peaks during
/// allocation churn — the actual rayon `vec![INF; ...]` allocations may
/// briefly hold ~2× the steady-state. 1.6× hits the measured profile on
/// macOS / Apple Silicon.
const TRANSIENT_PEAK_MULTIPLIER: f64 = 1.6;

/// Bytes the per-thread Dijeng state costs for a graph of `n` nodes.
/// Dual channel keeps two `Vec<f32>` of length `n` plus small touched/heap
/// vectors (counted in `WORKING_OVERHEAD`).
#[inline]
pub fn per_thread_state(graph_n: u32) -> u64 {
    2 * (graph_n as u64) * 4
}

/// Throughput saturation point: chunks beyond this size don't go faster.
/// On cache-constrained chips (Apple Silicon, 24-48 MB L3 per cluster) the
/// bucket scan overflows L3 around chunk = 1500–3000 and performance
/// actively *declines*. 1500 is empirically the knee for 50k×50k on London
/// CH; the planner stops growing past it to leave RAM for the caller.
const CHUNK_THROUGHPUT_SAT: usize = 1500;

/// Candidate thread counts to consider. Capped at the rayon pool's current
/// size, since we cannot exceed it via `ThreadPoolBuilder::install`.
const THREAD_CANDIDATES: [usize; 7] = [16, 11, 8, 4, 2, 1, 1];

/// Pick the (n_threads, chunk_size) combination that maximises predicted
/// throughput while staying within `budget.max_bytes`. Returns a plan even
/// for impossibly tight budgets: the smallest viable config (1 thread,
/// chunk = 1) — caller can inspect `estimated_peak_bytes` and bail.
///
/// Picks chunk_size ≤ n_dst since chunks bigger than the source count just
/// run as a single batch with no extra parallelism benefit. Use
/// [`plan_for_budget_with_n_src`] to clamp against an n_src that differs
/// from n_dst.
pub fn plan_for_budget(budget: &MatrixBudget, n_dst: u32) -> AutoChunkPlan {
    plan_for_budget_with_n_src(budget, n_dst, u32::MAX)
}

/// Same as [`plan_for_budget`] but additionally caps `chunk_size` at
/// `n_src` (since a single-batch run is the largest useful chunk).
pub fn plan_for_budget_with_n_src(
    budget: &MatrixBudget,
    n_dst: u32,
    n_src: u32,
) -> AutoChunkPlan {
    let max_threads_available = rayon::current_num_threads();
    let mut best: Option<(AutoChunkPlan, f64)> = None;

    for &t in &THREAD_CANDIDATES {
        let t = t.min(max_threads_available);
        if t == 0 {
            continue;
        }
        let thread_state = (t as u64) * per_thread_state(budget.graph_n);
        let fixed = thread_state + WORKING_OVERHEAD;
        if fixed >= budget.max_bytes {
            continue;
        }
        let variable_budget = budget.max_bytes - fixed;
        // Apply transient-peak multiplier when sizing chunk: pretend each
        // src costs more, so the planner picks a smaller chunk and the
        // *measured* peak (with allocator churn) stays under cap.
        let per_src_var = (((budget.bytes_per_output_cell as u64) * (n_dst as u64)
            + AVG_BUCKET_ENTRIES_PER_SRC * BUCKET_ENTRY_BYTES)
            as f64
            * TRANSIENT_PEAK_MULTIPLIER) as u64;
        let chunk_by_budget = (variable_budget / per_src_var.max(1)).max(1) as usize;
        // Cap chunk at the throughput saturation point — past it we burn
        // memory for no speed gain (and on cache-constrained chips,
        // negative gain). Also cap at n_src (one batch already runs it all).
        let chunk = chunk_by_budget
            .min(n_src as usize)
            .min(CHUNK_THROUGHPUT_SAT)
            .max(1);

        // Reported chunk_state uses the *modelled* (no multiplier) cost,
        // so the user sees the steady-state breakdown.
        let chunk_state = (chunk as u64)
            * (budget.bytes_per_output_cell as u64)
            * (n_dst as u64);
        let bucket_state =
            (chunk as u64) * AVG_BUCKET_ENTRIES_PER_SRC * BUCKET_ENTRY_BYTES;

        let breakdown = PeakBreakdown {
            thread_state,
            chunk_state,
            bucket_state,
            working_overhead: WORKING_OVERHEAD,
        };

        // Throughput model: T × f(chunk), where f saturates around K=5000.
        // f(K) = 0.3 + 0.7 × min(K, 5000) / 5000 — matches measured curve.
        let chunk_eff = (chunk as f64).min(5000.0) / 5000.0;
        let throughput = (t as f64) * (0.3 + 0.7 * chunk_eff);

        let plan = AutoChunkPlan {
            n_threads: t,
            chunk_size: chunk,
            estimated_peak_bytes: breakdown.total(),
            breakdown,
        };

        match &best {
            None => best = Some((plan, throughput)),
            Some((_, best_tp)) if throughput > *best_tp => best = Some((plan, throughput)),
            _ => {}
        }
    }

    // Fallback: 1 thread, chunk=1. Even tighter is impossible.
    let fallback = {
        let thread_state = per_thread_state(budget.graph_n);
        let chunk_state = (budget.bytes_per_output_cell as u64) * (n_dst as u64);
        let bucket_state = AVG_BUCKET_ENTRIES_PER_SRC * BUCKET_ENTRY_BYTES;
        let breakdown = PeakBreakdown {
            thread_state,
            chunk_state,
            bucket_state,
            working_overhead: WORKING_OVERHEAD,
        };
        AutoChunkPlan {
            n_threads: 1,
            chunk_size: 1,
            estimated_peak_bytes: breakdown.total(),
            breakdown,
        }
    };
    best.map(|(p, _)| p).unwrap_or(fallback)
}

/// Default peak-RAM cap for the matrix compute engine (MB).
pub const DEFAULT_MATRIX_BUDGET_MB: u64 = 500;

/// CLI/default budget, overridable by env `MPEE_MATRIX_BUDGET_MB`.
pub fn resolve_matrix_budget_mb(default_mb: u64) -> u64 {
    std::env::var("MPEE_MATRIX_BUDGET_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_mb)
}

/// Helper: format a byte count as a human-readable string ("123 MB").
pub fn fmt_bytes(b: u64) -> String {
    let mb = b as f64 / 1024.0 / 1024.0;
    let gb = mb / 1024.0;
    if gb >= 1.0 {
        format!("{gb:.2} GB")
    } else {
        format!("{mb:.1} MB")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_tight_budget_picks_single_thread_small_chunk() {
        // 60 MB on London (n=1.16M) — barely fits 1-2 threads.
        let budget = MatrixBudget {
            max_bytes: 60 * 1024 * 1024,
            graph_n: 1_160_000,
            bytes_per_output_cell: 8,
        };
        let plan = plan_for_budget(&budget, 50_000);
        assert!(
            plan.estimated_peak_bytes <= budget.max_bytes,
            "exceeded budget: peak={}, max={}",
            plan.estimated_peak_bytes,
            budget.max_bytes
        );
        assert!(plan.n_threads <= 4);
        assert!(plan.chunk_size >= 1);
    }

    #[test]
    fn plan_generous_budget_picks_more_threads_bigger_chunk() {
        // 4 GB — comfortable.
        let budget = MatrixBudget {
            max_bytes: 4 * 1024 * 1024 * 1024,
            graph_n: 1_160_000,
            bytes_per_output_cell: 8,
        };
        let plan = plan_for_budget(&budget, 50_000);
        assert!(plan.estimated_peak_bytes <= budget.max_bytes);
        assert!(plan.n_threads >= 4);
        assert!(plan.chunk_size >= 1000);
    }

    #[test]
    fn plan_scales_with_dst_count() {
        // Smaller n_dst → larger chunk fits the same budget.
        let budget = MatrixBudget {
            max_bytes: 500 * 1024 * 1024,
            graph_n: 1_160_000,
            bytes_per_output_cell: 8,
        };
        let small = plan_for_budget(&budget, 5_000);
        let big = plan_for_budget(&budget, 50_000);
        assert!(
            small.chunk_size > big.chunk_size,
            "smaller n_dst should give a bigger chunk; got small.chunk={}, big.chunk={}",
            small.chunk_size,
            big.chunk_size
        );
    }

    #[test]
    fn plan_handles_impossibly_tight_budget() {
        // 1 MB on London — fits nothing.
        let budget = MatrixBudget {
            max_bytes: 1024 * 1024,
            graph_n: 1_160_000,
            bytes_per_output_cell: 8,
        };
        let plan = plan_for_budget(&budget, 50_000);
        // We return the smallest viable plan even if it exceeds budget.
        assert_eq!(plan.n_threads, 1);
        assert_eq!(plan.chunk_size, 1);
    }
}
