//! Verify and benchmark the GPU 2-opt sweep against a CPU reference.
//!
//! Run:
//!   cargo run --release --example gpu_sweep_bench

use std::time::Instant;

use brooom::gpu_sweep::GpuSweep;

/// Reference CPU implementation: same formula, sequential.
fn cpu_2opt_deltas(tour: &[u32], matrix: &[i32], matrix_dim: usize) -> Vec<i32> {
    let n = tour.len();
    let mut out = vec![0i32; n * n];
    for i in 0..n.saturating_sub(2) {
        for j in (i + 2)..n {
            let a = tour[i] as usize;
            let b = tour[i + 1] as usize;
            let c = tour[j] as usize;
            let d = tour[(j + 1) % n] as usize;
            let old = matrix[a * matrix_dim + b] + matrix[c * matrix_dim + d];
            let new_ = matrix[a * matrix_dim + c] + matrix[b * matrix_dim + d];
            out[i * n + j] = new_ - old;
        }
    }
    out
}

fn random_problem(n: usize, seed: u64) -> (Vec<i32>, Vec<u32>) {
    // Deterministic LCG RNG so we don't pull rand for an example.
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut matrix = vec![0i32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                matrix[i * n + j] = ((next() % 100) + 1) as i32;
            }
        }
    }
    let tour: Vec<u32> = (0..n as u32).collect();
    (matrix, tour)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initialising GPU device...");
    let init_t = Instant::now();
    // Tiny matrix to bind during init; we'll re-instantiate per N below
    // since the matrix is uploaded once per GpuSweep instance.
    let (warm_matrix, _) = random_problem(16, 0);
    let _warmup = GpuSweep::new(&warm_matrix, 16)?;
    println!("  device ready in {:?}\n", init_t.elapsed());

    // Sizes to bench. Below ~500 the CPU sweep wins (kernel overhead).
    for n in [100, 500, 1000, 2000].iter().copied() {
        println!("=== N = {n} ===");
        let (matrix, tour) = random_problem(n, 42);
        let gpu = GpuSweep::new(&matrix, n as u32)?;

        // Warm-up GPU once to amortise pipeline caching.
        let _warm = gpu.eval_2opt(&tour)?;

        // Correctness check vs CPU reference.
        let cpu = cpu_2opt_deltas(&tour, &matrix, n);
        let gpu_out = gpu.eval_2opt(&tour)?;
        let mut diffs = 0usize;
        for k in 0..(n * n) {
            if cpu[k] != gpu_out[k] {
                diffs += 1;
                if diffs <= 3 {
                    let i = k / n;
                    let j = k % n;
                    println!("  mismatch i={i} j={j}: cpu={} gpu={}", cpu[k], gpu_out[k]);
                }
            }
        }
        if diffs == 0 {
            println!("  ✓ correctness verified ({} pairs)", n * n);
        } else {
            println!("  ✗ {diffs} mismatches");
        }

        // Timing: average over 50 runs (after warm-up).
        let runs = if n >= 1000 { 20 } else { 100 };
        let t0 = Instant::now();
        for _ in 0..runs {
            let _ = gpu.eval_2opt(&tour)?;
        }
        let gpu_us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;

        let t0 = Instant::now();
        for _ in 0..runs {
            let _ = cpu_2opt_deltas(&tour, &matrix, n);
        }
        let cpu_us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;

        let speedup = cpu_us / gpu_us;
        println!(
            "  CPU all-deltas:  {:>9.1} µs    GPU all-deltas:  {:>9.1} µs    speedup: {:>5.2}×",
            cpu_us, gpu_us, speedup
        );

        // Now: argmin variant — GPU returns just the best (delta, i, j).
        // Warm up + correctness vs CPU argmin.
        let _ = gpu.best_2opt(&tour)?;
        let gpu_best = gpu.best_2opt(&tour)?;
        // CPU argmin on the same data.
        let cpu_deltas = cpu_2opt_deltas(&tour, &matrix, n);
        let mut cpu_best_delta = i32::MAX;
        let mut cpu_best_i = 0u32;
        let mut cpu_best_j = 0u32;
        for i in 0..n {
            for j in 0..n {
                let d = cpu_deltas[i * n + j];
                if d != 0 && d < cpu_best_delta {
                    cpu_best_delta = d;
                    cpu_best_i = i as u32;
                    cpu_best_j = j as u32;
                }
            }
        }
        if gpu_best.delta == cpu_best_delta {
            println!("  ✓ argmin agrees on delta = {}", gpu_best.delta);
            if (gpu_best.i, gpu_best.j) != (cpu_best_i, cpu_best_j) {
                println!("    note: tie on delta — CPU picked ({},{}), GPU picked ({},{})",
                    cpu_best_i, cpu_best_j, gpu_best.i, gpu_best.j);
            }
        } else {
            println!("  ✗ argmin mismatch: CPU={} GPU={}", cpu_best_delta, gpu_best.delta);
        }

        let t0 = Instant::now();
        for _ in 0..runs {
            let _ = gpu.best_2opt(&tour)?;
        }
        let gpu_argmin_us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;
        let speedup_argmin = cpu_us / gpu_argmin_us;
        println!(
            "  CPU all-deltas:  {:>9.1} µs    GPU argmin:      {:>9.1} µs    speedup: {:>5.2}×",
            cpu_us, gpu_argmin_us, speedup_argmin
        );

        println!();
    }

    Ok(())
}
