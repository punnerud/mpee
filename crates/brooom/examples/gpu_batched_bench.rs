//! Bench: batched per-route 2-opt on GPU vs sequential CPU.
//!
//! Simulates the polish-pass scenario: many routes (say 67 for N=1000)
//! each ~10–20 stops. We need best 2-opt for ALL of them. Done sequentially
//! on CPU it's 67 × O(L²). Done batched on GPU it's one dispatch.

use std::time::Instant;

use brooom::gpu_sweep::{GpuSweep, BestMove};

fn random_problem(n_locs: usize, seed: u64) -> Vec<i32> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut matrix = vec![0i32; n_locs * n_locs];
    for i in 0..n_locs {
        for j in 0..n_locs {
            if i != j {
                matrix[i * n_locs + j] = ((next() % 100) + 1) as i32;
            }
        }
    }
    matrix
}

fn random_routes(n_locs: u32, n_routes: usize, len: u32, seed: u64) -> Vec<Vec<u32>> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut routes = Vec::with_capacity(n_routes);
    for _ in 0..n_routes {
        let mut r = Vec::with_capacity(len as usize);
        for _ in 0..len {
            r.push(next() % n_locs);
        }
        routes.push(r);
    }
    routes
}

fn cpu_best_2opt(tour: &[u32], matrix: &[i32], md: usize) -> BestMove {
    let len = tour.len();
    let mut best = BestMove { delta: i32::MAX, i: 0, j: 0 };
    for i in 0..len.saturating_sub(2) {
        for j in (i + 2)..len {
            let a = tour[i] as usize;
            let b = tour[i + 1] as usize;
            let c = tour[j] as usize;
            let d = tour[(j + 1) % len] as usize;
            let old_cost = matrix[a * md + b] + matrix[c * md + d];
            let new_cost = matrix[a * md + c] + matrix[b * md + d];
            let delta = new_cost - old_cost;
            if delta < best.delta {
                best = BestMove { delta, i: i as u32, j: j as u32 };
            }
        }
    }
    if best.delta == i32::MAX {
        best = BestMove { delta: 0, i: 0, j: 0 };
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Building random scenarios for batched 2-opt benchmarking.\n");

    let n_locs = 1000usize;
    let matrix = random_problem(n_locs, 42);

    println!("Initialising GPU + uploading matrix...");
    let t0 = Instant::now();
    let gpu = GpuSweep::new(&matrix, n_locs as u32)?;
    println!("  GPU ready in {:?}\n", t0.elapsed());

    // Test scenarios: N=1000 with ~67 routes of ~15 stops each (typical brooom output)
    // and a stress-test with longer routes.
    for &(n_routes, route_len, label) in &[
        (67usize, 15u32, "typical-brooom-N=1000"),
        (10, 100, "fewer-but-longer"),
        (200, 25, "many-medium"),
        (50, 50, "balanced"),
    ] {
        println!("=== {label}: {n_routes} routes × {route_len} stops ===");
        let routes = random_routes(n_locs as u32, n_routes, route_len, 1);

        // GPU correctness vs CPU on first 5 routes.
        let _ = gpu.batched_best_2opt(&routes)?; // warm up
        let gpu_results = gpu.batched_best_2opt(&routes)?;
        let mut mismatches = 0;
        for r_idx in 0..n_routes.min(5) {
            let cpu_best = cpu_best_2opt(&routes[r_idx], &matrix, n_locs);
            if cpu_best.delta != gpu_results[r_idx].delta {
                mismatches += 1;
                println!(
                    "  route {r_idx}: cpu_delta={} gpu_delta={} (mismatch)",
                    cpu_best.delta, gpu_results[r_idx].delta
                );
            }
        }
        if mismatches == 0 {
            println!("  ✓ correctness verified on first 5 routes");
        }

        // Bench: GPU one dispatch.
        let runs = 100;
        let t0 = Instant::now();
        for _ in 0..runs {
            let _ = gpu.batched_best_2opt(&routes)?;
        }
        let gpu_us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;

        // Bench: CPU sequential (one route at a time).
        let t0 = Instant::now();
        for _ in 0..runs {
            for r in &routes {
                let _ = cpu_best_2opt(r, &matrix, n_locs);
            }
        }
        let cpu_seq_us = t0.elapsed().as_secs_f64() * 1e6 / runs as f64;

        let speedup = cpu_seq_us / gpu_us;
        println!(
            "  CPU seq: {:>8.1} µs   GPU batch: {:>8.1} µs   speedup: {:>5.2}×\n",
            cpu_seq_us, gpu_us, speedup
        );
    }

    Ok(())
}
