//! Phase-1 verification: persistent multi-trajectory tour state on GPU.
//!
//! 1. Build a synthetic distance matrix and a population of `pop_size`
//!    random trajectories (each = several routes of varying length).
//! 2. Upload to `GpuPopulation`.
//! 3. Read back trajectory-by-trajectory and verify byte-identical match.
//! 4. Dispatch the proof-of-life distance-sum kernel and compare its
//!    per-route output to a CPU reference.
//!
//! Run with:
//!   cargo run --release --example gpu_population_roundtrip

use std::time::Instant;

use brooom::gpu_population::{GpuPopulation, TrajectoryTours};

fn random_matrix(n: usize, seed: u64) -> Vec<i32> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut m = vec![0i32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                m[i * n + j] = ((next() % 100) + 1) as i32;
            }
        }
    }
    m
}

fn random_population(
    n_loc: u32,
    pop_size: usize,
    routes_per_traj: usize,
    avg_route_len: u32,
    seed: u64,
) -> Vec<TrajectoryTours> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    let mut pop = Vec::with_capacity(pop_size);
    for _ in 0..pop_size {
        let mut traj = Vec::with_capacity(routes_per_traj);
        for _ in 0..routes_per_traj {
            let jitter = (next() % 7) as i32 - 3; // ±3
            let len = ((avg_route_len as i32) + jitter).max(2) as u32;
            let route: Vec<u32> = (0..len).map(|_| next() % n_loc).collect();
            traj.push(route);
        }
        pop.push(traj);
    }
    pop
}

fn cpu_route_distance(route: &[u32], matrix: &[i32], md: usize) -> i32 {
    let mut s: i32 = 0;
    for k in 0..route.len().saturating_sub(1) {
        let a = route[k] as usize;
        let b = route[k + 1] as usize;
        s += matrix[a * md + b];
    }
    s
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_loc = 200usize;
    let pop_size = 8usize;
    let routes_per_traj = 12usize;
    let avg_route_len = 15u32;
    let max_routes = 16u32;
    let tour_capacity = 400u32;

    println!(
        "Phase-1 round-trip: n_loc={n_loc}, pop_size={pop_size}, \
         routes_per_traj={routes_per_traj}, avg_len={avg_route_len}\n"
    );

    let matrix = random_matrix(n_loc, 42);
    let pop = random_population(n_loc as u32, pop_size, routes_per_traj, avg_route_len, 7);

    println!("Init GPU + uploading matrix...");
    let t0 = Instant::now();
    let gpu = GpuPopulation::new(
        &matrix,
        n_loc as u32,
        pop_size as u32,
        max_routes,
        tour_capacity,
    )?;
    println!("  GPU ready in {:?}\n", t0.elapsed());

    // 1. Upload + read-back round-trip.
    let t0 = Instant::now();
    gpu.upload(&pop)?;
    println!("Upload {pop_size} trajectories: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let back = gpu.read_back_all()?;
    println!("Read back all {pop_size} trajectories: {:?}\n", t0.elapsed());

    // Byte-identical check.
    let mut mismatches = 0;
    for t in 0..pop_size {
        if pop[t].len() != back[t].len() {
            println!("  traj {t}: route count {} != {}", pop[t].len(), back[t].len());
            mismatches += 1;
            continue;
        }
        for r in 0..pop[t].len() {
            if pop[t][r] != back[t][r] {
                println!(
                    "  traj {t} route {r}: lengths {} vs {} (or content differs)",
                    pop[t][r].len(),
                    back[t][r].len()
                );
                mismatches += 1;
            }
        }
    }
    if mismatches == 0 {
        println!("  ✓ all trajectories survived the round-trip byte-identical");
    } else {
        println!("  ✗ {mismatches} mismatches");
        return Err("round-trip mismatch".into());
    }

    // 2. Proof-of-life distance kernel.
    let t0 = Instant::now();
    let gpu_dist = gpu.route_distances()?;
    let dist_us = t0.elapsed().as_secs_f64() * 1e6;
    println!("\nGPU route_distances dispatch + readback: {:.1} µs", dist_us);

    let mut kernel_mismatches = 0;
    let mut compared = 0;
    for t in 0..pop_size {
        for r in 0..pop[t].len() {
            let cpu_d = cpu_route_distance(&pop[t][r], &matrix, n_loc);
            let gpu_d = gpu_dist[t * max_routes as usize + r];
            if cpu_d != gpu_d {
                if kernel_mismatches < 3 {
                    println!("  ✗ traj {t} route {r}: cpu={cpu_d} gpu={gpu_d}");
                }
                kernel_mismatches += 1;
            }
            compared += 1;
        }
    }
    if kernel_mismatches == 0 {
        println!("  ✓ proof-of-life kernel matches CPU on {compared} routes");
    } else {
        println!("  ✗ {kernel_mismatches} kernel mismatches");
        return Err("kernel mismatch".into());
    }

    // 3. Stress: re-upload a different population and confirm read-back follows.
    let pop2 = random_population(n_loc as u32, pop_size, routes_per_traj, avg_route_len, 99);
    gpu.upload(&pop2)?;
    let back2 = gpu.read_back_all()?;
    let mut m = 0;
    for t in 0..pop_size {
        if pop2[t] != back2[t] {
            m += 1;
        }
    }
    if m == 0 {
        println!("\n  ✓ re-upload + read-back works (state replaced cleanly)");
    } else {
        println!("\n  ✗ re-upload mismatch: {m} trajectories");
        return Err("re-upload mismatch".into());
    }

    println!("\nPhase 1 verified: persistent state survives upload + readback,");
    println!("kernel can read it correctly, replacement uploads work.");
    Ok(())
}
