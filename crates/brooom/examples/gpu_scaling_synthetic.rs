//! Synthetic-instance scaling test for the megakernel.
//!
//! Generates a CVRP instance directly (no JSON parsing) to push the GPU
//! megakernel to N=10K and beyond. Uses Euclidean distances scaled to i32,
//! wide TWs (effectively pure-distance CVRP), demand=1, capacity=20 → ~N/20
//! routes after a greedy split-by-K initial tour.
//!
//! When N ≥ 12000, switches to **coord-mode**: matrix is not materialized,
//! distances are computed on the fly in the shader from (x,y) coordinates.
//! This keeps N=50K feasible (would otherwise need 10 GB matrix).
//!
//! Run with:
//!   cargo run --release --example gpu_scaling_synthetic -- 10000
//!   cargo run --release --example gpu_scaling_synthetic -- 50000

use std::time::Instant;

use brooom::granular::Granular;
use brooom::gpu_population::GpuPopulation;
use brooom::matrix::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let granular_k: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    println!("=== Synthetic scaling test: N={n}, K={granular_k} ===\n");

    // 1. Generate Solomon-like clustered coordinates.
    let t0 = Instant::now();
    let n_loc = n + 1; // depot + N customers
    let mut coords: Vec<(f64, f64)> = Vec::with_capacity(n_loc);
    coords.push((500.0, 500.0)); // depot
    // Mix of clusters and random — gives interesting routing structure.
    let mut rng_state: u64 = 0xCAFEBABE;
    let mut next_rng = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };
    let n_clusters = (n / 50).max(10);
    let mut centers = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        centers.push((next_rng() * 1000.0, next_rng() * 1000.0));
    }
    for i in 0..n {
        // 70% cluster-around-a-center, 30% uniform.
        if next_rng() < 0.7 {
            let c = &centers[i % n_clusters];
            let r = 60.0;
            let x = c.0 + (next_rng() - 0.5) * r;
            let y = c.1 + (next_rng() - 0.5) * r;
            coords.push((x.clamp(0.0, 1000.0), y.clamp(0.0, 1000.0)));
        } else {
            coords.push((next_rng() * 1000.0, next_rng() * 1000.0));
        }
    }
    println!("Coords generated in {:?}", t0.elapsed());

    // 2. Build matrix only when needed. For N ≥ 12000, coord-mode bypasses
    // the matrix and computes distances on the fly. At N=50K, the i32
    // matrix would be 10 GB; coord-mode keeps it at 400 KB.
    let use_coord_mode = n_loc >= 12_000;
    let mem_mb = (n_loc * n_loc * 4) as f64 / (1024.0 * 1024.0);
    let matrix: Matrix;
    if use_coord_mode {
        println!(
            "Using coord-mode (matrix would be {:.1} MB — bypassing for N≥12K)",
            mem_mb
        );
        matrix = Matrix { n: n_loc, durations: Vec::new(), distances: None };
    } else {
        let t0 = Instant::now();
        println!("Building {}×{} matrix ({:.1} MB) ...", n_loc, n_loc, mem_mb);
        let mut durations = vec![0i32; n_loc * n_loc];
        for i in 0..n_loc {
            let (xi, yi) = coords[i];
            for j in 0..n_loc {
                let (xj, yj) = coords[j];
                let d = ((xi - xj).powi(2) + (yi - yj).powi(2)).sqrt();
                durations[i * n_loc + j] = (d * 100.0) as i32;
            }
        }
        println!("Matrix built in {:?}", t0.elapsed());
        matrix = Matrix { n: n_loc, durations, distances: None };
    }

    // 3. Build initial tours.
    let t0 = Instant::now();
    let cap_per_route = 20usize;
    let tours: Vec<Vec<u32>> = if use_coord_mode {
        // Angular sweep around the depot — O(N log N). For N≥12K nearest-
        // neighbour search would be O(N² × cap) which is prohibitive.
        build_initial_tours_sweep(&coords, cap_per_route)
    } else {
        build_initial_tours_nn(&matrix, n_loc, cap_per_route)
    };
    let n_routes = tours.len();
    println!(
        "Initial tour split ({}): {} routes built in {:?}",
        if use_coord_mode { "angular sweep" } else { "nearest-neighbour" },
        n_routes,
        t0.elapsed()
    );

    let initial_travel: i64 = tours.iter().map(|tour| {
        let mut s = 0i64;
        for w in tour.windows(2) {
            let a = w[0] as usize; let b = w[1] as usize;
            if use_coord_mode {
                let (ax, ay) = coords[a];
                let (bx, by) = coords[b];
                let d = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
                s += (d * 100.0) as i64;
            } else {
                s += matrix.durations[a * n_loc + b] as i64;
            }
        }
        s
    }).sum();
    println!("Initial total travel: {} ({}× scale)", initial_travel, 100);

    // 4. Build granular KNN — coord-based for large N (matrix-free).
    let t0 = Instant::now();
    let near_flat: Vec<u32>;
    let granular_k_eff: usize;
    if use_coord_mode {
        let (near, k_eff) = build_granular_from_coords(&coords, granular_k as usize);
        near_flat = near;
        granular_k_eff = k_eff;
    } else {
        let granular = Granular::build(&matrix, granular_k as usize);
        granular_k_eff = granular.k();
        let mut nf: Vec<u32> = Vec::with_capacity(n_loc * granular.k());
        for i in 0..n_loc {
            let mut found = 0;
            for nb in granular.neighbors(i) {
                nf.push(nb as u32);
                found += 1;
            }
            while found < granular.k() {
                nf.push(i as u32);
                found += 1;
            }
        }
        near_flat = nf;
    }
    println!("Granular K={} built in {:?}", granular_k_eff, t0.elapsed());

    // 5. Upload to GPU.
    let t0 = Instant::now();
    let max_route_len = tours.iter().map(|r| r.len() as u32).max().unwrap_or(0) + 8;
    let tour_capacity = n_routes as u32 * max_route_len;

    let gpu = GpuPopulation::new(
        if use_coord_mode { &[] } else { &matrix.durations[..] },
        n_loc as u32,
        1,
        n_routes as u32,
        tour_capacity,
    )?;
    gpu.upload(&[tours.clone()])?;

    let loc_service = vec![0i32; n_loc];
    let loc_demand = vec![1i32; n_loc]; // all customers demand 1 unit
    let loc_tw_s = vec![0i32; n_loc];   // wide TWs → effectively pure CVRP
    let loc_tw_e = vec![1_000_000i32; n_loc];
    gpu.upload_problem_data(&loc_service, &loc_demand, &loc_tw_s, &loc_tw_e)?;
    gpu.upload_vehicle_data(&[cap_per_route as i32], &[0i32], &[1_000_000i32])?;
    gpu.upload_granular(&near_flat, granular_k_eff as u32)?;
    if use_coord_mode {
        // Flatten coords to f32 pairs.
        let mut flat_xy: Vec<f32> = Vec::with_capacity(n_loc * 2);
        for &(x, y) in &coords {
            flat_xy.push(x as f32);
            flat_xy.push(y as f32);
        }
        gpu.upload_coords(&flat_xy)?;
    }
    println!("GPU upload: {:?}\n", t0.elapsed());

    // 6. Run chunked megakernel.
    // For very large N, per-iter cost in coord-mode is high enough that
    // even a few iters per dispatch hits Metal's GPU watchdog. Tune
    // chunk_iters down aggressively as N grows.
    println!("=== Megakernel LS-loop ===");
    let max_iter: u32 = std::env::var("MAX_ITER")
        .ok().and_then(|s| s.parse().ok())
        .unwrap_or(if n >= 25000 { 200 } else if n >= 10000 { 4000 } else { 2000 });
    let chunk: u32 = std::env::var("CHUNK_ITERS")
        .ok().and_then(|s| s.parse().ok())
        .unwrap_or(if n >= 25000 { 2 } else if n >= 10000 { 15 } else if n >= 5000 { 30 } else { 100 });
    println!("max_iter={max_iter}, chunk_iters={chunk}");

    let t0 = Instant::now();
    let (iters, applies, last_delta) = gpu.run_megakernel_2opt_chunked(0, max_iter, chunk)?;
    let t_ls = t0.elapsed();
    println!("Iterations:  {iters}");
    println!("Applies:     {applies}");
    println!("Final Δ:     {last_delta}");
    println!("Wallclock:   {:?}", t_ls);
    if iters > 0 {
        println!("Per iter:    {:?}", t_ls / iters);
    }

    // 7. Verify task preservation & compute final travel.
    let final_tours = gpu.read_back(0)?;
    let mut final_customers = 0usize;
    let mut final_travel: i64 = 0;
    for tour in &final_tours {
        if tour.len() >= 2 {
            final_customers += tour.len() - 2;
        }
        for w in tour.windows(2) {
            let a = w[0] as usize; let b = w[1] as usize;
            if use_coord_mode {
                let (ax, ay) = coords[a];
                let (bx, by) = coords[b];
                let d = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
                final_travel += (d * 100.0) as i64;
            } else {
                final_travel += matrix.durations[a * n_loc + b] as i64;
            }
        }
    }
    println!("\n=== Result ===");
    println!("Customers:   {} → {} ({})",
        n, final_customers,
        if final_customers == n { "✓ preserved" } else { "✗ LEAK" });
    println!("Travel:      {initial_travel} → {final_travel}");
    if initial_travel > 0 {
        println!("Improvement: {} ({:.2}%)",
            initial_travel - final_travel,
            100.0 * (initial_travel - final_travel) as f64 / initial_travel as f64);
    }

    Ok(())
}

/// Nearest-neighbour tour construction using matrix. O(N² × cap).
fn build_initial_tours_nn(matrix: &Matrix, n_loc: usize, cap: usize) -> Vec<Vec<u32>> {
    let n_routes = (n_loc - 1 + cap - 1) / cap;
    let mut tours: Vec<Vec<u32>> = vec![Vec::new(); n_routes];
    let mut assigned = vec![false; n_loc];
    assigned[0] = true;
    for r in 0..n_routes {
        tours[r].push(0);
        let mut current = 0usize;
        for _ in 0..cap {
            let mut best = usize::MAX;
            let mut best_d = i32::MAX;
            for j in 1..n_loc {
                if assigned[j] { continue; }
                let d = matrix.durations[current * n_loc + j];
                if d < best_d { best_d = d; best = j; }
            }
            if best == usize::MAX { break; }
            tours[r].push(best as u32);
            assigned[best] = true;
            current = best;
        }
        tours[r].push(0);
    }
    while let Some(t) = tours.last() {
        if t.len() <= 2 { tours.pop(); } else { break; }
    }
    tours
}

/// Angular sweep around depot — O(N log N). Used when matrix-free path is
/// needed (N≥12K). Quality is comparable to NN for our LS-baseline since
/// the megakernel re-optimizes anyway.
fn build_initial_tours_sweep(coords: &[(f64, f64)], cap: usize) -> Vec<Vec<u32>> {
    let depot = coords[0];
    let mut indexed: Vec<(usize, f64)> = (1..coords.len()).map(|i| {
        let (x, y) = coords[i];
        let angle = (y - depot.1).atan2(x - depot.0);
        (i, angle)
    }).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut tours: Vec<Vec<u32>> = Vec::new();
    let mut current: Vec<u32> = vec![0u32];
    for (i, _) in indexed {
        current.push(i as u32);
        if current.len() == cap + 1 {
            current.push(0);
            tours.push(std::mem::take(&mut current));
            current = vec![0u32];
        }
    }
    if current.len() > 1 {
        current.push(0);
        tours.push(current);
    }
    tours
}

/// Coord-based brute-force granular KNN. O(N² × log K) per build. For
/// N=50K, K=50: ~2.5 billion distance evaluations → ~20 s with parallel.
fn build_granular_from_coords(coords: &[(f64, f64)], k: usize) -> (Vec<u32>, usize) {
    use rayon::prelude::*;
    let n = coords.len();
    let k_eff = k.min(n - 1).max(1);
    let mut near = vec![0u32; n * k_eff];

    let rows: Vec<Vec<u32>> = (0..n).into_par_iter().map(|i| {
        let (xi, yi) = coords[i];
        let mut buf: Vec<(f64, u32)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if j == i { continue; }
            let (xj, yj) = coords[j];
            let d2 = (xi - xj).powi(2) + (yi - yj).powi(2);
            buf.push((d2, j as u32));
        }
        if buf.len() > k_eff {
            buf.select_nth_unstable_by(k_eff - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            buf.truncate(k_eff);
        }
        buf.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut row: Vec<u32> = buf.into_iter().map(|(_, j)| j).collect();
        // Pad with self if fewer than k.
        while row.len() < k_eff { row.push(i as u32); }
        row
    }).collect();

    for (i, row) in rows.iter().enumerate() {
        near[i * k_eff..(i + 1) * k_eff].copy_from_slice(row);
    }
    (near, k_eff)
}
