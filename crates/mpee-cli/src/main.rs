//! mpe — unified driver for the mpee workspace.
//!
//! Subcommands:
//!   gen       — random Vroom-compatible problem inside a region's bbox
//!   download  — fetch an OSM PBF from Geofabrik
//!   build     — preprocess an OSM PBF → CSR + PP + CH caches (delegates
//!               to the standalone bench_pp / bench_ch binaries)
//!   solve     — load CH cache, snap customer coords, build the N×N
//!               routing matrix via sssp_bench's bucket-MMM, hand it
//!               directly into brooom — no IPC, no disk
//!   pipeline  — gen + solve in one shot
//!
//! The integration is in-process: the matrix that sssp_bench produces is
//! a `Vec<f32>` that lives in this process; it is converted once to the
//! `Vec<i32>` brooom wants and handed in by `&Matrix`. No serialisation
//! after that.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "mpe", version, about = "mpee: routing + VRP in one process.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Generate a random Vroom-compatible VRP problem inside a region's bbox.
    Gen {
        /// Region name (currently supports: london, oslo, manhattan, paris).
        #[arg(long, default_value = "london")]
        region: String,

        /// Number of jobs (customers).
        #[arg(long, default_value_t = 500)]
        n_jobs: usize,

        /// Number of vehicles.
        #[arg(long, default_value_t = 20)]
        n_vehicles: usize,

        /// Capacity per vehicle (single-dimensional).
        #[arg(long, default_value_t = 100)]
        capacity: i64,

        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Output JSON path.
        #[arg(short = 'o', long)]
        output: PathBuf,
    },

    /// Download an OSM PBF extract from Geofabrik into ./data/.
    Download {
        /// Geofabrik path slug.
        region: String,
        #[arg(long, default_value = "data")]
        out_dir: PathBuf,
    },

    /// Preprocess a PBF into CSR + PP + CH caches (in-process).
    Build {
        pbf: PathBuf,
        #[arg(long, default_value = "car")]
        profile: String,
        /// Suppress the engine's parse/CH progress output.
        #[arg(long, short = 'q')]
        quiet: bool,
        /// Rebuild even if a cache already exists (default: reuse it).
        #[arg(long)]
        force: bool,
    },

    /// Solve a Vroom-compatible problem in-process against a CH cache.
    Solve {
        /// Vroom-style problem JSON (jobs + vehicles + lat/lon coordinates).
        problem: PathBuf,

        /// CH cache file (e.g. data/greater-london.osm.pbf.ch).
        #[arg(long)]
        ch: PathBuf,

        /// PP cache file (e.g. data/greater-london.osm.pbf.pp).
        /// Required for the coords used to snap lat/lon to graph vertices.
        #[arg(long)]
        pp: PathBuf,

        /// Wall-time budget in seconds for the solver. None = no limit.
        #[arg(long)]
        time_limit_s: Option<f64>,

        /// Parallel multi-start count. Default 8; smaller = faster.
        #[arg(long, default_value_t = 8)]
        multi_start: usize,

        /// Output JSON path (omit for stdout).
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
    },

    /// End-to-end: gen + solve.
    Pipeline {
        #[arg(long, default_value = "london")]
        region: String,
        #[arg(long, default_value_t = 500)]
        n_jobs: usize,
        #[arg(long, default_value_t = 20)]
        n_vehicles: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 100)]
        capacity: i64,
        #[arg(long)]
        ch: PathBuf,
        #[arg(long)]
        pp: PathBuf,
        #[arg(long)]
        time_limit_s: Option<f64>,
        #[arg(long, default_value_t = 8)]
        multi_start: usize,
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Gen { region, n_jobs, n_vehicles, capacity, seed, output } => {
            cmd_gen(&region, n_jobs, n_vehicles, capacity, seed, &output)
        }
        Cmd::Download { region, out_dir } => cmd_download(&region, &out_dir),
        Cmd::Build { pbf, profile, quiet, force } => cmd_build(&pbf, &profile, !quiet, force),
        Cmd::Solve { problem, ch, pp, time_limit_s, multi_start, output } => {
            cmd_solve(&problem, &ch, &pp, time_limit_s, multi_start, output.as_deref())
        }
        Cmd::Pipeline {
            region, n_jobs, n_vehicles, seed, capacity, ch, pp,
            time_limit_s, multi_start, output,
        } => {
            let tmp = std::env::temp_dir().join(format!("mpee-pipeline-{}.json", seed));
            cmd_gen(&region, n_jobs, n_vehicles, capacity, seed, &tmp)?;
            cmd_solve(&tmp, &ch, &pp, time_limit_s, multi_start, output.as_deref())
        }
    }
}

// -------------------------------------------------------------------------
// Region bounding boxes (lat_min, lat_max, lon_min, lon_max). These are
// rough rectangles; the snap layer will project points to the nearest road
// vertex, so points dropped over rivers or parks just snap to the nearest
// adjacent road.
// -------------------------------------------------------------------------

fn region_bbox(region: &str) -> Result<(f64, f64, f64, f64)> {
    // Tighter, road-dense rectangles. The looser Greater-London / Île-de-France
    // / Oslo-bygrense bboxes include water, parks, and isolated road
    // fragments where snap → car-routable node sometimes fails. For a clean
    // VRP demo we stick to the dense urban core. Callers wanting suburban
    // coverage should pass coords directly via the JSON instead of `gen`.
    Ok(match region {
        // Central London: Westminster + City + Camden + Islington + Lambeth.
        "london"    => (51.46, 51.56, -0.22,  0.02),
        // Inner Oslo.
        "oslo"      => (59.88, 59.95, 10.68, 10.82),
        // Manhattan from 23rd to 110th.
        "manhattan" => (40.74, 40.80, -74.00, -73.94),
        // Paris intra-muros.
        "paris"     => (48.82, 48.89,  2.27,  2.41),
        other => bail!("unknown region '{other}' (try: london, oslo, manhattan, paris)"),
    })
}

fn region_depot(region: &str) -> (f64, f64) {
    // (lat, lon) of a sensible city-centre depot.
    match region {
        "london"    => (51.5074, -0.1278),  // Charing Cross
        "oslo"      => (59.9139, 10.7522),  // Oslo Sentralstasjon
        "manhattan" => (40.7580, -73.9855), // Times Square
        "paris"     => (48.8566,  2.3522),  // Île de la Cité
        _ => (0.0, 0.0),
    }
}

// -------------------------------------------------------------------------
// gen: random problem inside a region's bbox.
// -------------------------------------------------------------------------

fn cmd_gen(
    region: &str,
    n_jobs: usize,
    n_vehicles: usize,
    capacity: i64,
    seed: u64,
    output: &Path,
) -> Result<()> {
    let (lat_min, lat_max, lon_min, lon_max) = region_bbox(region)?;
    let (depot_lat, depot_lon) = region_depot(region);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut jobs = Vec::with_capacity(n_jobs);
    for i in 0..n_jobs {
        let lat = rng.gen_range(lat_min..lat_max);
        let lon = rng.gen_range(lon_min..lon_max);
        let delivery = rng.gen_range(1..=10_i64);
        jobs.push(serde_json::json!({
            "id": (i + 1) as u64,
            "location": [lon, lat],
            "service": 60,
            "delivery": [delivery],
        }));
    }

    let mut vehicles = Vec::with_capacity(n_vehicles);
    for v in 0..n_vehicles {
        vehicles.push(serde_json::json!({
            "id": (v + 1) as u64,
            "start": [depot_lon, depot_lat],
            "end":   [depot_lon, depot_lat],
            "capacity": [capacity],
            "profile": "car",
        }));
    }

    let problem = serde_json::json!({
        "description": format!("mpe gen {region} n_jobs={n_jobs} n_vehicles={n_vehicles} seed={seed}"),
        "jobs": jobs,
        "vehicles": vehicles,
    });

    let bytes = serde_json::to_vec_pretty(&problem).context("serialise problem")?;
    std::fs::write(output, &bytes).with_context(|| format!("write {}", output.display()))?;
    eprintln!(
        "wrote {} ({} jobs, {} vehicles, {} bytes)",
        output.display(),
        n_jobs,
        n_vehicles,
        bytes.len()
    );
    Ok(())
}

// -------------------------------------------------------------------------
// download
// -------------------------------------------------------------------------

fn cmd_download(region: &str, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let url = format!("https://download.geofabrik.de/{}-latest.osm.pbf", region);
    let file_name = region.rsplit('/').next().unwrap_or("region");
    let out_path = out_dir.join(format!("{file_name}-latest.osm.pbf"));
    if out_path.exists() {
        eprintln!("already present: {}", out_path.display());
        return Ok(());
    }
    eprintln!("GET {url}");
    let resp = ureq::get(&url).call().context("Geofabrik request failed")?;
    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(&out_path)
        .with_context(|| format!("create {}", out_path.display()))?;
    let bytes = std::io::copy(&mut reader, &mut file).context("download copy")?;
    eprintln!("wrote {} ({} MB)", out_path.display(), bytes / 1_048_576);
    Ok(())
}

// -------------------------------------------------------------------------
// build: delegates to bench_pp / bench_ch — they're already feature-complete
// and parallelised. The unified CLI keeps them visible as one command.
// -------------------------------------------------------------------------

fn cmd_build(pbf: &Path, profile: &str, progress: bool, force: bool) -> Result<()> {
    // Build the cache IN-PROCESS via the shared sssp_bench helper — no
    // `cargo run` subprocess, no recompilation, and works as a distributed
    // binary (cargo/source not required). Same pipeline as Python's
    // Router.build, so the outputs (and their names) are identical.
    let prof = sssp_bench::osm_profile::Profile::from_name(profile)
        .ok_or_else(|| anyhow::anyhow!("unknown profile {profile:?} — use car/bicycle/foot"))?;
    eprintln!(
        "building cache from {} (profile={profile}) — seconds for a city, minutes for a country…",
        pbf.display()
    );
    // cmd_build's own status lines go to stderr, so they survive `--quiet`
    // (which only silences the engine's stdout progress chatter).
    let res = sssp_bench::build::build_cache(pbf, prof, progress, force).map_err(|e| anyhow::anyhow!(e))?;
    if res.cached {
        eprintln!("reused existing cache (pass --force to rebuild)");
    } else {
        eprintln!(
            "done: CH built in {:.1}s ({} nodes, {} edges)",
            res.build_secs, res.nodes, res.edges
        );
    }
    eprintln!("  pp: {}", res.pp_path.display());
    eprintln!("  ch: {}", res.ch_path.display());
    Ok(())
}

// -------------------------------------------------------------------------
// solve: the actual shared-memory integration.
// -------------------------------------------------------------------------

fn cmd_solve(
    problem_path: &Path,
    ch_path: &Path,
    pp_path: &Path,
    time_limit_s: Option<f64>,
    multi_start: usize,
    output: Option<&Path>,
) -> Result<()> {
    let t_total = std::time::Instant::now();

    // 1. Read the Vroom-style problem JSON.
    eprintln!("[1/5] reading problem {}", problem_path.display());
    let json = std::fs::read_to_string(problem_path)
        .with_context(|| format!("read {}", problem_path.display()))?;
    let mut problem: brooom::Problem = brooom::io::parse_input(&json).context("parse problem")?;
    eprintln!(
        "      {} jobs, {} shipments, {} vehicles",
        problem.jobs.len(),
        problem.shipments.len(),
        problem.vehicles.len()
    );

    // 2. mmap-load the dijeng caches (~20 µs each regardless of size).
    eprintln!("[2/5] mmap CH {}", ch_path.display());
    let t = std::time::Instant::now();
    let pp = sssp_bench::cache_pp::load_mmap(pp_path)
        .with_context(|| format!("load PP cache {}", pp_path.display()))?;
    let ch = sssp_bench::cache_ch::load_mmap(ch_path)
        .with_context(|| format!("load CH cache {}", ch_path.display()))?;
    eprintln!(
        "      loaded in {:.2} ms (graph_n={}, edges={})",
        t.elapsed().as_secs_f64() * 1000.0,
        ch.graph_fwd.n,
        ch.graph_fwd.head.last().copied().unwrap_or(0),
    );

    // 3. Collect every (lat, lon) referenced by the problem: vehicles first
    //    (so vehicle starts/ends land at indices 0..2V), then jobs (V..V+J).
    //    brooom-style JSON uses [lon, lat]; sssp_bench wants (lat, lon).
    let (coords_latlon, vehicle_starts, vehicle_ends, job_indices) = collect_coords(&problem)?;
    let n_points = coords_latlon.len();
    eprintln!("[3/5] {} coords to snap", n_points);

    // 4. Build RoutingService and the routing matrix in one shot.
    //    `matrix_with_distance` runs sssp_bench's bucket-MMM: forward sweep
    //    per src, backward sweep per dst, all in parallel, dual-channel
    //    (dur + dist) at no extra Dijeng cost.
    eprintln!("[4/5] sssp_bench::matrix_with_distance ({n_points}×{n_points})");
    let svc = sssp_bench::routing::RoutingService::new(ch, pp.coords);
    let t = std::time::Instant::now();
    let (durs_f32, dists_f32, _snap_src, _snap_dst) =
        svc.matrix_with_distance(&coords_latlon, &coords_latlon);
    let mmm_secs = t.elapsed().as_secs_f64();
    let cells = (n_points as u64).pow(2);
    eprintln!(
        "      matrix built in {:.2} s ({:.1} M cells/s)",
        mmm_secs,
        cells as f64 / mmm_secs / 1e6,
    );

    // 5. Convert f32 → i32 (brooom uses i32 for cache density). One pass,
    //    no extra allocations beyond the destination Vec. f32::INFINITY pairs
    //    (graph-disconnected after snap) get a 7-day sentinel — large enough
    //    that the solver routes around them when a finite alternative exists,
    //    small enough that summing a handful of them doesn't blow up route
    //    cost on the final summary.
    let n_inf = durs_f32.iter().filter(|d| !d.is_finite()).count();
    if n_inf > 0 {
        eprintln!(
            "      WARN: {} / {} matrix cells unreachable after snap (placed at 7-day sentinel)",
            n_inf, durs_f32.len(),
        );
    }
    let durations: Vec<i32> = durs_f32.iter().map(narrow_pos_i32).collect();
    let distances: Vec<i32> = dists_f32.iter().map(narrow_pos_i32).collect();
    drop(durs_f32);
    drop(dists_f32);
    let matrix = brooom::Matrix {
        n: n_points,
        durations,
        distances: Some(distances),
    };

    // 6. Drop jobs whose nearest road snap landed on a fragment that can't
    //    reach the depot (or vice-versa). Without this, the solver still
    //    inserts those jobs because their feasibility check is purely TW +
    //    capacity, and sentinel edges look like "very slow but legal".
    //    A handful of sentinel edges per route would dominate the summary.
    let depot_idx_opt = vehicle_starts.iter().copied().find(|x| x.is_some()).flatten();
    let dropped: Vec<usize> = if let Some(d) = depot_idx_opt {
        let mut drop = Vec::new();
        for (j, &idx) in job_indices.iter().enumerate() {
            let dur_out = matrix.durations[d * n_points + idx];
            let dur_in = matrix.durations[idx * n_points + d];
            if dur_out >= SENTINEL_I32 || dur_in >= SENTINEL_I32 {
                drop.push(j);
            }
        }
        drop
    } else {
        Vec::new()
    };
    if !dropped.is_empty() {
        eprintln!(
            "      WARN: dropping {} jobs unreachable from depot (snapped to road fragments)",
            dropped.len()
        );
        // Remove from back to keep earlier indices valid.
        let drop_set: std::collections::HashSet<usize> = dropped.iter().copied().collect();
        problem.jobs = problem
            .jobs
            .into_iter()
            .enumerate()
            .filter_map(|(j, job)| if drop_set.contains(&j) { None } else { Some(job) })
            .collect();
        // Rebuild job_indices to match the surviving order.
    }
    let kept_job_indices: Vec<usize> = job_indices
        .iter()
        .enumerate()
        .filter_map(|(j, &idx)| {
            if dropped.iter().any(|&d| d == j) { None } else { Some(idx) }
        })
        .collect();

    // 7. Switch every Location reference in the problem from coord-mode to
    //    index-mode. brooom's `solve_with_matrix` keys off matrix indices,
    //    not lat/lon. The indices match the order in which we collected
    //    coords in step 3.
    rebind_problem_to_indices(&mut problem, &vehicle_starts, &vehicle_ends, &kept_job_indices);

    // 8. Solve.
    eprintln!(
        "[5/5] brooom::solve_with_matrix (multi_start={}, time_limit={}s)",
        multi_start,
        time_limit_s.map(|s| format!("{:.0}", s)).unwrap_or_else(|| "∞".into())
    );
    let cfg = brooom::solver::SolverConfig {
        multi_start: multi_start.max(1),
        time_limit_ms: time_limit_s.map(|s| (s * 1000.0) as u64),
        verbose: true,
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let solution = brooom::solver::solve_with_matrix(&problem, &matrix, &cfg);
    let solve_secs = t.elapsed().as_secs_f64();

    // 9. Print a short summary + write Vroom output JSON.
    let total = t_total.elapsed().as_secs_f64();
    print_summary(&problem, &matrix, &solution, mmm_secs, solve_secs, total);

    let vroom_out = brooom::io::to_output(&problem, &solution, Some(&matrix));
    let output_json = serde_json::to_string_pretty(&vroom_out).context("serialise output")?;
    match output {
        Some(path) => {
            std::fs::write(path, &output_json).with_context(|| format!("write {}", path.display()))?;
            eprintln!("wrote {}", path.display());
        }
        None => {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(output_json.as_bytes())?;
            stdout.write_all(b"\n")?;
        }
    }
    Ok(())
}

/// Sentinel for "unreachable in the road graph after snap" — 7 days in
/// seconds (or 7 days × 1 m/s in metres). Large enough that the solver
/// avoids the cell when any finite alternative exists, but small enough
/// not to dominate aggregate cost when one accidentally lands in a route.
const SENTINEL_I32: i32 = 7 * 24 * 60 * 60; // 604_800

fn narrow_pos_i32(v: &f32) -> i32 {
    let v = *v;
    if !v.is_finite() || v < 0.0 {
        SENTINEL_I32
    } else if v > SENTINEL_I32 as f32 {
        SENTINEL_I32
    } else {
        v.round() as i32
    }
}

/// Returns:
///   - coords_latlon: every distinct (lat, lon) referenced, in the order
///     vehicle-start, vehicle-end, job, job, ...
///   - vehicle_starts[v] / vehicle_ends[v] / job_indices[j] = index into
///     coords_latlon for that entity.
fn collect_coords(
    problem: &brooom::Problem,
) -> Result<(Vec<(f32, f32)>, Vec<Option<usize>>, Vec<Option<usize>>, Vec<usize>)> {
    let mut coords: Vec<(f32, f32)> = Vec::new();
    let push = |coords: &mut Vec<(f32, f32)>, lonlat: [f64; 2]| -> usize {
        let idx = coords.len();
        coords.push((lonlat[1] as f32, lonlat[0] as f32)); // (lat, lon)
        idx
    };
    let mut vehicle_starts = Vec::with_capacity(problem.vehicles.len());
    let mut vehicle_ends = Vec::with_capacity(problem.vehicles.len());
    for v in &problem.vehicles {
        let s = v.start.as_ref().and_then(|l| l.coord).map(|c| push(&mut coords, c));
        let e = v.end.as_ref().and_then(|l| l.coord).map(|c| push(&mut coords, c));
        vehicle_starts.push(s);
        vehicle_ends.push(e);
    }
    let mut job_indices = Vec::with_capacity(problem.jobs.len());
    for j in &problem.jobs {
        let c = j
            .location
            .coord
            .ok_or_else(|| anyhow::anyhow!("job {} is missing coord", j.id))?;
        job_indices.push(push(&mut coords, c));
    }
    if !problem.shipments.is_empty() {
        bail!("shipments not yet wired into mpee-cli — use brooom directly for now");
    }
    Ok((coords, vehicle_starts, vehicle_ends, job_indices))
}

fn rebind_problem_to_indices(
    problem: &mut brooom::Problem,
    vehicle_starts: &[Option<usize>],
    vehicle_ends: &[Option<usize>],
    job_indices: &[usize],
) {
    for (v, vh) in problem.vehicles.iter_mut().enumerate() {
        if let (Some(start), Some(idx)) = (vh.start.as_mut(), vehicle_starts[v]) {
            start.coord = None;
            start.index = Some(idx);
        }
        if let (Some(end), Some(idx)) = (vh.end.as_mut(), vehicle_ends[v]) {
            end.coord = None;
            end.index = Some(idx);
        }
    }
    for (j, job) in problem.jobs.iter_mut().enumerate() {
        job.location.coord = None;
        job.location.index = Some(job_indices[j]);
    }
}

fn print_summary(
    problem: &brooom::Problem,
    matrix: &brooom::Matrix,
    solution: &brooom::solution::Solution,
    mmm_secs: f64,
    solve_secs: f64,
    total_secs: f64,
) {
    let n_routes = solution.routes.len();
    let used: usize = solution.routes.iter().filter(|r| !r.steps.is_empty()).count();
    let assigned_jobs: usize = solution
        .routes
        .iter()
        .map(|r| {
            r.steps
                .iter()
                .filter(|t| matches!(t, brooom::solution::TaskRef::Job(_)))
                .count()
        })
        .sum();
    let unassigned = problem.jobs.len() - assigned_jobs;

    // Sum each route's total duration (driving + service) via the matrix.
    let mut total_dur: i64 = 0;
    let mut total_dist: i64 = 0;
    for route in &solution.routes {
        if route.steps.is_empty() {
            continue;
        }
        let v = &problem.vehicles[route.vehicle_idx];
        let start_idx = v.start.as_ref().and_then(|l| l.index);
        let end_idx = v.end.as_ref().and_then(|l| l.index);
        let mut prev = start_idx;
        for task in &route.steps {
            let here = match task {
                brooom::solution::TaskRef::Job(j) => problem.jobs[*j].location.index,
                _ => continue,
            };
            if let (Some(p), Some(h)) = (prev, here) {
                total_dur += matrix.duration(p, h);
                total_dist += matrix.distance(p, h);
            }
            prev = here;
        }
        if let (Some(p), Some(e)) = (prev, end_idx) {
            total_dur += matrix.duration(p, e);
            total_dist += matrix.distance(p, e);
        }
    }

    eprintln!();
    eprintln!("─── mpe solve summary ─────────────────────────────────────");
    eprintln!("jobs              : {} (assigned {} / unassigned {})", problem.jobs.len(), assigned_jobs, unassigned);
    eprintln!("vehicles          : {} (used {})", n_routes, used);
    eprintln!("matrix MMM time   : {:.2} s", mmm_secs);
    eprintln!("solver time       : {:.2} s", solve_secs);
    eprintln!("wall time         : {:.2} s", total_secs);
    eprintln!("total drive time  : {} s  (≈ {:.1} h)", total_dur, total_dur as f64 / 3600.0);
    eprintln!("total drive dist  : {} m  (≈ {:.1} km)", total_dist, total_dist as f64 / 1000.0);
    eprintln!("───────────────────────────────────────────────────────────");
}
