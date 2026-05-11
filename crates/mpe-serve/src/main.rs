//! mpe-serve — live HTTP server that solves a VRP and shows it on a map.
//!
//! The whole pipeline runs in this process: sssp_bench loads the CH cache,
//! brooom solves the problem on top of the matrix sssp_bench produces, and
//! the HTTP handlers below read the same in-RAM `Problem` / `Solution` /
//! `Matrix` that the solver just finished with. No disk, no IPC, no
//! re-serialisation between steps — the JSON payload sent to the browser
//! is built once from those structs and held in an `Arc<String>`.
//!
//! Typical use:
//!
//!   mpe-serve --region london --n-jobs 5000 --n-vehicles 100 \
//!     --ch data/greater-london.osm.pbf.ch \
//!     --pp data/greater-london.osm.pbf.pp \
//!     --time-limit-s 20 --port 8032
//!
//! Then point a phone on the same network at http://<laptop-ip>:8032 .

use anyhow::{bail, Context, Result};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tiny_http::{Header, Method, Response, Server};

#[derive(Parser, Debug)]
#[command(name = "mpe-serve", version, about = "mpe-engine live map server")]
struct Args {
    /// Region for `gen` (london / oslo / manhattan / paris).
    #[arg(long, default_value = "london")]
    region: String,

    /// Number of random jobs to generate.
    #[arg(long, default_value_t = 5000)]
    n_jobs: usize,

    /// Number of vehicles.
    #[arg(long, default_value_t = 100)]
    n_vehicles: usize,

    /// Capacity per vehicle (one dimension).
    #[arg(long, default_value_t = 350)]
    capacity: i64,

    /// Random seed for the generator.
    #[arg(long, default_value_t = 7)]
    seed: u64,

    /// CH cache path (sssp_bench).
    #[arg(long)]
    ch: PathBuf,

    /// PP cache path (sssp_bench).
    #[arg(long)]
    pp: PathBuf,

    /// Solver wall-time budget in seconds.
    #[arg(long, default_value_t = 20.0)]
    time_limit_s: f64,

    /// Solver multi-start variants.
    #[arg(long, default_value_t = 2)]
    multi_start: usize,

    /// HTTP listen port.
    #[arg(long, default_value_t = 8032)]
    port: u16,

    /// HTTP bind host.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Instead of generating + solving, load these two files and just serve.
    /// Useful for revisiting a result without re-solving.
    #[arg(long)]
    load_problem: Option<PathBuf>,
    #[arg(long)]
    load_solution: Option<PathBuf>,
}

const INDEX_HTML: &str = include_str!("../static/index.html");
const SENTINEL_I32: i32 = 7 * 24 * 60 * 60;

fn main() -> Result<()> {
    let args = Args::parse();

    let bundle = if let (Some(p), Some(s)) = (&args.load_problem, &args.load_solution) {
        load_from_files(p, s).context("load problem + solution from disk")?
    } else {
        solve_in_process(&args).context("solve in process")?
    };

    let json = serde_json::to_string(&bundle).context("serialise dataset")?;
    eprintln!(
        "mpe-serve: dataset built ({} routes, {} stops, {:.1} km, {} unassigned)",
        bundle.vehicles.len(),
        bundle.total_stops,
        bundle.total_distance_km,
        bundle.unassigned.len(),
    );
    let shared = Arc::new(json);

    let addr = format!("{}:{}", args.host, args.port);
    let server = Server::http(&addr).map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
    eprintln!("mpe-serve listening on http://{}/", addr);
    if args.host == "0.0.0.0" {
        eprintln!("on a phone, point at  http://<laptop-ip>:{}/", args.port);
    }

    for req in server.incoming_requests() {
        let url = req.url().to_string();
        let _ = match (req.method(), url.as_str()) {
            (Method::Get, "/") | (Method::Get, "/index.html") => req.respond(
                Response::from_string(INDEX_HTML.to_string())
                    .with_header(header_text_html())
                    .with_header(header_cors()),
            ),
            (Method::Get, "/api/dataset") => req.respond(
                Response::from_string(shared.as_ref().clone())
                    .with_header(header_json())
                    .with_header(header_cors()),
            ),
            (Method::Get, "/api/health") => req.respond(
                Response::from_string("{\"ok\":true}")
                    .with_header(header_json())
                    .with_header(header_cors()),
            ),
            (Method::Options, _) => req.respond(
                Response::empty(204)
                    .with_header(header_cors())
                    .with_header(parse_header("Access-Control-Allow-Methods: GET, OPTIONS"))
                    .with_header(parse_header("Access-Control-Allow-Headers: *")),
            ),
            _ => req.respond(Response::from_string("Not Found").with_status_code(404)),
        };
    }
    Ok(())
}

fn header_text_html() -> Header {
    parse_header("Content-Type: text/html; charset=utf-8")
}
fn header_json() -> Header {
    parse_header("Content-Type: application/json")
}
fn header_cors() -> Header {
    parse_header("Access-Control-Allow-Origin: *")
}
fn parse_header(s: &str) -> Header {
    Header::from_bytes(
        s.split_once(':').map(|(k, _)| k.trim().as_bytes()).unwrap_or(b""),
        s.split_once(':').map(|(_, v)| v.trim().as_bytes()).unwrap_or(b""),
    )
    .expect("static header parse")
}

// -------------------------------------------------------------------------
// Generator: random jobs inside a region's bbox. Mirrors mpe-cli's `gen`.
// -------------------------------------------------------------------------

fn region_bbox(region: &str) -> Result<(f64, f64, f64, f64)> {
    Ok(match region {
        "london"    => (51.46, 51.56, -0.22,  0.02),
        "oslo"      => (59.88, 59.95, 10.68, 10.82),
        "manhattan" => (40.74, 40.80, -74.00, -73.94),
        "paris"     => (48.82, 48.89,  2.27,  2.41),
        other => bail!("unknown region '{other}'"),
    })
}

fn region_depot(region: &str) -> (f64, f64) {
    match region {
        "london"    => (51.5074, -0.1278),
        "oslo"      => (59.9139, 10.7522),
        "manhattan" => (40.7580, -73.9855),
        "paris"     => (48.8566,  2.3522),
        _ => (0.0, 0.0),
    }
}

// -------------------------------------------------------------------------
// In-process solve: gen → snap → matrix → brooom. The returned `Dataset`
// is the JSON-ready bundle for the frontend.
// -------------------------------------------------------------------------

fn solve_in_process(args: &Args) -> Result<Dataset> {
    let (lat_min, lat_max, lon_min, lon_max) = region_bbox(&args.region)?;
    let (depot_lat, depot_lon) = region_depot(&args.region);
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);

    // 1. Generate the problem in memory (no disk).
    eprintln!("[1/5] generating {} jobs in {} ({} vehicles)", args.n_jobs, args.region, args.n_vehicles);
    let mut problem = brooom::Problem::default();
    for i in 0..args.n_jobs {
        let lat = rng.gen_range(lat_min..lat_max);
        let lon = rng.gen_range(lon_min..lon_max);
        let delivery: i64 = rng.gen_range(1..=10);
        problem.jobs.push(brooom::Job {
            id: (i + 1) as u64,
            location: brooom::Location::from_coord(lon, lat),
            kind: Default::default(),
            service: 60,
            setup: 0,
            delivery: vec![delivery],
            pickup: vec![],
            skills: vec![],
            priority: 0,
            time_windows: vec![],
            description: None,
        });
    }
    for v in 0..args.n_vehicles {
        problem.vehicles.push(brooom::Vehicle {
            id: (v + 1) as u64,
            start: Some(brooom::Location::from_coord(depot_lon, depot_lat)),
            end: Some(brooom::Location::from_coord(depot_lon, depot_lat)),
            capacity: vec![args.capacity],
            skills: vec![],
            time_window: None,
            speed_factor: 1.0,
            max_tasks: None,
            max_travel_time: None,
            max_distance: None,
            fixed: 0.0,
            per_hour: 3600.0,
            profile: "car".to_string(),
            description: None,
        });
    }

    // 2. mmap CH + PP. ~20 µs regardless of size.
    eprintln!("[2/5] mmap CH cache");
    let pp = sssp_bench::cache_pp::load_mmap(&args.pp)
        .with_context(|| format!("load PP cache {}", args.pp.display()))?;
    let ch = sssp_bench::cache_ch::load_mmap(&args.ch)
        .with_context(|| format!("load CH cache {}", args.ch.display()))?;

    // 3. Collect coords (one entry per vehicle-start, per vehicle-end, per job).
    let (coords, vehicle_starts, vehicle_ends, job_indices) = collect_coords(&problem)?;
    let n_points = coords.len();
    eprintln!("[3/5] {} coords to snap", n_points);

    // 4. Build the full N×N duration+distance matrix via sssp_bench's MMM.
    eprintln!("[4/5] sssp_bench::matrix_with_distance ({n_points}×{n_points})");
    let svc = sssp_bench::routing::RoutingService::new(ch, pp.coords);
    let t = std::time::Instant::now();
    let (durs_f32, dists_f32, _, _) = svc.matrix_with_distance(&coords, &coords);
    let mmm_secs = t.elapsed().as_secs_f64();
    eprintln!(
        "      matrix built in {:.2} s ({:.1} M cells/s)",
        mmm_secs,
        (n_points as u64).pow(2) as f64 / mmm_secs / 1e6,
    );

    let n_inf = durs_f32.iter().filter(|d| !d.is_finite()).count();
    if n_inf > 0 {
        eprintln!("      {} / {} matrix cells unreachable", n_inf, durs_f32.len());
    }

    let durations: Vec<i32> = durs_f32.iter().map(narrow_pos_i32).collect();
    let distances: Vec<i32> = dists_f32.iter().map(narrow_pos_i32).collect();
    drop(durs_f32);
    drop(dists_f32);
    let matrix = brooom::Matrix { n: n_points, durations, distances: Some(distances) };

    // 5. Drop jobs unreachable from depot (snapped to isolated fragments).
    let depot_idx = vehicle_starts.iter().copied().find_map(|x| x);
    let mut dropped_idx_in_problem: Vec<usize> = Vec::new();
    if let Some(d) = depot_idx {
        for (j, &idx) in job_indices.iter().enumerate() {
            let out = matrix.durations[d * n_points + idx];
            let inb = matrix.durations[idx * n_points + d];
            if out >= SENTINEL_I32 || inb >= SENTINEL_I32 {
                dropped_idx_in_problem.push(j);
            }
        }
    }
    let drop_set: std::collections::HashSet<usize> =
        dropped_idx_in_problem.iter().copied().collect();
    let dropped_job_ids: Vec<u64> = dropped_idx_in_problem
        .iter()
        .map(|&j| problem.jobs[j].id)
        .collect();

    let kept_job_indices: Vec<usize> = job_indices
        .iter()
        .enumerate()
        .filter_map(|(j, &idx)| if drop_set.contains(&j) { None } else { Some(idx) })
        .collect();
    let kept_jobs_latlon: Vec<(f32, f32)> = job_indices
        .iter()
        .enumerate()
        .filter_map(|(j, &idx)| if drop_set.contains(&j) { None } else { Some(coords[idx]) })
        .collect();

    problem.jobs = problem
        .jobs
        .into_iter()
        .enumerate()
        .filter_map(|(j, job)| if drop_set.contains(&j) { None } else { Some(job) })
        .collect();
    rebind_to_indices(&mut problem, &vehicle_starts, &vehicle_ends, &kept_job_indices);

    if !dropped_job_ids.is_empty() {
        eprintln!("      dropped {} unreachable jobs", dropped_job_ids.len());
    }

    // 6. Solve.
    eprintln!(
        "[5/5] brooom::solve_with_matrix (multi_start={}, time_limit={}s)",
        args.multi_start, args.time_limit_s
    );
    let cfg = brooom::solver::SolverConfig {
        multi_start: args.multi_start.max(1),
        time_limit_ms: Some((args.time_limit_s * 1000.0) as u64),
        verbose: true,
        ..Default::default()
    };
    let solution = brooom::solver::solve_with_matrix(&problem, &matrix, &cfg);

    // 7. Materialise the rendering bundle from the SAME `Problem`,
    //    `Solution`, `Matrix`, and `kept_jobs_latlon` that live in our
    //    address space right now — no copy beyond the JSON serialiser.
    let bundle = build_dataset(
        &problem,
        &solution,
        &matrix,
        &kept_jobs_latlon,
        (depot_lat, depot_lon),
        &dropped_job_ids,
        &args.region,
    );

    Ok(bundle)
}

fn collect_coords(
    problem: &brooom::Problem,
) -> Result<(Vec<(f32, f32)>, Vec<Option<usize>>, Vec<Option<usize>>, Vec<usize>)> {
    let mut coords: Vec<(f32, f32)> = Vec::new();
    let push = |coords: &mut Vec<(f32, f32)>, lonlat: [f64; 2]| -> usize {
        let idx = coords.len();
        coords.push((lonlat[1] as f32, lonlat[0] as f32));
        idx
    };
    let mut vs = Vec::with_capacity(problem.vehicles.len());
    let mut ve = Vec::with_capacity(problem.vehicles.len());
    for v in &problem.vehicles {
        let s = v.start.as_ref().and_then(|l| l.coord).map(|c| push(&mut coords, c));
        let e = v.end.as_ref().and_then(|l| l.coord).map(|c| push(&mut coords, c));
        vs.push(s);
        ve.push(e);
    }
    let mut ji = Vec::with_capacity(problem.jobs.len());
    for j in &problem.jobs {
        let c = j
            .location
            .coord
            .ok_or_else(|| anyhow::anyhow!("job {} missing coord", j.id))?;
        ji.push(push(&mut coords, c));
    }
    Ok((coords, vs, ve, ji))
}

fn rebind_to_indices(
    problem: &mut brooom::Problem,
    vs: &[Option<usize>],
    ve: &[Option<usize>],
    kept_job_indices: &[usize],
) {
    for (v, vh) in problem.vehicles.iter_mut().enumerate() {
        if let (Some(start), Some(idx)) = (vh.start.as_mut(), vs[v]) {
            start.coord = None;
            start.index = Some(idx);
        }
        if let (Some(end), Some(idx)) = (vh.end.as_mut(), ve[v]) {
            end.coord = None;
            end.index = Some(idx);
        }
    }
    for (j, job) in problem.jobs.iter_mut().enumerate() {
        job.location.coord = None;
        job.location.index = Some(kept_job_indices[j]);
    }
}

fn narrow_pos_i32(v: &f32) -> i32 {
    let v = *v;
    if !v.is_finite() || v < 0.0 || v > SENTINEL_I32 as f32 {
        SENTINEL_I32
    } else {
        v.round() as i32
    }
}

// -------------------------------------------------------------------------
// Dataset = the JSON-ready bundle the browser fetches.
// -------------------------------------------------------------------------

#[derive(Serialize)]
struct Dataset {
    region: String,
    depot: LatLon,
    bbox: Bbox,
    total_jobs: usize,
    total_stops: usize,
    total_duration_h: f64,
    total_distance_km: f64,
    vehicles: Vec<VehicleOut>,
    unassigned: Vec<JobPoint>,
    dropped: Vec<u64>,
}

#[derive(Serialize)]
struct LatLon { lat: f32, lon: f32 }

#[derive(Serialize)]
struct Bbox { lat_min: f32, lat_max: f32, lon_min: f32, lon_max: f32 }

#[derive(Serialize)]
struct VehicleOut {
    id: u64,
    color: String,
    n_stops: usize,
    duration_s: i64,
    distance_m: i64,
    stops: Vec<StopOut>,
}

#[derive(Serialize)]
struct StopOut {
    job_id: u64,
    lat: f32,
    lon: f32,
    order: usize,
    load_after: i64,
}

#[derive(Serialize)]
struct JobPoint {
    job_id: u64,
    lat: f32,
    lon: f32,
}

fn build_dataset(
    problem: &brooom::Problem,
    solution: &brooom::solution::Solution,
    matrix: &brooom::Matrix,
    kept_jobs_latlon: &[(f32, f32)],
    depot_latlon: (f64, f64),
    dropped_ids: &[u64],
    region: &str,
) -> Dataset {
    use brooom::solution::TaskRef;

    // Bounding box over depot + every kept job.
    let mut lat_min = depot_latlon.0 as f32;
    let mut lat_max = lat_min;
    let mut lon_min = depot_latlon.1 as f32;
    let mut lon_max = lon_min;
    for &(la, lo) in kept_jobs_latlon {
        lat_min = lat_min.min(la); lat_max = lat_max.max(la);
        lon_min = lon_min.min(lo); lon_max = lon_max.max(lo);
    }

    let mut total_dur: i64 = 0;
    let mut total_dist: i64 = 0;
    let mut total_stops: usize = 0;

    let mut vehicles_out: Vec<VehicleOut> = Vec::with_capacity(solution.routes.len());
    for (ri, r) in solution.routes.iter().enumerate() {
        if r.steps.is_empty() {
            continue;
        }
        let vh = &problem.vehicles[r.vehicle_idx];
        let start_idx = vh.start.as_ref().and_then(|l| l.index);
        let end_idx = vh.end.as_ref().and_then(|l| l.index);

        let mut stops: Vec<StopOut> = Vec::with_capacity(r.steps.len());
        let mut prev = start_idx;
        let mut route_dur: i64 = 0;
        let mut route_dist: i64 = 0;
        let mut load: i64 = 0;
        let mut order = 0usize;

        for step in &r.steps {
            let job_idx = match step {
                TaskRef::Job(j) => *j,
                _ => continue,
            };
            let here_idx = problem.jobs[job_idx].location.index;
            if let (Some(p), Some(h)) = (prev, here_idx) {
                route_dur += matrix.duration(p, h);
                route_dist += matrix.distance(p, h);
            }
            prev = here_idx;
            // delivery → load goes UP from -delivery to 0 over the route;
            // for the UI we show how much has been delivered so far.
            let job = &problem.jobs[job_idx];
            let delivered: i64 = job.delivery.iter().sum();
            load += delivered;
            let (lat, lon) = kept_jobs_latlon[job_idx];
            stops.push(StopOut {
                job_id: job.id,
                lat,
                lon,
                order,
                load_after: load,
            });
            order += 1;
        }
        if let (Some(p), Some(e)) = (prev, end_idx) {
            route_dur += matrix.duration(p, e);
            route_dist += matrix.distance(p, e);
        }

        total_stops += stops.len();
        total_dur += route_dur;
        total_dist += route_dist;

        // Stable, well-spaced palette: golden-ratio hue stride.
        let hue = ((ri as f32 * 137.508) % 360.0).abs();
        let color = format!("hsl({:.0},75%,45%)", hue);

        vehicles_out.push(VehicleOut {
            id: vh.id,
            color,
            n_stops: stops.len(),
            duration_s: route_dur,
            distance_m: route_dist,
            stops,
        });
    }

    let unassigned = solution
        .unassigned
        .iter()
        .filter_map(|t| match t {
            TaskRef::Job(j) => {
                let job = &problem.jobs[*j];
                let (lat, lon) = kept_jobs_latlon[*j];
                Some(JobPoint { job_id: job.id, lat, lon })
            }
            _ => None,
        })
        .collect();

    Dataset {
        region: region.to_string(),
        depot: LatLon { lat: depot_latlon.0 as f32, lon: depot_latlon.1 as f32 },
        bbox: Bbox { lat_min, lat_max, lon_min, lon_max },
        total_jobs: problem.jobs.len() + dropped_ids.len(),
        total_stops,
        total_duration_h: total_dur as f64 / 3600.0,
        total_distance_km: total_dist as f64 / 1000.0,
        vehicles: vehicles_out,
        unassigned,
        dropped: dropped_ids.to_vec(),
    }
}

// -------------------------------------------------------------------------
// File fallback (used when --load-problem + --load-solution are given).
// -------------------------------------------------------------------------

fn load_from_files(problem_path: &Path, solution_path: &Path) -> Result<Dataset> {
    let _ = (problem_path, solution_path);
    bail!(
        "loading from disk is not implemented in this revision. Run without \
         --load-* to solve in-process — the integration is meant to share \
         memory with the solver, not round-trip through JSON."
    )
}
