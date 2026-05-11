//! PyO3 bindings — `import mpe_py` from Python and drive the in-process
//! VRP solver. The Python interpreter and the Rust solver share one
//! address space: the JSON bytes returned by `get_dataset_json()` come
//! straight out of the same `Arc<String>` that the solver thread just
//! populated.
//!
//! Build with maturin:
//!   cd crates/mpe-py
//!   python3 -m venv venv && source venv/bin/activate
//!   pip install maturin flask
//!   maturin develop --release
//!   python3 python/app.py
//!
//! API:
//!   import mpe_py
//!   eng = mpe_py.Engine()
//!   eng.start_solve(region="london", n_jobs=500, n_vehicles=20, ...)
//!   while not eng.is_done():
//!       status = json.loads(eng.get_status_json())
//!       ds = eng.get_dataset_json()  # None until first iter completes
//!       time.sleep(1)

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::sync::{Arc, RwLock};

const SENTINEL_I32: i32 = 7 * 24 * 60 * 60;

// -------------------------------------------------------------------------
// Shared state — exactly the same shape as mpe-serve's AppState.
// -------------------------------------------------------------------------

struct AppState {
    started_at_ms: u128,
    state: &'static str,        // "idle" | "solving" | "evolving" | "done" | "failed"
    phase: String,
    message: String,
    progress: f32,
    error: Option<String>,
    dataset_json: Option<Arc<String>>,
    dataset_iter: u32,
    config: ConfigOut,
}

#[derive(Clone, Serialize)]
struct ConfigOut {
    region: String,
    n_jobs: usize,
    n_vehicles: usize,
    capacity: i64,
    seed: u64,
    time_limit_s: f64,
    multi_start: usize,
}

#[derive(Serialize)]
struct StatusOut<'a> {
    state: &'a str,
    phase: &'a str,
    message: &'a str,
    progress: f32,
    elapsed_s: f64,
    dataset_iter: u32,
    config: &'a ConfigOut,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn set_phase(state: &Arc<RwLock<AppState>>, phase: &str, message: &str, progress: f32) {
    let mut s = state.write().unwrap();
    s.phase = phase.into();
    s.message = message.into();
    s.progress = progress;
    eprintln!("[mpe_py {:>10}] {}", phase, message);
}

// -------------------------------------------------------------------------
// PyO3 Engine class
// -------------------------------------------------------------------------

#[pyclass]
struct Engine {
    state: Arc<RwLock<AppState>>,
}

#[pymethods]
impl Engine {
    #[new]
    fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(AppState {
                started_at_ms: now_ms(),
                state: "idle",
                phase: "idle".into(),
                message: "no solve started".into(),
                progress: 0.0,
                error: None,
                dataset_json: None,
                dataset_iter: 0,
                config: ConfigOut {
                    region: "".into(),
                    n_jobs: 0,
                    n_vehicles: 0,
                    capacity: 0,
                    seed: 0,
                    time_limit_s: 0.0,
                    multi_start: 0,
                },
            })),
        }
    }

    /// Spawn a background thread that runs the in-process VRP pipeline
    /// (gen → snap → matrix → iterative brooom solve), publishing the
    /// best-known dataset after every chunk. Returns immediately.
    /// `radius_km` > 0 generates jobs uniformly inside a disk of that
    /// radius around the region's depot. `radius_km` = 0 (default) uses
    /// the region's bbox.
    #[pyo3(signature = (
        region, n_jobs, n_vehicles, capacity, seed, ch, pp,
        time_limit_s=45.0, multi_start=1, radius_km=0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn start_solve(
        &self,
        region: &str,
        n_jobs: usize,
        n_vehicles: usize,
        capacity: i64,
        seed: u64,
        ch: &str,
        pp: &str,
        time_limit_s: f64,
        multi_start: usize,
        radius_km: f64,
    ) -> PyResult<()> {
        // Reset state to "solving".
        {
            let mut s = self.state.write().unwrap();
            if s.state == "solving" || s.state == "evolving" {
                return Err(PyRuntimeError::new_err(
                    "solver already running — call wait_done() or create a new Engine",
                ));
            }
            s.started_at_ms = now_ms();
            s.state = "solving";
            s.phase = "init".into();
            s.message = "starting".into();
            s.progress = 0.0;
            s.error = None;
            s.dataset_json = None;
            s.dataset_iter = 0;
            s.config = ConfigOut {
                region: region.into(),
                n_jobs,
                n_vehicles,
                capacity,
                seed,
                time_limit_s,
                multi_start,
            };
        }

        let solver_state = self.state.clone();
        let (region, ch, pp) = (region.to_string(), ch.to_string(), pp.to_string());
        std::thread::Builder::new()
            .name("solver".into())
            .spawn(move || {
                let args = SolverArgs {
                    region, n_jobs, n_vehicles, capacity, seed,
                    ch, pp, time_limit_s, multi_start, radius_km,
                };
                if let Err(e) = solve_in_process(&args, &solver_state) {
                    let msg = format!("{:#}", e);
                    eprintln!("[mpe_py] solver failed: {msg}");
                    let mut s = solver_state.write().unwrap();
                    s.state = "failed";
                    s.error = Some(msg);
                    s.message = "failed".into();
                    s.progress = 0.0;
                }
            })
            .map_err(|e| PyRuntimeError::new_err(format!("spawn solver: {e}")))?;
        Ok(())
    }

    /// Return the current status as a JSON string.
    fn get_status_json(&self) -> String {
        let s = self.state.read().unwrap();
        let out = StatusOut {
            state: s.state,
            phase: &s.phase,
            message: &s.message,
            progress: s.progress,
            elapsed_s: (now_ms() - s.started_at_ms) as f64 / 1000.0,
            dataset_iter: s.dataset_iter,
            config: &s.config,
            error: s.error.as_deref(),
        };
        serde_json::to_string(&out).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    /// Return the latest dataset JSON, or `None` if no iteration has
    /// completed yet. The string is cloned out of the shared `Arc<String>`,
    /// so subsequent reads after a solver update will return new bytes
    /// reflecting the improved solution.
    fn get_dataset_json(&self) -> Option<String> {
        let s = self.state.read().unwrap();
        s.dataset_json.as_ref().map(|a| a.as_ref().clone())
    }

    /// Convenience: which dataset_iter is currently published.
    fn dataset_iter(&self) -> u32 {
        self.state.read().unwrap().dataset_iter
    }

    /// True once the solver thread has finished (either "done" or "failed").
    fn is_done(&self) -> bool {
        let s = self.state.read().unwrap();
        s.state == "done" || s.state == "failed"
    }

    /// Current state string: "idle" | "solving" | "evolving" | "done" | "failed".
    fn state(&self) -> &'static str {
        self.state.read().unwrap().state
    }
}

#[pymodule]
fn mpe_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}

// -------------------------------------------------------------------------
// solver — mirrors mpe-serve's solve_in_process. The boundary between
// Python and Rust does NOT cross here; the solver thread is pure Rust
// and only the published `Arc<String>` is read by Python on demand.
// -------------------------------------------------------------------------

struct SolverArgs {
    region: String,
    n_jobs: usize,
    n_vehicles: usize,
    capacity: i64,
    seed: u64,
    ch: String,
    pp: String,
    time_limit_s: f64,
    multi_start: usize,
    radius_km: f64,
}

fn region_bbox(region: &str) -> anyhow::Result<(f64, f64, f64, f64)> {
    Ok(match region {
        "london"    => (51.46, 51.56, -0.22,  0.02),
        "oslo"      => (59.88, 59.95, 10.68, 10.82),
        "manhattan" => (40.74, 40.80, -74.00, -73.94),
        "paris"     => (48.82, 48.89,  2.27,  2.41),
        other => anyhow::bail!("unknown region '{other}'"),
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

fn solve_in_process(args: &SolverArgs, state: &Arc<RwLock<AppState>>) -> anyhow::Result<()> {
    use anyhow::Context;

    let (depot_lat, depot_lon) = region_depot(&args.region);
    let shape_desc = if args.radius_km > 0.0 {
        format!("circle r={:.1}km around depot", args.radius_km)
    } else {
        format!("bbox of {}", args.region)
    };
    set_phase(state, "gen", &format!("generating {} jobs in {}", args.n_jobs, shape_desc), 0.05);
    let (lat_min, lat_max, lon_min, lon_max) = region_bbox(&args.region)?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);

    let mut problem = brooom::Problem::default();
    for i in 0..args.n_jobs {
        let (lat, lon) = if args.radius_km > 0.0 {
            // Uniform sample within a disk of radius_km around the depot.
            // r = sqrt(u) * R gives uniform area-density; theta uniform.
            use std::f64::consts::PI;
            let u: f64 = rng.gen_range(0.0..1.0);
            let r = u.sqrt() * (args.radius_km * 1000.0);
            let theta: f64 = rng.gen_range(0.0..(2.0 * PI));
            let dy = r * theta.sin();
            let dx = r * theta.cos();
            let lat = depot_lat + dy / 111_000.0;
            let lon_per_m = 1.0 / (111_000.0 * depot_lat.to_radians().cos().max(1e-6));
            let lon = depot_lon + dx * lon_per_m;
            (lat, lon)
        } else {
            (rng.gen_range(lat_min..lat_max), rng.gen_range(lon_min..lon_max))
        };
        let delivery: i64 = rng.gen_range(1..=10);
        problem.jobs.push(brooom::Job {
            id: (i + 1) as u64,
            location: brooom::Location::from_coord(lon, lat),
            kind: Default::default(),
            service: 60, setup: 0,
            delivery: vec![delivery], pickup: vec![],
            skills: vec![], priority: 0,
            time_windows: vec![], description: None,
        });
    }
    for v in 0..args.n_vehicles {
        problem.vehicles.push(brooom::Vehicle {
            id: (v + 1) as u64,
            start: Some(brooom::Location::from_coord(depot_lon, depot_lat)),
            end:   Some(brooom::Location::from_coord(depot_lon, depot_lat)),
            capacity: vec![args.capacity],
            skills: vec![], time_window: None,
            speed_factor: 1.0,
            max_tasks: None, max_travel_time: None, max_distance: None,
            fixed: 0.0, per_hour: 3600.0,
            profile: "car".into(), description: None,
        });
    }

    set_phase(state, "mmap", "mmap CH + PP caches", 0.10);
    let pp = sssp_bench::cache_pp::load_mmap(&args.pp)
        .with_context(|| format!("load PP cache {}", args.pp))?;
    let ch = sssp_bench::cache_ch::load_mmap(&args.ch)
        .with_context(|| format!("load CH cache {}", args.ch))?;

    let (coords, vehicle_starts, vehicle_ends, job_indices) = collect_coords(&problem)?;
    let n_points = coords.len();
    set_phase(state, "snap", &format!("{} coords ready", n_points), 0.15);

    let svc = sssp_bench::routing::RoutingService::new(ch, pp.coords);
    set_phase(state, "matrix", &format!("building {n_points}×{n_points} routing matrix"), 0.20);
    let t = std::time::Instant::now();
    let (durs_f32, dists_f32, _, _) = svc.matrix_with_distance(&coords, &coords);
    let mmm_secs = t.elapsed().as_secs_f64();
    set_phase(
        state, "matrix",
        &format!("matrix built in {:.2}s ({:.1} M cells/s)",
            mmm_secs, (n_points as u64).pow(2) as f64 / mmm_secs / 1e6),
        0.30,
    );

    let durations: Vec<i32> = durs_f32.iter().map(narrow_pos_i32).collect();
    let distances: Vec<i32> = dists_f32.iter().map(narrow_pos_i32).collect();
    drop(durs_f32); drop(dists_f32);
    let matrix = brooom::Matrix { n: n_points, durations, distances: Some(distances) };

    set_phase(state, "filter", "dropping jobs unreachable from depot", 0.32);
    let depot_idx = vehicle_starts.iter().copied().find_map(|x| x);
    let mut dropped_in_problem: Vec<usize> = Vec::new();
    if let Some(d) = depot_idx {
        for (j, &idx) in job_indices.iter().enumerate() {
            let out = matrix.durations[d * n_points + idx];
            let inb = matrix.durations[idx * n_points + d];
            if out >= SENTINEL_I32 || inb >= SENTINEL_I32 {
                dropped_in_problem.push(j);
            }
        }
    }
    let drop_set: std::collections::HashSet<usize> = dropped_in_problem.iter().copied().collect();
    let dropped_job_ids: Vec<u64> = dropped_in_problem.iter().map(|&j| problem.jobs[j].id).collect();
    let kept_job_indices: Vec<usize> = job_indices.iter().enumerate()
        .filter_map(|(j, &idx)| if drop_set.contains(&j) { None } else { Some(idx) }).collect();
    let kept_jobs_latlon: Vec<(f32, f32)> = job_indices.iter().enumerate()
        .filter_map(|(j, &idx)| if drop_set.contains(&j) { None } else { Some(coords[idx]) }).collect();
    problem.jobs = problem.jobs.into_iter().enumerate()
        .filter_map(|(j, job)| if drop_set.contains(&j) { None } else { Some(job) }).collect();
    rebind_to_indices(&mut problem, &vehicle_starts, &vehicle_ends, &kept_job_indices);

    let chunk_ms: u64 = match args.n_jobs {
        0..=200 => 800,
        201..=1000 => 1500,
        1001..=3000 => 2500,
        _ => 4000,
    };
    let total_budget_ms = (args.time_limit_s * 1000.0) as u64;
    let solve_start = std::time::Instant::now();
    let mut warm: Option<brooom::solution::Solution> = None;
    let mut iter: u32 = 0;

    loop {
        iter += 1;
        // Iter 1 has no warm-start, so brooom does:
        //   1) greedy insertion of all jobs (O(N²)) — slow for N≥1000
        //   2) up to max_local_search_passes of full LS
        //   3) ILS bounded by time_limit_ms
        // For N=2000 the first two phases alone can take 30-60 s; the
        // time_limit_ms budget only applies after (1)+(2). Cap LS passes
        // for iter 1 to keep the wait short — subsequent iters use the
        // default 50 since warm-start lands near a local optimum and
        // each pass terminates fast.
        let max_passes = if iter == 1 { 10 } else { 50 };
        let cfg = brooom::solver::SolverConfig {
            multi_start: if iter == 1 { args.multi_start.max(1) } else { 1 },
            time_limit_ms: Some(chunk_ms),
            warm_start: warm.clone(),
            max_local_search_passes: max_passes,
            verbose: false,
            ..Default::default()
        };
        let msg = if iter == 1 {
            format!(
                "iter 1 · initial insertion + LS (N={}, slow first solve)",
                args.n_jobs
            )
        } else {
            format!(
                "iter {iter} · refining (chunk {:.1}s · total {:.0}s/{:.0}s)",
                chunk_ms as f64 / 1000.0,
                solve_start.elapsed().as_secs_f64(),
                args.time_limit_s,
            )
        };
        set_phase(
            state, "solve", &msg,
            0.40 + 0.55 * (solve_start.elapsed().as_millis() as f32 / total_budget_ms.max(1) as f32).min(1.0),
        );
        let sol = brooom::solver::solve_with_matrix(&problem, &matrix, &cfg);
        let elapsed_ms = solve_start.elapsed().as_millis() as u64;
        let is_final = elapsed_ms >= total_budget_ms;

        let bundle = build_dataset(
            &problem, &sol, &matrix, &kept_jobs_latlon,
            (depot_lat, depot_lon), &dropped_job_ids, &args.region,
            iter, sol.summary.cost, solve_start.elapsed().as_secs_f64(), is_final,
        );
        let json = serde_json::to_string(&bundle).unwrap_or_default();
        let summary = format!(
            "{} routes · {} stops · {:.1} km · cost {:.0}{}",
            bundle.vehicles.len(), bundle.total_stops, bundle.total_distance_km, sol.summary.cost,
            if is_final { " (final)" } else { "" },
        );
        {
            let mut s = state.write().unwrap();
            s.dataset_json = Some(Arc::new(json));
            s.dataset_iter = iter;
            s.state = if is_final { "done" } else { "evolving" };
            s.phase = if is_final { "done".into() } else { "solve".into() };
            s.progress = if is_final { 1.0 } else {
                0.40 + 0.55 * (elapsed_ms as f32 / total_budget_ms.max(1) as f32).min(1.0)
            };
            s.message = summary;
        }
        warm = Some(sol);
        if is_final { break; }
    }
    Ok(())
}

fn collect_coords(
    problem: &brooom::Problem,
) -> anyhow::Result<(Vec<(f32, f32)>, Vec<Option<usize>>, Vec<Option<usize>>, Vec<usize>)> {
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
        vs.push(s); ve.push(e);
    }
    let mut ji = Vec::with_capacity(problem.jobs.len());
    for j in &problem.jobs {
        let c = j.location.coord.ok_or_else(|| anyhow::anyhow!("job {} missing coord", j.id))?;
        ji.push(push(&mut coords, c));
    }
    Ok((coords, vs, ve, ji))
}

fn rebind_to_indices(
    problem: &mut brooom::Problem,
    vs: &[Option<usize>], ve: &[Option<usize>],
    kept_job_indices: &[usize],
) {
    for (v, vh) in problem.vehicles.iter_mut().enumerate() {
        if let (Some(start), Some(idx)) = (vh.start.as_mut(), vs[v]) {
            start.coord = None; start.index = Some(idx);
        }
        if let (Some(end), Some(idx)) = (vh.end.as_mut(), ve[v]) {
            end.coord = None; end.index = Some(idx);
        }
    }
    for (j, job) in problem.jobs.iter_mut().enumerate() {
        job.location.coord = None;
        job.location.index = Some(kept_job_indices[j]);
    }
}

fn narrow_pos_i32(v: &f32) -> i32 {
    let v = *v;
    if !v.is_finite() || v < 0.0 || v > SENTINEL_I32 as f32 { SENTINEL_I32 } else { v.round() as i32 }
}

// -------------------------------------------------------------------------
// Dataset = the JSON-ready bundle the frontend renders.
// Same shape as mpe-serve's bundle.
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
    iter: u32,
    cost: f64,
    solve_elapsed_s: f64,
    #[serde(rename = "final")]
    is_final: bool,
    vehicles: Vec<VehicleOut>,
    unassigned: Vec<JobPoint>,
    dropped: Vec<u64>,
}
#[derive(Serialize)] struct LatLon { lat: f32, lon: f32 }
#[derive(Serialize)] struct Bbox { lat_min: f32, lat_max: f32, lon_min: f32, lon_max: f32 }
#[derive(Serialize)]
struct VehicleOut {
    id: u64, color: String, n_stops: usize,
    duration_s: i64, distance_m: i64,
    stops: Vec<StopOut>,
}
#[derive(Serialize)]
struct StopOut {
    job_id: u64, lat: f32, lon: f32,
    order: usize, load_after: i64,
}
#[derive(Serialize)]
struct JobPoint { job_id: u64, lat: f32, lon: f32 }

#[allow(clippy::too_many_arguments)]
fn build_dataset(
    problem: &brooom::Problem,
    solution: &brooom::solution::Solution,
    matrix: &brooom::Matrix,
    kept_jobs_latlon: &[(f32, f32)],
    depot_latlon: (f64, f64),
    dropped_ids: &[u64],
    region: &str,
    iter: u32,
    cost: f64,
    solve_elapsed_s: f64,
    is_final: bool,
) -> Dataset {
    use brooom::solution::TaskRef;
    let mut lat_min = depot_latlon.0 as f32; let mut lat_max = lat_min;
    let mut lon_min = depot_latlon.1 as f32; let mut lon_max = lon_min;
    for &(la, lo) in kept_jobs_latlon {
        lat_min = lat_min.min(la); lat_max = lat_max.max(la);
        lon_min = lon_min.min(lo); lon_max = lon_max.max(lo);
    }

    let mut total_dur: i64 = 0;
    let mut total_dist: i64 = 0;
    let mut total_stops: usize = 0;
    let mut vehicles_out: Vec<VehicleOut> = Vec::with_capacity(solution.routes.len());
    for (ri, r) in solution.routes.iter().enumerate() {
        if r.steps.is_empty() { continue; }
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
            let job_idx = match step { TaskRef::Job(j) => *j, _ => continue };
            let here_idx = problem.jobs[job_idx].location.index;
            if let (Some(p), Some(h)) = (prev, here_idx) {
                route_dur += matrix.duration(p, h);
                route_dist += matrix.distance(p, h);
            }
            prev = here_idx;
            let job = &problem.jobs[job_idx];
            let delivered: i64 = job.delivery.iter().sum();
            load += delivered;
            let (lat, lon) = kept_jobs_latlon[job_idx];
            stops.push(StopOut { job_id: job.id, lat, lon, order, load_after: load });
            order += 1;
        }
        if let (Some(p), Some(e)) = (prev, end_idx) {
            route_dur += matrix.duration(p, e);
            route_dist += matrix.distance(p, e);
        }

        total_stops += stops.len();
        total_dur += route_dur;
        total_dist += route_dist;

        let hue = ((ri as f32 * 137.508) % 360.0).abs();
        let color = format!("hsl({:.0},75%,45%)", hue);
        vehicles_out.push(VehicleOut {
            id: vh.id, color, n_stops: stops.len(),
            duration_s: route_dur, distance_m: route_dist, stops,
        });
    }

    let unassigned = solution.unassigned.iter().filter_map(|t| match t {
        TaskRef::Job(j) => {
            let job = &problem.jobs[*j];
            let (lat, lon) = kept_jobs_latlon[*j];
            Some(JobPoint { job_id: job.id, lat, lon })
        }
        _ => None,
    }).collect();

    Dataset {
        region: region.into(),
        depot: LatLon { lat: depot_latlon.0 as f32, lon: depot_latlon.1 as f32 },
        bbox: Bbox { lat_min, lat_max, lon_min, lon_max },
        total_jobs: problem.jobs.len() + dropped_ids.len(),
        total_stops,
        total_duration_h: total_dur as f64 / 3600.0,
        total_distance_km: total_dist as f64 / 1000.0,
        iter, cost, solve_elapsed_s, is_final,
        vehicles: vehicles_out, unassigned, dropped: dropped_ids.to_vec(),
    }
}
