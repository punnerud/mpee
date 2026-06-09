//! PyO3 bindings — `import mpee` from Python and drive the in-process
//! VRP solver. The Python interpreter and the Rust solver share one
//! address space: the JSON bytes returned by `get_dataset_json()` come
//! straight out of the same `Arc<String>` that the solver thread just
//! populated.
//!
//! Build with maturin:
//!   cd crates/mpee-py
//!   python3 -m venv venv && source venv/bin/activate
//!   pip install maturin flask
//!   maturin develop --release
//!   python3 examples/flask_app.py
//!
//! API:
//!   import mpee
//!   eng = mpee.Engine()
//!   eng.start_solve(region="london", n_jobs=500, n_vehicles=20, ...)
//!   while not eng.is_done():
//!       status = json.loads(eng.get_status_json())
//!       ds = eng.get_dataset_json()  # None until first iter completes
//!       time.sleep(1)

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyDict, PyList};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::sync::{Arc, RwLock};

const SENTINEL_I32: i32 = 7 * 24 * 60 * 60;

// -------------------------------------------------------------------------
// Shared state — exactly the same shape as mpee-viz's AppState.
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
    eprintln!("[mpee {:>10}] {}", phase, message);
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
    /// the region's bbox. `max_travel_time_s` and `max_distance_m` cap
    /// individual route length — set to 0 (default) to leave unbounded.
    #[pyo3(signature = (
        region, n_jobs, n_vehicles, capacity, seed, ch, pp,
        time_limit_s=45.0, multi_start=1, radius_km=0.0,
        max_travel_time_s=0i64, max_distance_m=0i64, matrix_budget_mb=500,
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
        max_travel_time_s: i64,
        max_distance_m: i64,
        matrix_budget_mb: u64,
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
                    max_travel_time_s, max_distance_m, matrix_budget_mb,
                };
                if let Err(e) = solve_in_process(&args, &solver_state) {
                    let msg = format!("{:#}", e);
                    eprintln!("[mpee] solver failed: {msg}");
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

// =========================================================================
// Router — the standalone routing API. Loads a prebuilt CH + PP cache and
// answers point-to-point routes, snaps, N×N tables, and multi-stop VRP
// optimisation entirely offline (no network, no external service).
// Mirrors the engine calls the iOS C ABI uses (mpe_route / mpe_snap /
// mpe_solve_vrp / mpe_build_ch).
// =========================================================================

#[pyclass]
struct Router {
    routing: dijeng::routing::RoutingService,
    // Keep the mmap-backed PP cache alive for the Router's lifetime.
    _pp: dijeng::cache_pp::PpFull,
}

/// Wrap a Python callable as a brooom custom constraint. The callable is invoked
/// on every completed candidate route with a dict
/// `{vehicle_id, job_ids, cost, duration_s, distance_m, service_s, waiting_s}`
/// and returns:
///   * `None` / `True`  → feasible
///   * `False`          → infeasible (hard reject — the route is never used)
///   * a number         → soft penalty added to that route's cost
/// A raised exception is printed and treated as feasible, so a buggy constraint
/// can't silently wipe out every route. Note: the callable runs under the GIL on
/// the solver's worker threads, so it is best suited to small/medium instances;
/// registering any constraint also keeps the solve on the CPU (no GPU polish).
fn wrap_py_constraint(cb: Py<PyAny>) -> Arc<brooom::constraint::CustomConstraintFn> {
    use brooom::constraint::Verdict;
    Arc::new(move |view: &brooom::constraint::RouteView| {
        Python::with_gil(|py| {
            let route = PyDict::new_bound(py);
            let _ = route.set_item("vehicle_id", view.vehicle.id);
            let _ = route.set_item("job_ids", view.stop_ids());
            let _ = route.set_item("cost", view.metrics.cost);
            let _ = route.set_item("duration_s", view.metrics.travel_time);
            let _ = route.set_item("distance_m", view.metrics.distance);
            let _ = route.set_item("service_s", view.metrics.service_time);
            let _ = route.set_item("waiting_s", view.metrics.waiting_time);
            match cb.call1(py, (route,)) {
                Ok(ret) => {
                    if ret.is_none(py) {
                        Verdict::Feasible
                    } else if let Ok(b) = ret.extract::<bool>(py) {
                        if b { Verdict::Feasible } else { Verdict::Infeasible }
                    } else if let Ok(p) = ret.extract::<f64>(py) {
                        Verdict::Penalty(p)
                    } else {
                        Verdict::Feasible
                    }
                }
                Err(e) => { e.print(py); Verdict::Feasible }
            }
        })
    })
}

/// Parse the Python `objective=` argument into a [`brooom::ObjectiveMode`].
/// Accepts `None` (default scalar), the string `"scalar"`/`"lexicographic"`, or
/// a list of level-name strings (e.g. `["vehicles", "cost"]`) → lexicographic.
/// Reuses brooom's name→level mapping so every surface agrees on the spelling.
fn parse_objective(py: Python<'_>, objective: Option<Py<PyAny>>) -> PyResult<brooom::ObjectiveMode> {
    let Some(obj) = objective else { return Ok(brooom::ObjectiveMode::Scalar) };
    // A bare string: "scalar" | "lexicographic".
    if let Ok(s) = obj.extract::<String>(py) {
        return match s.trim().to_ascii_lowercase().as_str() {
            "scalar" => Ok(brooom::ObjectiveMode::Scalar),
            "lexicographic" => Ok(brooom::ObjectiveMode::Lexicographic { levels: Vec::new() }),
            other => Err(PyRuntimeError::new_err(format!(
                "objective {other:?} must be \"scalar\", \"lexicographic\", or a list of level names"
            ))),
        };
    }
    // A list of level names → lexicographic in that order.
    if let Ok(names) = obj.extract::<Vec<String>>(py) {
        let levels = names
            .iter()
            .map(|n| brooom::options::lex_objective_from_name(n))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyRuntimeError::new_err(format!("objective: {e}")))?;
        return Ok(brooom::ObjectiveMode::Lexicographic { levels });
    }
    Err(PyRuntimeError::new_err(
        "objective must be a string (\"scalar\"/\"lexicographic\") or a list of level-name strings",
    ))
}

/// Parse the Python `dimensions=` argument (a list of dicts in the
/// `options.dimensions` schema) into compiled [`brooom::CustomDimension`]s by
/// round-tripping through `brooom::SolverOptions` — so the Python surface uses the
/// exact same parser, transit DSL, and defaults as the JSON/CLI surfaces.
fn parse_dimensions(
    py: Python<'_>,
    dimensions: Option<Vec<Py<PyAny>>>,
) -> PyResult<Vec<brooom::CustomDimension>> {
    let Some(dims) = dimensions else { return Ok(Vec::new()) };
    if dims.is_empty() {
        return Ok(Vec::new());
    }
    // Convert each dict to JSON via the `json` module, then assemble an
    // {"dimensions": [...]} options object and reuse brooom's SolverOptions.
    let json_mod = py.import_bound("json")?;
    let mut items = Vec::with_capacity(dims.len());
    for d in dims {
        let s: String = json_mod.call_method1("dumps", (d,))?.extract()?;
        let v: serde_json::Value = serde_json::from_str(&s)
            .map_err(|e| PyRuntimeError::new_err(format!("dimension JSON: {e}")))?;
        items.push(v);
    }
    let opts_val = serde_json::json!({ "dimensions": serde_json::Value::Array(items) });
    let opts = brooom::SolverOptions::from_value(Some(&opts_val))
        .map_err(|e| PyRuntimeError::new_err(format!("dimensions: {e}")))?;
    opts.build_dimensions()
        .map_err(|e| PyRuntimeError::new_err(format!("dimensions: {e}")))
}

#[pymethods]
impl Router {
    /// Open a prebuilt cache pair. `pp_path` is the `.pp` file, `ch_path`
    /// the `.ch` file (build them once with `Router.build(pbf)` or the
    /// `mpee build` CLI). Both are mmap'd, so opening is near-instant and
    /// peak RAM stays low.
    #[new]
    #[pyo3(signature = (pp_path, ch_path = None))]
    fn new(pp_path: &str, ch_path: Option<&str>) -> PyResult<Self> {
        let pp = dijeng::cache_pp::load_mmap(pp_path)
            .map_err(|e| PyRuntimeError::new_err(format!("load PP cache {pp_path}: {e}")))?;
        let n = pp.coords.as_slice().len();
        let coords = dijeng::buffer::Buffer::from(pp.coords.as_slice().to_vec());
        // `ch_path=None` opens a geocoding-only Router (reverse/geocode/
        // intersection/snap) without loading the large `.ch` — handy for a
        // pure geocoding service. routing methods then raise a clear error.
        let mut routing = match ch_path {
            Some(cp) => {
                let ch = dijeng::cache_ch::load_mmap(cp)
                    .map_err(|e| PyRuntimeError::new_err(format!("load CH cache {cp}: {e}")))?;
                dijeng::routing::RoutingService::new(ch, coords)
            }
            None => dijeng::routing::RoutingService::new_geocoding(coords),
        };

        // Auto-attach the street-name sidecar if present (`x.osm.pbf.names`,
        // derived from the `.pp` path), enabling offline geocoding. Absent or
        // mismatched → routing still works, geocoding just returns None.
        let names_path = pp_path
            .strip_suffix(".pp")
            .map(|base| format!("{base}.names"))
            .unwrap_or_else(|| format!("{pp_path}.names"));
        if std::path::Path::new(&names_path).is_file() {
            match dijeng::names::NameTable::load_mmap(&names_path, n) {
                Ok(nt) => routing.set_names(nt),
                Err(e) => eprintln!("[mpee] ignoring names sidecar {names_path}: {e}"),
            }
        }
        // House-number address sidecar (optional; enables address geocoding).
        let addr_path = pp_path
            .strip_suffix(".pp")
            .map(|base| format!("{base}.addr"))
            .unwrap_or_else(|| format!("{pp_path}.addr"));
        if std::path::Path::new(&addr_path).is_file() {
            match dijeng::addresses::AddressIndex::load_mmap(&addr_path) {
                Ok(ai) => routing.set_addresses(ai),
                Err(e) => eprintln!("[mpee] ignoring address sidecar {addr_path}: {e}"),
            }
        }
        Ok(Self { routing, _pp: pp })
    }

    /// Build a `.pp` + `.ch` cache from an OSM `.pbf` extract. This is the
    /// slow, one-time offline preprocessing step (seconds for a city,
    /// minutes for a country). `profile` is "car" | "bicycle" | "foot".
    /// Returns a dict with the output paths and graph size.
    /// `progress` (default True) prints the engine's parse/CH progress to
    /// stdout — pass False to silence it (e.g. when driving the library from
    /// another program). `force=False` reuses an existing `.pp`+`.ch` cache
    /// instead of rebuilding, so repeated calls return instantly.
    #[staticmethod]
    #[pyo3(signature = (pbf, profile = "car", progress = true, force = false, keep_csr = false))]
    fn build<'py>(
        py: Python<'py>,
        pbf: &str,
        profile: &str,
        progress: bool,
        force: bool,
        keep_csr: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        use dijeng::osm_profile::Profile;
        let prof = Profile::from_name(profile).ok_or_else(|| {
            PyRuntimeError::new_err(format!("unknown profile {profile:?} (use car|bicycle|foot)"))
        })?;
        let pbf_owned = pbf.to_string();
        // The whole pipeline (parse → preprocess → CH) runs in-process in the
        // shared `dijeng::build` helper; release the GIL for it. The .csr
        // intermediate is deleted unless keep_csr (routing only needs .pp/.ch).
        let res = py
            .allow_threads(move || {
                dijeng::build::build_cache(std::path::Path::new(&pbf_owned), prof, progress, force, keep_csr)
            })
            .map_err(PyRuntimeError::new_err)?;

        let d = PyDict::new_bound(py);
        d.set_item("pp_path", res.pp_path.to_string_lossy().to_string())?;
        d.set_item("ch_path", res.ch_path.to_string_lossy().to_string())?;
        d.set_item("names_path", res.names_path.to_string_lossy().to_string())?;
        d.set_item("addr_path", res.addr_path.to_string_lossy().to_string())?;
        d.set_item("nodes", res.nodes)?;
        d.set_item("edges", res.edges)?;
        d.set_item("build_secs", res.build_secs)?;
        d.set_item("cached", res.cached)?;
        Ok(d)
    }

    /// Driving route between two (lat, lon) points. Returns a dict with
    /// `distance_m` / `distance_km`, `duration_s` / `duration_min`, the
    /// snapped endpoints, and (if `geometry=True`) the [[lat, lon], …] path.
    #[pyo3(signature = (from_lat, from_lon, to_lat, to_lon, geometry = false))]
    fn route<'py>(
        &self, py: Python<'py>,
        from_lat: f32, from_lon: f32, to_lat: f32, to_lon: f32, geometry: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.require_routing()?;
        let resp = self.routing.route(from_lat, from_lon, to_lat, to_lon)
            .ok_or_else(|| PyRuntimeError::new_err("no route found between the snapped points"))?;
        let d = PyDict::new_bound(py);
        d.set_item("distance_m", resp.distance_m)?;
        d.set_item("distance_km", resp.distance_m / 1000.0)?;
        d.set_item("duration_s", resp.duration_s)?;
        d.set_item("duration_min", resp.duration_s / 60.0)?;
        d.set_item("source_snapped", (resp.source_snapped.0, resp.source_snapped.1))?;
        d.set_item("destination_snapped", (resp.destination_snapped.0, resp.destination_snapped.1))?;
        if geometry {
            let geo: Vec<(f32, f32)> = resp.geometry;
            d.set_item("geometry", geo)?;
        }
        Ok(d)
    }

    /// Snap a (lat, lon) to the nearest routable road node. Returns (lat, lon).
    fn snap(&self, lat: f32, lon: f32) -> (f32, f32) {
        let csr = self.routing.nearest_node(lat, lon);
        self.routing.coords[csr as usize]
    }

    /// Reverse-geocode: the street name nearest to a (lat, lon). Returns the
    /// street name string, or `None` if no `.names` sidecar is loaded or the
    /// nearest road node has no name. The sidecar is produced automatically by
    /// `Router.build` / `mpee build` (no separate indexing step).
    fn reverse(&self, lat: f32, lon: f32) -> Option<String> {
        // Prefer a full "Street 42, 0123 City" address when the .addr sidecar is
        // loaded; otherwise fall back to the nearest street name.
        if let Some(h) = self.routing.reverse_address(lat, lon) {
            let mut s = format!("{} {}", h.street, h.housenumber);
            match (h.postcode, h.city) {
                (Some(pc), Some(c)) => s.push_str(&format!(", {pc} {c}")),
                (None, Some(c)) => s.push_str(&format!(", {c}")),
                (Some(pc), None) => s.push_str(&format!(", {pc}")),
                (None, None) => {}
            }
            return Some(s);
        }
        self.routing.reverse(lat, lon).map(|s| s.to_string())
    }

    /// Forward-geocode: look up a street by name (case-insensitive; a
    /// substring matches, e.g. `"karl johan"` finds `"Karl Johans gate"`).
    /// Returns a dict `{"name", "lat", "lon"}` for the matched street, or
    /// `None` if nothing matches / no sidecar is loaded.
    ///
    /// On a multi-city cache the same name exists in several towns; pass
    /// `near=(lat, lon)` to return the match nearest that point (e.g. to pick
    /// "Munkegata in Trondheim" rather than an arbitrary first hit).
    #[pyo3(signature = (query, near = None))]
    fn geocode<'py>(
        &self, py: Python<'py>, query: &str, near: Option<(f32, f32)>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        // If the query carries a house number ("Storgata 42") and an address
        // sidecar is loaded, resolve to address level; the dict then also has
        // "housenumber"/"city"/"postcode"/"approximate".
        let (street_part, number) = dijeng::routing::split_house_number(query);
        if number.is_some() {
            if let Some(h) = self.routing.address_query(query, near) {
                let d = PyDict::new_bound(py);
                d.set_item("name", h.street)?;
                d.set_item("housenumber", h.housenumber)?;
                d.set_item("lat", h.lat)?;
                d.set_item("lon", h.lon)?;
                d.set_item("city", h.city)?;
                d.set_item("postcode", h.postcode)?;
                d.set_item("approximate", h.approximate)?;
                return Ok(Some(d));
            }
        }
        // Street-level fallback (strip the number if one was present).
        let q = if number.is_some() { street_part } else { query };
        let hit = match near {
            Some((la, lo)) => self.routing.geocode_near(q, la, lo),
            None => self.routing.geocode(q),
        };
        match hit {
            Some((lat, lon, name)) => {
                let d = PyDict::new_bound(py);
                d.set_item("name", name)?;
                d.set_item("lat", lat)?;
                d.set_item("lon", lon)?;
                Ok(Some(d))
            }
            None => Ok(None),
        }
    }

    /// Forward-geocode a street + house number explicitly. Returns a dict
    /// `{"name","housenumber","lat","lon","city","postcode","approximate"}` or
    /// `None`. `near=(lat,lon)` disambiguates a name shared by several towns.
    /// Exact number → `approximate=False`; if missing, the nearest number on the
    /// street is returned with `approximate=True`.
    #[pyo3(signature = (street, number, near = None))]
    fn geocode_address<'py>(
        &self, py: Python<'py>, street: &str, number: &str, near: Option<(f32, f32)>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.routing.geocode_address(street, number, near) {
            Some(h) => {
                let d = PyDict::new_bound(py);
                d.set_item("name", h.street)?;
                d.set_item("housenumber", h.housenumber)?;
                d.set_item("lat", h.lat)?;
                d.set_item("lon", h.lon)?;
                d.set_item("city", h.city)?;
                d.set_item("postcode", h.postcode)?;
                d.set_item("approximate", h.approximate)?;
                Ok(Some(d))
            }
            None => Ok(None),
        }
    }

    /// Intersection search: every coordinate where two streets meet. Returns a
    /// list of `{"lat", "lon"}` dicts (often one; several when the streets
    /// cross more than once or share a stretch). Empty if no sidecar is loaded,
    /// a name doesn't resolve, or the streets share no node. Names are matched
    /// case-insensitively (substring), e.g. `("Karl Johans gate", "Kongens gate")`.
    ///
    /// On a multi-city cache, pass `near=(lat, lon)` to sort crossings
    /// nearest-first to that point, and `radius_km` to keep only those within
    /// it (e.g. "… near Trondheim" instead of every same-named crossing).
    #[pyo3(signature = (a, b, near = None, radius_km = None))]
    fn intersection<'py>(
        &self, py: Python<'py>, a: &str, b: &str,
        near: Option<(f32, f32)>, radius_km: Option<f64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let hits = match near {
            Some((la, lo)) => self.routing.intersection_near(a, b, la, lo, radius_km),
            None => self.routing.intersection(a, b),
        };
        let out = PyList::empty_bound(py);
        for (lat, lon) in hits {
            let d = PyDict::new_bound(py);
            d.set_item("lat", lat)?;
            d.set_item("lon", lon)?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// Whether a street-name sidecar is loaded (i.e. geocoding is available).
    fn has_names(&self) -> bool {
        self.routing.has_names()
    }

    /// Whether a house-number address sidecar is loaded (address geocoding).
    fn has_addresses(&self) -> bool {
        self.routing.has_addresses()
    }

    /// Whether a `.ch` cache is loaded — i.e. `route`/`optimize`/`solve`/`table`
    /// are available. False when the Router was opened geocoding-only as
    /// `Router(pp_path)` (no `ch_path`).
    fn has_routing(&self) -> bool {
        self.routing.has_routing()
    }

    /// Bounding box of the loaded graph as a dict
    /// `{"min_lat", "min_lon", "max_lat", "max_lon"}`.
    fn bbox<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (mut mn_la, mut mx_la, mut mn_lo, mut mx_lo) =
            (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        for &(la, lo) in self.routing.coords.as_slice() {
            if la.is_finite() && lo.is_finite() {
                mn_la = mn_la.min(la); mx_la = mx_la.max(la);
                mn_lo = mn_lo.min(lo); mx_lo = mx_lo.max(lo);
            }
        }
        if !mn_la.is_finite() { return Err(PyRuntimeError::new_err("empty graph")); }
        let d = PyDict::new_bound(py);
        d.set_item("min_lat", mn_la)?; d.set_item("min_lon", mn_lo)?;
        d.set_item("max_lat", mx_la)?; d.set_item("max_lon", mx_lo)?;
        Ok(d)
    }

    /// Full duration+distance table among `points` (list of (lat, lon)).
    /// Returns {"n", "durations_s": flat n×n, "distances_m": flat n×n}.
    #[pyo3(signature = (points, matrix_budget_mb = 500))]
    fn table<'py>(
        &self,
        py: Python<'py>,
        points: Vec<(f32, f32)>,
        matrix_budget_mb: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.require_routing()?;
        let budget = dijeng::budget::resolve_matrix_budget_mb(matrix_budget_mb);
        let (durs, dists, _, _) =
            self.routing
                .matrix_with_distance_budgeted_full(&points, &points, budget);
        let d = PyDict::new_bound(py);
        d.set_item("n", points.len())?;
        d.set_item("durations_s", durs)?;
        d.set_item("distances_m", dists)?;
        Ok(d)
    }

    /// Optimise a multi-stop delivery plan (VRP) over `stops` (list of
    /// (lat, lon)). Vehicles start and end at `depot` (defaults to the
    /// centroid of the stops). Returns a dict with one entry per used
    /// vehicle (ordered stops + leg distances) plus totals and any
    /// unassigned stops.
    ///
    /// `objective` selects the optimisation mode: `None`/`"scalar"` (default,
    /// today's behaviour), `"lexicographic"`, or a list of level names
    /// (`["vehicles", "cost"]`, `["unassigned", "vehicles", "cost"]`, …) for
    /// N-level lexicographic search (levels: `unassigned`, `vehicles`, `cost`,
    /// `makespan`, `distance`). `dimensions` is a list of custom accumulator
    /// dimensions in the JSON `options.dimensions` schema — each a dict with a
    /// `name`, a pyspell `transit` expression over the arc context (`distance`,
    /// `duration`, `cumul`/`cumul_before`, `from`, `to`, `arrival`), and optional
    /// `start`/`min`/`max`/`soft_max`/`soft_min`/`soft_weight`/`monotonicity`.
    #[pyo3(signature = (
        stops, vehicles = 1, capacity = 1_000_000i64, depot = None,
        time_limit_s = 5.0, use_gpu = false, objective = None, dimensions = None,
        soft_tw = None, matrix_budget_mb = 500,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn optimize<'py>(
        &self, py: Python<'py>,
        stops: Vec<(f32, f32)>, vehicles: usize, capacity: i64,
        depot: Option<(f32, f32)>, time_limit_s: f64, use_gpu: bool,
        objective: Option<Py<PyAny>>, dimensions: Option<Vec<Py<PyAny>>>,
        soft_tw: Option<bool>, matrix_budget_mb: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.require_routing()?;
        if stops.is_empty() { return Err(PyRuntimeError::new_err("no stops given")); }
        if vehicles == 0 { return Err(PyRuntimeError::new_err("vehicles must be >= 1")); }

        // Map the optional objective/dimensions args to the same engine surfaces
        // the JSON/CLI paths use. Default (both None) reproduces today's behaviour.
        let objective_mode = parse_objective(py, objective)?;
        let dims = parse_dimensions(py, dimensions)?;
        // Install the custom dimensions for the duration of this solve; the guard
        // clears the global registry on drop. Empty list ⇒ a no-op (guard is None).
        let _dim_guard = if dims.is_empty() {
            None
        } else {
            Some(brooom::DimensionGuard::install(dims))
        };

        // Depot defaults to the centroid of the stops.
        let depot = depot.unwrap_or_else(|| {
            let n = stops.len() as f32;
            let (sla, slo) = stops.iter().fold((0.0f32, 0.0f32), |(a, b), &(la, lo)| (a + la, b + lo));
            (sla / n, slo / n)
        });

        // coords[0] = depot, coords[1..=N] = stops (lat, lon).
        let mut coords: Vec<(f32, f32)> = Vec::with_capacity(stops.len() + 1);
        coords.push(depot);
        coords.extend_from_slice(&stops);

        // Build the matrix + snapped coords off the GIL.
        let n = coords.len();
        let budget = dijeng::budget::resolve_matrix_budget_mb(matrix_budget_mb);
        let (durs_f, dists_f, snapped, sol_routes, unassigned, solve_s) =
            py.allow_threads(|| {
                let (durs_f, dists_f, snapped, _) =
                    self.routing
                        .matrix_with_distance_budgeted_full(&coords, &coords, budget);
                let to_i = |v: &[f32]| -> Vec<i32> {
                    v.iter()
                        .map(|&d| if d.is_finite() { d.round().max(0.0) as i32 } else { i32::MAX / 2 })
                        .collect()
                };
                let matrix = brooom::Matrix {
                    n,
                    durations: to_i(&durs_f),
                    distances: Some(to_i(&dists_f)),
                };

                let mut problem = brooom::Problem::default();
                for v in 0..vehicles {
                    problem.vehicles.push(brooom::Vehicle {
                        id: (v + 1) as u64,
                        start: Some(brooom::Location::from_index(0)),
                        end: Some(brooom::Location::from_index(0)),
                        capacity: vec![capacity],
                        skills: vec![], time_window: None, speed_factor: 1.0,
                        max_tasks: None, max_travel_time: None, max_distance: None,
                        fixed: 0.0, per_hour: 3600.0,
                        span_cost: 0.0, distance_weight: 0.0, time_weight: 1.0,
                        profile: "car".into(),
                        breaks: vec![], max_trips: 1, description: None,
                    });
                }
                for i in 0..stops.len() {
                    problem.jobs.push(brooom::Job {
                        id: (i + 1) as u64,
                        location: brooom::Location::from_index(i + 1),
                        kind: Default::default(), service: 0, setup: 0, release: 0,
                        delivery: vec![1], pickup: vec![], skills: vec![], allowed_vehicles: None, priority: 0,
                        time_windows: vec![], prize: brooom::problem::DEFAULT_PRIZE,
                        disjunction_penalty: None, group: None,
                        description: None,
                    });
                }
                let cfg = brooom::solver::SolverConfig {
                    multi_start: 4,
                    granular_k: Some(40),
                    max_local_search_passes: 50,
                    time_limit_ms: Some((time_limit_s * 1000.0) as u64),
                    verbose: false,
                    use_gpu,
                    objective_mode: objective_mode.clone(),
                    // Penalty-managed soft constraints: None ⇒ AUTO (on when the
                    // problem has time windows), Some(_) forces it on/off.
                    soft_search: soft_tw,
                    ..Default::default()
                };
                let t = std::time::Instant::now();
                let sol = brooom::solver::solve_with_matrix(&problem, &matrix, &cfg);
                let elapsed = t.elapsed().as_secs_f64();

                // Flatten routes to (vehicle_id, [(stop_index, leg_dist_m, leg_dur_s)], total_d, total_t).
                let dist = matrix.distances.as_ref().unwrap();
                let mut routes_out: Vec<(u64, Vec<(usize, i32, i32)>, i64, i64)> = Vec::new();
                for r in &sol.routes {
                    if r.steps.is_empty() { continue; }
                    let vid = problem.vehicles[r.vehicle_idx].id;
                    let mut legs: Vec<(usize, i32, i32)> = Vec::with_capacity(r.steps.len());
                    let (mut td, mut tt) = (0i64, 0i64);
                    let mut prev = 0usize; // depot index
                    for step in &r.steps {
                        let mi = match step { brooom::solution::TaskRef::Job(j) => *j + 1, _ => continue };
                        let ld = matrix.durations[prev * n + mi];
                        let dm = dist[prev * n + mi];
                        td += dm as i64; tt += ld as i64;
                        legs.push((mi, dm, ld));
                        prev = mi;
                    }
                    // return leg to depot
                    td += dist[prev * n] as i64; tt += matrix.durations[prev * n] as i64;
                    routes_out.push((vid, legs, td, tt));
                }
                let unassigned: Vec<usize> = sol.unassigned.iter().filter_map(|t| match t {
                    brooom::solution::TaskRef::Job(j) => Some(*j), _ => None,
                }).collect();
                (durs_f, dists_f, snapped, routes_out, unassigned, elapsed)
            });
        let _ = durs_f; // matrix already consumed for output; dists_f used below

        // Assemble the Python result.
        let routes_list = PyList::empty_bound(py);
        let (mut grand_d, mut grand_t, mut grand_stops) = (0i64, 0i64, 0usize);
        for (vid, legs, td, tt) in &sol_routes {
            let rd = PyDict::new_bound(py);
            rd.set_item("vehicle_id", *vid)?;
            rd.set_item("n_stops", legs.len())?;
            rd.set_item("distance_m", *td)?;
            rd.set_item("distance_km", *td as f64 / 1000.0)?;
            rd.set_item("duration_s", *tt)?;
            rd.set_item("duration_min", *tt as f64 / 60.0)?;
            let stops_list = PyList::empty_bound(py);
            for (order, (mi, leg_d, leg_t)) in legs.iter().enumerate() {
                let (la, lo) = snapped[*mi];
                let sd = PyDict::new_bound(py);
                sd.set_item("order", order)?;
                sd.set_item("stop_index", *mi - 1)?; // index into the input `stops`
                sd.set_item("lat", la)?;
                sd.set_item("lon", lo)?;
                sd.set_item("leg_distance_m", *leg_d)?;
                sd.set_item("leg_duration_s", *leg_t)?;
                stops_list.append(sd)?;
            }
            rd.set_item("stops", stops_list)?;
            routes_list.append(rd)?;
            grand_d += *td; grand_t += *tt; grand_stops += legs.len();
        }

        let out = PyDict::new_bound(py);
        out.set_item("routes", routes_list)?;
        out.set_item("vehicles_used", sol_routes.len())?;
        out.set_item("total_stops", grand_stops)?;
        out.set_item("total_distance_m", grand_d)?;
        out.set_item("total_distance_km", grand_d as f64 / 1000.0)?;
        out.set_item("total_duration_s", grand_t)?;
        out.set_item("total_duration_min", grand_t as f64 / 60.0)?;
        out.set_item("depot", (depot.0, depot.1))?;
        // Categorise each unassigned stop by *why* it couldn't be placed.
        let unassigned_detail = PyList::empty_bound(py);
        for &jidx in &unassigned {
            let mi = jidx + 1; // depot is index 0, stops are 1..=N
            let reason = if dists_f[mi] >= 1e8_f32 || dists_f[mi * n] >= 1e8_f32 {
                "unreachable" // no road connects this stop to the depot
            } else if 1i64 > capacity {
                "exceeds_capacity"
            } else {
                "no_room" // reachable & fits, but the fleet ran out of capacity
            };
            let sd = PyDict::new_bound(py);
            sd.set_item("stop_index", jidx)?;
            sd.set_item("reason", reason)?;
            unassigned_detail.append(sd)?;
        }
        out.set_item("unassigned", unassigned)?;
        out.set_item("unassigned_detail", unassigned_detail)?;
        out.set_item("solve_s", solve_s)?;
        Ok(out)
    }

    /// Solve a full VROOM-compatible problem given as a JSON string. Unlike
    /// `optimize()`, this exposes the engine's whole constraint model.
    ///
    /// Per vehicle: capacity (multi-dim), `skills`, `time_window`, `speed_factor`
    /// (mixed fleets), `max_travel_time`/`max_distance`/`max_tasks`, distinct
    /// `start`/`end`, `breaks` (`{id, service, time_windows:[[s,e]]}`), and
    /// `max_trips` (>1 = multi-trip/reloading). Per job: `delivery`/`pickup`
    /// (multi-dim; a pickup-only job is a backhaul), `skills`, `time_windows`,
    /// `service`, `setup`, `priority`, `release` (earliest service, s), `prize`
    /// (finite ⇒ optional/prize-collecting), and `group` (visit exactly one).
    ///
    /// Keyword args: `constraints=[...]` (DSL strings and/or Python callables),
    /// `max_vehicles`, `fairness_weight` + `fairness_metric` ("duration"/"load"),
    /// `use_gpu`, `time_limit_s`, `objective` (`"scalar"` | `"lexicographic"` | a
    /// list of level names) and `dimensions` (list of dicts in the JSON
    /// `options.dimensions` schema). (Top-level `shipments` are not yet routed by
    /// this binding — model each half as a job, or use the Rust API.)
    ///
    /// Locations carry `[lon, lat]` coords and
    /// are snapped + turned into a routing matrix here. Returns a dict with
    /// one entry per used vehicle (ordered job_ids + coords + leg metrics),
    /// plus any unassigned job ids.
    #[pyo3(signature = (problem_json, time_limit_s = 5.0, use_gpu = false, constraints = None, max_vehicles = None, fairness_weight = 0.0, fairness_metric = "duration", objective = None, dimensions = None, soft_tw = None, balance_spread = None, group_cardinality = None, propagate = true, matrix_budget_mb = 500))]
    #[allow(clippy::too_many_arguments)]
    fn solve<'py>(
        &self, py: Python<'py>, problem_json: &str, time_limit_s: f64, use_gpu: bool,
        constraints: Option<Vec<Py<PyAny>>>,
        max_vehicles: Option<usize>, fairness_weight: f64, fairness_metric: &str,
        objective: Option<Py<PyAny>>, dimensions: Option<Vec<Py<PyAny>>>,
        soft_tw: Option<bool>,
        balance_spread: Option<i64>, group_cardinality: Option<(u32, u32)>,
        propagate: bool, matrix_budget_mb: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.require_routing()?;
        let mut problem: brooom::Problem = serde_json::from_str(problem_json)
            .map_err(|e| PyRuntimeError::new_err(format!("problem JSON: {e}")))?;
        problem.validate().map_err(|e| PyRuntimeError::new_err(format!("invalid problem: {e}")))?;

        // Map optional objective/dimensions to the engine surfaces (same parser as
        // the JSON/CLI paths). Both None ⇒ today's behaviour. The dimension guard
        // is held until method end, scoping the registration to this solve.
        let objective_mode = parse_objective(py, objective)?;
        let dims = parse_dimensions(py, dimensions)?;
        let _dim_guard = if dims.is_empty() {
            None
        } else {
            Some(brooom::DimensionGuard::install(dims))
        };

        // The Python binding routes only `jobs` (it snaps/indexes jobs + vehicle
        // depots, and the result builder surfaces job stops). Top-level VROOM
        // `shipments` are supported by the Rust core but not yet wired through
        // this binding — error loudly rather than silently dropping them.
        if !problem.shipments.is_empty() {
            return Err(PyRuntimeError::new_err(
                "shipments (paired pickup→delivery) aren't yet routed by the Python solve(); \
                 model each half as a job for now, or use the brooom Rust API which supports \
                 shipments natively",
            ));
        }

        let fairness_metric = match fairness_metric {
            "load" => brooom::FairnessMetric::Load,
            "duration" => brooom::FairnessMetric::Duration,
            other => return Err(PyRuntimeError::new_err(format!(
                "fairness_metric must be \"duration\" or \"load\", got {other:?}"
            ))),
        };

        // Install any custom constraints (code) for the duration of this solve.
        // Each item is either a Python callable (invoked per route) or a string
        // in the constraint DSL (compiled once to native IR — far faster, and it
        // can be mirrored into the insertion probe). The guard clears the global
        // registry when it drops at method end.
        let _cguard = match constraints {
            None => None,
            Some(cs) => {
                let mut closures = Vec::with_capacity(cs.len());
                let mut bounds = Vec::new();
                for obj in cs {
                    if let Ok(src) = obj.extract::<String>(py) {
                        let (c, b) = brooom::pyspell::compiled_python(&src)
                            .map_err(|e| PyRuntimeError::new_err(format!("constraint: {e}")))?;
                        if let Some(b) = b {
                            bounds.push(b);
                        }
                        closures.push(c);
                    } else {
                        closures.push(wrap_py_constraint(obj));
                    }
                }
                brooom::constraint::set_probe_bounds(bounds);
                Some(brooom::constraint::ConstraintGuard::install(closures))
            }
        };

        // Snap every vehicle start/end + job coord, build the matrix, and
        // rewrite the problem's Locations to matrix indices (off the GIL).
        let budget = dijeng::budget::resolve_matrix_budget_mb(matrix_budget_mb);
        let (sol, ji, vs, ve, snapped, n, dur_i, dist_i, solve_s) =
            py.allow_threads(|| -> Result<_, String> {
                let (coords, vs, ve, ji) = collect_coords(&problem).map_err(|e| e.to_string())?;
                let n = coords.len();
                let (dur_i, dist_i, snapped, _) = self.routing.matrix_with_distance_budgeted_mapped(
                    &coords,
                    &coords,
                    budget,
                    SENTINEL_I32,
                    |v| narrow_pos_i32(&v),
                );
                let matrix = brooom::Matrix {
                    n,
                    durations: dur_i.clone(),
                    distances: Some(dist_i.clone()),
                };
                rebind_to_indices(&mut problem, &vs, &ve, &ji);
                let cfg = brooom::solver::SolverConfig {
                    multi_start: 4,
                    granular_k: Some(40),
                    max_local_search_passes: 50,
                    time_limit_ms: Some((time_limit_s * 1000.0) as u64),
                    verbose: false,
                    use_gpu,
                    max_vehicles,
                    fairness_weight,
                    fairness_metric,
                    objective_mode: objective_mode.clone(),
                    // Penalty-managed soft constraints: None ⇒ AUTO (on when the
                    // problem has time windows), Some(_) forces it on/off.
                    soft_search: soft_tw,
                    // HARD balance cap + k-of-N group cardinality (constraint parity).
                    balance_spread,
                    group_cardinality,
                    propagate,
                    ..Default::default()
                };
                // mpee-py calls solve_with_matrix directly (bypassing solve_full),
                // so run the propagation pre-pass here too when enabled.
                if cfg.propagate {
                    let soft = cfg.soft_search.unwrap_or_else(|| {
                        problem.jobs.iter().any(|j| {
                            j.time_windows.iter().any(|w| w != &brooom::problem::TimeWindow::FOREVER)
                        })
                    });
                    let _ = brooom::propagate::tighten(&mut problem, &matrix, soft);
                }
                let t = std::time::Instant::now();
                let sol = brooom::solver::solve_with_matrix(&problem, &matrix, &cfg);
                Ok((sol, ji, vs, ve, snapped, n, dur_i, dist_i, t.elapsed().as_secs_f64()))
            }).map_err(PyRuntimeError::new_err)?;

        // Assemble the Python result, keyed by the caller's job ids.
        let routes_list = PyList::empty_bound(py);
        let (mut grand_d, mut grand_t, mut grand_stops) = (0i64, 0i64, 0usize);
        for r in &sol.routes {
            if r.steps.is_empty() { continue; }
            let v = r.vehicle_idx;
            let vid = problem.vehicles[v].id;
            let start_mi = vs.get(v).copied().flatten();
            let end_mi = ve.get(v).copied().flatten();
            let rd = PyDict::new_bound(py);
            let stops_list = PyList::empty_bound(py);
            let (mut td, mut tt) = (0i64, 0i64);
            let mut prev = start_mi;
            for (order, step) in r.steps.iter().enumerate() {
                let jidx = match step { brooom::solution::TaskRef::Job(j) => *j, _ => continue };
                let mi = ji[jidx];
                if let Some(p) = prev {
                    td += dist_i[p * n + mi] as i64;
                    tt += dur_i[p * n + mi] as i64;
                }
                prev = Some(mi);
                let (la, lo) = snapped[mi];
                let sd = PyDict::new_bound(py);
                sd.set_item("order", order)?;
                sd.set_item("job_id", problem.jobs[jidx].id)?;
                sd.set_item("lat", la)?;
                sd.set_item("lon", lo)?;
                stops_list.append(sd)?;
            }
            if let (Some(p), Some(e)) = (prev, end_mi) {
                td += dist_i[p * n + e] as i64;
                tt += dur_i[p * n + e] as i64;
            }
            rd.set_item("vehicle_id", vid)?;
            rd.set_item("n_stops", r.steps.len())?;
            rd.set_item("distance_m", td)?;
            rd.set_item("distance_km", td as f64 / 1000.0)?;
            rd.set_item("duration_s", tt)?;
            rd.set_item("duration_min", tt as f64 / 60.0)?;
            rd.set_item("stops", stops_list)?;
            routes_list.append(rd)?;
            grand_d += td; grand_t += tt; grand_stops += r.steps.len();
        }
        let unassigned: Vec<u64> = sol.unassigned.iter().filter_map(|t| match t {
            brooom::solution::TaskRef::Job(j) => Some(problem.jobs[*j].id), _ => None,
        }).collect();
        // Categorise each unassigned job by *why* it couldn't be placed.
        let unassigned_detail = PyList::empty_bound(py);
        for t in &sol.unassigned {
            if let brooom::solution::TaskRef::Job(j) = t {
                let reason = unassigned_reason(
                    &problem.jobs[*j], ji[*j], n, &dist_i, &problem.vehicles, &vs, &ve,
                );
                let sd = PyDict::new_bound(py);
                sd.set_item("job_id", problem.jobs[*j].id)?;
                sd.set_item("reason", reason)?;
                unassigned_detail.append(sd)?;
            }
        }

        let out = PyDict::new_bound(py);
        let vehicles_used = routes_list.len();
        out.set_item("routes", routes_list)?;
        out.set_item("vehicles_used", vehicles_used)?;
        out.set_item("total_stops", grand_stops)?;
        out.set_item("total_distance_m", grand_d)?;
        out.set_item("total_distance_km", grand_d as f64 / 1000.0)?;
        out.set_item("total_duration_s", grand_t)?;
        out.set_item("total_duration_min", grand_t as f64 / 60.0)?;
        out.set_item("unassigned", unassigned)?;
        out.set_item("unassigned_detail", unassigned_detail)?;
        out.set_item("solve_s", solve_s)?;
        Ok(out)
    }
}

/// Distance (metres) at or above which a matrix cell is treated as "no road"
/// (the routing engine's unreachable sentinel is ~2.1e9).
const UNREACHABLE_I32: i32 = 100_000_000;

/// Classify *why* a job ended up unassigned, for the categorized
/// `unassigned_detail` output. `dist` is the row-major n×n distance matrix;
/// `vstart`/`vend` are each vehicle's matrix index for its start/end.
fn unassigned_reason(
    job: &brooom::Job,
    mi: usize,
    n: usize,
    dist: &[i32],
    vehicles: &[brooom::Vehicle],
    vstart: &[Option<usize>],
    vend: &[Option<usize>],
) -> &'static str {
    let req = &job.skills;
    let (mut reachable, mut skilled, mut fits) = (false, false, false);
    for v in 0..vehicles.len() {
        let veh = &vehicles[v];
        let has = veh.has_skills(req);
        skilled |= has;
        let reach_v = matches!(
            (vstart.get(v).copied().flatten(), vend.get(v).copied().flatten()),
            (Some(s), Some(e)) if dist[s * n + mi] < UNREACHABLE_I32 && dist[mi * n + e] < UNREACHABLE_I32
        );
        reachable |= reach_v;
        let cap_ok = job
            .delivery
            .iter()
            .enumerate()
            .all(|(i, &d)| d <= veh.capacity.get(i).copied().unwrap_or(0));
        if has && reach_v && cap_ok {
            fits = true;
        }
    }
    if !reachable {
        "unreachable" // no road connects this stop to any vehicle's depot
    } else if !skilled {
        "missing_skill" // no vehicle has the required skill(s)
    } else if !fits {
        "exceeds_capacity" // larger than any single capable+reachable vehicle
    } else {
        "no_room" // serviceable alone, but the fleet/time windows had no room left
    }
}

impl Router {
    /// Error out if this Router was opened geocoding-only (no `.ch`), so the
    /// routing methods give a clear message instead of a confusing failure.
    fn require_routing(&self) -> PyResult<()> {
        if self.routing.has_routing() {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(
                "this Router was opened without a .ch cache (geocoding-only); \
                 routing/optimization need it — open Router(pp_path, ch_path)",
            ))
        }
    }
}

#[pymodule]
fn _mpee(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<Router>()?;
    Ok(())
}

// -------------------------------------------------------------------------
// solver — mirrors mpee-viz's solve_in_process. The boundary between
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
    max_travel_time_s: i64,
    max_distance_m: i64,
    matrix_budget_mb: u64,
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
            service: 60, setup: 0, release: 0,
            delivery: vec![delivery], pickup: vec![],
            skills: vec![], allowed_vehicles: None, priority: 0,
            time_windows: vec![], prize: brooom::problem::DEFAULT_PRIZE,
            disjunction_penalty: None, group: None,
            description: None,
        });
    }
    let max_tt = if args.max_travel_time_s > 0 { Some(args.max_travel_time_s) } else { None };
    let max_d  = if args.max_distance_m > 0 { Some(args.max_distance_m) } else { None };
    for v in 0..args.n_vehicles {
        problem.vehicles.push(brooom::Vehicle {
            id: (v + 1) as u64,
            start: Some(brooom::Location::from_coord(depot_lon, depot_lat)),
            end:   Some(brooom::Location::from_coord(depot_lon, depot_lat)),
            capacity: vec![args.capacity],
            skills: vec![], time_window: None,
            speed_factor: 1.0,
            max_tasks: None,
            max_travel_time: max_tt,
            max_distance: max_d,
            fixed: 0.0, per_hour: 3600.0,
            span_cost: 0.0, distance_weight: 0.0, time_weight: 1.0,
            profile: "car".into(), breaks: vec![], max_trips: 1, description: None,
        });
    }

    set_phase(state, "mmap", "mmap CH + PP caches", 0.10);
    let pp = dijeng::cache_pp::load_mmap(&args.pp)
        .with_context(|| format!("load PP cache {}", args.pp))?;
    let ch = dijeng::cache_ch::load_mmap(&args.ch)
        .with_context(|| format!("load CH cache {}", args.ch))?;

    let (coords, vehicle_starts, vehicle_ends, job_indices) = collect_coords(&problem)?;
    let n_points = coords.len();
    set_phase(state, "snap", &format!("{} coords ready", n_points), 0.15);

    let svc = dijeng::routing::RoutingService::new(ch, pp.coords);
    let matrix_budget_mb =
        dijeng::budget::resolve_matrix_budget_mb(args.matrix_budget_mb);
    set_phase(
        state,
        "matrix",
        &format!(
            "building {n_points}×{n_points} routing matrix (budget={matrix_budget_mb} MB)"
        ),
        0.20,
    );
    let t = std::time::Instant::now();
    let (durations, distances, _, _) = svc.matrix_with_distance_budgeted_mapped(
        &coords,
        &coords,
        matrix_budget_mb,
        SENTINEL_I32,
        |v| narrow_pos_i32(&v),
    );
    let mmm_secs = t.elapsed().as_secs_f64();
    set_phase(
        state, "matrix",
        &format!("matrix built in {:.2}s ({:.1} M cells/s)",
            mmm_secs, (n_points as u64).pow(2) as f64 / mmm_secs / 1e6),
        0.30,
    );

    let matrix = brooom::Matrix {
        n: n_points,
        durations,
        distances: Some(distances),
    };

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

    // (Sweep-based warm-start was tried and reverted: brooom's local-search
    // does not recompute Solution.summary or Route.metrics from a manually
    // constructed warm_start — it consumes them as-is, which made the LS
    // think the warm-start was already at cost 0 and refuse to budge. Left
    // `build_sweep_warm_start` in the file for reference; re-enable once
    // brooom exposes a `recompute_metrics(&mut Solution, &Matrix)` hook.)
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
        // With sweep warm-start, iter 1 doesn't need 50 LS passes — sweep
        // is already a strong local optimum. Bumping granular_k from
        // 20 → 40 lets brooom's LS consider twice as many candidate
        // swap partners per customer, which is what unsticks the
        // "long cross-route segments" that visually screamed at us
        // when granular was too small for N≥1000.
        let max_passes = if iter == 1 { 10 } else { 50 };
        let cfg = brooom::solver::SolverConfig {
            multi_start: if iter == 1 { args.multi_start.max(1) } else { 1 },
            time_limit_ms: Some(chunk_ms),
            warm_start: warm.clone(),
            max_local_search_passes: max_passes,
            granular_k: Some(40),
            verbose: false,
            ..Default::default()
        };
        let msg = if iter == 1 {
            format!(
                "iter 1 · initial insertion + LS (N={}, granular K=40)",
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

        // Geometric-crossing post-pass: brooom's granular LS often misses
        // pairs of cross-route segments whose endpoints aren't in each
        // other's K-nearest sets. Detecting them by literal segment
        // intersection on lat/lon and swapping the suffixes uncrosses
        // the visual mess and usually drops a few percent of distance.
        // The post-processed solution is only used for the rendering
        // bundle below; brooom keeps its own (un-postprocessed) `sol`
        // as the warm-start for the next iter so its internal state
        // and metrics stay consistent.
        let mut pub_sol = sol.clone();
        let n_swaps = uncross_pass(&mut pub_sol, &problem, &matrix, &kept_jobs_latlon);
        let n_relocs = relocate_pass(&mut pub_sol, &problem, &matrix);
        let n_2opts = intra_route_2opt_pass(&mut pub_sol, &problem, &matrix);
        if n_swaps + n_relocs + n_2opts > 0 {
            eprintln!(
                "[mpee     fixup] iter {iter}: {n_swaps} 2-opt* swap(s) + {n_relocs} cross-route relocate(s) + {n_2opts} intra-route 2-opt(s)"
            );
        }

        // Feed the post-processed (un-crossed + relocated) solution
        // back to brooom as warm-start for the next iter. brooom's LS
        // operators compute Δ-cost from the matrix on the fly, so the
        // stale Solution.summary inherited from a manually edited
        // warm-start doesn't matter for search quality — only the
        // route topology counts. This way our cross-route fixes
        // compound across iters instead of being rediscovered every
        // time on top of an identical brooom output.
        let warm_next = pub_sol.clone();

        let bundle = build_dataset(
            &problem, &pub_sol, &matrix, &kept_jobs_latlon,
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
        warm = Some(warm_next);
        if is_final { break; }
    }
    Ok(())
}

// -------------------------------------------------------------------------
// Inter-route 2-opt* on geometric crossings.
//
// brooom's granular LS considers K=40 nearest neighbours per customer.
// When two routes happen to "cross" geographically (their stop-to-stop
// polylines literally intersect on the map), the operator pair that
// would untangle them often isn't in each other's K-NN set — neither
// endpoint of the crossing segment is among the other endpoint's 40
// graph-nearest customers. So those crossings survive every iter,
// producing the "long routes across many other routes" visual the user
// flagged.
//
// This pass enumerates every pair of routes, walks consecutive
// (lat,lon) segments in both, and looks for literal segment-segment
// intersections (cross-product orientation test on the lat/lon plane;
// fine for ≤30 km radii, where the projection error is millimetres).
// When found, it tries the 2-opt* suffix swap and applies it only if
// (a) capacity holds for both vehicles and (b) the matrix-distance sum
// of the two new edges is strictly smaller than the two old edges.
// Depot legs cancel because both vehicles share the same depot.
//
// O(R² · S̄²) per sweep where R is the route count and S̄ is the mean
// stops per route. For R=54, S̄=37 that's ~1.4 M segment-pair checks,
// ~20 ms on M3 Pro. We iterate the sweep until a pass finds no
// improvement (or 5 passes hit), so total cost stays under ~100 ms.
// -------------------------------------------------------------------------

fn uncross_pass(
    solution: &mut brooom::solution::Solution,
    problem: &brooom::Problem,
    matrix: &brooom::Matrix,
    kept_jobs_latlon: &[(f32, f32)],
) -> usize {
    let n_routes = solution.routes.len();
    let mut applied = 0usize;
    let mut improved = true;
    let mut sweep = 0;
    while improved && sweep < 5 {
        sweep += 1;
        improved = false;
        for a in 0..n_routes {
            for b in (a + 1)..n_routes {
                if try_2opt_star(solution, a, b, problem, matrix, kept_jobs_latlon) {
                    improved = true;
                    applied += 1;
                }
            }
        }
    }
    applied
}

fn try_2opt_star(
    solution: &mut brooom::solution::Solution,
    a: usize, b: usize,
    problem: &brooom::Problem,
    matrix: &brooom::Matrix,
    kept_jobs_latlon: &[(f32, f32)],
) -> bool {
    let na = solution.routes[a].steps.len();
    let nb = solution.routes[b].steps.len();
    if na < 2 || nb < 2 { return false; }

    let cap_a: i64 = problem.vehicles[solution.routes[a].vehicle_idx]
        .capacity.iter().sum::<i64>().max(1);
    let cap_b: i64 = problem.vehicles[solution.routes[b].vehicle_idx]
        .capacity.iter().sum::<i64>().max(1);

    // Cumulative deliveries (length = steps.len() + 1, prefix_loads[k] =
    // sum of deliveries of steps[..k]).
    let loads_a = cumulative_loads(&solution.routes[a].steps, problem);
    let loads_b = cumulative_loads(&solution.routes[b].steps, problem);
    let total_a = *loads_a.last().unwrap_or(&0);
    let total_b = *loads_b.last().unwrap_or(&0);

    let mut best_delta: i64 = 0;
    let mut best_swap: Option<(usize, usize)> = None;

    for i in 0..(na - 1) {
        let ji  = job_idx_of(solution.routes[a].steps[i]);
        let ji1 = job_idx_of(solution.routes[a].steps[i + 1]);
        let pi  = kept_jobs_latlon[ji];
        let pi1 = kept_jobs_latlon[ji1];
        let mi  = problem.jobs[ji].location.index.unwrap_or(0);
        let mi1 = problem.jobs[ji1].location.index.unwrap_or(0);

        for j in 0..(nb - 1) {
            let jb  = job_idx_of(solution.routes[b].steps[j]);
            let jb1 = job_idx_of(solution.routes[b].steps[j + 1]);
            let pj  = kept_jobs_latlon[jb];
            let pj1 = kept_jobs_latlon[jb1];

            if !segments_cross(pi, pi1, pj, pj1) { continue; }

            let mj  = problem.jobs[jb].location.index.unwrap_or(0);
            let mj1 = problem.jobs[jb1].location.index.unwrap_or(0);

            let old_dist = matrix.distance(mi, mi1) + matrix.distance(mj, mj1);
            let new_dist = matrix.distance(mi, mj1) + matrix.distance(mj, mi1);
            let delta = new_dist - old_dist;
            if delta >= best_delta { continue; }

            // After 2-opt* (suffix swap after positions i / j):
            //   new route A = a[..=i]  +  b[(j+1)..]
            //   new route B = b[..=j]  +  a[(i+1)..]
            // capacity sums:
            let load_a_new = loads_a[i + 1] + (total_b - loads_b[j + 1]);
            let load_b_new = loads_b[j + 1] + (total_a - loads_a[i + 1]);
            if load_a_new > cap_a || load_b_new > cap_b { continue; }

            best_delta = delta;
            best_swap = Some((i, j));
        }
    }

    if let Some((i, j)) = best_swap {
        let a_suffix: Vec<_> = solution.routes[a].steps.split_off(i + 1);
        let b_suffix: Vec<_> = solution.routes[b].steps.split_off(j + 1);
        solution.routes[a].steps.extend(b_suffix);
        solution.routes[b].steps.extend(a_suffix);
        true
    } else {
        false
    }
}

#[inline]
fn job_idx_of(step: brooom::solution::TaskRef) -> usize {
    match step {
        brooom::solution::TaskRef::Job(j) => j,
        _ => 0,  // shipments are not generated by this CLI
    }
}

fn cumulative_loads(steps: &[brooom::solution::TaskRef], problem: &brooom::Problem) -> Vec<i64> {
    let mut out = Vec::with_capacity(steps.len() + 1);
    out.push(0);
    let mut acc = 0i64;
    for &s in steps {
        let j = job_idx_of(s);
        acc += problem.jobs[j].delivery.iter().sum::<i64>();
        out.push(acc);
    }
    out
}

// -------------------------------------------------------------------------
// Intra-route 2-opt, NO granular restriction.
//
// brooom does intra-route 2-opt but its candidate set is filtered by
// the K=40 granular neighbourhood, so long zigzag-style segments in
// the visiting order whose fix would require swapping with a stop
// outside the K-nearest list survive. This pass enumerates every
// (i, j) edge pair within a route (including the two depot legs as
// boundary edges) and applies the best reversal.
//
// Reversing path[i+1..=j] turns edges (path[i], path[i+1]) and
// (path[j], path[j+1]) into (path[i], path[j]) and (path[i+1], path[j+1]).
// path = [depot_start, s_0, s_1, ..., s_{n-1}, depot_end].
// In `steps` terms: reversing steps[i..j] (exclusive on the right).
//
// O(R · n²) per sweep; for R=54, n=37 that's <80k ops, <1 ms total.
// -------------------------------------------------------------------------

fn intra_route_2opt_pass(
    solution: &mut brooom::solution::Solution,
    problem: &brooom::Problem,
    matrix: &brooom::Matrix,
) -> usize {
    let mut applied = 0usize;
    for r in 0..solution.routes.len() {
        let vh = &problem.vehicles[solution.routes[r].vehicle_idx];
        let dep_s = vh.start.as_ref().and_then(|l| l.index).unwrap_or(0);
        let dep_e = vh.end.as_ref().and_then(|l| l.index).unwrap_or(dep_s);
        let mut sweep_local = 0usize;
        let mut improved = true;
        while improved && sweep_local < 10 {
            sweep_local += 1;
            improved = false;
            let n = solution.routes[r].steps.len();
            if n < 2 { break; }
            // Build node-id path including depot at both ends.
            let path: Vec<usize> = std::iter::once(dep_s)
                .chain(solution.routes[r].steps.iter()
                    .map(|s| problem.jobs[job_idx_of(*s)].location.index.unwrap_or(0)))
                .chain(std::iter::once(dep_e))
                .collect();
            let plen = path.len();  // = n + 2

            let mut best_delta: i64 = 0;
            let mut best: Option<(usize, usize)> = None;
            for i in 0..(plen - 2) {
                for j in (i + 2)..(plen - 1) {
                    let old = matrix.distance(path[i], path[i + 1])
                            + matrix.distance(path[j], path[j + 1]);
                    let new_d = matrix.distance(path[i], path[j])
                              + matrix.distance(path[i + 1], path[j + 1]);
                    let delta = new_d - old;
                    if delta < best_delta {
                        best_delta = delta;
                        best = Some((i, j));
                    }
                }
            }
            if let Some((i, j)) = best {
                // path[i+1..=j] reverses → in `steps` indices that is steps[i..j].
                solution.routes[r].steps[i..j].reverse();
                applied += 1;
                improved = true;
            }
        }
    }
    applied
}

// -------------------------------------------------------------------------
// Inter-route relocate, NO granular restriction.
//
// brooom's `relocate` operator only considers a customer's K=40 graph-
// nearest neighbours as candidate destinations, which is why the
// "long-haul stop in the wrong route" pattern survives even after LS
// converges: the right destination route isn't in the customer's K-NN.
//
// This pass walks every customer × every other route × every insertion
// position. For each candidate move we compute:
//   Δ = (insertion cost in target route) - (removal saving in source)
// and apply the best strictly-negative one. Capacity is the only
// constraint checked (no TWs in the generated problems).
//
// O(N × R × S̄) per sweep = O(N²) overall. For N=2000 that's ~4M
// matrix-distance lookups; ~50 ms on M3 Pro. We iterate sweeps until a
// pass finds nothing (capped at 5).
// -------------------------------------------------------------------------

fn relocate_pass(
    solution: &mut brooom::solution::Solution,
    problem: &brooom::Problem,
    matrix: &brooom::Matrix,
) -> usize {
    let n_routes = solution.routes.len();
    let mut applied = 0usize;
    let mut improved = true;
    let mut sweep = 0;
    while improved && sweep < 5 {
        sweep += 1;
        improved = false;
        for from_route in 0..n_routes {
            // Walk the FROM-route in reverse so removing a stop doesn't
            // shift the indices we haven't visited yet.
            let mut from_pos = solution.routes[from_route].steps.len();
            while from_pos > 0 {
                from_pos -= 1;
                if try_relocate(solution, from_route, from_pos, problem, matrix) {
                    improved = true;
                    applied += 1;
                }
            }
        }
    }
    applied
}

fn try_relocate(
    solution: &mut brooom::solution::Solution,
    from_route: usize, from_pos: usize,
    problem: &brooom::Problem,
    matrix: &brooom::Matrix,
) -> bool {
    if solution.routes[from_route].steps.is_empty() { return false; }
    let n_routes = solution.routes.len();

    let job_move = job_idx_of(solution.routes[from_route].steps[from_pos]);
    let m_move = problem.jobs[job_move].location.index.unwrap_or(0);
    let delivery_move: i64 = problem.jobs[job_move].delivery.iter().sum();

    let from_v = &problem.vehicles[solution.routes[from_route].vehicle_idx];
    let from_dep_s = from_v.start.as_ref().and_then(|l| l.index).unwrap_or(0);
    let from_dep_e = from_v.end.as_ref().and_then(|l| l.index).unwrap_or(from_dep_s);

    let from_steps = &solution.routes[from_route].steps;
    let prev_from = if from_pos > 0 {
        problem.jobs[job_idx_of(from_steps[from_pos - 1])].location.index.unwrap_or(0)
    } else { from_dep_s };
    let next_from = if from_pos + 1 < from_steps.len() {
        problem.jobs[job_idx_of(from_steps[from_pos + 1])].location.index.unwrap_or(0)
    } else { from_dep_e };

    // Distance saved by removing `m_move` from from_route.
    let removal_save = matrix.distance(prev_from, m_move)
                     + matrix.distance(m_move, next_from)
                     - matrix.distance(prev_from, next_from);

    let mut best_delta: i64 = 0;
    let mut best_target: Option<(usize, usize)> = None;

    for to_route in 0..n_routes {
        if to_route == from_route { continue; }
        let to_v = &problem.vehicles[solution.routes[to_route].vehicle_idx];
        let to_cap: i64 = to_v.capacity.iter().sum::<i64>().max(1);
        let to_dep_s = to_v.start.as_ref().and_then(|l| l.index).unwrap_or(0);
        let to_dep_e = to_v.end.as_ref().and_then(|l| l.index).unwrap_or(to_dep_s);

        // Capacity check: current load of to_route + this delivery.
        let to_load: i64 = solution.routes[to_route].steps.iter()
            .map(|s| problem.jobs[job_idx_of(*s)].delivery.iter().sum::<i64>()).sum();
        if to_load + delivery_move > to_cap { continue; }

        let to_steps = &solution.routes[to_route].steps;
        let ts_len = to_steps.len();

        for to_pos in 0..=ts_len {
            let prev_to = if to_pos > 0 {
                problem.jobs[job_idx_of(to_steps[to_pos - 1])].location.index.unwrap_or(0)
            } else { to_dep_s };
            let next_to = if to_pos < ts_len {
                problem.jobs[job_idx_of(to_steps[to_pos])].location.index.unwrap_or(0)
            } else { to_dep_e };

            let insertion_cost = matrix.distance(prev_to, m_move)
                               + matrix.distance(m_move, next_to)
                               - matrix.distance(prev_to, next_to);
            let delta = insertion_cost - removal_save;
            if delta < best_delta {
                best_delta = delta;
                best_target = Some((to_route, to_pos));
            }
        }
    }

    if let Some((to_route, to_pos)) = best_target {
        let task = solution.routes[from_route].steps.remove(from_pos);
        solution.routes[to_route].steps.insert(to_pos, task);
        true
    } else {
        false
    }
}

/// Cross-product based segment-intersection test on the lat/lon plane.
/// Treats (lat, lon) as Cartesian — fine for a ≤30 km radius problem
/// (projection error is sub-metre at this scale).
fn segments_cross(p1: (f32, f32), p2: (f32, f32), p3: (f32, f32), p4: (f32, f32)) -> bool {
    let dir = |a: (f32, f32), b: (f32, f32), c: (f32, f32)| -> f32 {
        (c.0 - a.0) * (b.1 - a.1) - (b.0 - a.0) * (c.1 - a.1)
    };
    let d1 = dir(p3, p4, p1);
    let d2 = dir(p3, p4, p2);
    let d3 = dir(p1, p2, p3);
    let d4 = dir(p1, p2, p4);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) &&
    ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

// -------------------------------------------------------------------------
// Sweep heuristic: every customer is mapped to a polar angle around the
// depot, then the customers are walked in angle order and packed into
// each vehicle until its capacity is reached. The result is N_routes
// disjoint angular sectors, which is provably near-optimal for radial
// VRP instances and a much better starting point for LS than greedy
// cheapest-insertion (which produces tangled cross-route segments).
// -------------------------------------------------------------------------
fn build_sweep_warm_start(
    problem: &brooom::Problem,
    depot_latlon: (f64, f64),
    kept_jobs_latlon: &[(f32, f32)],
) -> brooom::solution::Solution {
    use brooom::solution::{Route, RouteMetrics, Solution, Summary, TaskRef};

    let n_jobs = problem.jobs.len();
    let n_vehicles = problem.vehicles.len();
    if n_jobs == 0 || n_vehicles == 0 {
        return Solution::default();
    }

    let lat0 = depot_latlon.0;
    let cos_lat = lat0.to_radians().cos().max(1e-6);
    let mut angled: Vec<(f64, usize)> = (0..n_jobs)
        .map(|j| {
            let (lat, lon) = kept_jobs_latlon[j];
            let dy = lat as f64 - lat0;
            let dx = (lon as f64 - depot_latlon.1) * cos_lat;
            (dy.atan2(dx), j)
        })
        .collect();
    angled.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Use the first vehicle's capacity as a representative — the generator
    // in this CLI gives every vehicle the same capacity. For mixed fleets
    // brooom's LS will rebalance during the first iteration anyway.
    let cap: i64 = problem.vehicles[0].capacity.iter().sum::<i64>().max(1);

    let mut routes: Vec<Route> = Vec::with_capacity(n_vehicles);
    let mut current_steps: Vec<TaskRef> = Vec::new();
    let mut current_load: i64 = 0;
    let mut veh_idx: usize = 0;

    for (_, j) in angled {
        if veh_idx >= n_vehicles { break; }
        let delivery: i64 = problem.jobs[j].delivery.iter().sum();
        if current_load + delivery > cap && !current_steps.is_empty() {
            routes.push(Route {
                vehicle_idx: veh_idx,
                steps: std::mem::take(&mut current_steps),
                metrics: RouteMetrics::default(),
            });
            current_load = 0;
            veh_idx += 1;
            if veh_idx >= n_vehicles { break; }
        }
        current_steps.push(TaskRef::Job(j));
        current_load += delivery;
    }
    if !current_steps.is_empty() && veh_idx < n_vehicles {
        routes.push(Route {
            vehicle_idx: veh_idx,
            steps: current_steps,
            metrics: RouteMetrics::default(),
        });
    }

    Solution { routes, unassigned: Vec::new(), summary: Summary::default() }
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
// Same shape as mpee-viz's bundle.
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
    /// Matrix distance (metres) from the previous stop in this route.
    /// For order=0 this is the depot leg. The frontend uses these to
    /// surface the top-K longest customer-to-customer segments —
    /// those are the visually suspicious "long across" lines.
    dist_from_prev_m: i32,
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
            let seg_dist: i32 = if let (Some(p), Some(h)) = (prev, here_idx) {
                let d = matrix.distance(p, h);
                route_dur += matrix.duration(p, h);
                route_dist += d;
                d as i32
            } else { 0 };
            prev = here_idx;
            let job = &problem.jobs[job_idx];
            let delivered: i64 = job.delivery.iter().sum();
            load += delivered;
            let (lat, lon) = kept_jobs_latlon[job_idx];
            stops.push(StopOut {
                job_id: job.id, lat, lon, order, load_after: load,
                dist_from_prev_m: seg_dist,
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
