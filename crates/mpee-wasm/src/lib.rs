//! WebAssembly bindings for the MPEE engine — offline routing, geocoding and
//! VRP entirely in the browser. The three cache files (`.pp` / `.ch` /
//! `.names`) are fetched by JS and handed in as byte slices; everything else
//! (snap, route, reverse/forward geocode, street crossing, multi-vehicle
//! optimization) runs in-process via the same Rust crates as the native CLI.
//!
//! Build: `wasm-pack build --target web --release`.

use wasm_bindgen::prelude::*;

use brooom::solution::TaskRef;
use brooom::solver::{solve_with_matrix, SolverConfig};
use brooom::{Job, Location, Matrix, Problem, Vehicle};
use dijeng::buffer::Buffer;
use dijeng::routing::RoutingService;

fn err_to_js<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// The in-browser engine. Holds a memory-loaded routing + geocoding service.
#[wasm_bindgen]
pub struct Engine {
    routing: RoutingService,
}

#[wasm_bindgen]
impl Engine {
    /// Build the engine from the three cache files' bytes. `names` may be empty
    /// (`Uint8Array(0)`) to load a routing-only engine without geocoding.
    #[wasm_bindgen(constructor)]
    pub fn new(pp: &[u8], ch: &[u8], names: &[u8]) -> Result<Engine, JsValue> {
        console_error_panic_hook::set_once();
        let pp = dijeng::cache_pp::load_bytes(pp).map_err(err_to_js)?;
        let ch = dijeng::cache_ch::load_bytes(ch).map_err(err_to_js)?;
        let n = pp.coords.as_slice().len();
        let coords = Buffer::from(pp.coords.as_slice().to_vec());
        let mut routing = RoutingService::new(ch, coords);
        if !names.is_empty() {
            match dijeng::names::NameTable::load_bytes(names, n) {
                Ok(nt) => routing.set_names(nt),
                Err(e) => web_log(&format!("[mpee] ignoring names sidecar: {e}")),
            }
        }
        Ok(Engine { routing })
    }

    /// Number of road nodes in the loaded graph.
    pub fn node_count(&self) -> usize {
        self.routing.node_count()
    }

    /// Whether forward/reverse geocoding is available (a `.names` sidecar loaded).
    pub fn has_names(&self) -> bool {
        self.routing.has_names()
    }

    /// Bounding box of the loaded area as JSON `{min_lat,min_lon,max_lat,max_lon}`.
    pub fn bbox(&self) -> String {
        let (mut mnla, mut mxla, mut mnlo, mut mxlo) =
            (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        for &(la, lo) in self.routing.coords.as_slice() {
            if la.is_finite() && lo.is_finite() {
                mnla = mnla.min(la); mxla = mxla.max(la);
                mnlo = mnlo.min(lo); mxlo = mxlo.max(lo);
            }
        }
        format!(
            "{{\"min_lat\":{mnla},\"min_lon\":{mnlo},\"max_lat\":{mxla},\"max_lon\":{mxlo}}}"
        )
    }

    /// Snap a point to the nearest road node. JSON `{lat,lon}`.
    pub fn snap(&self, lat: f32, lon: f32) -> String {
        let (la, lo) = self.routing.coords[self.routing.nearest_node(lat, lon) as usize];
        format!("{{\"lat\":{la},\"lon\":{lo}}}")
    }

    /// Driving route between two points. Returns JSON with distance/duration,
    /// snapped endpoints and the `[[lat,lon],…]` road geometry.
    pub fn route(
        &self,
        from_lat: f32,
        from_lon: f32,
        to_lat: f32,
        to_lon: f32,
    ) -> Result<String, JsValue> {
        let r = self
            .routing
            .route(from_lat, from_lon, to_lat, to_lon)
            .ok_or_else(|| JsValue::from_str("no route between the snapped points"))?;
        let geom: Vec<[f32; 2]> = r.geometry.iter().map(|&(la, lo)| [la, lo]).collect();
        let out = serde_json::json!({
            "distance_m": r.distance_m,
            "distance_km": r.distance_m / 1000.0,
            "duration_s": r.duration_s,
            "duration_min": r.duration_s / 60.0,
            "source_snapped": [r.source_snapped.0, r.source_snapped.1],
            "destination_snapped": [r.destination_snapped.0, r.destination_snapped.1],
            "geometry": geom,
        });
        Ok(out.to_string())
    }

    /// Reverse-geocode: nearest street name to a point. Returns the name, or an
    /// empty string if none / no sidecar.
    pub fn reverse(&self, lat: f32, lon: f32) -> String {
        self.routing.reverse(lat, lon).unwrap_or("").to_string()
    }

    /// Forward-geocode: street name → JSON `{name,lat,lon}`, or `null`.
    /// `near_lat`/`near_lon` finite → pick the match nearest that point
    /// (multi-city disambiguation); pass NaN to ignore.
    pub fn geocode(&self, query: &str, near_lat: f32, near_lon: f32) -> String {
        let hit = if near_lat.is_finite() && near_lon.is_finite() {
            self.routing.geocode_near(query, near_lat, near_lon)
        } else {
            self.routing.geocode(query)
        };
        match hit {
            Some((lat, lon, name)) => serde_json::json!({
                "name": name, "lat": lat, "lon": lon
            })
            .to_string(),
            None => "null".to_string(),
        }
    }

    /// Intersection search: every coordinate where two streets cross. JSON
    /// `[{lat,lon},…]`. `near_*` finite → sort nearest-first to that point.
    pub fn intersection(&self, a: &str, b: &str, near_lat: f32, near_lon: f32) -> String {
        let hits = if near_lat.is_finite() && near_lon.is_finite() {
            self.routing.intersection_near(a, b, near_lat, near_lon, None)
        } else {
            self.routing.intersection(a, b)
        };
        let arr: Vec<serde_json::Value> = hits
            .into_iter()
            .map(|(la, lo)| serde_json::json!({"lat": la, "lon": lo}))
            .collect();
        serde_json::Value::Array(arr).to_string()
    }

    /// Optimize a multi-vehicle delivery run over `stops` (JSON `[[lat,lon],…]`).
    /// Vehicles start/end at `depot` (JSON `[lat,lon]`, or `null` → centroid).
    /// Returns JSON with one entry per used vehicle (ordered stops + coords),
    /// totals and any unassigned stops. CPU solver (serial multi-start).
    pub fn optimize(
        &self,
        stops_json: &str,
        depot_json: &str,
        vehicles: usize,
        capacity: i32,
        time_limit_s: f64,
    ) -> Result<String, JsValue> {
        let capacity = capacity as i64; // i32 maps to a JS number; widen for brooom
        let stops: Vec<[f32; 2]> = serde_json::from_str(stops_json).map_err(err_to_js)?;
        if stops.is_empty() {
            return Err(JsValue::from_str("no stops given"));
        }
        if vehicles == 0 {
            return Err(JsValue::from_str("vehicles must be >= 1"));
        }

        // Depot: explicit [lat,lon] or centroid of the stops.
        let depot: (f32, f32) = match serde_json::from_str::<Option<[f32; 2]>>(depot_json) {
            Ok(Some(d)) => (d[0], d[1]),
            _ => {
                let n = stops.len() as f32;
                let (sla, slo) = stops
                    .iter()
                    .fold((0.0f32, 0.0f32), |(a, b), p| (a + p[0], b + p[1]));
                (sla / n, slo / n)
            }
        };

        // coords[0] = depot, coords[1..=N] = stops.
        let mut coords: Vec<(f32, f32)> = Vec::with_capacity(stops.len() + 1);
        coords.push(depot);
        coords.extend(stops.iter().map(|p| (p[0], p[1])));

        let n = coords.len();
        let (durs_f, dists_f, snapped, _) = self.routing.matrix_with_distance(&coords, &coords);
        let to_i = |v: &[f32]| -> Vec<i32> {
            v.iter()
                .map(|&d| if d.is_finite() { d.round().max(0.0) as i32 } else { i32::MAX / 2 })
                .collect()
        };
        let matrix = Matrix { n, durations: to_i(&durs_f), distances: Some(to_i(&dists_f)) };

        let mut problem = Problem::default();
        for v in 0..vehicles {
            problem.vehicles.push(Vehicle {
                id: (v + 1) as u64,
                start: Some(Location::from_index(0)),
                end: Some(Location::from_index(0)),
                capacity: vec![capacity],
                skills: vec![],
                time_window: None,
                speed_factor: 1.0,
                max_tasks: None,
                max_travel_time: None,
                max_distance: None,
                fixed: 0.0,
                per_hour: 3600.0,
                profile: "car".into(),
                description: None,
            });
        }
        for i in 0..stops.len() {
            problem.jobs.push(Job {
                id: (i + 1) as u64,
                location: Location::from_index(i + 1),
                kind: Default::default(),
                service: 0,
                setup: 0,
                delivery: vec![1],
                pickup: vec![],
                skills: vec![],
                priority: 0,
                time_windows: vec![],
                description: None,
            });
        }

        let cfg = SolverConfig {
            multi_start: 4,
            granular_k: Some(40),
            max_local_search_passes: 50,
            time_limit_ms: Some((time_limit_s * 1000.0) as u64),
            verbose: false,
            use_gpu: false,
            ..Default::default()
        };
        let sol = solve_with_matrix(&problem, &matrix, &cfg);

        // Assemble per-vehicle ordered stops (snapped coords for drawing).
        let dist = matrix.distances.as_ref().unwrap();
        let mut routes_out: Vec<serde_json::Value> = Vec::new();
        let (mut grand_d, mut grand_t, mut grand_stops) = (0i64, 0i64, 0usize);
        for r in &sol.routes {
            if r.steps.is_empty() {
                continue;
            }
            let vid = problem.vehicles[r.vehicle_idx].id;
            let mut steps: Vec<serde_json::Value> = Vec::new();
            let (mut td, mut tt) = (0i64, 0i64);
            let mut prev = 0usize; // depot
            for (order, step) in r.steps.iter().enumerate() {
                let mi = match step {
                    TaskRef::Job(j) => *j + 1,
                    _ => continue,
                };
                td += dist[prev * n + mi] as i64;
                tt += matrix.durations[prev * n + mi] as i64;
                prev = mi;
                let (la, lo) = snapped[mi];
                steps.push(serde_json::json!({
                    "order": order, "stop_index": mi - 1, "lat": la, "lon": lo
                }));
            }
            td += dist[prev * n] as i64;
            tt += matrix.durations[prev * n] as i64;
            grand_d += td;
            grand_t += tt;
            grand_stops += steps.len();
            routes_out.push(serde_json::json!({
                "vehicle_id": vid,
                "n_stops": steps.len(),
                "distance_km": td as f64 / 1000.0,
                "duration_min": tt as f64 / 60.0,
                "stops": steps,
            }));
        }
        let unassigned: Vec<usize> = sol
            .unassigned
            .iter()
            .filter_map(|t| match t {
                TaskRef::Job(j) => Some(*j),
                _ => None,
            })
            .collect();

        let out = serde_json::json!({
            "routes": routes_out,
            "vehicles_used": routes_out.len(),
            "total_stops": grand_stops,
            "total_distance_km": grand_d as f64 / 1000.0,
            "total_duration_min": grand_t as f64 / 60.0,
            "depot": [depot.0, depot.1],
            "unassigned": unassigned,
        });
        Ok(out.to_string())
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn web_log(s: &str);
}
