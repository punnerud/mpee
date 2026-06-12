//! WebAssembly bindings for the MPEE engine — offline routing, geocoding and
//! VRP entirely in the browser. The three cache files (`.pp` / `.ch` /
//! `.names`) are fetched by JS and handed in as byte slices; everything else
//! (snap, route, reverse/forward geocode, street crossing, multi-vehicle
//! optimization) runs in-process via the same Rust crates as the native CLI.
//!
//! Build: `wasm-pack build --target web --release`.

use wasm_bindgen::prelude::*;

use brooom::solution::TaskRef;
use brooom::solver::{solve_with_matrix, ObjectiveMode, SolverConfig};
use brooom::{Job, Location, Matrix, Problem, Vehicle};
use dijeng::buffer::Buffer;
use dijeng::routing::RoutingService;

fn err_to_js<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Parse the `objective` argument shared by the JS surface into an
/// [`ObjectiveMode`]. `""` / `"scalar"` ⇒ the default single-cost solve; a
/// comma list of level names ("vehicles,cost", "unassigned,vehicles,cost", …)
/// ⇒ an N-level lexicographic objective. Level names match the CLI/JSON/Python
/// surfaces (see `brooom::options::lex_objective_from_name`).
fn parse_objective(spec: &str) -> Result<ObjectiveMode, String> {
    let spec = spec.trim();
    if spec.is_empty() || spec.eq_ignore_ascii_case("scalar") {
        return Ok(ObjectiveMode::Scalar);
    }
    let mut levels = Vec::new();
    for name in spec.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        levels.push(brooom::options::lex_objective_from_name(name).map_err(|e| e.to_string())?);
    }
    if levels.is_empty() {
        Ok(ObjectiveMode::Scalar)
    } else {
        Ok(ObjectiveMode::Lexicographic { levels })
    }
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
        // Keep the plain road adjacency (same node order as coords) so a whole
        // street can be drawn via `street_segments`.
        let road_graph = pp.graph;
        let mut routing = RoutingService::new(ch, coords);
        routing.set_road_graph(road_graph);
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

    /// Attach the optional house-number address sidecar (`.addr` bytes fetched by
    /// JS). Call after construction; enables address-level forward/reverse. A
    /// no-op for empty input — kept separate from the constructor for back-compat.
    pub fn load_addresses(&mut self, addr: &[u8]) {
        if addr.is_empty() {
            return;
        }
        match dijeng::addresses::AddressIndex::load_bytes(addr) {
            Ok(ai) => self.routing.set_addresses(ai),
            Err(e) => web_log(&format!("[mpee] ignoring address sidecar: {e}")),
        }
    }

    /// Whether a house-number address sidecar is loaded.
    pub fn has_addresses(&self) -> bool {
        self.routing.has_addresses()
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

    /// Reverse-geocode: nearest address ("Street 42, 0123 City") when an address
    /// sidecar is loaded, else the nearest street name; empty string if none.
    pub fn reverse(&self, lat: f32, lon: f32) -> String {
        if let Some(h) = self.routing.reverse_address(lat, lon) {
            let mut s = format!("{} {}", h.street, h.housenumber);
            match (h.postcode, h.city) {
                (Some(pc), Some(c)) => s.push_str(&format!(", {pc} {c}")),
                (None, Some(c)) => s.push_str(&format!(", {c}")),
                (Some(pc), None) => s.push_str(&format!(", {pc}")),
                (None, None) => {}
            }
            return s;
        }
        self.routing.reverse(lat, lon).unwrap_or("").to_string()
    }

    /// Nearest STREET NAME only (no house number) — used by crossing selection,
    /// which matches on street names. `reverse` (above) returns the full address.
    pub fn reverse_street(&self, lat: f32, lon: f32) -> String {
        self.routing.reverse(lat, lon).unwrap_or("").to_string()
    }

    /// Forward-geocode: street (optionally "Street 42") → JSON. Address-level
    /// when the query has a number and an address sidecar is loaded (adds
    /// `housenumber`/`city`/`postcode`/`approximate`), else `{name,lat,lon}`.
    /// `near_*` finite → multi-city disambiguation; pass NaN to ignore.
    pub fn geocode(&self, query: &str, near_lat: f32, near_lon: f32) -> String {
        let near = if near_lat.is_finite() && near_lon.is_finite() {
            Some((near_lat, near_lon))
        } else {
            None
        };
        let (street_part, number) = dijeng::routing::split_house_number(query);
        if number.is_some() {
            if let Some(h) = self.routing.address_query(query, near) {
                return serde_json::json!({
                    "name": h.street, "housenumber": h.housenumber,
                    "lat": h.lat, "lon": h.lon,
                    "city": h.city, "postcode": h.postcode,
                    "approximate": h.approximate,
                })
                .to_string();
            }
        }
        let q = if number.is_some() { street_part } else { query };
        let hit = match near {
            Some((la, lo)) => self.routing.geocode_near(q, la, lo),
            None => self.routing.geocode(q),
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

    /// Type-ahead suggestions: up to `limit` street names matching `query`
    /// (prefix first, then substring). JSON array of strings.
    pub fn suggest(&self, query: &str, limit: usize) -> String {
        serde_json::to_string(&self.routing.suggest(query, limit)).unwrap_or_else(|_| "[]".into())
    }

    /// All road segments of a named street, as JSON `[[[lat,lon],[lat,lon]],…]`
    /// — the whole street drawn as a polyline set. Empty array if the name
    /// doesn't resolve or no sidecar/road-graph is loaded.
    pub fn street_segments(&self, query: &str) -> String {
        let segs = self.routing.street_segments(query);
        let arr: Vec<serde_json::Value> = segs
            .into_iter()
            .map(|(la, lo, lb, lob)| serde_json::json!([[la, lo], [lb, lob]]))
            .collect();
        serde_json::Value::Array(arr).to_string()
    }

    /// Optimize a multi-vehicle delivery run over `stops` (JSON `[[lat,lon],…]`).
    /// Vehicles start/end at `depot` (JSON `[lat,lon]`, or `null` → centroid).
    /// Returns JSON with one entry per used vehicle (ordered stops + coords),
    /// totals and any unassigned stops. CPU solver (serial multi-start).
    ///
    /// Working-hours model: `max_route_min > 0` caps each driver's route —
    /// driving + service + the return to the depot must fit within that many
    /// minutes (a shift; the UI subtracts the break before calling). The cap
    /// is hard: stops that can't fit any driver's remaining shift come back
    /// in `unassigned`. `service_min > 0` adds that many minutes of work at
    /// every stop. `capacity <= 0` means uncapacitated (hours bind instead).
    pub fn optimize(
        &self,
        stops_json: &str,
        depot_json: &str,
        vehicles: usize,
        capacity: i32,
        time_limit_s: f64,
        objective: &str,
        max_route_min: f64,
        service_min: f64,
    ) -> Result<String, JsValue> {
        let capacity = capacity as i64; // i32 maps to a JS number; widen for brooom
        // `objective`: "" / "scalar" → today's single-cost solve; otherwise a
        // comma list of lexicographic levels ("vehicles,cost", …) parsed to an
        // N-level objective. Same names as the CLI/JSON/Python surfaces.
        let objective_mode = parse_objective(objective).map_err(err_to_js)?;
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

        // Shift cap as a vehicle time window [0, work_s]: the evaluator then
        // enforces driving + service + waiting + the return leg <= work_s.
        let work_s: Option<i64> = (max_route_min > 0.0)
            .then(|| (max_route_min * 60.0).round() as i64)
            .filter(|&s| s > 0);
        let service_s: i64 = if service_min > 0.0 { (service_min * 60.0).round() as i64 } else { 0 };
        // capacity <= 0 ⇒ uncapacitated: every job fits, hours bind instead.
        let capacity = if capacity <= 0 { stops.len() as i64 } else { capacity };

        let mut problem = Problem::default();
        for v in 0..vehicles {
            problem.vehicles.push(Vehicle {
                id: (v + 1) as u64,
                start: Some(Location::from_index(0)),
                end: Some(Location::from_index(0)),
                capacity: vec![capacity],
                skills: vec![],
                time_window: work_s
                    .map(|s| brooom::problem::TimeWindow { start: 0, end: s }),
                speed_factor: 1.0,
                max_tasks: None,
                max_travel_time: None,
                max_distance: None,
                fixed: 0.0,
                per_hour: 3600.0,
                span_cost: 0.0,
                distance_weight: 0.0,
                time_weight: 1.0,
                profile: "car".into(),
                breaks: vec![],
                max_trips: 1,
                description: None,
            });
        }
        // Skip stops unreachable from the depot (disconnected road fragments /
        // across water) — they'd otherwise be assigned with sentinel-distance
        // legs (absurd totals + straight-line gaps). `job_mi[j]` maps a brooom
        // job index back to its matrix index; `dropped` are the unreachable ones.
        const UNREACH_I: i32 = 100_000_000;
        let dist0 = matrix.distances.as_ref().unwrap();
        let mut job_mi: Vec<usize> = Vec::new();
        let mut dropped: Vec<usize> = Vec::new();
        for i in 0..stops.len() {
            if dist0[i + 1] >= UNREACH_I || dist0[(i + 1) * n] >= UNREACH_I {
                dropped.push(i);
                continue;
            }
            problem.jobs.push(Job {
                id: (i + 1) as u64,
                location: Location::from_index(i + 1),
                kind: Default::default(),
                service: service_s,
                setup: 0,
                release: 0,
                delivery: vec![1],
                pickup: vec![],
                skills: vec![],
                allowed_vehicles: None,
                priority: 0,
                time_windows: vec![],
                prize: brooom::problem::DEFAULT_PRIZE,
                disjunction_penalty: None,
                group: None,
                description: None,
            });
            job_mi.push(i + 1);
        }

        let cfg = SolverConfig {
            multi_start: 4,
            granular_k: Some(40),
            max_local_search_passes: 50,
            time_limit_ms: Some((time_limit_s * 1000.0) as u64),
            verbose: false,
            use_gpu: false,
            objective_mode,
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
                    TaskRef::Job(j) => job_mi[*j],
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
            // Route duration = driving + service: the working time the shift
            // cap judges (waiting can't occur here — no job time windows).
            tt += service_s * steps.len() as i64;
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
        // Unassigned = brooom's leftovers (mapped back to original stop index)
        // plus the unreachable stops we dropped up front.
        let mut unassigned: Vec<usize> = sol
            .unassigned
            .iter()
            .filter_map(|t| match t {
                TaskRef::Job(j) => Some(job_mi[*j] - 1),
                _ => None,
            })
            .collect();
        unassigned.extend(dropped);

        let out = serde_json::json!({
            "routes": routes_out,
            "vehicles_used": routes_out.len(),
            "total_stops": grand_stops,
            "total_distance_km": grand_d as f64 / 1000.0,
            "total_duration_min": grand_t as f64 / 60.0,
            "depot": [depot.0, depot.1],
            "unassigned": unassigned,
            // Echo of the working-hours model, so the UI can render
            // "6h 51m / 7h 30m" without re-deriving its own inputs.
            "max_route_min": max_route_min,
            "service_min": service_min,
        });
        Ok(out.to_string())
    }

    /// Stage 2b — GPU-accelerated optimize. Builds the matrix + a quick CPU
    /// construction (greedy insertion, **no** local search), then runs
    /// intra-route **2-opt on the GPU** (WebGPU compute, one workgroup per
    /// route) to improve each route's visiting order. Fully async — nothing
    /// blocks the main thread. Same JSON as `optimize`, plus `solver:"gpu-2opt"`
    /// and `before_distance_km` / `after_distance_km` so the GPU's improvement
    /// over the raw construction is visible.
    pub async fn optimize_gpu(
        &self,
        stops_json: &str,
        depot_json: &str,
        vehicles: usize,
        capacity: i32,
    ) -> Result<String, JsValue> {
        let capacity = capacity as i64;
        let stops: Vec<[f32; 2]> = serde_json::from_str(stops_json).map_err(err_to_js)?;
        if stops.is_empty() { return Err(JsValue::from_str("no stops given")); }
        if vehicles == 0 { return Err(JsValue::from_str("vehicles must be >= 1")); }
        let depot: (f32, f32) = match serde_json::from_str::<Option<[f32; 2]>>(depot_json) {
            Ok(Some(d)) => (d[0], d[1]),
            _ => {
                let n = stops.len() as f32;
                let (a, b) = stops.iter().fold((0.0f32, 0.0f32), |(a, b), p| (a + p[0], b + p[1]));
                (a / n, b / n)
            }
        };
        let mut coords: Vec<(f32, f32)> = Vec::with_capacity(stops.len() + 1);
        coords.push(depot);
        coords.extend(stops.iter().map(|p| (p[0], p[1])));
        let n = coords.len();
        let (durs_f, dists_f, snapped, _) = self.routing.matrix_with_distance(&coords, &coords);
        let to_i = |v: &[f32]| -> Vec<i32> {
            v.iter().map(|&d| if d.is_finite() { d.round().max(0.0) as i32 } else { i32::MAX / 2 }).collect()
        };
        let dur_i = to_i(&durs_f);
        let dist_i = to_i(&dists_f);

        let route_dist = |seq: &[u32]| -> i64 {
            seq.windows(2).map(|w| dist_i[w[0] as usize * n + w[1] as usize] as i64).sum()
        };

        // Naive round-robin assignment (respecting capacity) — deliberately a
        // poor start so the GPU local search has real cross-route + intra-route
        // work to do, and its improvement is clearly visible. Each route is
        // [depot, stops…, depot] in matrix indices.
        let cap = capacity.max(1) as usize;
        let mut routes: Vec<Vec<u32>> = (0..vehicles).map(|_| vec![0u32]).collect();
        let mut loads = vec![0usize; vehicles];
        let mut unassigned: Vec<usize> = Vec::new();
        // A stop that snaps to a road fragment disconnected from the depot (e.g.
        // across the bay) has a sentinel distance — never route to it, mark it
        // unassigned. Then every routed stop shares the depot's component, so no
        // leg is unreachable (no absurd distances, no straight-line gaps).
        const UNREACH: i32 = 100_000_000;
        for i in 0..stops.len() {
            let mi = (i + 1) as u32;
            if dist_i[i + 1] >= UNREACH || dist_i[(i + 1) * n] >= UNREACH {
                unassigned.push(i);
                continue;
            }
            let mut placed = false;
            for k in 0..vehicles {
                let v = (i + k) % vehicles;
                if loads[v] < cap { routes[v].push(mi); loads[v] += 1; placed = true; break; }
            }
            if !placed { unassigned.push(i); }
        }
        for r in routes.iter_mut() { r.push(0); } // closing depot
        routes.retain(|r| r.len() > 2);            // drop empty routes

        // Flatten the Vec<Vec> into the GPU layout (seqs + offsets + route_of).
        let flatten = |routes: &Vec<Vec<u32>>| -> (Vec<u32>, Vec<u32>, Vec<u32>) {
            let mut seqs = Vec::new();
            let mut offsets = vec![0u32];
            let mut route_of = Vec::new();
            for (ri, r) in routes.iter().enumerate() {
                for _ in 0..r.len() { route_of.push(ri as u32); }
                seqs.extend_from_slice(r);
                offsets.push(seqs.len() as u32);
            }
            (seqs, offsets, route_of)
        };
        let total_dist = |routes: &Vec<Vec<u32>>| -> i64 { routes.iter().map(|r| route_dist(r)).sum() };
        let before: i64 = total_dist(&routes);

        // One WebGPU device for the whole solve (the relocate loop dispatches
        // dozens of kernels — re-creating a device each time would dominate).
        let (device, queue) = acquire_device().await?;

        // ---- GPU cross-route relocate (steepest descent) ----
        // The GPU evaluates the whole relocate neighbourhood each round; the CPU
        // applies the single best improving move. Relocate is exact on the true
        // asymmetric matrix. Bounded rounds so it always terminates.
        if routes.len() >= 2 {
            // Each round is one GPU dispatch + async readback; cap the count so
            // the latency-bound loop stays fast even on software WebGPU. Steepest
            // descent converges quickly, so a modest cap keeps most of the gain.
            let max_rounds = (stops.len() / 3 + 4).min(12);
            for _ in 0..max_rounds {
                let (seqs, offsets, route_of) = flatten(&routes);
                let (delta, src, dr, dp) =
                    run_relocate_eval(&device, &queue, &dist_i, n as u32, &seqs, &offsets, &route_of, cap as u32).await?;
                if delta >= 0 { break; } // no improving move
                // Guard against an out-of-range move (keeps a valid solution).
                let a = route_of[src] as usize;
                if a >= routes.len() || dr >= routes.len() || a == dr { break; }
                let li = src - offsets[a] as usize;
                if li == 0 || li + 1 >= routes[a].len() { break; }  // must be interior
                let local_pos = dp - offsets[dr] as usize;
                if local_pos + 1 >= routes[dr].len() { break; }
                let s = routes[a][li];
                routes[a].remove(li);
                routes[dr].insert(local_pos + 1, s); // insert s after bp
            }
            routes.retain(|r| r.len() > 2);
        }

        // ---- GPU intra-route 2-opt (one workgroup per route) ----
        // 2-opt reverses a segment → invalid on asymmetric distances, so use a
        // symmetrised matrix for the kernel + a per-route safety net on the true
        // distance (never worse than the pre-2-opt order).
        let mut sym = vec![0i32; n * n];
        for a in 0..n {
            for b in 0..n {
                sym[a * n + b] = dist_i[a * n + b].min(dist_i[b * n + a]);
            }
        }
        let (mut seqs, offsets, _route_of) = flatten(&routes);
        let route_vehicle: Vec<u64> = (0..routes.len() as u64).map(|i| i + 1).collect();
        let num_routes = routes.len() as u32;
        let seqs_before = seqs.clone();
        if num_routes > 0 {
            run_2opt_gpu(&device, &queue, &sym, n as u32, &mut seqs, &offsets, num_routes).await?;
        }
        for ri in 0..routes.len() {
            let (a, b) = (offsets[ri] as usize, offsets[ri + 1] as usize);
            if route_dist(&seqs[a..b]) > route_dist(&seqs_before[a..b]) {
                seqs[a..b].copy_from_slice(&seqs_before[a..b]);
            }
        }

        // Rebuild the plan from the improved sequences.
        let mut routes_out: Vec<serde_json::Value> = Vec::new();
        let (mut grand_d, mut grand_t, mut grand_stops) = (0i64, 0i64, 0usize);
        for ri in 0..route_vehicle.len() {
            let seq = &seqs[offsets[ri] as usize..offsets[ri + 1] as usize];
            let mut td = 0i64;
            let mut tt = 0i64;
            for w in seq.windows(2) {
                td += dist_i[w[0] as usize * n + w[1] as usize] as i64;
                tt += dur_i[w[0] as usize * n + w[1] as usize] as i64;
            }
            let mut steps: Vec<serde_json::Value> = Vec::new();
            for (order, &mi) in seq[1..seq.len() - 1].iter().enumerate() {
                let (la, lo) = snapped[mi as usize];
                steps.push(serde_json::json!({ "order": order, "stop_index": (mi as usize) - 1, "lat": la, "lon": lo }));
            }
            grand_d += td; grand_t += tt; grand_stops += steps.len();
            routes_out.push(serde_json::json!({
                "vehicle_id": route_vehicle[ri], "n_stops": steps.len(),
                "distance_km": td as f64 / 1000.0, "duration_min": tt as f64 / 60.0, "stops": steps,
            }));
        }
        // `unassigned` was collected during the naive assignment (over-capacity).

        let out = serde_json::json!({
            "routes": routes_out,
            "vehicles_used": routes_out.len(),
            "total_stops": grand_stops,
            "total_distance_km": grand_d as f64 / 1000.0,
            "total_duration_min": grand_t as f64 / 60.0,
            "depot": [depot.0, depot.1],
            "unassigned": unassigned,
            "solver": "gpu-2opt",
            "before_distance_km": before as f64 / 1000.0,
            "after_distance_km": grand_d as f64 / 1000.0,
        });
        Ok(out.to_string())
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn web_log(s: &str);
}

// =========================================================================
// Stage 2a — WebGPU spike. An async probe that initialises a real WebGPU
// device via wgpu (BROWSER_WEBGPU backend), runs a trivial compute kernel
// (out[i] = in[i]²) and reads the result back — all async, nothing blocking
// the browser's main thread. Proves the hardest GPU-on-wasm infrastructure
// before porting the brooom VRP megakernel (stages 2b/2c). Returns a JSON
// string `{ok, backend, adapter, sample}` or rejects with the failure reason.
// =========================================================================
#[wasm_bindgen]
pub async fn webgpu_probe() -> Result<String, JsValue> {
    use wgpu::util::DeviceExt;
    console_error_panic_hook::set_once();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok_or_else(|| JsValue::from_str("no WebGPU adapter (navigator.gpu present but no usable device)"))?;
    let info = adapter.get_info();

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("mpee-probe"),
            required_features: wgpu::Features::empty(),
            // Conservative limits, and (on this wgpu) without the removed
            // maxInterStageShaderComponents that current Chrome rejects.
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
        }, None)
        .await
        .map_err(|e| JsValue::from_str(&format!("request_device failed: {e}")))?;

    // in[i] = i, expect out[i] = i².
    let n: u32 = 64;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let bytes = (n as usize * std::mem::size_of::<f32>()) as u64;

    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("in"),
        contents: bytemuck_cast(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("square"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&inp)) { outp[i] = inp[i] * inp[i]; }
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("square-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: in_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1); // n=64 == one workgroup
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, bytes);
    queue.submit(Some(enc.finish()));

    // Async readback (no device.poll(Wait) on wasm — the browser drives it).
    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
    device.poll(wgpu::Maintain::Poll);
    rx.await
        .map_err(|_| JsValue::from_str("map_async channel dropped"))?
        .map_err(|e| JsValue::from_str(&format!("buffer map failed: {e:?}")))?;
    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck_from(&data);
    drop(data);
    staging.unmap();

    // Correctness: out[3] should be 9, out[7] = 49.
    let ok = (out.get(3).copied().unwrap_or(0.0) - 9.0).abs() < 1e-3
        && (out.get(7).copied().unwrap_or(0.0) - 49.0).abs() < 1e-3;
    let sample: Vec<f32> = out.iter().take(8).copied().collect();
    Ok(serde_json::json!({
        "ok": ok,
        "backend": format!("{:?}", info.backend),
        "adapter": if info.name.is_empty() { "WebGPU".to_string() } else { info.name },
        "sample": sample,
    })
    .to_string())
}

// Minimal POD casts (avoid pulling bytemuck just for the probe).
fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn bytemuck_from(b: &[u8]) -> Vec<f32> {
    let mut out = vec![0f32; b.len() / 4];
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), out.as_mut_ptr() as *mut u8, b.len()); }
    out
}
fn as_bytes<T>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}
fn vec_u32(b: &[u8]) -> Vec<u32> {
    let mut v = vec![0u32; b.len() / 4];
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, b.len()); }
    v
}

// Acquire a WebGPU device + queue once, then reuse across many kernel
// dispatches (the relocate loop runs dozens of them — re-creating a device each
// time would dominate the runtime).
async fn acquire_device() -> Result<(wgpu::Device, wgpu::Queue), JsValue> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU, ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false, compatible_surface: None,
        })
        .await
        .ok_or_else(|| JsValue::from_str("no WebGPU adapter"))?;
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("mpee-gpu"), required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(), memory_hints: wgpu::MemoryHints::Performance,
        }, None)
        .await
        .map_err(|e| JsValue::from_str(&format!("request_device: {e}")))
}

// GPU intra-route 2-opt: one workgroup per route. Each route's node sequence
// (matrix indices, depot at both ends) lives in workgroup memory; threads scan
// edge-pair reversals against the distance matrix, reduce to the single best
// improving move, apply it, and repeat until no improvement (best-improvement
// 2-opt). Routes longer than MAXLEN are left unchanged.
const TWO_OPT_WGSL: &str = r#"
struct Params { n: u32, max_sweeps: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>        matrix:  array<i32>;
@group(0) @binding(1) var<storage, read_write>  seqs:    array<u32>;
@group(0) @binding(2) var<storage, read>         offsets: array<u32>;
@group(0) @binding(3) var<uniform>               P:       Params;

const WG: u32 = 64u;
const MAXLEN: u32 = 256u;
var<workgroup> path: array<u32, 256>;
var<workgroup> plen: u32;
var<workgroup> bd: array<i32, 64>;
var<workgroup> bi: array<u32, 64>;
var<workgroup> bj: array<u32, 64>;

fn d(a: u32, b: u32) -> i32 { return matrix[a * P.n + b]; }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.x;
    let s = offsets[r];
    let e = offsets[r + 1u];
    let len = e - s;
    let t = lid.x;
    if (t == 0u) { plen = len; }
    workgroupBarrier();
    if (len < 4u || len > MAXLEN) { return; }
    for (var k = t; k < len; k = k + WG) { path[k] = seqs[s + k]; }
    workgroupBarrier();

    for (var sweep = 0u; sweep < P.max_sweeps; sweep = sweep + 1u) {
        var localBest: i32 = 0;
        var lI: u32 = 0u;
        var lJ: u32 = 0u;
        // i indexes the first edge (path[i],path[i+1]); j the second
        // (path[j],path[j+1]). j+1 must stay in-bounds (<= plen-1), so j <=
        // plen-2 — reversing path[i+1..=j] never moves the trailing depot.
        for (var i = 0u; i + 1u < plen; i = i + 1u) {
            for (var j = i + 2u; j + 1u < plen; j = j + 1u) {
                if (((i * plen + j) % WG) == t) {
                    let a = path[i];
                    let b = path[i + 1u];
                    let c = path[j];
                    let f = path[j + 1u];
                    let delta = d(a, c) + d(b, f) - d(a, b) - d(c, f);
                    if (delta < localBest) { localBest = delta; lI = i; lJ = j; }
                }
            }
        }
        bd[t] = localBest; bi[t] = lI; bj[t] = lJ;
        workgroupBarrier();
        // Best-improvement: thread 0 picks the single best move and applies it.
        // No early break (a barrier after a workgroup-var-dependent break is
        // non-uniform and rejected); extra sweeps after convergence are no-ops.
        if (t == 0u) {
            var best: i32 = 0;
            var Ii: u32 = 0u;
            var Jj: u32 = 0u;
            for (var k = 0u; k < WG; k = k + 1u) {
                if (bd[k] < best) { best = bd[k]; Ii = bi[k]; Jj = bj[k]; }
            }
            if (best < 0) {
                var lo = Ii + 1u;
                var hi = Jj;
                loop {
                    if (lo >= hi) { break; }
                    let tmp = path[lo]; path[lo] = path[hi]; path[hi] = tmp;
                    lo = lo + 1u; hi = hi - 1u;
                }
            }
        }
        workgroupBarrier();
    }
    for (var k = t; k < plen; k = k + WG) { seqs[s + k] = path[k]; }
}
"#;

async fn run_2opt_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    matrix: &[i32],
    n: u32,
    seqs: &mut Vec<u32>,
    offsets: &[u32],
    num_routes: u32,
) -> Result<(), JsValue> {
    use wgpu::util::DeviceExt;

    let mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matrix"), contents: as_bytes(matrix), usage: wgpu::BufferUsages::STORAGE,
    });
    let seq_bytes = as_bytes(seqs).to_vec();
    let seq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seqs"), contents: &seq_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let off_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offsets"), contents: as_bytes(offsets), usage: wgpu::BufferUsages::STORAGE,
    });
    let params: [u32; 4] = [n, 300, 0, 0];
    let par_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: as_bytes(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"), size: seq_bytes.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("2opt"), source: wgpu::ShaderSource::Wgsl(TWO_OPT_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("2opt-pipeline"), layout: None, module: &shader,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: mat_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: seq_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: off_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: par_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(num_routes, 1, 1);
    }
    enc.copy_buffer_to_buffer(&seq_buf, 0, &staging, 0, seq_bytes.len() as u64);
    queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::Maintain::Poll);
    rx.await
        .map_err(|_| JsValue::from_str("2opt map channel dropped"))?
        .map_err(|e| JsValue::from_str(&format!("2opt buffer map: {e:?}")))?;
    let data = slice.get_mapped_range();
    let updated = vec_u32(&data);
    drop(data);
    staging.unmap();
    seqs.copy_from_slice(&updated);
    Ok(())
}

// GPU cross-route relocate: evaluate the WHOLE relocate neighbourhood in
// parallel (move each interior stop into every position of every other route),
// reduce to the single best improving move. The CPU applies it (trivial Vec
// edit) and re-evaluates — so the heavy O(stops × routes × positions) search is
// on the GPU while in-kernel mutation (the bug-prone part) is avoided. Relocate
// is *exact* on asymmetric road distances (no segment reversal), so it uses the
// true matrix. Returns (best_delta, src_pos, dst_route, dst_pos); delta ≥ 0
// means no improving move.
const RELOCATE_WGSL: &str = r#"
struct RP { n: u32, total: u32, num_routes: u32, cap: u32 };
@group(0) @binding(0) var<storage, read>        matrix:   array<i32>;
@group(0) @binding(1) var<storage, read>        seqs:     array<u32>;
@group(0) @binding(2) var<storage, read>        offsets:  array<u32>;
@group(0) @binding(3) var<storage, read>        route_of: array<u32>;
@group(0) @binding(4) var<storage, read_write>  outm:     array<i32>;
@group(0) @binding(5) var<uniform>              P:        RP;

const WGN: u32 = 256u;
var<workgroup> bD: array<i32, 256>;
var<workgroup> bS: array<u32, 256>;
var<workgroup> bR: array<u32, 256>;
var<workgroup> bP: array<u32, 256>;
fn d(a: u32, b: u32) -> i32 { return matrix[a * P.n + b]; }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var bestD: i32 = 0;
    var bs: u32 = 0u; var br: u32 = 0u; var bp: u32 = 0u;
    for (var si = 0u; si < P.total; si = si + 1u) {
        let a = route_of[si];
        let aStart = offsets[a];
        let aEnd = offsets[a + 1u] - 1u;
        if (si == aStart || si == aEnd) { continue; }   // skip depot ends
        let prev = seqs[si - 1u];
        let s = seqs[si];
        let nxt = seqs[si + 1u];
        let removal = d(prev, nxt) - d(prev, s) - d(s, nxt);   // saving from A
        for (var rb = 0u; rb < P.num_routes; rb = rb + 1u) {
            if (rb == a) { continue; }
            let loadB = (offsets[rb + 1u] - offsets[rb]) - 2u;
            if (loadB + 1u > P.cap) { continue; }              // capacity
            let lo = offsets[rb];
            let hi = offsets[rb + 1u] - 1u;
            for (var pos = lo; pos < hi; pos = pos + 1u) {
                if (((si * 1000003u + rb * 9176u + pos) % WGN) == t) {
                    let bpn = seqs[pos];
                    let bp1 = seqs[pos + 1u];
                    let ins = d(bpn, s) + d(s, bp1) - d(bpn, bp1);  // cost into B
                    let delta = ins - removal;
                    if (delta < bestD) { bestD = delta; bs = si; br = rb; bp = pos; }
                }
            }
        }
    }
    bD[t] = bestD; bS[t] = bs; bR[t] = br; bP[t] = bp;
    workgroupBarrier();
    if (t == 0u) {
        var best: i32 = 0; var s_: u32 = 0u; var r_: u32 = 0u; var p_: u32 = 0u;
        for (var k = 0u; k < WGN; k = k + 1u) {
            if (bD[k] < best) { best = bD[k]; s_ = bS[k]; r_ = bR[k]; p_ = bP[k]; }
        }
        outm[0] = best; outm[1] = i32(s_); outm[2] = i32(r_); outm[3] = i32(p_);
    }
}
"#;

async fn run_relocate_eval(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    matrix: &[i32],
    n: u32,
    seqs: &[u32],
    offsets: &[u32],
    route_of: &[u32],
    cap: u32,
) -> Result<(i32, usize, usize, usize), JsValue> {
    use wgpu::util::DeviceExt;
    let mk = |label, contents: &[u8]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label), contents, usage: wgpu::BufferUsages::STORAGE,
    });
    let mat_buf = mk("matrix", as_bytes(matrix));
    let seq_buf = mk("seqs", as_bytes(seqs));
    let off_buf = mk("offsets", as_bytes(offsets));
    let rof_buf = mk("route_of", as_bytes(route_of));
    let out_init: [i32; 4] = [0, 0, 0, 0];
    let out_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("outm"), contents: as_bytes(&out_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let params: [u32; 4] = [n, seqs.len() as u32, (offsets.len() - 1) as u32, cap];
    let par_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: as_bytes(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("relocate"), source: wgpu::ShaderSource::Wgsl(RELOCATE_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("relocate-pipeline"), layout: None, module: &shader,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: mat_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: seq_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: off_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: rof_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: par_buf.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, 16);
    queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::Maintain::Poll);
    rx.await
        .map_err(|_| JsValue::from_str("relocate map channel dropped"))?
        .map_err(|e| JsValue::from_str(&format!("relocate buffer map: {e:?}")))?;
    let data = slice.get_mapped_range();
    let out: Vec<i32> = {
        let mut v = vec![0i32; 4];
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), v.as_mut_ptr() as *mut u8, 16); }
        v
    };
    drop(data);
    staging.unmap();
    Ok((out[0], out[1] as usize, out[2] as usize, out[3] as usize))
}
