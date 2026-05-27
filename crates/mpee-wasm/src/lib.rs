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

        // Construction only (max_local_search_passes = 0) — the GPU does the 2-opt.
        let mut problem = Problem::default();
        for v in 0..vehicles {
            problem.vehicles.push(Vehicle {
                id: (v + 1) as u64,
                start: Some(Location::from_index(0)),
                end: Some(Location::from_index(0)),
                capacity: vec![capacity], skills: vec![], time_window: None, speed_factor: 1.0,
                max_tasks: None, max_travel_time: None, max_distance: None,
                fixed: 0.0, per_hour: 3600.0, profile: "car".into(), description: None,
            });
        }
        for i in 0..stops.len() {
            problem.jobs.push(Job {
                id: (i + 1) as u64, location: Location::from_index(i + 1), kind: Default::default(),
                service: 0, setup: 0, delivery: vec![1], pickup: vec![], skills: vec![],
                priority: 0, time_windows: vec![], description: None,
            });
        }
        let matrix = Matrix { n, durations: dur_i.clone(), distances: Some(dist_i.clone()) };
        let cfg = SolverConfig {
            multi_start: 1, granular_k: Some(40), max_local_search_passes: 0,
            time_limit_ms: Some(50), verbose: false, use_gpu: false, ..Default::default()
        };
        let sol = solve_with_matrix(&problem, &matrix, &cfg);

        // Flatten non-empty routes into GPU sequences (matrix indices, depot=0
        // at both ends): seqs = [0, j+1, …, 0] per route, offsets delimit them.
        let mut route_vehicle: Vec<u64> = Vec::new();
        let mut seqs: Vec<u32> = Vec::new();
        let mut offsets: Vec<u32> = vec![0];
        // brooom decides the assignment (which stops each vehicle gets); the
        // GPU's job is the *visiting order*. Start each route from the naive
        // input order (the order the stops were listed) so the GPU's 2-opt has
        // real work and its improvement is visible.
        for r in &sol.routes {
            if r.steps.is_empty() { continue; }
            route_vehicle.push(problem.vehicles[r.vehicle_idx].id);
            let mut jobs_mi: Vec<u32> = r.steps.iter()
                .filter_map(|s| if let TaskRef::Job(j) = s { Some((*j as u32) + 1) } else { None })
                .collect();
            jobs_mi.sort_unstable(); // input order (job id = input index + 1)
            seqs.push(0);
            seqs.extend_from_slice(&jobs_mi);
            seqs.push(0);
            offsets.push(seqs.len() as u32);
        }
        let num_routes = route_vehicle.len() as u32;

        let route_dist = |seq: &[u32]| -> i64 {
            seq.windows(2).map(|w| dist_i[w[0] as usize * n + w[1] as usize] as i64).sum()
        };
        let before: i64 = (0..route_vehicle.len())
            .map(|ri| route_dist(&seqs[offsets[ri] as usize..offsets[ri + 1] as usize]))
            .sum();

        // ---- GPU 2-opt over all routes (one workgroup each) ----
        // Road distances are asymmetric (one-way streets), but a 2-opt edge
        // swap reverses a segment's travel direction — so feed the kernel a
        // symmetrised matrix (min of both directions) to keep its delta valid.
        let mut sym = vec![0i32; n * n];
        for a in 0..n {
            for b in 0..n {
                sym[a * n + b] = dist_i[a * n + b].min(dist_i[b * n + a]);
            }
        }
        let seqs_before = seqs.clone();
        if num_routes > 0 {
            run_2opt_gpu(&sym, n as u32, &mut seqs, &offsets, num_routes).await?;
        }
        // Safety net: keep, per route, whichever of (construction, GPU result)
        // is shorter on the TRUE asymmetric distance — so the GPU pass can
        // never make a route worse, even when the symmetric approximation is off.
        for ri in 0..route_vehicle.len() {
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
        let unassigned: Vec<usize> = sol.unassigned.iter().filter_map(|t| match t {
            TaskRef::Job(j) => Some(*j), _ => None,
        }).collect();

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
    matrix: &[i32],
    n: u32,
    seqs: &mut Vec<u32>,
    offsets: &[u32],
    num_routes: u32,
) -> Result<(), JsValue> {
    use wgpu::util::DeviceExt;
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
        .ok_or_else(|| JsValue::from_str("no WebGPU adapter for 2-opt"))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("mpee-2opt"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
        }, None)
        .await
        .map_err(|e| JsValue::from_str(&format!("request_device (2opt): {e}")))?;

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
