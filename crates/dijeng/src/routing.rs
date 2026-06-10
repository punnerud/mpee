//! High-level routing service: snap (lat,lon) → CSR vertex, run a CH query,
//! unpack the path to a polyline, return distance + duration estimate.
//!
//! Wraps `ContractionHierarchy` + per-vertex coordinates. Holds the inverse
//! permutation needed to map internal CH IDs back to CSR IDs (and hence
//! coords).
//!
//! Duration is currently a global `speed_kmh * distance` estimate; per-edge
//! speeds (from OSM `highway=*` and `maxspeed=*`) are a follow-up.

use crate::buffer::Buffer;
use crate::ch::{self, ContractionHierarchy};
#[cfg(feature = "native")]
use crate::ch::PathScratch;
use crate::geo_index::LatLonGrid;

/// Panic message when a routing matrix is requested on a geocoding-only
/// service (opened via `new_geocoding`, i.e. without a `.ch` cache).
const CH_REQUIRED: &str =
    "matrix requires a CH cache — this service was opened for geocoding only (no .ch)";

/// Membership mask (by INTERNAL id) of the largest weakly-connected component
/// of the CH graph. Undirected BFS treating fwd+bwd edges as one edge set;
/// shortcuts only mirror base-graph connectivity, so this equals the base
/// graph's weak components.
fn largest_component(ch: &ContractionHierarchy) -> Vec<bool> {
    let n = ch.graph_fwd.n;
    let mut comp_of: Vec<u32> = vec![u32::MAX; n];
    let mut sizes: Vec<u32> = Vec::new();
    let mut stack: Vec<u32> = Vec::with_capacity(1024);
    for start in 0..n {
        if comp_of[start] != u32::MAX {
            continue;
        }
        let cid = sizes.len() as u32;
        let mut size = 0u32;
        comp_of[start] = cid;
        stack.push(start as u32);
        while let Some(v) = stack.pop() {
            size += 1;
            let vu = v as usize;
            for (g_ref, _) in [(&ch.graph_fwd, ()), (&ch.graph_bwd, ())] {
                let s = g_ref.head[vu] as usize;
                let e = g_ref.head[vu + 1] as usize;
                for k in s..e {
                    let w = g_ref.edge_to[k] as usize;
                    if comp_of[w] == u32::MAX {
                        comp_of[w] = cid;
                        stack.push(w as u32);
                    }
                }
            }
        }
        sizes.push(size);
    }
    let largest = sizes
        .iter()
        .enumerate()
        .max_by_key(|&(_, &s)| s)
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    comp_of.into_iter().map(|c| c == largest).collect()
}

pub struct RoutingService {
    /// The contraction hierarchy needed for `route`/`matrix`. `None` when the
    /// service was opened for **geocoding only** (no `.ch`), which lets a pure
    /// reverse/forward/intersection service skip loading the largest cache file.
    ch: Option<ContractionHierarchy>,
    pub coords: Buffer<(f32, f32)>,
    /// `inv_perm[internal_id] = csr_id`. Built at construction time from
    /// `ch.perm` so we can map a CH-path back to coordinates (empty without ch).
    inv_perm: Buffer<u32>,
    /// Spatial index for sub-100 µs nearest-vertex lookup. For a ROUTING
    /// service this indexes only nodes that are routable in the mounted
    /// profile (degree ≥ 1 in the CH graph) — central-city coordinates would
    /// otherwise snap to footpath/rail nodes with no car edges and every
    /// route from them would be "no path" (long-standing bug surfaced by the
    /// trip service). Geocoding-only services index every node.
    snap_grid: LatLonGrid,
    /// Grid-index → CSR node id for the filtered grid. Empty = identity
    /// (geocoding-only service, grid built over all of `coords`).
    snap_ids: Vec<u32>,
    /// Coordinates parallel to `snap_ids` (what the filtered grid indexes).
    snap_coords: Vec<(f32, f32)>,
    /// Optional street-name sidecar, enabling offline geocoding. Reverse
    /// lookups reuse `snap_grid`; forward lookups scan the distinct names.
    /// `None` when no `.names` sidecar was attached.
    names: Option<crate::names::NameTable>,
    /// Optional house-number address index (`.addr` sidecar). Independent of the
    /// routing graph (own coords + grid). Enables address-level forward/reverse.
    addresses: Option<crate::addresses::AddressIndex>,
    /// Optional plain road adjacency (the `.pp` forward CSR, same node order as
    /// `coords`/`names`). Used by `street_segments` to draw a whole street.
    road_graph: Option<crate::graph::CsrGraph>,
}

#[derive(Debug, Clone)]
pub struct RouteResponse {
    /// Distance along the path, in metres.
    pub distance_m: f32,
    /// Duration estimate, in seconds.
    pub duration_s: f32,
    /// Path geometry as (lat, lon) pairs, including the snapped start and end.
    pub geometry: Vec<(f32, f32)>,
    /// Snapped source location (nearest road node to the requested point).
    pub source_snapped: (f32, f32),
    /// Snapped destination location.
    pub destination_snapped: (f32, f32),
}

/// Result of the trip service: the optimised visiting order over the input
/// waypoints plus one routed leg per consecutive pair (closing leg included
/// for roundtrips).
pub struct TripResponse {
    /// Visiting order as indices into the input waypoint slice. Starts at 0;
    /// open paths end at `len - 1`.
    pub order: Vec<usize>,
    /// One routed leg per consecutive pair in `order` (+ the closing leg when
    /// roundtrip), in travel order.
    pub legs: Vec<RouteResponse>,
    pub duration_s: f32,
    pub distance_m: f32,
}

impl RoutingService {
    /// `coords[csr_id] = (lat, lon)`. `ch` was built from the same CSR graph,
    /// so `ch.perm[csr_id]` gives the internal (rank-ordered) id.
    /// CH `edge_w` must be **duration in seconds** (i.e. cache magic
    /// `SSSPCH1C`); distance is recomputed by haversine over the path.
    pub fn new(ch: ContractionHierarchy, coords: Buffer<(f32, f32)>) -> Self {
        let n = coords.len();
        assert_eq!(n, ch.graph_fwd.n, "coords.len() must equal ch.graph_fwd.n");
        let mut inv = vec![0u32; n];
        for csr_id in 0..n {
            let internal = ch.perm[csr_id] as usize;
            inv[internal] = csr_id as u32;
        }
        // Cell sizing: aim for ~10–30 vertices/cell on typical road graphs.
        // 0.005° (~550 m at mid-latitudes) hits this sweet spot for London;
        // for a continent-scale graph use slightly larger (0.01°).
        let cell_size_deg = if n > 5_000_000 { 0.01 } else { 0.005 };
        // Snap only to nodes the profile can actually route between: members
        // of the LARGEST weakly-connected component of the profile graph.
        // OSM car graphs are full of tiny disconnected fragments (service-road
        // stubs, parking aisles, pedestrian-zone slivers) — in central London
        // the nearest node to a landmark is frequently a 2-node stub, and
        // every route from it is "no path" (long-standing bug surfaced by the
        // trip service). One undirected BFS over the CH edges (originals +
        // shortcuts share the base graph's connectivity) finds the giant
        // component in ~100 ms at mount time.
        let comp = largest_component(&ch);
        let mut snap_ids: Vec<u32> = Vec::with_capacity(n);
        let mut snap_coords: Vec<(f32, f32)> = Vec::with_capacity(n);
        for csr_id in 0..n {
            if comp[ch.perm[csr_id] as usize] {
                snap_ids.push(csr_id as u32);
                snap_coords.push(coords[csr_id]);
            }
        }
        let snap_grid = LatLonGrid::from_coords(&snap_coords, cell_size_deg);
        Self {
            ch: Some(ch),
            coords,
            inv_perm: inv.into(),
            snap_grid,
            snap_ids,
            snap_coords,
            names: None,
            addresses: None,
            road_graph: None,
        }
    }

    /// Open a **geocoding-only** service from coordinates alone (the `.pp`
    /// cache) — no contraction hierarchy. `reverse`/`geocode`/`intersection`
    /// (after `set_names`) and `snap`/`nearest_node` work; `route`/`matrix`
    /// do not (they need a `.ch`). This avoids loading the largest cache file
    /// when all you want is street ⇄ coordinate lookups.
    pub fn new_geocoding(coords: Buffer<(f32, f32)>) -> Self {
        let n = coords.len();
        let cell_size_deg = if n > 5_000_000 { 0.01 } else { 0.005 };
        let snap_grid = LatLonGrid::from_coords(coords.as_slice(), cell_size_deg);
        Self {
            ch: None,
            coords,
            inv_perm: Buffer::from(Vec::new()),
            snap_grid,
            snap_ids: Vec::new(),   // identity: grid indexes all of `coords`
            snap_coords: Vec::new(),
            names: None,
            addresses: None,
            road_graph: None,
        }
    }

    /// Attach the plain road adjacency (`.pp` forward CSR) so `street_segments`
    /// can return a whole street's geometry. Same node order as `coords`.
    pub fn set_road_graph(&mut self, g: crate::graph::CsrGraph) {
        self.road_graph = Some(g);
    }

    /// All road edges belonging to a named street, as `(lat1,lon1,lat2,lon2)`
    /// segments — i.e. the street drawn as a polyline set. Resolves the name
    /// like `geocode`, takes the street's node set, and keeps the graph edges
    /// whose both endpoints are in that set. Empty without a names sidecar +
    /// road graph, or if the name doesn't resolve.
    pub fn street_segments(&self, query: &str) -> Vec<(f32, f32, f32, f32)> {
        let (names, g) = match (self.names.as_ref(), self.road_graph.as_ref()) {
            (Some(n), Some(g)) => (n, g),
            _ => return Vec::new(),
        };
        let id = match names.find_id(query) {
            Some(i) => i,
            None => return Vec::new(),
        };
        let nodes = names.street_nodes(id);
        let set: std::collections::HashSet<u32> = nodes.iter().copied().collect();
        let head = g.head.as_slice();
        let edge_to = g.edge_to.as_slice();
        let coords = self.coords.as_slice();
        let mut out = Vec::new();
        for &u in nodes {
            let (a, b) = (head[u as usize] as usize, head[u as usize + 1] as usize);
            for &v in &edge_to[a..b] {
                if u < v && set.contains(&v) {
                    let (la, lo) = coords[u as usize];
                    let (lb, lob) = coords[v as usize];
                    out.push((la, lo, lb, lob));
                }
            }
        }
        out
    }

    /// Whether a CH is loaded — i.e. `route`/`matrix` are available. False for
    /// a geocoding-only service opened with [`new_geocoding`](Self::new_geocoding).
    pub fn has_routing(&self) -> bool {
        self.ch.is_some()
    }

    /// Node count of the loaded graph.
    pub fn node_count(&self) -> usize {
        self.coords.len()
    }

    /// Attach a street-name sidecar (built next to the `.pp`/`.ch` caches),
    /// enabling `reverse` and `geocode`. The sidecar's node count must match
    /// the loaded graph; `load_mmap` already enforces that.
    pub fn set_names(&mut self, names: crate::names::NameTable) {
        self.names = Some(names);
    }

    /// Whether a street-name sidecar is loaded (geocoding available).
    pub fn has_names(&self) -> bool {
        self.names.is_some()
    }

    /// Attach a house-number address index (the `.addr` sidecar). Enables
    /// `geocode_address`/`reverse_address`/`address_query`. Independent of the
    /// graph, so no node-count check.
    pub fn set_addresses(&mut self, idx: crate::addresses::AddressIndex) {
        self.addresses = Some(idx);
    }

    /// Whether a house-number address index is loaded.
    pub fn has_addresses(&self) -> bool {
        self.addresses.is_some()
    }

    /// Forward address geocode: `street` + `number` → a coordinate-bearing hit.
    /// `near` disambiguates a street name shared by several towns. Exact number
    /// wins; otherwise the nearest number on the street (with `approximate`
    /// set). `None` if no address sidecar / the street doesn't resolve.
    pub fn geocode_address(
        &self,
        street: &str,
        number: &str,
        near: Option<(f32, f32)>,
    ) -> Option<crate::addresses::AddressHit> {
        self.addresses.as_ref()?.forward(street, number, near)
    }

    /// Reverse address geocode: the nearest address point to `(lat, lon)`.
    /// `None` if no address sidecar is loaded.
    pub fn reverse_address(&self, lat: f32, lon: f32) -> Option<crate::addresses::AddressHit> {
        self.addresses.as_ref()?.reverse(lat, lon)
    }

    /// Convenience: resolve a *combined* query like "Karl Johans gate 42" at
    /// address level. Returns `None` when the query has no trailing number or no
    /// address matches — so callers can fall back to street-level [`geocode`].
    pub fn address_query(
        &self,
        query: &str,
        near: Option<(f32, f32)>,
    ) -> Option<crate::addresses::AddressHit> {
        let (street, number) = split_house_number(query);
        self.geocode_address(street, number?, near)
    }

    /// Reverse-geocode: the street name nearest to `(lat, lon)`. Snaps to the
    /// nearest road node (the same grid `route` uses) and returns that node's
    /// street name. `None` if no sidecar is loaded or the node has no name.
    pub fn reverse(&self, lat: f32, lon: f32) -> Option<&str> {
        let node = self.nearest_node(lat, lon);
        self.names.as_ref()?.name_of(node)
    }

    /// Forward-geocode: find a street by name and return its coordinate plus
    /// the matched street name. Case-insensitive; an exact match wins, else
    /// the first street whose name contains the query. `None` if no sidecar
    /// is loaded or nothing matches.
    pub fn geocode(&self, query: &str) -> Option<(f32, f32, &str)> {
        let names = self.names.as_ref()?;
        let node = names.find(query)?;
        let (lat, lon) = self.coords[node as usize];
        let name = names.name_of(node)?;
        Some((lat, lon, name))
    }

    /// Intersection search: every coordinate where streets `a` and `b` meet
    /// (set intersection of their road-node lists). Names are resolved like
    /// `geocode` (case-insensitive, substring). Empty if no sidecar is loaded,
    /// a name doesn't resolve, or the streets share no node. Several results
    /// are possible (streets that cross more than once or run together).
    pub fn intersection(&self, a: &str, b: &str) -> Vec<(f32, f32)> {
        match self.names.as_ref() {
            Some(names) => names
                .intersections(a, b)
                .into_iter()
                .map(|node| self.coords[node as usize])
                .collect(),
            None => Vec::new(),
        }
    }

    /// Up to `limit` street-name suggestions for `query` (type-ahead). Empty
    /// without a names sidecar.
    pub fn suggest(&self, query: &str, limit: usize) -> Vec<String> {
        self.names.as_ref().map(|n| n.suggest(query, limit)).unwrap_or_default()
    }

    /// Forward-geocode disambiguated by a reference point: among all road nodes
    /// of the matched street (which, on a multi-city cache, may span several
    /// towns sharing the name), return the one nearest `(ref_lat, ref_lon)`.
    /// Use this to pick "Munkegata in Trondheim" rather than an arbitrary first
    /// hit. `None` if no sidecar / the name doesn't resolve.
    pub fn geocode_near(&self, query: &str, ref_lat: f32, ref_lon: f32) -> Option<(f32, f32, &str)> {
        let names = self.names.as_ref()?;
        let id = names.find_id(query)?;
        let nodes = names.street_nodes(id);
        let best = *nodes.iter().min_by(|&&a, &&b| {
            let (la, lo) = self.coords[a as usize];
            let (lb, lob) = self.coords[b as usize];
            haversine_m(ref_lat, ref_lon, la, lo)
                .partial_cmp(&haversine_m(ref_lat, ref_lon, lb, lob))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        let (lat, lon) = self.coords[best as usize];
        Some((lat, lon, names.name_by_id(id)?))
    }

    /// Intersection search disambiguated by a reference point: crossings of `a`
    /// and `b` sorted nearest-first to `(ref_lat, ref_lon)`, and (when
    /// `radius_km` is given) filtered to that radius. Lets a country cache
    /// answer "Prinsens gate × Kongens gate near Trondheim" instead of
    /// returning every same-named crossing nationwide.
    pub fn intersection_near(
        &self,
        a: &str,
        b: &str,
        ref_lat: f32,
        ref_lon: f32,
        radius_km: Option<f64>,
    ) -> Vec<(f32, f32)> {
        let mut hits = self.intersection(a, b);
        hits.sort_by(|p, q| {
            haversine_m(ref_lat, ref_lon, p.0, p.1)
                .partial_cmp(&haversine_m(ref_lat, ref_lon, q.0, q.1))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(r_km) = radius_km {
            let r_m = (r_km * 1000.0) as f32;
            hits.retain(|p| haversine_m(ref_lat, ref_lon, p.0, p.1) <= r_m);
        }
        hits
    }

    /// Nearest road node by squared planar distance (longitude scaled by
    /// `cos(lat)` for correct ordering). Backed by a uniform grid index;
    /// typically ~50 µs on city/country-scale graphs.
    pub fn nearest_node(&self, lat: f32, lon: f32) -> u32 {
        if self.snap_ids.is_empty() {
            self.snap_grid
                .nearest(lat, lon, self.coords.as_slice())
                .unwrap_or(0)
        } else {
            self.snap_grid
                .nearest(lat, lon, &self.snap_coords)
                .map(|i| self.snap_ids[i as usize])
                .unwrap_or(0)
        }
    }

    /// K nearest road nodes to (lat, lon), ascending by ground distance —
    /// the OSRM `/nearest?number=K` service. Each hit is
    /// `(node_id, snapped_lat, snapped_lon)`.
    pub fn nearest_nodes(&self, lat: f32, lon: f32, k: usize) -> Vec<(u32, f32, f32)> {
        let hits = if self.snap_ids.is_empty() {
            self.snap_grid.nearest_k(lat, lon, self.coords.as_slice(), k)
        } else {
            self.snap_grid
                .nearest_k(lat, lon, &self.snap_coords, k)
                .into_iter()
                .map(|i| self.snap_ids[i as usize])
                .collect()
        };
        hits.into_iter()
            .map(|id| {
                let (la, lo) = self.coords[id as usize];
                (id, la, lo)
            })
            .collect()
    }

    /// Many-to-many duration matrix: `out[i*dsts.len() + j]` is the
    /// shortest-time route from `srcs[i]` to `dsts[j]` (in seconds).
    /// Inputs are (lat, lon); the service snaps each to its nearest road
    /// node before running the bucket-based CH MMM algorithm.
    ///
    /// Returns `(durations, snapped_sources, snapped_destinations)`.
    #[cfg(feature = "native")]
    pub fn matrix(
        &self,
        srcs: &[(f32, f32)],
        dsts: &[(f32, f32)],
    ) -> (Vec<f32>, Vec<(f32, f32)>, Vec<(f32, f32)>) {
        let ch = self.ch.as_ref().expect(CH_REQUIRED);
        let snap_srcs: Vec<u32> = srcs.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let snap_dsts: Vec<u32> = dsts.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let int_srcs: Vec<u32> = snap_srcs
            .iter()
            .map(|&csr| ch.perm[csr as usize])
            .collect();
        let int_dsts: Vec<u32> = snap_dsts
            .iter()
            .map(|&csr| ch.perm[csr as usize])
            .collect();
        let durations = ch::matrix(ch, &int_srcs, &int_dsts);
        let snapped_src_coords = snap_srcs
            .iter()
            .map(|&csr| self.coords[csr as usize])
            .collect();
        let snapped_dst_coords = snap_dsts
            .iter()
            .map(|&csr| self.coords[csr as usize])
            .collect();
        (durations, snapped_src_coords, snapped_dst_coords)
    }

    /// Trip service (OSRM `/trip`): order `pts` into the shortest visiting
    /// sequence and return per-leg routes. `roundtrip = true` gives a closed
    /// tour starting/ending at `pts[0]`; `false` gives an open path pinned to
    /// start at `pts[0]` and end at the last point. Returns `None` when there
    /// is no CH, fewer than 2 points, or no feasible ordering (disconnected).
    pub fn trip(&self, pts: &[(f32, f32)], roundtrip: bool) -> Option<TripResponse> {
        self.ch.as_ref()?;
        let n = pts.len();
        if n < 2 {
            return None;
        }
        // Waypoint duration matrix (tiny: n ≤ a few dozen in practice).
        let (durations, _, _) = self.matrix(pts, pts);
        let order = crate::trip::tsp_order(&durations, n, roundtrip);
        if !crate::trip::tour_cost(&durations, n, &order, roundtrip).is_finite() {
            return None;
        }
        // Per-leg routes with geometry, following the chosen order.
        let mut legs: Vec<RouteResponse> = Vec::with_capacity(n);
        let mut total_dur = 0.0f32;
        let mut total_dist = 0.0f32;
        let leg_pairs = order
            .windows(2)
            .map(|w| (w[0], w[1]))
            .chain(roundtrip.then(|| (*order.last().unwrap(), order[0])));
        for (a, b) in leg_pairs {
            let (sla, slo) = pts[a];
            let (dla, dlo) = pts[b];
            let r = self.route(sla, slo, dla, dlo)?;
            total_dur += r.duration_s;
            total_dist += r.distance_m;
            legs.push(r);
        }
        Some(TripResponse { order, legs, duration_s: total_dur, distance_m: total_dist })
    }

    /// Map matching (Newson-Krumm HMM): reconstruct the road path a noisy GPS
    /// trace was driven on. `sigma_m` is the GPS noise (≈15 m urban). Returns
    /// one matched road point per input ping plus a confidence in [0, 1].
    pub fn match_trace(
        &self,
        trace: &[(f32, f32)],
        k: usize,
        sigma_m: f32,
    ) -> Option<crate::matching::MatchResult> {
        let ch = self.ch.as_ref()?;
        Some(crate::matching::match_trace(
            ch,
            |la, lo| self.nearest_nodes(la, lo, k),
            |csr| ch.perm[csr as usize],
            trace,
            sigma_m,
        ))
    }

    /// Isochrone bands from (lat, lon): polygons covering everything reachable
    /// within each of `limits` (seconds, ascending; metres when `metric_dist`).
    /// `cell_deg` controls polygon resolution (0.0015° ≈ 150 m). Rings come
    /// back closed, in (lat, lon).
    pub fn isochrone(
        &self,
        lat: f32,
        lon: f32,
        limits: &[f32],
        cell_deg: f32,
        metric_dist: bool,
    ) -> Option<Vec<crate::isochrone::IsochroneBand>> {
        let ch = self.ch.as_ref()?;
        let src = self.nearest_node(lat, lon);
        let src_int = ch.perm[src as usize];
        Some(crate::isochrone::isochrone(
            ch,
            |internal| self.coords[self.inv_perm[internal as usize] as usize],
            src_int,
            limits,
            cell_deg,
            metric_dist,
        ))
    }

    /// Variant of `matrix` that also returns per-cell distances. With a
    /// dual-channel CH (`edge_dist_*` populated, SSSPCH1D format), this is
    /// just a single bucket-MMM sweep that accumulates both metrics —
    /// 30–100× faster than per-cell path-unpack on large matrices.
    #[cfg(feature = "native")]
    pub fn matrix_with_distance(
        &self,
        srcs: &[(f32, f32)],
        dsts: &[(f32, f32)],
    ) -> (Vec<f32>, Vec<f32>, Vec<(f32, f32)>, Vec<(f32, f32)>) {
        let ch = self.ch.as_ref().expect(CH_REQUIRED);
        let snap_srcs: Vec<u32> = srcs.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let snap_dsts: Vec<u32> = dsts.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let int_srcs: Vec<u32> = snap_srcs
            .iter()
            .map(|&csr| ch.perm[csr as usize])
            .collect();
        let int_dsts: Vec<u32> = snap_dsts
            .iter()
            .map(|&csr| ch.perm[csr as usize])
            .collect();
        let (durations, distances) = ch::matrix_with_dist(ch, &int_srcs, &int_dsts);
        // matrix_with_dist returns INF where the CH didn't carry a distance
        // channel; clean those up to 0 for downstream consumers.
        let _ = (PathScratch::new(0), &self.inv_perm); // keep imports used
        let snapped_src_coords = snap_srcs
            .iter()
            .map(|&csr| self.coords[csr as usize])
            .collect();
        let snapped_dst_coords = snap_dsts
            .iter()
            .map(|&csr| self.coords[csr as usize])
            .collect();
        (durations, distances, snapped_src_coords, snapped_dst_coords)
    }

    /// Chunked many-to-many matrix with a hard RAM budget for the compute
    /// engine. Streams each src-batch via `on_chunk(s_start, s_end, dur, dist)`
    /// without materialising the full `n_src × n_dst` f32 buffers up front.
    ///
    /// `budget_mb = 0` falls back to one-shot [`matrix_with_dist`] (legacy
    /// in-memory path). Otherwise [`crate::budget::plan_for_budget_with_n_src`]
    /// picks `(n_threads, chunk_size)` and runs inside a sized rayon pool.
    #[cfg(feature = "native")]
    pub fn matrix_with_distance_budgeted<F>(
        &self,
        srcs: &[(f32, f32)],
        dsts: &[(f32, f32)],
        budget_mb: u64,
        mut on_chunk: F,
    ) -> (Vec<(f32, f32)>, Vec<(f32, f32)>)
    where
        F: FnMut(usize, usize, &[f32], &[f32]) + Send,
    {
        use crate::budget::{plan_for_budget_with_n_src, MatrixBudget};

        let ch = self.ch.as_ref().expect(CH_REQUIRED);
        let snap_srcs: Vec<u32> = srcs.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let snap_dsts: Vec<u32> = dsts.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let int_srcs: Vec<u32> = snap_srcs
            .iter()
            .map(|&csr| ch.perm[csr as usize])
            .collect();
        let int_dsts: Vec<u32> = snap_dsts
            .iter()
            .map(|&csr| ch.perm[csr as usize])
            .collect();
        let n_src = int_srcs.len();
        let n_dst = int_dsts.len();
        let snapped_src_coords: Vec<(f32, f32)> = snap_srcs
            .iter()
            .map(|&csr| self.coords[csr as usize])
            .collect();
        let snapped_dst_coords: Vec<(f32, f32)> = snap_dsts
            .iter()
            .map(|&csr| self.coords[csr as usize])
            .collect();

        if budget_mb == 0 {
            let (dur, dist) = ch::matrix_with_dist(ch, &int_srcs, &int_dsts);
            on_chunk(0, n_src, &dur, &dist);
            return (snapped_src_coords, snapped_dst_coords);
        }

        let budget = MatrixBudget {
            max_bytes: budget_mb * 1024 * 1024,
            graph_n: ch.graph_fwd.n as u32,
            bytes_per_output_cell: 8,
        };
        let plan = plan_for_budget_with_n_src(&budget, n_dst as u32, n_src as u32);
        let chunk = plan.chunk_size;
        let threads = plan.n_threads;

        let mut run_chunked = || {
            ch::matrix_with_dist_chunked(ch, &int_srcs, &int_dsts, chunk, &mut on_chunk);
        };
        if threads != rayon::current_num_threads() {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build rayon pool for matrix budget");
            pool.install(run_chunked);
        } else {
            run_chunked();
        }

        (snapped_src_coords, snapped_dst_coords)
    }

    /// Materialise a full duration+distance matrix using the budget-capped
    /// chunked engine. Peak *compute* RAM stays near the planner estimate;
    /// the returned `Vec<f32>` buffers are still `n_src × n_dst`.
    #[cfg(feature = "native")]
    pub fn matrix_with_distance_budgeted_full(
        &self,
        srcs: &[(f32, f32)],
        dsts: &[(f32, f32)],
        budget_mb: u64,
    ) -> (Vec<f32>, Vec<f32>, Vec<(f32, f32)>, Vec<(f32, f32)>) {
        let n_src = srcs.len();
        let n_dst = dsts.len();
        let n_cells = n_src * n_dst;
        let mut durations = vec![f32::INFINITY; n_cells];
        let mut distances = vec![f32::INFINITY; n_cells];
        let (snapped_src, snapped_dst) =
            self.matrix_with_distance_budgeted(srcs, dsts, budget_mb, |s_start, s_end, dur, dist| {
                for s_local in 0..(s_end - s_start) {
                    let s_global = s_start + s_local;
                    let row_off = s_local * n_dst;
                    let base = s_global * n_dst;
                    for j in 0..n_dst {
                        let idx = base + j;
                        durations[idx] = dur[row_off + j];
                        distances[idx] = dist[row_off + j];
                    }
                }
            });
        (durations, distances, snapped_src, snapped_dst)
    }

    /// Like [`Self::matrix_with_distance_budgeted_full`] but maps each cell
    /// through `map` into a caller-chosen type (e.g. `i32` for brooom).
    #[cfg(feature = "native")]
    pub fn matrix_with_distance_budgeted_mapped<T>(
        &self,
        srcs: &[(f32, f32)],
        dsts: &[(f32, f32)],
        budget_mb: u64,
        init: T,
        map: impl Fn(f32) -> T + Copy + Send + Sync,
    ) -> (Vec<T>, Vec<T>, Vec<(f32, f32)>, Vec<(f32, f32)>)
    where
        T: Copy + Send + Sync,
    {
        let n_src = srcs.len();
        let n_dst = dsts.len();
        let n_cells = n_src * n_dst;
        let mut durations = vec![init; n_cells];
        let mut distances = vec![init; n_cells];
        let (snapped_src, snapped_dst) =
            self.matrix_with_distance_budgeted(srcs, dsts, budget_mb, |s_start, s_end, dur, dist| {
                for s_local in 0..(s_end - s_start) {
                    let s_global = s_start + s_local;
                    let row_off = s_local * n_dst;
                    let base = s_global * n_dst;
                    for j in 0..n_dst {
                        let idx = base + j;
                        durations[idx] = map(dur[row_off + j]);
                        distances[idx] = map(dist[row_off + j]);
                    }
                }
            });
        (durations, distances, snapped_src, snapped_dst)
    }

    /// Serial (single-threaded) `matrix_with_distance` for the wasm build,
    /// which has no rayon and no bucket-MMM. Snaps each input, then runs one
    /// CH path query per (src, dst) cell, summing haversine over the unpacked
    /// path for the distance. O(N²) path-unpacks — fine for the dozens-of-stops
    /// demo sizes the browser optimiser handles. Same signature/return shape as
    /// the native version so callers are identical.
    #[cfg(not(feature = "native"))]
    pub fn matrix_with_distance(
        &self,
        srcs: &[(f32, f32)],
        dsts: &[(f32, f32)],
    ) -> (Vec<f32>, Vec<f32>, Vec<(f32, f32)>, Vec<(f32, f32)>) {
        let ch = self.ch.as_ref().expect(CH_REQUIRED);
        let snap_srcs: Vec<u32> = srcs.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let snap_dsts: Vec<u32> = dsts.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let (ns, nd) = (snap_srcs.len(), snap_dsts.len());
        let mut durations = vec![f32::INFINITY; ns * nd];
        let mut distances = vec![f32::INFINITY; ns * nd];
        for (i, &s_csr) in snap_srcs.iter().enumerate() {
            let s_int = ch.perm[s_csr as usize];
            for (j, &d_csr) in snap_dsts.iter().enumerate() {
                if s_csr == d_csr {
                    durations[i * nd + j] = 0.0;
                    distances[i * nd + j] = 0.0;
                    continue;
                }
                let d_int = ch.perm[d_csr as usize];
                if let Some((dur, path)) = ch::query_with_path(ch, s_int, d_int) {
                    let mut dist = 0.0f32;
                    for w in path.windows(2) {
                        let a = self.coords[self.inv_perm[w[0] as usize] as usize];
                        let b = self.coords[self.inv_perm[w[1] as usize] as usize];
                        dist += haversine_m(a.0, a.1, b.0, b.1);
                    }
                    durations[i * nd + j] = dur;
                    distances[i * nd + j] = dist;
                }
            }
        }
        let snapped_src_coords = snap_srcs.iter().map(|&c| self.coords[c as usize]).collect();
        let snapped_dst_coords = snap_dsts.iter().map(|&c| self.coords[c as usize]).collect();
        (durations, distances, snapped_src_coords, snapped_dst_coords)
    }

    pub fn route(
        &self,
        src_lat: f32,
        src_lon: f32,
        dst_lat: f32,
        dst_lon: f32,
    ) -> Option<RouteResponse> {
        // Geocoding-only service (no CH) → no routing.
        let ch = self.ch.as_ref()?;
        let src_csr = self.nearest_node(src_lat, src_lon);
        let dst_csr = self.nearest_node(dst_lat, dst_lon);
        let src_int = ch.perm[src_csr as usize];
        let dst_int = ch.perm[dst_csr as usize];
        // CH weight is duration (seconds). Path is in CH-internal IDs.
        let (duration_s, path_internal) = ch::query_with_path(ch, src_int, dst_int)?;
        let geometry: Vec<(f32, f32)> = path_internal
            .iter()
            .map(|&iid| {
                let csr = self.inv_perm[iid as usize] as usize;
                self.coords[csr]
            })
            .collect();
        // Sum haversine over consecutive points to get the actual road distance.
        let mut distance_m = 0.0_f32;
        for w in geometry.windows(2) {
            let (la, lo) = w[0];
            let (lb, lob) = w[1];
            distance_m += haversine_m(la, lo, lb, lob);
        }
        let source_snapped = self.coords[src_csr as usize];
        let destination_snapped = self.coords[dst_csr as usize];
        Some(RouteResponse {
            distance_m,
            duration_s,
            geometry,
            source_snapped,
            destination_snapped,
        })
    }

    /// Best route plus up to `alts` alternatives (plateau/via-node method:
    /// every double-settled vertex of the bidirectional search is a candidate
    /// detour). Alternatives are ≤ `max_stretch` longer than the optimum
    /// (e.g. 0.25) and share ≤ `max_share` of their edges with any
    /// already-returned route (e.g. 0.6). Best route first.
    pub fn route_alternatives(
        &self,
        src_lat: f32,
        src_lon: f32,
        dst_lat: f32,
        dst_lon: f32,
        alts: usize,
        max_stretch: f32,
        max_share: f32,
    ) -> Option<Vec<RouteResponse>> {
        let ch = self.ch.as_ref()?;
        let src_csr = self.nearest_node(src_lat, src_lon);
        let dst_csr = self.nearest_node(dst_lat, dst_lon);
        let src_int = ch.perm[src_csr as usize];
        let dst_int = ch.perm[dst_csr as usize];
        let mut scratch = ch::PathScratch::new(ch.graph_fwd.n);
        let routes = ch::query_alternatives(
            ch, src_int, dst_int, &mut scratch, alts, max_stretch, max_share,
        );
        if routes.is_empty() {
            return None;
        }
        let source_snapped = self.coords[src_csr as usize];
        let destination_snapped = self.coords[dst_csr as usize];
        Some(
            routes
                .into_iter()
                .map(|(duration_s, path_internal)| {
                    let geometry: Vec<(f32, f32)> = path_internal
                        .iter()
                        .map(|&iid| self.coords[self.inv_perm[iid as usize] as usize])
                        .collect();
                    let mut distance_m = 0.0_f32;
                    for w in geometry.windows(2) {
                        distance_m += haversine_m(w[0].0, w[0].1, w[1].0, w[1].1);
                    }
                    RouteResponse {
                        distance_m,
                        duration_s,
                        geometry,
                        source_snapped,
                        destination_snapped,
                    }
                })
                .collect(),
        )
    }
}

#[inline]
fn haversine_m(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    let r = 6_371_000.0_f64;
    let l1 = (lat1 as f64).to_radians();
    let l2 = (lat2 as f64).to_radians();
    let dlat = (lat2 as f64 - lat1 as f64).to_radians();
    let dlon = (lon2 as f64 - lon1 as f64).to_radians();
    let a = (dlat / 2.0).sin().powi(2) + l1.cos() * l2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    (r * c) as f32
}

/// Split a combined address query ("Karl Johans gate 42") into
/// `(street, Some("42"))`. The number is the last whitespace-separated token
/// that starts with an ASCII digit (covers "42", "42B", "42-44"); otherwise the
/// whole input is the street and the number is `None`. (Number-last convention,
/// as in Norwegian/most European addresses.)
pub fn split_house_number(query: &str) -> (&str, Option<&str>) {
    let q = query.trim();
    if let Some(pos) = q.rfind(char::is_whitespace) {
        let last = q[pos + 1..].trim();
        if last.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            return (q[..pos].trim_end(), Some(last));
        }
    }
    (q, None)
}

#[cfg(test)]
mod tests {
    use super::split_house_number;

    #[test]
    fn splits_trailing_house_number() {
        assert_eq!(split_house_number("Karl Johans gate 42"), ("Karl Johans gate", Some("42")));
        assert_eq!(split_house_number("Storgata 1A"), ("Storgata", Some("1A")));
        assert_eq!(split_house_number("Storgata 42-44"), ("Storgata", Some("42-44")));
        assert_eq!(split_house_number("Main St"), ("Main St", None));
        assert_eq!(split_house_number("Bjørndal"), ("Bjørndal", None));
        assert_eq!(split_house_number("  Foo bar 7 "), ("Foo bar", Some("7")));
    }
}
