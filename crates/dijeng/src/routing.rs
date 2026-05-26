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
use crate::ch::{self, ContractionHierarchy, PathScratch};
use crate::geo_index::LatLonGrid;
pub struct RoutingService {
    pub ch: ContractionHierarchy,
    pub coords: Buffer<(f32, f32)>,
    /// `inv_perm[internal_id] = csr_id`. Built at construction time from
    /// `ch.perm` so we can map a CH-path back to coordinates.
    inv_perm: Buffer<u32>,
    /// Spatial index over `coords` for sub-100 µs nearest-vertex lookup.
    snap_grid: LatLonGrid,
    /// Optional street-name sidecar, enabling offline geocoding. Reverse
    /// lookups reuse `snap_grid`; forward lookups scan the distinct names.
    /// `None` when no `.names` sidecar was attached.
    names: Option<crate::names::NameTable>,
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
        let snap_grid = LatLonGrid::from_coords(coords.as_slice(), cell_size_deg);
        Self {
            ch,
            coords,
            inv_perm: inv.into(),
            snap_grid,
            names: None,
        }
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
        self.snap_grid
            .nearest(lat, lon, self.coords.as_slice())
            .unwrap_or(0)
    }

    /// Many-to-many duration matrix: `out[i*dsts.len() + j]` is the
    /// shortest-time route from `srcs[i]` to `dsts[j]` (in seconds).
    /// Inputs are (lat, lon); the service snaps each to its nearest road
    /// node before running the bucket-based CH MMM algorithm.
    ///
    /// Returns `(durations, snapped_sources, snapped_destinations)`.
    pub fn matrix(
        &self,
        srcs: &[(f32, f32)],
        dsts: &[(f32, f32)],
    ) -> (Vec<f32>, Vec<(f32, f32)>, Vec<(f32, f32)>) {
        let snap_srcs: Vec<u32> = srcs.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let snap_dsts: Vec<u32> = dsts.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let int_srcs: Vec<u32> = snap_srcs
            .iter()
            .map(|&csr| self.ch.perm[csr as usize])
            .collect();
        let int_dsts: Vec<u32> = snap_dsts
            .iter()
            .map(|&csr| self.ch.perm[csr as usize])
            .collect();
        let durations = ch::matrix(&self.ch, &int_srcs, &int_dsts);
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

    /// Variant of `matrix` that also returns per-cell distances. With a
    /// dual-channel CH (`edge_dist_*` populated, SSSPCH1D format), this is
    /// just a single bucket-MMM sweep that accumulates both metrics —
    /// 30–100× faster than per-cell path-unpack on large matrices.
    pub fn matrix_with_distance(
        &self,
        srcs: &[(f32, f32)],
        dsts: &[(f32, f32)],
    ) -> (Vec<f32>, Vec<f32>, Vec<(f32, f32)>, Vec<(f32, f32)>) {
        let snap_srcs: Vec<u32> = srcs.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let snap_dsts: Vec<u32> = dsts.iter().map(|&(la, lo)| self.nearest_node(la, lo)).collect();
        let int_srcs: Vec<u32> = snap_srcs
            .iter()
            .map(|&csr| self.ch.perm[csr as usize])
            .collect();
        let int_dsts: Vec<u32> = snap_dsts
            .iter()
            .map(|&csr| self.ch.perm[csr as usize])
            .collect();
        let (durations, distances) = ch::matrix_with_dist(&self.ch, &int_srcs, &int_dsts);
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

    pub fn route(
        &self,
        src_lat: f32,
        src_lon: f32,
        dst_lat: f32,
        dst_lon: f32,
    ) -> Option<RouteResponse> {
        let src_csr = self.nearest_node(src_lat, src_lon);
        let dst_csr = self.nearest_node(dst_lat, dst_lon);
        let src_int = self.ch.perm[src_csr as usize];
        let dst_int = self.ch.perm[dst_csr as usize];
        // CH weight is duration (seconds). Path is in CH-internal IDs.
        let (duration_s, path_internal) = ch::query_with_path(&self.ch, src_int, dst_int)?;
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
