//! Distance/duration matrix abstraction and several concrete sources.
//!
//! - `Matrix` is a dense N×N table of durations (seconds) and optional
//!   distances (meters), addressed by a single integer per location.
//! - `MatrixSource` builds such a matrix from a list of `[lon, lat]` coords.
//! - Concrete sources: `HaversineMatrix` (no network) and `OsrmClient`
//!   (talks to an OSRM `/table` endpoint).
//!
//! For routing-engine-free use, `HaversineMatrix` is enough; for realistic
//! road distances, `OsrmClient` is the production choice.

#[cfg(feature = "osrm")]
use serde::Deserialize;

use crate::error::{Error, Result};
use crate::problem::{Idx, Problem, ProvidedMatrix, Time};

/// Dense routing matrix.
///
/// Storage is `i32` rather than `i64`: durations cap at ~68 years of seconds
/// and distances at ~2 million km, both well above any plausible VRP scale.
/// Halving the cell width also halves the working set, which matters more
/// than the saved RAM — at N=1000 the matrix fits in L2; at N=2000 it nearly
/// fits in L3. Lookups widen back to `i64` so no caller needs to change.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub n: usize,
    /// Durations in seconds, indexed `durations[i*n + j]`.
    pub durations: Vec<i32>,
    /// Optional distances in meters. Same layout.
    pub distances: Option<Vec<i32>>,
}

impl Matrix {
    #[inline]
    pub fn duration(&self, i: Idx, j: Idx) -> Time {
        self.durations[i * self.n + j] as Time
    }
    #[inline]
    pub fn distance(&self, i: Idx, j: Idx) -> i64 {
        self.distances
            .as_ref()
            .map(|d| d[i * self.n + j] as i64)
            .unwrap_or(0)
    }
    pub fn from_provided(p: &ProvidedMatrix) -> Result<Self> {
        let n = p.durations.len();
        if p.durations.iter().any(|row| row.len() != n) {
            return Err(Error::Matrix(format!(
                "durations matrix is not square ({}×?)",
                n
            )));
        }
        let durations = p
            .durations
            .iter()
            .flat_map(|r| r.iter().map(|&v| narrow_i32(v)))
            .collect();
        let distances = if let Some(dist) = &p.distances {
            if dist.len() != n || dist.iter().any(|r| r.len() != n) {
                return Err(Error::Matrix("distances matrix shape mismatch".into()));
            }
            Some(
                dist.iter()
                    .flat_map(|r| r.iter().map(|&v| narrow_i32(v)))
                    .collect(),
            )
        } else {
            None
        };
        Ok(Self { n, durations, distances })
    }
}

#[inline]
fn narrow_i32(v: i64) -> i32 {
    v.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Anything that can build a `Matrix` from a list of coordinates.
pub trait MatrixSource: Send + Sync {
    fn build(&self, coords: &[[f64; 2]]) -> Result<Matrix>;
}

/// A subset of matrix cells to fetch — origin/destination index pairs. The
/// cost-aware broker (see `crate::broker`) asks a provider for only the cells
/// the solver will actually read instead of a full N×N.
#[derive(Debug, Clone, Default)]
pub struct CellRequest {
    pub pairs: Vec<(u32, u32)>,
}

/// Durations (+ optional distances) parallel to a `CellRequest`'s `pairs`.
#[derive(Debug, Clone, Default)]
pub struct CellResponse {
    pub dur: Vec<i32>,
    pub dist: Option<Vec<i32>>,
}

/// A `MatrixSource` that can serve a SUBSET of cells. The default implementation
/// just builds the full matrix and gathers the requested cells, so any existing
/// source works unchanged; a metered provider (Google, batched OSRM) overrides
/// `fetch_cells` to request only what's asked (Stage D). Implementors opt in
/// explicitly (no blanket impl) so future providers can supply a real one.
pub trait CellSource: MatrixSource {
    fn fetch_cells(&self, coords: &[[f64; 2]], req: &CellRequest) -> Result<CellResponse> {
        let full = self.build(coords)?;
        Ok(gather_cells(&full, req))
    }
}

impl CellSource for HaversineMatrix {}

// Let a boxed provider be used generically (e.g. wrapped by the broker).
impl MatrixSource for Box<dyn CellSource> {
    fn build(&self, coords: &[[f64; 2]]) -> Result<Matrix> {
        (**self).build(coords)
    }
}
impl CellSource for Box<dyn CellSource> {
    fn fetch_cells(&self, coords: &[[f64; 2]], req: &CellRequest) -> Result<CellResponse> {
        (**self).fetch_cells(coords, req)
    }
}

/// Generic per-origin batching for rectangular table APIs (OSRM, Google, any
/// metered provider). Groups a `CellRequest` by origin and asks
/// `fetch_row(origin, &dests)` for each, so the provider pays for ONLY the
/// requested cells (the broker's skeleton) instead of a full N×N rectangle.
/// This is the seam that makes the broker provider-agnostic: a new provider just
/// supplies a per-origin row fetcher.
pub fn per_origin_fetch<F>(req: &CellRequest, want_dist: bool, mut fetch_row: F) -> Result<CellResponse>
where
    F: FnMut(u32, &[u32]) -> Result<(Vec<i32>, Option<Vec<i32>>)>,
{
    use std::collections::HashMap;
    let mut by_origin: HashMap<u32, Vec<(u32, usize)>> = HashMap::new();
    for (idx, &(i, j)) in req.pairs.iter().enumerate() {
        by_origin.entry(i).or_default().push((j, idx));
    }
    let mut dur = vec![0i32; req.pairs.len()];
    let mut dist = if want_dist { Some(vec![0i32; req.pairs.len()]) } else { None };
    for (origin, dests) in by_origin {
        let dlist: Vec<u32> = dests.iter().map(|&(j, _)| j).collect();
        let (rdur, rdist) = fetch_row(origin, &dlist)?;
        for (k, &(_, pi)) in dests.iter().enumerate() {
            dur[pi] = rdur[k];
            if let (Some(out), Some(rd)) = (dist.as_mut(), rdist.as_ref()) {
                out[pi] = rd[k];
            }
        }
    }
    Ok(CellResponse { dur, dist })
}

#[cfg(feature = "osrm")]
impl CellSource for OsrmClient {
    /// Per-origin OSRM `/table` (sources=[origin], destinations=[…]) so only the
    /// requested cells are fetched, not a full N×N. Destinations are tiled to
    /// keep the URL bounded.
    fn fetch_cells(&self, coords: &[[f64; 2]], req: &CellRequest) -> Result<CellResponse> {
        per_origin_fetch(req, true, |origin, dests| self.table_row(coords, origin, dests))
    }
}

#[cfg(feature = "osrm")]
impl OsrmClient {
    /// One OSRM `/table` row: durations+distances from `origin` to `dests`.
    fn table_row(
        &self,
        coords: &[[f64; 2]],
        origin: u32,
        dests: &[u32],
    ) -> Result<(Vec<i32>, Option<Vec<i32>>)> {
        const TILE: usize = 90; // keep the coordinate list / URL bounded
        let mut dur = Vec::with_capacity(dests.len());
        let mut dist = Vec::with_capacity(dests.len());
        for chunk in dests.chunks(TILE) {
            // Coordinate list = [origin, dest0, dest1, …]; sources=0, destinations=1..
            let o = coords[origin as usize];
            let mut cs = format!("{:.6},{:.6}", o[0], o[1]);
            for &d in chunk {
                let c = coords[d as usize];
                cs.push(';');
                cs.push_str(&format!("{:.6},{:.6}", c[0], c[1]));
            }
            let dst_idx = (1..=chunk.len()).map(|i| i.to_string()).collect::<Vec<_>>().join(";");
            let url = format!(
                "{}/table/v1/{}/{}?annotations=duration,distance&sources=0&destinations={}",
                self.host.trim_end_matches('/'),
                self.profile,
                cs,
                dst_idx
            );
            let resp: TableResp = ureq::get(&url)
                .timeout(std::time::Duration::from_secs(60))
                .call()
                .map_err(|e| Error::Http(format!("OSRM table row failed: {e}")))?
                .into_json()
                .map_err(|e| Error::Http(format!("OSRM JSON decode: {e}")))?;
            if resp.code != "Ok" {
                return Err(Error::Http(format!("OSRM error: {}", resp.code)));
            }
            let row = resp
                .durations
                .and_then(|d| d.into_iter().next())
                .ok_or_else(|| Error::Http("OSRM missing durations row".into()))?;
            for c in row {
                dur.push(narrow_i32(c.unwrap_or(0.0).round() as i64));
            }
            if let Some(drow) = resp.distances.and_then(|d| d.into_iter().next()) {
                for c in drow {
                    dist.push(narrow_i32(c.unwrap_or(0.0).round() as i64));
                }
            }
        }
        let dist = if dist.len() == dur.len() { Some(dist) } else { None };
        Ok((dur, dist))
    }
}

/// Pull the requested cells out of a dense matrix (the `CellSource` default).
pub fn gather_cells(full: &Matrix, req: &CellRequest) -> CellResponse {
    let n = full.n;
    let mut dur = Vec::with_capacity(req.pairs.len());
    let mut dist = full.distances.as_ref().map(|_| Vec::with_capacity(req.pairs.len()));
    for &(i, j) in &req.pairs {
        dur.push(full.durations[i as usize * n + j as usize]);
        if let (Some(out), Some(d)) = (dist.as_mut(), full.distances.as_ref()) {
            out.push(d[i as usize * n + j as usize]);
        }
    }
    CellResponse { dur, dist }
}

// -------------------------------------------------------------------------
// Haversine — straight-line distance, no external service.
// -------------------------------------------------------------------------

/// Great-circle distance in meters, multiplied by a detour factor to stand
/// in for actual road distance.
#[derive(Debug, Clone, Copy)]
pub struct HaversineMatrix {
    /// Travel speed in meters per second (default 13.9 ≈ 50 km/h).
    pub speed_mps: f64,
    /// Multiplier on great-circle distance to approximate road detour
    /// (default 1.3).
    pub detour: f64,
}

impl Default for HaversineMatrix {
    fn default() -> Self {
        Self { speed_mps: 13.9, detour: 1.3 }
    }
}

impl MatrixSource for HaversineMatrix {
    fn build(&self, coords: &[[f64; 2]]) -> Result<Matrix> {
        let n = coords.len();
        let mut durations = vec![0i32; n * n];
        let mut distances = vec![0i32; n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let d = haversine_m(coords[i], coords[j]) * self.detour;
                distances[i * n + j] = narrow_i32(d.round() as i64);
                durations[i * n + j] = narrow_i32((d / self.speed_mps).round() as i64);
            }
        }
        Ok(Matrix { n, durations, distances: Some(distances) })
    }
}

/// Great-circle distance in meters between two `[lon, lat]` points.
pub fn haversine_m(a: [f64; 2], b: [f64; 2]) -> f64 {
    const R: f64 = 6_371_000.0;
    let (lon1, lat1) = (a[0].to_radians(), a[1].to_radians());
    let (lon2, lat2) = (b[0].to_radians(), b[1].to_radians());
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let h = (dlat / 2.0).sin().powi(2)
        + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * R * h.sqrt().asin()
}

// -------------------------------------------------------------------------
// OSRM /table client. Gated behind `osrm` because it pulls in `ureq` and
// makes the binary larger; downstream consumers (including the iOS
// embedding) build with `default-features = false` and turn it back on
// only when they actually need it.
// -------------------------------------------------------------------------

#[cfg(feature = "osrm")]
/// HTTP client for an OSRM `/table` endpoint. Works against the OSM-hosted
/// demo server at `https://router.project-osrm.org` or any self-hosted OSRM.
#[derive(Debug, Clone)]
pub struct OsrmClient {
    pub host: String,
    pub profile: String,
}

#[cfg(feature = "osrm")]
impl OsrmClient {
    pub fn new(host: impl Into<String>, profile: impl Into<String>) -> Self {
        Self { host: host.into(), profile: profile.into() }
    }
}

#[cfg(feature = "osrm")]
#[derive(Deserialize)]
struct TableResp {
    code: String,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    durations: Option<Vec<Vec<Option<f64>>>>,
    #[serde(default)]
    distances: Option<Vec<Vec<Option<f64>>>>,
}

#[cfg(feature = "osrm")]
impl MatrixSource for OsrmClient {
    fn build(&self, coords: &[[f64; 2]]) -> Result<Matrix> {
        let n = coords.len();
        if n == 0 {
            return Ok(Matrix { n: 0, durations: vec![], distances: Some(vec![]) });
        }
        let coord_str = coords
            .iter()
            .map(|c| format!("{:.6},{:.6}", c[0], c[1]))
            .collect::<Vec<_>>()
            .join(";");
        let url = format!(
            "{}/table/v1/{}/{}?annotations=duration,distance",
            self.host.trim_end_matches('/'),
            self.profile,
            coord_str
        );
        let resp: TableResp = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(60))
            .call()
            .map_err(|e| Error::Http(format!("OSRM table request failed: {e}")))?
            .into_json()
            .map_err(|e| Error::Http(format!("OSRM JSON decode: {e}")))?;
        if resp.code != "Ok" {
            return Err(Error::Http(format!(
                "OSRM error: {} — {}",
                resp.code,
                resp.message.unwrap_or_default()
            )));
        }
        let dur = resp.durations.ok_or_else(|| Error::Http("OSRM missing durations".into()))?;
        if dur.len() != n {
            return Err(Error::Http(format!(
                "OSRM returned {}×_ durations, expected {n}×{n}",
                dur.len()
            )));
        }
        let mut durations = Vec::with_capacity(n * n);
        for row in &dur {
            if row.len() != n {
                return Err(Error::Http("OSRM duration row width mismatch".into()));
            }
            for cell in row {
                durations.push(narrow_i32(cell.unwrap_or(0.0).round() as i64));
            }
        }
        let distances = resp.distances.map(|dist| {
            dist.iter()
                .flat_map(|row| {
                    row.iter()
                        .map(|c| narrow_i32(c.unwrap_or(0.0).round() as i64))
                })
                .collect::<Vec<i32>>()
        });
        Ok(Matrix { n, durations, distances })
    }
}

// -------------------------------------------------------------------------
// Resolve coords → indices for a problem.
// -------------------------------------------------------------------------

/// Walk every `Location` in the problem and produce the coord list keyed by
/// matrix index. Coord-only locations get a fresh index assigned. Indexed
/// locations with a coord plant that coord at their slot. Idempotent: calling
/// it twice on the same problem yields the same indices and coord vector.
///
/// If a problem mixes indexed locations with coord-only locations, the
/// indexed ones win — their slots are reserved first, and coord-only ones
/// fill any remaining gaps (or extend the tail).
pub fn resolve_coords(problem: &mut Problem) -> Vec<[f64; 2]> {
    // PASS 1: figure out the maximum existing index.
    let mut max_idx: Option<usize> = None;
    let visit_idx = |loc: &crate::problem::Location, max_idx: &mut Option<usize>| {
        if let Some(i) = loc.index {
            *max_idx = Some(max_idx.map_or(i, |m| m.max(i)));
        }
    };
    for v in &problem.vehicles {
        if let Some(s) = &v.start { visit_idx(s, &mut max_idx); }
        if let Some(e) = &v.end { visit_idx(e, &mut max_idx); }
    }
    for j in &problem.jobs { visit_idx(&j.location, &mut max_idx); }
    for s in &problem.shipments {
        visit_idx(&s.pickup.location, &mut max_idx);
        visit_idx(&s.delivery.location, &mut max_idx);
    }

    let initial_len = max_idx.map_or(0, |m| m + 1);
    let mut coords: Vec<[f64; 2]> = vec![[0.0, 0.0]; initial_len];
    let mut filled: Vec<bool> = vec![false; initial_len];

    // PASS 2: place coords at their fixed indices; intern free coords.
    let intern = |loc: &mut crate::problem::Location,
                   coords: &mut Vec<[f64; 2]>,
                   filled: &mut Vec<bool>| {
        match (loc.coord, loc.index) {
            (Some(c), Some(i)) => {
                if i >= coords.len() {
                    coords.resize(i + 1, [0.0, 0.0]);
                    filled.resize(i + 1, false);
                }
                coords[i] = c;
                filled[i] = true;
            }
            (Some(c), None) => {
                let pos = coords.iter().enumerate().find(|(idx, &existing)| {
                    filled[*idx]
                        && (existing[0] - c[0]).abs() < 1e-7
                        && (existing[1] - c[1]).abs() < 1e-7
                }).map(|(i, _)| i);
                let i = pos.unwrap_or_else(|| {
                    coords.push(c);
                    filled.push(true);
                    coords.len() - 1
                });
                loc.index = Some(i);
            }
            (None, Some(_)) | (None, None) => {}
        }
    };

    for v in &mut problem.vehicles {
        if let Some(s) = &mut v.start { intern(s, &mut coords, &mut filled); }
        if let Some(e) = &mut v.end { intern(e, &mut coords, &mut filled); }
    }
    for j in &mut problem.jobs {
        intern(&mut j.location, &mut coords, &mut filled);
    }
    for s in &mut problem.shipments {
        intern(&mut s.pickup.location, &mut coords, &mut filled);
        intern(&mut s.delivery.location, &mut coords, &mut filled);
    }

    coords
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haversine_zero_when_same_point() {
        assert!(haversine_m([10.0, 60.0], [10.0, 60.0]).abs() < 1e-9);
    }

    #[test]
    fn haversine_oslo_bergen_roughly() {
        // Rough straight-line: Oslo (10.75, 59.91) to Bergen (5.32, 60.39)
        // is ~ 305 km great-circle.
        let d = haversine_m([10.75, 59.91], [5.32, 60.39]);
        assert!((d - 305_000.0).abs() < 20_000.0, "got {d}");
    }

    #[test]
    fn haversine_matrix_diagonal_zero() {
        let m = HaversineMatrix::default()
            .build(&[[10.0, 60.0], [11.0, 60.0], [10.0, 61.0]])
            .unwrap();
        for i in 0..3 {
            assert_eq!(m.duration(i, i), 0);
        }
    }
}
