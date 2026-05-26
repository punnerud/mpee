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
