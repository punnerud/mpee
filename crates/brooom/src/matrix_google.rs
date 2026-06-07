//! Google Distance Matrix provider for the cost-aware broker (feature `google`).
//!
//! Google bills **per element** (one origin×destination pair), so requesting
//! only the broker's skeleton cells is a direct money saving — the broker's
//! whole point. This is just one provider: the broker is **generic over any
//! [`CellSource`]** (Haversine, OSRM, this, or your own), and they all reuse the
//! same per-origin batching (`crate::matrix::per_origin_fetch`).
//!
//! Needs an API key. Untested in CI (no key); the request/parse shape follows
//! the Distance Matrix API. Destinations are tiled at 25 per request (the API
//! cap); each request fetches exactly one origin's needed destinations, so you
//! pay for the requested elements only — never a full N×N.

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::matrix::{per_origin_fetch, CellRequest, CellResponse, CellSource, Matrix, MatrixSource};

#[inline]
fn narrow(v: f64) -> i32 {
    (v.round() as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Distance Matrix API client. `mode` is "driving" | "walking" | "bicycling".
#[derive(Debug, Clone)]
pub struct GoogleDistanceMatrix {
    pub api_key: String,
    pub mode: String,
}

impl GoogleDistanceMatrix {
    pub fn new(api_key: impl Into<String>, mode: impl Into<String>) -> Self {
        Self { api_key: api_key.into(), mode: mode.into() }
    }

    /// One origin → its destinations, tiled at the API's 25-destination cap.
    fn row(&self, coords: &[[f64; 2]], origin: u32, dests: &[u32]) -> Result<(Vec<i32>, Option<Vec<i32>>)> {
        const TILE: usize = 25;
        let o = coords[origin as usize]; // [lon, lat]
        let mut dur = Vec::with_capacity(dests.len());
        let mut dist = Vec::with_capacity(dests.len());
        for chunk in dests.chunks(TILE) {
            let origins = format!("{:.6},{:.6}", o[1], o[0]); // lat,lng
            let destinations = chunk
                .iter()
                .map(|&d| {
                    let c = coords[d as usize];
                    format!("{:.6},{:.6}", c[1], c[0])
                })
                .collect::<Vec<_>>()
                .join("|");
            let url = format!(
                "https://maps.googleapis.com/maps/api/distancematrix/json?origins={}&destinations={}&mode={}&key={}",
                origins, destinations, self.mode, self.api_key
            );
            let resp: GResp = ureq::get(&url)
                .timeout(std::time::Duration::from_secs(60))
                .call()
                .map_err(|e| Error::Http(format!("Google DM request failed: {e}")))?
                .into_json()
                .map_err(|e| Error::Http(format!("Google DM JSON decode: {e}")))?;
            if resp.status != "OK" {
                return Err(Error::Http(format!("Google DM status: {}", resp.status)));
            }
            let row = resp
                .rows
                .into_iter()
                .next()
                .ok_or_else(|| Error::Http("Google DM missing row".into()))?;
            for el in row.elements {
                // ZERO_RESULTS / NOT_FOUND ⇒ 0 (the broker will derive/skip it).
                dur.push(narrow(el.duration.map(|v| v.value).unwrap_or(0.0)));
                dist.push(narrow(el.distance.map(|v| v.value).unwrap_or(0.0)));
            }
        }
        Ok((dur, Some(dist)))
    }
}

impl MatrixSource for GoogleDistanceMatrix {
    /// Full N×N (only used when the broker is bypassed). Tiles the whole grid
    /// through `fetch_cells` — expensive; prefer the broker.
    fn build(&self, coords: &[[f64; 2]]) -> Result<Matrix> {
        let n = coords.len();
        let pairs = (0..n as u32)
            .flat_map(|i| (0..n as u32).filter(move |&j| j != i).map(move |j| (i, j)))
            .collect();
        let resp = self.fetch_cells(coords, &CellRequest { pairs: pairs })?;
        let mut durations = vec![0i32; n * n];
        let mut distances = vec![0i32; n * n];
        let mut k = 0usize;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                durations[i * n + j] = resp.dur[k];
                if let Some(d) = &resp.dist {
                    distances[i * n + j] = d[k];
                }
                k += 1;
            }
        }
        Ok(Matrix { n, durations, distances: Some(distances) })
    }
}

impl CellSource for GoogleDistanceMatrix {
    fn fetch_cells(&self, coords: &[[f64; 2]], req: &CellRequest) -> Result<CellResponse> {
        per_origin_fetch(req, true, |origin, dests| self.row(coords, origin, dests))
    }
}

#[derive(Deserialize)]
struct GResp {
    status: String,
    #[serde(default)]
    rows: Vec<GRow>,
}
#[derive(Deserialize)]
struct GRow {
    #[serde(default)]
    elements: Vec<GElem>,
}
#[derive(Deserialize)]
struct GElem {
    #[serde(default)]
    duration: Option<GVal>,
    #[serde(default)]
    distance: Option<GVal>,
}
#[derive(Deserialize)]
struct GVal {
    value: f64,
}
