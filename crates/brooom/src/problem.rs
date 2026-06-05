//! Problem definition: vehicles, jobs, shipments, capacities, time windows.
//!
//! All times are in seconds. All distances in meters. Costs are unitless f64.
//! Locations are referenced by an integer index into the routing matrix; jobs
//! and vehicles may also carry raw `[lon, lat]` coordinates so a matrix can
//! be built from an external routing engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Error, Result};

pub type Idx = usize;
pub type Time = i64;
pub type Cost = f64;
pub type SkillSet = Vec<u32>;

/// Multidimensional capacity / load vector.
pub type Capacity = Vec<i64>;

/// A geographic point. At least one of `coord` or `index` must be set.
///
/// Deserialization is forgiving so hand-written JSON doesn't have to remember
/// the coordinate order — all of these parse to the same point:
///   * `[lon, lat]`              — VROOM's bare array
///   * `{"lon": .., "lat": ..}`  — explicit keys (unambiguous; recommended)
///   * `{"coord": [lon, lat]}`   — the struct form
///   * `{"index": n}`            — a matrix index instead of a coordinate
/// Always serializes back to `{"coord": [lon, lat]}` / `{"index": n}`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Location {
    /// `[lon, lat]` in WGS84.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coord: Option<[f64; 2]>,
    /// Index into the routing matrix.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<Idx>,
}

impl<'de> Deserialize<'de> for Location {
    fn deserialize<D>(d: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct LocMap {
            #[serde(default)]
            coord: Option<[f64; 2]>,
            #[serde(default)]
            index: Option<Idx>,
            #[serde(default)]
            lat: Option<f64>,
            #[serde(default)]
            lon: Option<f64>,
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Arr([f64; 2]),
            Map(LocMap),
        }
        Ok(match Raw::deserialize(d)? {
            Raw::Arr([lon, lat]) => Location { coord: Some([lon, lat]), index: None },
            Raw::Map(m) => {
                let coord = m.coord.or(match (m.lon, m.lat) {
                    (Some(lon), Some(lat)) => Some([lon, lat]),
                    _ => None,
                });
                Location { coord, index: m.index }
            }
        })
    }
}

impl Location {
    pub fn from_coord(lon: f64, lat: f64) -> Self {
        Self { coord: Some([lon, lat]), index: None }
    }
    pub fn from_index(idx: Idx) -> Self {
        Self { coord: None, index: Some(idx) }
    }
    pub fn require_index(&self) -> Result<Idx> {
        self.index.ok_or_else(|| Error::Invalid("location is missing matrix index".into()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start: Time,
    pub end: Time,
}

impl TimeWindow {
    pub const FOREVER: TimeWindow = TimeWindow { start: 0, end: i64::MAX / 4 };

    pub fn contains(&self, t: Time) -> bool {
        t >= self.start && t <= self.end
    }
}

/// Whether a task is a single job, a pickup, or a delivery half of a shipment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobKind {
    Single,
    Pickup,
    Delivery,
}

/// A single visit. Deliveries decrement vehicle load, pickups increment it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: u64,
    pub location: Location,
    #[serde(default)]
    pub kind: JobKindOpt,

    /// Time spent at the location, in seconds.
    #[serde(default)]
    pub service: Time,
    /// Setup time when arriving from a *different* location, in seconds.
    #[serde(default)]
    pub setup: Time,
    /// Earliest time service may begin here, regardless of time windows; the
    /// vehicle waits if it arrives earlier. Default 0 (no effect).
    #[serde(default)]
    pub release: Time,

    /// Goods delivered (subtracted from vehicle load on arrival).
    #[serde(default)]
    pub delivery: Capacity,
    /// Goods picked up (added to vehicle load on departure).
    #[serde(default)]
    pub pickup: Capacity,

    #[serde(default)]
    pub skills: SkillSet,
    /// Priority 0..=100. Higher means more important to schedule.
    #[serde(default)]
    pub priority: u8,
    #[serde(default)]
    pub time_windows: Vec<TimeWindow>,
    /// Prize collected for serving this job (prize-collecting VRP). A finite
    /// value makes the job optional, worth this much; if left unserved the
    /// objective is charged `prize`. Defaults to a large sentinel so unset jobs
    /// stay effectively mandatory (dropping one costs the same as before).
    #[serde(default = "default_prize")]
    pub prize: Cost,
    /// Client-group id. The "exactly one per group" global constraint serves
    /// exactly one member of each non-None group. None = ungrouped.
    #[serde(default)]
    pub group: Option<u32>,
    #[serde(default)]
    pub description: Option<String>,
}

/// Sentinel prize that keeps an unset job effectively mandatory (matches the
/// historical flat unassigned penalty).
pub const DEFAULT_PRIZE: Cost = 1e9;
fn default_prize() -> Cost { DEFAULT_PRIZE }

/// Default kind when only `Job` is present in input is `Single`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobKindOpt {
    #[default]
    Single,
    Pickup,
    Delivery,
}

impl From<JobKindOpt> for JobKind {
    fn from(k: JobKindOpt) -> Self {
        match k {
            JobKindOpt::Single => JobKind::Single,
            JobKindOpt::Pickup => JobKind::Pickup,
            JobKindOpt::Delivery => JobKind::Delivery,
        }
    }
}

/// Pickup-and-delivery shipment: a pair of jobs that must be served by the
/// same vehicle, in pickup-then-delivery order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shipment {
    pub pickup: Job,
    pub delivery: Job,
    /// Amount carried between pickup and delivery (defaults to `pickup.pickup`).
    #[serde(default)]
    pub amount: Capacity,
    #[serde(default)]
    pub skills: SkillSet,
    #[serde(default)]
    pub priority: u8,
}

/// Vehicle endpoint (start or end depot).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleStep {
    pub location: Location,
    /// Earliest the vehicle may depart its start (or arrive at its end).
    #[serde(default)]
    pub time: Option<Time>,
}

/// A mandatory driver break (rest/lunch). The break must be taken somewhere in
/// the route such that it starts within one of its `time_windows`. Mirrors
/// Vroom's per-vehicle `breaks`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Break {
    pub id: u64,
    /// Break duration in seconds.
    #[serde(default)]
    pub service: Time,
    /// Windows the break may start within. Empty means any time (FOREVER).
    #[serde(default)]
    pub time_windows: Vec<TimeWindow>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vehicle {
    pub id: u64,
    #[serde(default)]
    pub start: Option<Location>,
    #[serde(default)]
    pub end: Option<Location>,
    #[serde(default)]
    pub capacity: Capacity,
    #[serde(default)]
    pub skills: SkillSet,
    #[serde(default)]
    pub time_window: Option<TimeWindow>,
    /// Multiplier on matrix durations. 1.0 means use as-is, 2.0 means twice as slow.
    #[serde(default = "one")]
    pub speed_factor: f64,
    #[serde(default)]
    pub max_tasks: Option<usize>,
    #[serde(default)]
    pub max_travel_time: Option<Time>,
    #[serde(default)]
    pub max_distance: Option<i64>,
    #[serde(default)]
    pub fixed: Cost,
    /// Cost per hour of route duration.
    #[serde(default = "default_per_hour")]
    pub per_hour: Cost,
    #[serde(default = "default_profile")]
    pub profile: String,
    /// Mandatory driver breaks; each must be taken within one of its windows.
    #[serde(default)]
    pub breaks: Vec<Break>,
    /// Maximum number of trips this vehicle may make from its depot within one
    /// shift (multi-trip / reloading). 1 (default) = a single trip; >1 lets the
    /// solver return to the depot to reload and continue.
    #[serde(default = "default_max_trips")]
    pub max_trips: usize,
    #[serde(default)]
    pub description: Option<String>,
}

fn one() -> f64 { 1.0 }
fn default_per_hour() -> f64 { 3600.0 }
fn default_profile() -> String { "car".to_string() }
fn default_max_trips() -> usize { 1 }

impl Vehicle {
    pub fn time_window(&self) -> TimeWindow {
        self.time_window.unwrap_or(TimeWindow::FOREVER)
    }
    pub fn capacity_dim(&self) -> usize {
        self.capacity.len()
    }
    /// Vehicle dominates a job's skills iff the vehicle has every skill the job requires.
    pub fn has_skills(&self, required: &[u32]) -> bool {
        required.iter().all(|s| self.skills.contains(s))
    }
    /// Whether this vehicle may make more than one trip (reload at the depot).
    pub fn is_multi_trip(&self) -> bool {
        self.max_trips > 1
    }
}

/// A complete VRP problem instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Problem {
    #[serde(default)]
    pub jobs: Vec<Job>,
    #[serde(default)]
    pub shipments: Vec<Shipment>,
    pub vehicles: Vec<Vehicle>,
    /// Optional matrices keyed by routing profile (e.g. "car", "bike").
    #[serde(default)]
    pub matrices: HashMap<String, ProvidedMatrix>,
    /// Optional name/description of the problem.
    #[serde(default)]
    pub description: Option<String>,
}

/// Matrix as provided in the JSON input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvidedMatrix {
    /// Square matrix of durations in seconds.
    pub durations: Vec<Vec<Time>>,
    /// Optional square matrix of distances in meters.
    #[serde(default)]
    pub distances: Option<Vec<Vec<i64>>>,
}

impl Problem {
    /// Whether any vehicle is configured for multiple trips (reloading).
    pub fn any_multi_trip(&self) -> bool {
        self.vehicles.iter().any(|v| v.is_multi_trip())
    }

    pub fn validate(&self) -> Result<()> {
        if self.vehicles.is_empty() {
            return Err(Error::Invalid("at least one vehicle is required".into()));
        }
        let dim = self.capacity_dim();
        for v in &self.vehicles {
            if !v.capacity.is_empty() && v.capacity.len() != dim {
                return Err(Error::Invalid(format!(
                    "vehicle {} capacity dim {} != problem dim {}",
                    v.id,
                    v.capacity.len(),
                    dim
                )));
            }
        }
        Ok(())
    }

    /// Capacity dimensionality, taken as the max non-empty length seen.
    pub fn capacity_dim(&self) -> usize {
        self.vehicles
            .iter()
            .map(|v| v.capacity.len())
            .chain(self.jobs.iter().map(|j| j.delivery.len().max(j.pickup.len())))
            .max()
            .unwrap_or(0)
    }
}

/// Pad/normalize a capacity vector to the problem dimension.
pub fn pad(c: &Capacity, dim: usize) -> Capacity {
    let mut out = c.clone();
    out.resize(dim, 0);
    out
}

pub fn cap_add(a: &Capacity, b: &Capacity) -> Capacity {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0))
        .collect()
}

pub fn cap_sub(a: &Capacity, b: &Capacity) -> Capacity {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| a.get(i).copied().unwrap_or(0) - b.get(i).copied().unwrap_or(0))
        .collect()
}

pub fn cap_le(a: &Capacity, b: &Capacity) -> bool {
    let n = a.len().max(b.len());
    (0..n).all(|i| a.get(i).copied().unwrap_or(0) <= b.get(i).copied().unwrap_or(0))
}

pub fn cap_nonneg(a: &Capacity) -> bool {
    a.iter().all(|&x| x >= 0)
}
