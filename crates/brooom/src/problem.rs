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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Location {
    /// `[lon, lat]` in WGS84.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coord: Option<[f64; 2]>,
    /// Index into the routing matrix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<Idx>,
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
    #[serde(default)]
    pub description: Option<String>,
}

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
    #[serde(default)]
    pub description: Option<String>,
}

fn one() -> f64 { 1.0 }
fn default_per_hour() -> f64 { 3600.0 }
fn default_profile() -> String { "car".to_string() }

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
