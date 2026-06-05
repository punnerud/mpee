//! brooom — VRP solver library.
//!
//! Open Rust alternative to Vroom. Solves TSP / CVRP / VRPTW / PDPTW
//! and accepts Vroom-compatible JSON.

pub mod constraint;
#[cfg(feature = "pyspell")]
pub mod pyspell;
pub mod error;
pub mod problem;
pub mod matrix;
pub mod solution;
pub mod eval;
pub mod granular;
pub mod insertion;
pub mod local_search;
pub mod solver;
pub mod io;
pub mod cache;
pub mod embedding;
pub mod regression;
#[cfg(feature = "neural")]
pub mod neural;
pub mod pattern_db;
pub mod warm_start;
pub mod cluster_decompose;
#[cfg(feature = "gpu")]
pub mod gpu_sweep;
#[cfg(feature = "gpu")]
pub mod gpu_population;
#[cfg(feature = "gpu")]
pub mod gpu_polish;
#[cfg(feature = "gpu")]
pub mod hgs;
pub mod route_exact;
pub mod population;

pub use error::{Error, Result};
pub use problem::{
    Break, Capacity, Job, Location, Problem, Shipment, TimeWindow, Vehicle, VehicleStep, JobKind,
};
pub use matrix::{Matrix, MatrixSource, HaversineMatrix};
#[cfg(feature = "osrm")]
pub use matrix::OsrmClient;
pub use constraint::{RouteView, Verdict};
pub use solution::{Route, Solution, Step, StepKind, Summary, TaskRef};
pub use solver::{solve, solve_full, solve_with_matrix, Solved, SolverConfig};
