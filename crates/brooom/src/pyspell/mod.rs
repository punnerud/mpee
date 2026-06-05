//! Constraint DSL: write a constraint as code (Rust or Python expression
//! syntax), compile it once to a small native IR, and run it inside the solver
//! via the existing per-route hook (`crate::constraint`).
//!
//! The whole module is gated behind the `pyspell` feature so the default build
//! gains nothing. The Rust front-end needs only `syn` (already in the tree);
//! the Python front-end (`pyspell-python`) adds `rustpython-parser`.
//!
//! ## Result contract
//! The constraint's final value becomes a [`Verdict`]:
//! * `bool`   → `true` is feasible, `false` rejects the route (hard)
//! * number   → `<= 0` is feasible, `> 0` is a soft penalty added to the cost
//!
//! ## Schema (the only inputs)
//! `route.{travel_time, service_time, waiting_time, setup_time, start_time,
//! end_time, distance, cost, duration, stop_count, job_ids}` and
//! `vehicle.{id, capacity, max_tasks, fixed, per_hour}`.

pub mod error;
pub mod eval;
pub mod ir;
pub(crate) mod lower;
pub mod rust_frontend;
#[cfg(feature = "pyspell-python")]
pub mod py_frontend;

use std::sync::Arc;

use crate::constraint::{ConstraintGuard, CustomConstraintFn, ProbeBound, RouteView, Verdict};

pub use error::DslError;
pub use ir::Program;

/// Compile + install a set of Rust-syntax constraints for the duration of the
/// returned guard, registering their probe bounds so the fast insertion probe
/// prunes early. The guard clears everything on drop.
pub fn install_rust(srcs: &[&str]) -> Result<ConstraintGuard, DslError> {
    let mut closures = Vec::with_capacity(srcs.len());
    let mut bounds = Vec::new();
    for s in srcs {
        let program = Arc::new(rust_frontend::compile_rust(s)?);
        if let Some(b) = probe_bound_of(&program) {
            bounds.push(b);
        }
        closures.push(wrap(program));
    }
    crate::constraint::set_probe_bounds(bounds);
    Ok(ConstraintGuard::install(closures))
}

/// Python-syntax counterpart of [`install_rust`].
#[cfg(feature = "pyspell-python")]
pub fn install_python(srcs: &[&str]) -> Result<ConstraintGuard, DslError> {
    let mut closures = Vec::with_capacity(srcs.len());
    let mut bounds = Vec::new();
    for s in srcs {
        let program = Arc::new(py_frontend::compile_python(s)?);
        if let Some(b) = probe_bound_of(&program) {
            bounds.push(b);
        }
        closures.push(wrap(program));
    }
    crate::constraint::set_probe_bounds(bounds);
    Ok(ConstraintGuard::install(closures))
}

/// Compile a constraint written in **Rust expression syntax** into an
/// installable constraint function. Compile errors are returned here, before
/// any solve — never as a panic during search.
pub fn constraint_from_rust(src: &str) -> Result<Arc<CustomConstraintFn>, DslError> {
    Ok(wrap(Arc::new(rust_frontend::compile_rust(src)?)))
}

/// Compile a constraint written in **Python expression syntax**.
#[cfg(feature = "pyspell-python")]
pub fn constraint_from_python(src: &str) -> Result<Arc<CustomConstraintFn>, DslError> {
    Ok(wrap(Arc::new(py_frontend::compile_python(src)?)))
}

/// Compile a Python-syntax constraint into a closure plus its optional probe
/// bound (for callers that install constraints themselves, e.g. mpee-py).
#[cfg(feature = "pyspell-python")]
pub fn compiled_python(src: &str) -> Result<(Arc<CustomConstraintFn>, Option<ProbeBound>), DslError> {
    let program = Arc::new(py_frontend::compile_python(src)?);
    let bound = probe_bound_of(&program);
    Ok((wrap(program), bound))
}

/// Probe bound a program can contribute to the fast insertion probe, if it is a
/// single `field <= const` hard bound on a probe-visible field.
pub fn probe_bound_of(program: &Program) -> Option<ProbeBound> {
    let (field, max) = program.mirror_bound?;
    let metric = match field {
        ir::Field::RouteTravelTime => crate::constraint::ProbeMetric::TravelTime,
        ir::Field::RouteDistance => crate::constraint::ProbeMetric::Distance,
        ir::Field::RouteDuration => crate::constraint::ProbeMetric::Duration,
        _ => return None,
    };
    Some(ProbeBound { metric, max })
}

fn wrap(program: Arc<Program>) -> Arc<CustomConstraintFn> {
    Arc::new(move |view: &RouteView| match eval::run(&program, view) {
        Ok(v) => v,
        // A runtime/sandbox error rejects the route (conservative, never panics).
        Err(_) => Verdict::Infeasible,
    })
}
