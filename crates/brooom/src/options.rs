//! Parsing of the Vroom-style `options` object into solver configuration.
//!
//! `VroomInput.options` is a free-form JSON object that, when present, can carry
//! two Phase-1 engine features:
//!
//! * `options.objective` — selects scalar (default) vs. N-level lexicographic
//!   optimisation, mapping to [`crate::solver::ObjectiveMode`].
//! * `options.dimensions` — a list of user-defined accumulator dimensions, each
//!   with a pyspell **transit expression** over the arc context, mapping to
//!   [`crate::dimension::CustomDimension`].
//!
//! Both are optional. An absent `options` (or an `options` with neither key)
//! reproduces today's behaviour byte-for-byte: [`ObjectiveMode::Scalar`] and no
//! registered dimensions.
//!
//! The transit field is a sandboxed DSL **expression** (NOT arbitrary code); it
//! is compiled via the existing pyspell front-end, so it requires the `pyspell`
//! feature. When the feature is off, a problem that declares `options.dimensions`
//! with a transit is rejected with a clear error rather than silently ignored.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::solver::{LexObjective, ObjectiveMode};

/// The parsed `options` object. Every field is optional; an all-default value
/// reproduces today's behaviour (scalar objective, no dimensions).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverOptions {
    /// Objective selector. Either the string `"scalar"`, or an object
    /// `{ "levels": [..] }` for lexicographic mode. Absent ⇒ scalar.
    pub objective: Option<ObjectiveOpt>,
    /// Custom accumulator dimensions. Absent / empty ⇒ none registered.
    pub dimensions: Vec<DimensionOpt>,
    /// Penalty-managed soft constraints (PyVRP-style time-warp). `null`/absent =
    /// AUTO (on when the problem has time windows). `true`/`false` force it. Maps
    /// to [`crate::solver::SolverConfig::soft_search`].
    #[serde(default)]
    pub soft_time_windows: Option<bool>,
}

/// `options.objective`: a bare `"scalar"` string or a `{ "levels": [...] }` map.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ObjectiveOpt {
    /// `"scalar"` (the default) — any other bare string is rejected at mapping.
    Mode(String),
    /// `{ "levels": ["unassigned", "vehicles", "cost", ...] }`.
    Levels { levels: Vec<String> },
}

/// One entry of `options.dimensions`. Mirrors the [`crate::dimension::CustomDimension`]
/// builder surface. `transit` is a pyspell expression over the arc context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionOpt {
    /// The name a DSL constraint reads it by (`route.<name>`).
    pub name: String,
    /// Pyspell transit **expression** over the arc context (reads `distance`,
    /// `duration`, `cumul`/`cumul_before`, `from`, `to`, `arrival`).
    pub transit: String,
    /// Cumul value at the start depot. Default 0.
    #[serde(default)]
    pub start: i64,
    /// Optional hard lower bound on every cumul (checked at full eval).
    #[serde(default)]
    pub min: Option<i64>,
    /// Optional hard upper bound on every cumul (checked at full eval).
    #[serde(default)]
    pub max: Option<i64>,
    /// Optional soft upper bound (penalty above this, within the hard max).
    #[serde(default)]
    pub soft_max: Option<i64>,
    /// Optional soft lower bound (penalty below this, within the hard min).
    #[serde(default)]
    pub soft_min: Option<i64>,
    /// Per-unit soft-bound violation weight. Default 0.0 (no penalty).
    #[serde(default)]
    pub soft_weight: f64,
    /// Declared monotonicity: `"non_decreasing"`, `"non_increasing"`, or
    /// `"none"` (default). Drives the O(1) probe mirror.
    #[serde(default)]
    pub monotonicity: Option<String>,
}

/// Map one objective-level name to a [`LexObjective`]. Names match the wire
/// spelling: `unassigned`, `vehicles`, `cost`, `makespan`, `distance`.
pub fn lex_objective_from_name(name: &str) -> Result<LexObjective> {
    Ok(match name.trim().to_ascii_lowercase().as_str() {
        "unassigned" | "unassigned_count" => LexObjective::UnassignedCount,
        "vehicles" | "vehicle_count" => LexObjective::Vehicles,
        "cost" => LexObjective::Cost,
        "makespan" => LexObjective::Makespan,
        "distance" => LexObjective::Distance,
        other => {
            return Err(Error::Invalid(format!(
                "unknown objective level {other:?} (expected one of: \
                 unassigned, vehicles, cost, makespan, distance)"
            )))
        }
    })
}

impl SolverOptions {
    /// Parse a `SolverOptions` from the raw `options` JSON value. `None` (absent
    /// `options`) yields the default (scalar, no dimensions).
    pub fn from_value(v: Option<&serde_json::Value>) -> Result<Self> {
        match v {
            None => Ok(Self::default()),
            Some(val) => serde_json::from_value(val.clone())
                .map_err(|e| Error::Invalid(format!("options: {e}"))),
        }
    }

    /// Build the [`ObjectiveMode`] this options object selects. Absent / `"scalar"`
    /// ⇒ [`ObjectiveMode::Scalar`] (unchanged default).
    pub fn objective_mode(&self) -> Result<ObjectiveMode> {
        match &self.objective {
            None => Ok(ObjectiveMode::Scalar),
            Some(ObjectiveOpt::Mode(s)) => match s.trim().to_ascii_lowercase().as_str() {
                "scalar" => Ok(ObjectiveMode::Scalar),
                "lexicographic" => Ok(ObjectiveMode::Lexicographic { levels: Vec::new() }),
                other => Err(Error::Invalid(format!(
                    "options.objective {other:?} must be \"scalar\", \"lexicographic\", \
                     or an object with a \"levels\" list"
                ))),
            },
            Some(ObjectiveOpt::Levels { levels }) => {
                let levels = levels
                    .iter()
                    .map(|n| lex_objective_from_name(n))
                    .collect::<Result<Vec<_>>>()?;
                Ok(ObjectiveMode::Lexicographic { levels })
            }
        }
    }

    /// Whether any dimensions are declared (cheap; avoids the feature-gated build
    /// path when none are present).
    pub fn has_dimensions(&self) -> bool {
        !self.dimensions.is_empty()
    }

    /// Compile the declared dimensions into [`crate::dimension::CustomDimension`]s,
    /// compiling each transit expression via the pyspell front-end. Requires the
    /// `pyspell` feature; without it, a non-empty `dimensions` list is an error
    /// (we never silently drop a declared constraint).
    #[cfg(feature = "pyspell")]
    pub fn build_dimensions(&self) -> Result<Vec<crate::dimension::CustomDimension>> {
        use crate::dimension::{CustomDimension, Monotonicity};
        let mut out = Vec::with_capacity(self.dimensions.len());
        for d in &self.dimensions {
            let transit = crate::pyspell::arc_transit_from_rust(&d.transit)
                .map_err(|e| Error::Invalid(format!("dimension {:?} transit: {e}", d.name)))?;
            let mut dim = CustomDimension::new(d.name.clone(), transit).with_start(d.start);
            if let Some(m) = d.min {
                dim = dim.with_min(m);
            }
            if let Some(m) = d.max {
                dim = dim.with_max(m);
            }
            if let Some(sm) = d.soft_max {
                dim = dim.with_soft_max(sm, d.soft_weight);
            }
            if let Some(sm) = d.soft_min {
                dim = dim.with_soft_min(sm, d.soft_weight);
            }
            dim.monotonicity = match d.monotonicity.as_deref() {
                None | Some("none") | Some("") => Monotonicity::None,
                Some("non_decreasing") | Some("nondecreasing") | Some("monotone") => {
                    Monotonicity::NonDecreasing
                }
                Some("non_increasing") | Some("nonincreasing") | Some("draining") => {
                    Monotonicity::NonIncreasing
                }
                Some(other) => {
                    return Err(Error::Invalid(format!(
                        "dimension {:?} monotonicity {other:?} must be one of: \
                         non_decreasing, non_increasing, none",
                        d.name
                    )))
                }
            };
            out.push(dim);
        }
        Ok(out)
    }

    /// Without the `pyspell` feature a declared transit cannot be compiled — fail
    /// loudly rather than ignore it.
    #[cfg(not(feature = "pyspell"))]
    pub fn build_dimensions(&self) -> Result<Vec<crate::dimension::CustomDimension>> {
        if self.dimensions.is_empty() {
            Ok(Vec::new())
        } else {
            Err(Error::Invalid(
                "options.dimensions requires the `pyspell` feature (the transit DSL); \
                 rebuild brooom with --features pyspell"
                    .into(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(json: &str) -> SolverOptions {
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        SolverOptions::from_value(Some(&v)).unwrap()
    }

    #[test]
    fn absent_options_is_scalar_no_dimensions() {
        let o = SolverOptions::from_value(None).unwrap();
        assert!(matches!(o.objective_mode().unwrap(), ObjectiveMode::Scalar));
        assert!(!o.has_dimensions());
    }

    #[test]
    fn empty_object_is_scalar() {
        let o = opts("{}");
        assert!(matches!(o.objective_mode().unwrap(), ObjectiveMode::Scalar));
        assert!(!o.has_dimensions());
    }

    #[test]
    fn scalar_string_is_scalar() {
        let o = opts(r#"{"objective": "scalar"}"#);
        assert!(matches!(o.objective_mode().unwrap(), ObjectiveMode::Scalar));
    }

    #[test]
    fn levels_map_to_lexicographic_in_order() {
        let o = opts(r#"{"objective": {"levels": ["unassigned", "vehicles", "cost"]}}"#);
        match o.objective_mode().unwrap() {
            ObjectiveMode::Lexicographic { levels } => {
                assert_eq!(
                    levels,
                    vec![
                        LexObjective::UnassignedCount,
                        LexObjective::Vehicles,
                        LexObjective::Cost
                    ]
                );
            }
            _ => panic!("expected lexicographic"),
        }
    }

    #[test]
    fn all_five_level_names_parse() {
        let o = opts(
            r#"{"objective": {"levels": ["vehicles","unassigned","cost","makespan","distance"]}}"#,
        );
        match o.objective_mode().unwrap() {
            ObjectiveMode::Lexicographic { levels } => assert_eq!(levels.len(), 5),
            _ => panic!("expected lexicographic"),
        }
    }

    #[test]
    fn unknown_level_is_an_error() {
        let o = opts(r#"{"objective": {"levels": ["bogus"]}}"#);
        assert!(o.objective_mode().is_err());
    }

    #[test]
    fn dimensions_parse_all_fields() {
        let o = opts(
            r#"{"dimensions": [{
                "name": "fuel",
                "transit": "-(distance / 10)",
                "start": 500,
                "min": 0,
                "max": null,
                "monotonicity": "non_increasing",
                "soft_max": 400,
                "soft_min": 50,
                "soft_weight": 2.0
            }]}"#,
        );
        assert!(o.has_dimensions());
        let d = &o.dimensions[0];
        assert_eq!(d.name, "fuel");
        assert_eq!(d.start, 500);
        assert_eq!(d.min, Some(0));
        assert_eq!(d.max, None);
        assert_eq!(d.soft_max, Some(400));
        assert_eq!(d.soft_min, Some(50));
        assert_eq!(d.soft_weight, 2.0);
        assert_eq!(d.monotonicity.as_deref(), Some("non_increasing"));
    }

    #[cfg(feature = "pyspell")]
    #[test]
    fn build_dimensions_compiles_transit() {
        let o = opts(
            r#"{"dimensions": [{
                "name": "fuel", "transit": "-(distance / 10)",
                "start": 100, "min": 0, "monotonicity": "non_increasing"
            }]}"#,
        );
        let dims = o.build_dimensions().unwrap();
        assert_eq!(dims.len(), 1);
        assert_eq!(dims[0].name, "fuel");
        assert_eq!(dims[0].start, 100);
        assert_eq!(dims[0].min, Some(0));
        assert_eq!(dims[0].monotonicity, crate::dimension::Monotonicity::NonIncreasing);
        // The compiled transit burns distance/10 per arc.
        let ctx = crate::dimension::ArcCtx {
            from: 0, to: 1, cumul_before: 100, arrival: 0, distance: 200, duration: 0,
        };
        assert_eq!((dims[0].transit)(&ctx), -20);
    }
}
