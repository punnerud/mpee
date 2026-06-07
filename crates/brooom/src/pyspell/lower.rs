//! Front-end-independent lowering helpers shared by the Rust and Python
//! front-ends: the route/vehicle field schema, the builtin table, the local
//! scope, and probe-mirror detection. Keeping these here guarantees both
//! syntaxes lower to exactly the same IR and schema.

use std::collections::HashMap;

use super::error::DslError;
use super::ir::{
    ArcField, ArcProgram, BrokerField, Builtin, CmpOp, Expr, Field, GlobalProgram, LetBinding,
    ListField, Program, SolutionField, Value, DEFAULT_MAX_STEPS,
};

pub(crate) struct Ctx {
    pub locals: HashMap<String, u16>,
    pub next_slot: u16,
    pub body: Vec<LetBinding>,
    pub reads: Vec<Field>,
    /// Registered custom-dimension names, in index order (P5). A `route.<name>`
    /// that is not a built-in route field is resolved against this list at
    /// compile time → `Field::CustomDimension(index)`. Captured from the global
    /// dimension registry when the context is created (dimensions are registered
    /// before constraints are compiled, exactly like the constraint hook).
    pub dim_names: Vec<String>,
    /// True while compiling a dimension-transit expression: a bare identifier
    /// that names an arc field (`distance`, `duration`, `cumul`, …) resolves to
    /// an [`ArcField`] instead of being rejected as an unknown name.
    pub in_arc: bool,
}

impl Ctx {
    pub fn new() -> Self {
        Ctx {
            locals: HashMap::new(),
            next_slot: 0,
            body: Vec::new(),
            reads: Vec::new(),
            dim_names: crate::dimension::dimension_names(),
            in_arc: false,
        }
    }
    /// A context for compiling a dimension-transit expression.
    pub fn new_arc() -> Self {
        let mut c = Ctx::new();
        c.in_arc = true;
        c
    }
    pub fn declare(&mut self, name: String) -> u16 {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.locals.insert(name, slot);
        slot
    }
    /// Index of a registered custom dimension by name, if any.
    pub fn dimension_index(&self, name: &str) -> Option<u32> {
        self.dim_names.iter().position(|n| n == name).map(|i| i as u32)
    }
}

/// Resolve `route.<field>` / `vehicle.<field>` to a scalar `Field` or a
/// `ListField` expression. The only attribute access the sandbox allows.
pub(crate) fn resolve_field(base: &str, field: &str, ctx: &mut Ctx) -> Result<Expr, DslError> {
    let scalar = |ctx: &mut Ctx, f: Field| {
        ctx.reads.push(f);
        Ok(Expr::Field(f))
    };
    match base {
        "route" => match field {
            "travel_time" => scalar(ctx, Field::RouteTravelTime),
            "service_time" => scalar(ctx, Field::RouteServiceTime),
            "waiting_time" => scalar(ctx, Field::RouteWaitingTime),
            "setup_time" => scalar(ctx, Field::RouteSetupTime),
            "start_time" => scalar(ctx, Field::RouteStartTime),
            "end_time" => scalar(ctx, Field::RouteEndTime),
            "distance" => scalar(ctx, Field::RouteDistance),
            "cost" => scalar(ctx, Field::RouteCost),
            "cost_travel" => scalar(ctx, Field::RouteCostTravel),
            "cost_span" => scalar(ctx, Field::RouteCostSpan),
            "cost_custom" => scalar(ctx, Field::RouteCostCustom),
            "span" => scalar(ctx, Field::RouteSpan),
            "duration" => scalar(ctx, Field::RouteDuration),
            "stop_count" => scalar(ctx, Field::RouteStopCount),
            "has_break" => scalar(ctx, Field::RouteHasBreak),
            "break_count" => scalar(ctx, Field::RouteBreakCount),
            "break_duration" => scalar(ctx, Field::RouteBreakDuration),
            "job_ids" => Ok(Expr::ListField(ListField::RouteJobIds)),
            // Custom dimensions (P5): a `route.<name>` that is not a built-in
            // field is resolved against the registered dimension names. The bare
            // form reads the whole-route aggregate scalar; `route.<name>[k]` is
            // rewritten to the cumul list by the index lowering (see the Rust /
            // Python front-ends' `Index` handling).
            other => match ctx.dimension_index(other) {
                Some(idx) => scalar(ctx, Field::CustomDimension(idx)),
                None => Err(DslError::UnknownField { base: "route", field: field.to_string() }),
            },
        },
        "vehicle" => match field {
            "id" => scalar(ctx, Field::VehicleId),
            "max_tasks" => scalar(ctx, Field::VehicleMaxTasks),
            "fixed" => scalar(ctx, Field::VehicleFixed),
            "per_hour" => scalar(ctx, Field::VehiclePerHour),
            "capacity" => Ok(Expr::ListField(ListField::VehicleCapacity)),
            _ => Err(DslError::UnknownField { base: "vehicle", field: field.to_string() }),
        },
        // Arc fields, only meaningful inside a dimension-transit program. The
        // evaluator's context (route vs solution vs arc) decides whether they read.
        "arc" => arc_field(field).map(Expr::ArcField).ok_or(DslError::UnknownField {
            base: "arc",
            field: field.to_string(),
        }),
        // Cross-route fields, only meaningful inside a global program. The
        // evaluator's context (route vs solution) decides whether they read.
        "solution" => {
            let sf = |s: SolutionField| Ok(Expr::SolutionField(s));
            match field {
                "vehicles_used" => sf(SolutionField::VehiclesUsed),
                "route_count" => sf(SolutionField::RouteCount),
                "unassigned_count" => sf(SolutionField::UnassignedCount),
                "cost" => sf(SolutionField::SolutionCost),
                "total_load" => sf(SolutionField::TotalLoad),
                "max_route_load" => sf(SolutionField::MaxRouteLoad),
                "average_duration" => sf(SolutionField::AverageDuration),
                _ => Err(DslError::UnknownField { base: "solution", field: field.to_string() }),
            }
        }
        // Broker cost/policy fields, only meaningful inside a broker program. The
        // evaluator's context decides whether they read.
        "broker" => {
            let bf = |b: BrokerField| Ok(Expr::BrokerField(b));
            match field {
                "n" => bf(BrokerField::N),
                "batch_size" => bf(BrokerField::BatchSize),
                "tier" => bf(BrokerField::Tier),
                "budget_remaining" => bf(BrokerField::BudgetRemaining),
                "cells_known" => bf(BrokerField::CellsKnown),
                "crossing_count" => bf(BrokerField::CrossingCount),
                "haversine_km" => bf(BrokerField::HaversineKm),
                "departure_hour" => bf(BrokerField::DepartureHour),
                "weekday_class" => bf(BrokerField::WeekdayClass),
                _ => Err(DslError::UnknownField { base: "broker", field: field.to_string() }),
            }
        }
        _ => Err(DslError::UnknownName(base.to_string())),
    }
}

/// Rewrite the base of an index expression so a bare custom-dimension aggregate
/// (`route.<dim>`, lowered to `Field::CustomDimension`) becomes its per-position
/// cumul list (`ListField::CustomDimension`) when it is being subscripted —
/// `route.<dim>[k]`. Any other base passes through unchanged. Keeps the indexing
/// path in both front-ends a single shared call.
pub(crate) fn index_base(base: Expr) -> Expr {
    if let Expr::Field(Field::CustomDimension(idx)) = base {
        Expr::ListField(ListField::CustomDimension(idx))
    } else {
        base
    }
}

/// Map an `arc.<field>` name to its [`ArcField`]. Returns `None` for unknowns.
fn arc_field(field: &str) -> Option<ArcField> {
    Some(match field {
        "from" => ArcField::ArcFrom,
        "to" => ArcField::ArcTo,
        "cumul" | "cumul_before" => ArcField::ArcCumulBefore,
        "arrival" => ArcField::ArcArrival,
        "distance" => ArcField::ArcDistance,
        "duration" => ArcField::ArcDuration,
        _ => return None,
    })
}

/// Resolve a *bare* identifier inside a dimension-transit expression to an
/// [`ArcField`], so a transit can be written `distance / 10` rather than
/// `arc.distance / 10`. Returns `None` if the name is not an arc field (the
/// caller then falls back to its usual "unknown name" handling). The qualified
/// `arc.<field>` form is also accepted (via [`resolve_field`]) and is identical.
pub(crate) fn resolve_bare_arc(name: &str) -> Option<Expr> {
    arc_field(name).map(Expr::ArcField)
}

pub(crate) fn builtin_from(name: &str) -> Result<Builtin, DslError> {
    Ok(match name {
        "len" => Builtin::Len,
        "abs" => Builtin::Abs,
        "min" => Builtin::Min,
        "max" => Builtin::Max,
        "sum" => Builtin::Sum,
        "any" => Builtin::Any,
        "all" => Builtin::All,
        "round" => Builtin::Round,
        "int" => Builtin::Int,
        "float" => Builtin::Float,
        "bool" => Builtin::Bool,
        "index" => Builtin::IndexOf,
        "before" => Builtin::Before,
        "first" => Builtin::First,
        "last" => Builtin::Last,
        _ => return Err(DslError::Forbidden(format!("function `{name}()`"))),
    })
}

/// Assemble a `Program` from a finished context and return expression,
/// detecting a probe-mirrorable hard bound.
pub(crate) fn finish(ctx: Ctx, ret: Expr) -> Program {
    let mirror_bound = detect_mirror(&ret);
    Program {
        body: ctx.body,
        ret,
        n_locals: ctx.next_slot,
        max_steps: DEFAULT_MAX_STEPS,
        reads: ctx.reads,
        mirror_bound,
    }
}

/// Assemble a [`GlobalProgram`] from a finished context and return expression.
/// Globals run only on the cold path, so they carry no probe metadata.
pub(crate) fn finish_global(ctx: Ctx, ret: Expr) -> GlobalProgram {
    GlobalProgram { body: ctx.body, ret, n_locals: ctx.next_slot, max_steps: DEFAULT_MAX_STEPS }
}

/// Assemble an [`ArcProgram`] from a finished context and return expression.
/// Transit programs read `arc.*` only, so they carry no probe metadata.
pub(crate) fn finish_arc(ctx: Ctx, ret: Expr) -> ArcProgram {
    ArcProgram { body: ctx.body, ret, n_locals: ctx.next_slot, max_steps: DEFAULT_MAX_STEPS }
}

/// A single `field <= const` / `field < const` on a probe-visible field can be
/// mirrored into the fast insertion probe.
fn detect_mirror(ret: &Expr) -> Option<(Field, f64)> {
    if let Expr::Cmp(CmpOp::Le | CmpOp::Lt, lhs, rhs) = ret {
        if let (Expr::Field(f), Expr::Const(c)) = (&**lhs, &**rhs) {
            if f.probe_safe() {
                let max = match c {
                    Value::Int(n) => *n as f64,
                    Value::Float(x) => *x,
                    _ => return None,
                };
                return Some((*f, max));
            }
        }
    }
    None
}
