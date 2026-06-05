//! Front-end-independent lowering helpers shared by the Rust and Python
//! front-ends: the route/vehicle field schema, the builtin table, the local
//! scope, and probe-mirror detection. Keeping these here guarantees both
//! syntaxes lower to exactly the same IR and schema.

use std::collections::HashMap;

use super::error::DslError;
use super::ir::{Builtin, CmpOp, Expr, Field, LetBinding, ListField, Program, Value, DEFAULT_MAX_STEPS};

pub(crate) struct Ctx {
    pub locals: HashMap<String, u16>,
    pub next_slot: u16,
    pub body: Vec<LetBinding>,
    pub reads: Vec<Field>,
}

impl Ctx {
    pub fn new() -> Self {
        Ctx { locals: HashMap::new(), next_slot: 0, body: Vec::new(), reads: Vec::new() }
    }
    pub fn declare(&mut self, name: String) -> u16 {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.locals.insert(name, slot);
        slot
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
            "duration" => scalar(ctx, Field::RouteDuration),
            "stop_count" => scalar(ctx, Field::RouteStopCount),
            "has_break" => scalar(ctx, Field::RouteHasBreak),
            "break_count" => scalar(ctx, Field::RouteBreakCount),
            "break_duration" => scalar(ctx, Field::RouteBreakDuration),
            "job_ids" => Ok(Expr::ListField(ListField::RouteJobIds)),
            _ => Err(DslError::UnknownField { base: "route", field: field.to_string() }),
        },
        "vehicle" => match field {
            "id" => scalar(ctx, Field::VehicleId),
            "max_tasks" => scalar(ctx, Field::VehicleMaxTasks),
            "fixed" => scalar(ctx, Field::VehicleFixed),
            "per_hour" => scalar(ctx, Field::VehiclePerHour),
            "capacity" => Ok(Expr::ListField(ListField::VehicleCapacity)),
            _ => Err(DslError::UnknownField { base: "vehicle", field: field.to_string() }),
        },
        _ => Err(DslError::UnknownName(base.to_string())),
    }
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
