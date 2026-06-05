//! Native tree-walk evaluator for a compiled constraint `Program`.
//!
//! No FFI, no GIL, no allocation in the scalar path — a typical predicate is a
//! handful of node visits, cheap enough to run on every route inside the
//! optimization loop. A per-call step budget guards against runaway programs.

use std::sync::Arc;

use crate::constraint::{RouteView, Verdict};

use super::error::DslError;
use super::ir::{BinOp, BoolOp, Builtin, CmpOp, Expr, Field, ListField, Program, UnOp, Value};

struct Frame<'a> {
    view: &'a RouteView<'a>,
    locals: Vec<Value>,
    budget: u32,
}

/// Evaluate a compiled program against one finished route, mapping its final
/// value to a `Verdict`:
/// * `bool`   → `Feasible` / `Infeasible`
/// * number   → `<= 0` is `Feasible`, `> 0` is `Penalty(x)`
pub fn run(program: &Program, view: &RouteView) -> Result<Verdict, DslError> {
    let mut f = Frame {
        view,
        locals: vec![Value::Int(0); program.n_locals as usize],
        budget: program.max_steps,
    };
    for b in &program.body {
        let v = eval(&b.expr, &mut f)?;
        f.locals[b.slot as usize] = v;
    }
    let out = eval(&program.ret, &mut f)?;
    Ok(match out {
        Value::Bool(true) => Verdict::Feasible,
        Value::Bool(false) => Verdict::Infeasible,
        Value::Int(n) => {
            if n <= 0 {
                Verdict::Feasible
            } else {
                Verdict::Penalty(n as f64)
            }
        }
        Value::Float(x) => {
            if x <= 0.0 {
                Verdict::Feasible
            } else {
                Verdict::Penalty(x)
            }
        }
        Value::List(_) => return Err(DslError::ResultType),
    })
}

fn eval(e: &Expr, f: &mut Frame) -> Result<Value, DslError> {
    if f.budget == 0 {
        return Err(DslError::Budget);
    }
    f.budget -= 1;
    match e {
        Expr::Const(v) => Ok(v.clone()),
        Expr::Local(i) => Ok(f.locals[*i as usize].clone()),
        Expr::Field(fld) => Ok(read_field(*fld, f.view)),
        Expr::ListField(lf) => Ok(read_list_field(*lf, f.view)),
        Expr::Bin(op, a, b) => {
            let (x, y) = (eval(a, f)?, eval(b, f)?);
            num_binop(*op, x, y)
        }
        Expr::Cmp(op, a, b) => {
            let (x, y) = (eval(a, f)?, eval(b, f)?);
            Ok(Value::Bool(compare(*op, x, y)?))
        }
        Expr::Bool(op, a, b) => {
            let l = as_bool(&eval(a, f)?)?;
            match (op, l) {
                (BoolOp::And, false) => Ok(Value::Bool(false)),
                (BoolOp::Or, true) => Ok(Value::Bool(true)),
                _ => Ok(Value::Bool(as_bool(&eval(b, f)?)?)),
            }
        }
        Expr::Unary(op, a) => {
            let v = eval(a, f)?;
            match op {
                UnOp::Neg => match v {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(x) => Ok(Value::Float(-x)),
                    _ => Err(DslError::Type("cannot negate a non-number".into())),
                },
                UnOp::Not => Ok(Value::Bool(!as_bool(&v)?)),
            }
        }
        Expr::Index(l, i) => {
            let list = eval(l, f)?;
            let idx = eval(i, f)?;
            index(list, idx)
        }
        Expr::If(c, t, e2) => {
            if as_bool(&eval(c, f)?)? {
                eval(t, f)
            } else {
                eval(e2, f)
            }
        }
        Expr::Call(b, args) => call_builtin(*b, args, f),
        Expr::List(items) => {
            let mut v = Vec::with_capacity(items.len());
            for it in items {
                v.push(eval(it, f)?);
            }
            Ok(Value::List(v.into()))
        }
    }
}

fn read_field(fld: Field, view: &RouteView) -> Value {
    let m = view.metrics;
    let v = view.vehicle;
    match fld {
        Field::RouteTravelTime => Value::Int(m.travel_time),
        Field::RouteServiceTime => Value::Int(m.service_time),
        Field::RouteWaitingTime => Value::Int(m.waiting_time),
        Field::RouteSetupTime => Value::Int(m.setup_time),
        Field::RouteStartTime => Value::Int(m.start_time),
        Field::RouteEndTime => Value::Int(m.end_time),
        Field::RouteDistance => Value::Int(m.distance),
        Field::RouteCost => Value::Float(m.cost),
        Field::RouteCostTravel => Value::Float(m.cost_travel),
        Field::RouteCostSpan => Value::Float(m.cost_span),
        Field::RouteCostCustom => Value::Float(m.cost_custom),
        Field::RouteSpan => Value::Int(m.end_time - m.start_time),
        Field::RouteDuration => Value::Int(m.end_time - m.start_time),
        Field::RouteStopCount => Value::Int(view.steps.len() as i64),
        Field::VehicleId => Value::Int(v.id as i64),
        Field::VehicleMaxTasks => Value::Int(v.max_tasks.map(|n| n as i64).unwrap_or(i64::MAX)),
        Field::VehicleFixed => Value::Float(v.fixed),
        Field::VehiclePerHour => Value::Float(v.per_hour),
    }
}

fn read_list_field(lf: ListField, view: &RouteView) -> Value {
    match lf {
        ListField::RouteJobIds => {
            let ids: Arc<[Value]> =
                view.stop_ids().into_iter().map(|id| Value::Int(id as i64)).collect();
            Value::List(ids)
        }
        ListField::VehicleCapacity => {
            let cap: Arc<[Value]> =
                view.vehicle.capacity.iter().map(|&c| Value::Int(c)).collect();
            Value::List(cap)
        }
    }
}

// ---- value helpers -------------------------------------------------------

fn as_bool(v: &Value) -> Result<bool, DslError> {
    Ok(match v {
        Value::Bool(b) => *b,
        Value::Int(n) => *n != 0,
        Value::Float(x) => *x != 0.0,
        Value::List(l) => !l.is_empty(),
    })
}

fn as_f64(v: &Value) -> Result<f64, DslError> {
    match v {
        Value::Int(n) => Ok(*n as f64),
        Value::Float(x) => Ok(*x),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::List(_) => Err(DslError::Type("expected a number, got a list".into())),
    }
}

fn num_binop(op: BinOp, a: Value, b: Value) -> Result<Value, DslError> {
    // If both are ints, stay integral (truncating div/rem); otherwise float.
    if let (Value::Int(x), Value::Int(y)) = (&a, &b) {
        let (x, y) = (*x, *y);
        return Ok(match op {
            BinOp::Add => Value::Int(x + y),
            BinOp::Sub => Value::Int(x - y),
            BinOp::Mul => Value::Int(x * y),
            BinOp::Div => {
                if y == 0 {
                    return Err(DslError::Type("division by zero".into()));
                }
                Value::Int(x / y)
            }
            BinOp::Rem => {
                if y == 0 {
                    return Err(DslError::Type("modulo by zero".into()));
                }
                Value::Int(x % y)
            }
        });
    }
    let (x, y) = (as_f64(&a)?, as_f64(&b)?);
    Ok(Value::Float(match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul => x * y,
        BinOp::Div => x / y,
        BinOp::Rem => x % y,
    }))
}

fn compare(op: CmpOp, a: Value, b: Value) -> Result<bool, DslError> {
    // Bool == Bool / Bool != Bool handled directly; everything else numerically.
    if let (Value::Bool(x), Value::Bool(y)) = (&a, &b) {
        return match op {
            CmpOp::Eq => Ok(x == y),
            CmpOp::Ne => Ok(x != y),
            _ => Err(DslError::Type("booleans support only == and !=".into())),
        };
    }
    let (x, y) = (as_f64(&a)?, as_f64(&b)?);
    Ok(match op {
        CmpOp::Eq => x == y,
        CmpOp::Ne => x != y,
        CmpOp::Lt => x < y,
        CmpOp::Le => x <= y,
        CmpOp::Gt => x > y,
        CmpOp::Ge => x >= y,
    })
}

fn index(list: Value, idx: Value) -> Result<Value, DslError> {
    let items = match list {
        Value::List(l) => l,
        _ => return Err(DslError::Type("cannot index a non-list".into())),
    };
    let i = match idx {
        Value::Int(n) => n,
        _ => return Err(DslError::Type("list index must be an integer".into())),
    };
    // Support Python-style negative indexing.
    let len = items.len() as i64;
    let real = if i < 0 { len + i } else { i };
    if real < 0 || real >= len {
        return Err(DslError::Type("list index out of range".into()));
    }
    Ok(items[real as usize].clone())
}

fn call_builtin(b: Builtin, args: &[Expr], f: &mut Frame) -> Result<Value, DslError> {
    let name = builtin_name(b);
    // Evaluate args once.
    let mut vals: Vec<Value> = Vec::with_capacity(args.len());
    for a in args {
        vals.push(eval(a, f)?);
    }
    let arity_err = |got: usize| DslError::Arity { builtin: name, got };

    match b {
        Builtin::Len => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            match &vals[0] {
                Value::List(l) => Ok(Value::Int(l.len() as i64)),
                _ => Err(DslError::Type("len() expects a list".into())),
            }
        }
        Builtin::Abs => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            match &vals[0] {
                Value::Int(n) => Ok(Value::Int(n.abs())),
                Value::Float(x) => Ok(Value::Float(x.abs())),
                _ => Err(DslError::Type("abs() expects a number".into())),
            }
        }
        Builtin::Round => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            Ok(Value::Int(as_f64(&vals[0])?.round() as i64))
        }
        Builtin::Int => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            Ok(Value::Int(as_f64(&vals[0])? as i64))
        }
        Builtin::Float => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            Ok(Value::Float(as_f64(&vals[0])?))
        }
        Builtin::Bool => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            Ok(Value::Bool(as_bool(&vals[0])?))
        }
        Builtin::Min | Builtin::Max => reduce_minmax(b, vals, name),
        Builtin::Sum => {
            let items = single_list(&vals, name)?;
            let mut int_acc: i64 = 0;
            let mut float_acc: f64 = 0.0;
            let mut any_float = false;
            for v in items.iter() {
                match v {
                    Value::Int(n) => int_acc += *n,
                    Value::Float(x) => {
                        any_float = true;
                        float_acc += *x;
                    }
                    _ => return Err(DslError::Type("sum() expects a list of numbers".into())),
                }
            }
            Ok(if any_float {
                Value::Float(float_acc + int_acc as f64)
            } else {
                Value::Int(int_acc)
            })
        }
        Builtin::Any => {
            let items = single_list(&vals, name)?;
            for v in items.iter() {
                if as_bool(v)? {
                    return Ok(Value::Bool(true));
                }
            }
            Ok(Value::Bool(false))
        }
        Builtin::All => {
            let items = single_list(&vals, name)?;
            for v in items.iter() {
                if !as_bool(v)? {
                    return Ok(Value::Bool(false));
                }
            }
            Ok(Value::Bool(true))
        }
        Builtin::Contains => {
            if vals.len() != 2 {
                return Err(arity_err(vals.len()));
            }
            let items = match &vals[0] {
                Value::List(l) => l.clone(),
                _ => return Err(DslError::Type("contains expects a list".into())),
            };
            let needle = as_f64(&vals[1])?;
            for v in items.iter() {
                if let Ok(x) = as_f64(v) {
                    if x == needle {
                        return Ok(Value::Bool(true));
                    }
                }
            }
            Ok(Value::Bool(false))
        }
        Builtin::IndexOf => {
            if vals.len() != 2 {
                return Err(arity_err(vals.len()));
            }
            let items = list_of(&vals[0], "index")?;
            let needle = as_f64(&vals[1])?;
            Ok(Value::Int(
                position_of(&items, needle).map(|i| i as i64).unwrap_or(-1),
            ))
        }
        Builtin::Before => {
            if vals.len() != 3 {
                return Err(arity_err(vals.len()));
            }
            let items = list_of(&vals[0], "before")?;
            let a = as_f64(&vals[1])?;
            let b = as_f64(&vals[2])?;
            // Vacuous (false) when either stop is not on this route.
            let verdict = match (position_of(&items, a), position_of(&items, b)) {
                (Some(ia), Some(ib)) => ia < ib,
                _ => false,
            };
            Ok(Value::Bool(verdict))
        }
        Builtin::First => {
            let items = single_list(&vals, "first")?;
            Ok(items.first().cloned().unwrap_or(Value::Int(-1)))
        }
        Builtin::Last => {
            let items = single_list(&vals, "last")?;
            Ok(items.last().cloned().unwrap_or(Value::Int(-1)))
        }
    }
}

/// First element of `items` numerically equal to `needle`, by position.
fn position_of(items: &[Value], needle: f64) -> Option<usize> {
    items.iter().position(|v| as_f64(v).map(|x| x == needle).unwrap_or(false))
}

fn list_of(v: &Value, name: &'static str) -> Result<Arc<[Value]>, DslError> {
    match v {
        Value::List(l) => Ok(l.clone()),
        _ => Err(DslError::Type(format!("{name}() expects a list as its first argument"))),
    }
}

fn single_list<'a>(vals: &'a [Value], name: &'static str) -> Result<Arc<[Value]>, DslError> {
    if vals.len() != 1 {
        return Err(DslError::Arity { builtin: name, got: vals.len() });
    }
    match &vals[0] {
        Value::List(l) => Ok(l.clone()),
        _ => Err(DslError::Type(format!("{name}() expects a list"))),
    }
}

fn reduce_minmax(b: Builtin, vals: Vec<Value>, name: &'static str) -> Result<Value, DslError> {
    // min/max accept either a single list or 2+ scalar args (Python-like).
    let items: Vec<Value> = if vals.len() == 1 {
        match &vals[0] {
            Value::List(l) => l.to_vec(),
            _ => return Err(DslError::Type(format!("{name}() of a single non-list"))),
        }
    } else if vals.len() >= 2 {
        vals
    } else {
        return Err(DslError::Arity { builtin: name, got: vals.len() });
    };
    if items.is_empty() {
        return Err(DslError::Type(format!("{name}() of an empty list")));
    }
    let mut best = items[0].clone();
    let mut best_f = as_f64(&best)?;
    for v in items.iter().skip(1) {
        let f = as_f64(v)?;
        let take = match b {
            Builtin::Min => f < best_f,
            _ => f > best_f,
        };
        if take {
            best = v.clone();
            best_f = f;
        }
    }
    Ok(best)
}

fn builtin_name(b: Builtin) -> &'static str {
    match b {
        Builtin::Len => "len",
        Builtin::Abs => "abs",
        Builtin::Min => "min",
        Builtin::Max => "max",
        Builtin::Sum => "sum",
        Builtin::Any => "any",
        Builtin::All => "all",
        Builtin::Round => "round",
        Builtin::Int => "int",
        Builtin::Float => "float",
        Builtin::Bool => "bool",
        Builtin::Contains => "contains",
        Builtin::IndexOf => "index",
        Builtin::Before => "before",
        Builtin::First => "first",
        Builtin::Last => "last",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::RouteView;
    use crate::problem::{Job, Location, Problem, Vehicle};
    use crate::solution::{RouteMetrics, TaskRef};
    use crate::pyspell::ir::{DEFAULT_MAX_STEPS, LetBinding};

    fn problem() -> Problem {
        let job = |id: u64| Job {
            id,
            location: Location { coord: None, index: Some(0) },
            kind: Default::default(),
            service: 0,
            setup: 0,
            release: 0,
            delivery: vec![1],
            pickup: vec![],
            skills: vec![],
            priority: 0,
            time_windows: vec![],
            prize: crate::problem::DEFAULT_PRIZE,
            disjunction_penalty: None,
            group: None,
            description: None,
        };
        let mut p = Problem::default();
        p.jobs = vec![job(10), job(20)];
        p
    }

    fn vehicle() -> Vehicle {
        Vehicle {
            id: 7,
            start: None,
            end: None,
            capacity: vec![100, 5],
            skills: vec![],
            time_window: None,
            speed_factor: 1.0,
            max_tasks: Some(3),
            max_travel_time: None,
            max_distance: None,
            fixed: 10.0,
            per_hour: 3600.0,
            span_cost: 0.0,
            distance_weight: 0.0,
            time_weight: 1.0,
            profile: "car".into(),
            breaks: vec![],
            max_trips: 1,
            description: None,
        }
    }

    fn metrics() -> RouteMetrics {
        RouteMetrics {
            start_time: 100,
            end_time: 1100,
            travel_time: 900,
            service_time: 60,
            waiting_time: 40,
            setup_time: 0,
            distance: 5000,
            cost: 42.0,
            cost_travel: 42.0,
            cost_span: 0.0,
            cost_custom: 0.0,
        }
    }

    fn prog(ret: Expr) -> Program {
        Program { body: vec![], ret, n_locals: 0, max_steps: DEFAULT_MAX_STEPS, reads: vec![], mirror_bound: None }
    }

    fn eval_verdict(p: &Program) -> Verdict {
        let problem = problem();
        let veh = vehicle();
        let m = metrics();
        let steps = [TaskRef::Job(0), TaskRef::Job(1)];
        let view = RouteView { problem: &problem, vehicle: &veh, steps: &steps, metrics: &m };
        run(p, &view).unwrap()
    }

    #[test]
    fn bool_result_maps_to_feasibility() {
        // travel_time (900) <= 1000 → true → Feasible
        let p = prog(Expr::Cmp(
            CmpOp::Le,
            Box::new(Expr::Field(Field::RouteTravelTime)),
            Box::new(Expr::Const(Value::Int(1000))),
        ));
        assert_eq!(eval_verdict(&p), Verdict::Feasible);

        let p = prog(Expr::Cmp(
            CmpOp::Gt,
            Box::new(Expr::Field(Field::RouteTravelTime)),
            Box::new(Expr::Const(Value::Int(1000))),
        ));
        assert_eq!(eval_verdict(&p), Verdict::Infeasible);
    }

    #[test]
    fn numeric_result_is_penalty_above_zero() {
        // const 250 → Penalty(250); const 0 → Feasible
        assert_eq!(eval_verdict(&prog(Expr::Const(Value::Int(250)))), Verdict::Penalty(250.0));
        assert_eq!(eval_verdict(&prog(Expr::Const(Value::Int(0)))), Verdict::Feasible);
        assert_eq!(eval_verdict(&prog(Expr::Const(Value::Float(-3.0)))), Verdict::Feasible);
    }

    #[test]
    fn sequencing_builtins() {
        // route job ids are [10, 20] in visiting order.
        let jobids = || Expr::ListField(ListField::RouteJobIds);
        // index(job_ids, 20) == 1 → Penalty(1)
        assert_eq!(
            eval_verdict(&prog(Expr::Call(Builtin::IndexOf, vec![jobids(), Expr::Const(Value::Int(20))]))),
            Verdict::Penalty(1.0)
        );
        // index of absent (99) → -1 → Feasible
        assert_eq!(
            eval_verdict(&prog(Expr::Call(Builtin::IndexOf, vec![jobids(), Expr::Const(Value::Int(99))]))),
            Verdict::Feasible
        );
        // before(job_ids, 10, 20) → true → Feasible
        assert_eq!(
            eval_verdict(&prog(Expr::Call(Builtin::Before, vec![jobids(), Expr::Const(Value::Int(10)), Expr::Const(Value::Int(20))]))),
            Verdict::Feasible
        );
        // before reversed → false → Infeasible
        assert_eq!(
            eval_verdict(&prog(Expr::Call(Builtin::Before, vec![jobids(), Expr::Const(Value::Int(20)), Expr::Const(Value::Int(10))]))),
            Verdict::Infeasible
        );
        // before with an absent stop → false (vacuous) → Infeasible
        assert_eq!(
            eval_verdict(&prog(Expr::Call(Builtin::Before, vec![jobids(), Expr::Const(Value::Int(10)), Expr::Const(Value::Int(99))]))),
            Verdict::Infeasible
        );
        // last(job_ids) == 20 → true
        assert_eq!(
            eval_verdict(&prog(Expr::Cmp(
                CmpOp::Eq,
                Box::new(Expr::Call(Builtin::Last, vec![jobids()])),
                Box::new(Expr::Const(Value::Int(20))),
            ))),
            Verdict::Feasible
        );
        // first(job_ids) == 10 → true
        assert_eq!(
            eval_verdict(&prog(Expr::Cmp(
                CmpOp::Eq,
                Box::new(Expr::Call(Builtin::First, vec![jobids()])),
                Box::new(Expr::Const(Value::Int(10))),
            ))),
            Verdict::Feasible
        );
    }

    #[test]
    fn derived_duration_and_stop_count() {
        // duration = 1100 - 100 = 1000; stop_count = 2
        let p = prog(Expr::Field(Field::RouteDuration));
        assert_eq!(eval_verdict(&p), Verdict::Penalty(1000.0));
        let p = prog(Expr::Field(Field::RouteStopCount));
        assert_eq!(eval_verdict(&p), Verdict::Penalty(2.0));
    }

    #[test]
    fn short_circuit_and_or() {
        // false && (1/0) must not evaluate the rhs (no div-by-zero error)
        let p = prog(Expr::Bool(
            BoolOp::And,
            Box::new(Expr::Const(Value::Bool(false))),
            Box::new(Expr::Bin(
                BinOp::Div,
                Box::new(Expr::Const(Value::Int(1))),
                Box::new(Expr::Const(Value::Int(0))),
            )),
        ));
        assert_eq!(eval_verdict(&p), Verdict::Infeasible);
    }

    #[test]
    fn list_builtins_and_contains() {
        // job_ids has the two job ids; len == 2
        let p = prog(Expr::Call(Builtin::Len, vec![Expr::ListField(ListField::RouteJobIds)]));
        assert_eq!(eval_verdict(&p), Verdict::Penalty(2.0));
        // capacity[1] == 5
        let p = prog(Expr::Index(
            Box::new(Expr::ListField(ListField::VehicleCapacity)),
            Box::new(Expr::Const(Value::Int(1))),
        ));
        assert_eq!(eval_verdict(&p), Verdict::Penalty(5.0));
    }

    #[test]
    fn budget_exhaustion_errors() {
        let mut p = prog(Expr::Const(Value::Bool(true)));
        p.max_steps = 0;
        // run() returns Err(Budget); the wrapper maps that to Infeasible, but
        // here we assert the raw error.
        let problem = problem();
        let veh = vehicle();
        let m = metrics();
        let steps = [TaskRef::Job(0)];
        let view = RouteView { problem: &problem, vehicle: &veh, steps: &steps, metrics: &m };
        assert_eq!(run(&p, &view), Err(DslError::Budget));
    }

    #[test]
    fn let_bindings() {
        // let d = end_time - start_time; d <= 1000  → true
        let p = Program {
            body: vec![LetBinding {
                slot: 0,
                expr: Expr::Bin(
                    BinOp::Sub,
                    Box::new(Expr::Field(Field::RouteEndTime)),
                    Box::new(Expr::Field(Field::RouteStartTime)),
                ),
            }],
            ret: Expr::Cmp(
                CmpOp::Le,
                Box::new(Expr::Local(0)),
                Box::new(Expr::Const(Value::Int(1000))),
            ),
            n_locals: 1,
            max_steps: DEFAULT_MAX_STEPS,
            reads: vec![],
            mirror_bound: None,
        };
        assert_eq!(eval_verdict(&p), Verdict::Feasible);
    }
}
