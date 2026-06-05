//! Python-expression front-end (feature `pyspell-python`): parse with
//! rustpython-parser (pure Rust, no CPython, no GIL) and lower a whitelisted
//! subset to the **same** shared IR as the Rust front-end. Deny-by-default.
//!
//! Python is a single expression (no `let` bindings). Membership uses Python's
//! `x in list`; integer `/` truncates (it is not Python true division) — a
//! deliberate simplification documented for the constraint DSL.

use rustpython_parser::{ast, parse_expression};

use super::error::DslError;
use super::ir::{BinOp, BoolOp, Builtin, CmpOp, Expr, UnOp, Value};
use super::lower::{builtin_from, finish, resolve_field, Ctx};

pub fn compile_python(src: &str) -> Result<super::ir::Program, DslError> {
    let expr = parse_expression(src, "<constraint>").map_err(|e| DslError::Parse(e.to_string()))?;
    let mut ctx = Ctx::new();
    let ret = lower(&expr, &mut ctx)?;
    Ok(finish(ctx, ret))
}

fn lower(e: &ast::Expr, ctx: &mut Ctx) -> Result<Expr, DslError> {
    match e {
        ast::Expr::Constant(c) => lower_const(&c.value),

        ast::Expr::Name(n) => {
            let name = n.id.as_str();
            if let Some(&slot) = ctx.locals.get(name) {
                Ok(Expr::Local(slot))
            } else if name == "route" || name == "vehicle" {
                Err(DslError::Forbidden(format!(
                    "`{name}` must be used with a field, e.g. `{name}.<field>`"
                )))
            } else {
                Err(DslError::UnknownName(name.to_string()))
            }
        }

        ast::Expr::Attribute(a) => {
            let base = match &*a.value {
                ast::Expr::Name(n) => n.id.as_str().to_string(),
                _ => return Err(DslError::Forbidden("nested attribute access".into())),
            };
            resolve_field(&base, a.attr.as_str(), ctx)
        }

        ast::Expr::BinOp(b) => {
            let l = Box::new(lower(&b.left, ctx)?);
            let r = Box::new(lower(&b.right, ctx)?);
            let op = match b.op {
                ast::Operator::Add => BinOp::Add,
                ast::Operator::Sub => BinOp::Sub,
                ast::Operator::Mult => BinOp::Mul,
                ast::Operator::Div | ast::Operator::FloorDiv => BinOp::Div,
                ast::Operator::Mod => BinOp::Rem,
                _ => return Err(DslError::Forbidden("arithmetic operator".into())),
            };
            Ok(Expr::Bin(op, l, r))
        }

        ast::Expr::UnaryOp(u) => {
            let v = Box::new(lower(&u.operand, ctx)?);
            match u.op {
                ast::UnaryOp::Not => Ok(Expr::Unary(UnOp::Not, v)),
                ast::UnaryOp::USub => Ok(Expr::Unary(UnOp::Neg, v)),
                ast::UnaryOp::UAdd => Ok(*v),
                ast::UnaryOp::Invert => Err(DslError::Forbidden("bitwise `~`".into())),
            }
        }

        ast::Expr::BoolOp(b) => {
            let op = match b.op {
                ast::BoolOp::And => BoolOp::And,
                ast::BoolOp::Or => BoolOp::Or,
            };
            let mut it = b.values.iter();
            let first = it
                .next()
                .ok_or_else(|| DslError::Parse("empty boolean expression".into()))?;
            let mut acc = lower(first, ctx)?;
            for v in it {
                acc = Expr::Bool(op, Box::new(acc), Box::new(lower(v, ctx)?));
            }
            Ok(acc)
        }

        ast::Expr::Compare(c) => lower_compare(c, ctx),

        ast::Expr::IfExp(i) => Ok(Expr::If(
            Box::new(lower(&i.test, ctx)?),
            Box::new(lower(&i.body, ctx)?),
            Box::new(lower(&i.orelse, ctx)?),
        )),

        ast::Expr::Subscript(s) => {
            if matches!(&*s.slice, ast::Expr::Slice(_)) {
                return Err(DslError::Forbidden("slice".into()));
            }
            Ok(Expr::Index(
                Box::new(lower(&s.value, ctx)?),
                Box::new(lower(&s.slice, ctx)?),
            ))
        }

        ast::Expr::List(l) => {
            let mut items = Vec::with_capacity(l.elts.len());
            for el in &l.elts {
                items.push(lower(el, ctx)?);
            }
            Ok(Expr::List(items))
        }

        ast::Expr::Call(c) => {
            let name = match &*c.func {
                ast::Expr::Name(n) => n.id.as_str().to_string(),
                _ => return Err(DslError::Forbidden("call target".into())),
            };
            if !c.keywords.is_empty() {
                return Err(DslError::Forbidden("keyword arguments".into()));
            }
            let b = builtin_from(&name)?;
            let mut args = Vec::with_capacity(c.args.len());
            for a in &c.args {
                args.push(lower(a, ctx)?);
            }
            Ok(Expr::Call(b, args))
        }

        other => Err(DslError::Forbidden(format!("{} expression", expr_kind(other)))),
    }
}

fn lower_compare(c: &ast::ExprCompare, ctx: &mut Ctx) -> Result<Expr, DslError> {
    // Lower all operands once, then fold chained comparisons (a < b < c) into
    // an AND of pairwise comparisons.
    let mut operands = Vec::with_capacity(c.comparators.len() + 1);
    operands.push(lower(&c.left, ctx)?);
    for cmp in &c.comparators {
        operands.push(lower(cmp, ctx)?);
    }
    let mut terms = Vec::with_capacity(c.ops.len());
    for (i, op) in c.ops.iter().enumerate() {
        terms.push(one_cmp(op, operands[i].clone(), operands[i + 1].clone())?);
    }
    let mut it = terms.into_iter();
    let mut acc = it
        .next()
        .ok_or_else(|| DslError::Parse("empty comparison".into()))?;
    for t in it {
        acc = Expr::Bool(BoolOp::And, Box::new(acc), Box::new(t));
    }
    Ok(acc)
}

fn one_cmp(op: &ast::CmpOp, lhs: Expr, rhs: Expr) -> Result<Expr, DslError> {
    Ok(match op {
        ast::CmpOp::Eq => Expr::Cmp(CmpOp::Eq, Box::new(lhs), Box::new(rhs)),
        ast::CmpOp::NotEq => Expr::Cmp(CmpOp::Ne, Box::new(lhs), Box::new(rhs)),
        ast::CmpOp::Lt => Expr::Cmp(CmpOp::Lt, Box::new(lhs), Box::new(rhs)),
        ast::CmpOp::LtE => Expr::Cmp(CmpOp::Le, Box::new(lhs), Box::new(rhs)),
        ast::CmpOp::Gt => Expr::Cmp(CmpOp::Gt, Box::new(lhs), Box::new(rhs)),
        ast::CmpOp::GtE => Expr::Cmp(CmpOp::Ge, Box::new(lhs), Box::new(rhs)),
        // `x in list` → contains(list, x)
        ast::CmpOp::In => Expr::Call(Builtin::Contains, vec![rhs, lhs]),
        ast::CmpOp::NotIn => {
            Expr::Unary(UnOp::Not, Box::new(Expr::Call(Builtin::Contains, vec![rhs, lhs])))
        }
        ast::CmpOp::Is | ast::CmpOp::IsNot => {
            return Err(DslError::Forbidden("`is` / `is not`".into()))
        }
    })
}

fn lower_const(c: &ast::Constant) -> Result<Expr, DslError> {
    match c {
        ast::Constant::Int(big) => big
            .to_string()
            .parse::<i64>()
            .map(|n| Expr::Const(Value::Int(n)))
            .map_err(|_| DslError::Forbidden("integer literal out of range".into())),
        ast::Constant::Float(x) => Ok(Expr::Const(Value::Float(*x))),
        ast::Constant::Bool(b) => Ok(Expr::Const(Value::Bool(*b))),
        ast::Constant::Str(_) => Err(DslError::Forbidden("string literal".into())),
        _ => Err(DslError::Forbidden("constant".into())),
    }
}

fn expr_kind(e: &ast::Expr) -> &'static str {
    match e {
        ast::Expr::Lambda(_) => "lambda",
        ast::Expr::Dict(_) => "dict",
        ast::Expr::Set(_) => "set",
        ast::Expr::ListComp(_) | ast::Expr::SetComp(_) | ast::Expr::DictComp(_) => "comprehension",
        ast::Expr::GeneratorExp(_) => "generator",
        ast::Expr::Await(_) => "await",
        ast::Expr::Yield(_) | ast::Expr::YieldFrom(_) => "yield",
        ast::Expr::NamedExpr(_) => "walrus `:=`",
        ast::Expr::Starred(_) => "starred",
        ast::Expr::Tuple(_) => "tuple",
        ast::Expr::Slice(_) => "slice",
        _ => "unsupported",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(src: &str) -> super::super::ir::Program {
        compile_python(src).unwrap_or_else(|e| panic!("expected ok for `{src}`: {e}"))
    }
    fn err(src: &str) -> DslError {
        compile_python(src).unwrap_err()
    }

    #[test]
    fn accepts_python_constraints() {
        ok("route.travel_time <= 28800");
        ok("abs(route.cost) < 100.0");
        ok("20 not in route.job_ids");
        ok("250 if route.distance > 1000 else 0");
        ok("route.stop_count <= vehicle.max_tasks and route.distance < 50000");
        ok("vehicle.capacity[0] >= 1");
        ok("0 < route.travel_time < 28800"); // chained
        ok("sum(vehicle.capacity) > 0");
    }

    #[test]
    fn rejects_unsafe_or_unknown() {
        assert!(matches!(err("__import__('os')"), DslError::UnknownName(_) | DslError::Forbidden(_)));
        assert!(matches!(err("route.foo"), DslError::UnknownField { .. }));
        assert!(matches!(err("nope + 1"), DslError::UnknownName(_)));
        assert!(matches!(err("[x for x in route.job_ids]"), DslError::Forbidden(_)));
        assert!(matches!(err("lambda x: x"), DslError::Forbidden(_)));
        assert!(matches!(err("'hi'"), DslError::Forbidden(_)));
    }

    #[test]
    fn mirror_bound_matches_rust_frontend() {
        assert!(ok("route.travel_time <= 28800").mirror_bound.is_some());
        assert!(ok("route.cost <= 100.0").mirror_bound.is_none());
    }
}
