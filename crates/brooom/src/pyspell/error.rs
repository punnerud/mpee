//! Errors for the constraint DSL — split into compile-time and eval-time.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum DslError {
    /// The front-end parser rejected the source.
    Parse(String),
    /// A syntactic construct outside the sandboxed subset.
    Forbidden(String),
    /// A name that is neither a bound local nor a known schema base.
    UnknownName(String),
    /// `route.<field>` / `vehicle.<field>` where `<field>` isn't in the schema.
    UnknownField { base: &'static str, field: String },
    /// A builtin called with the wrong number of arguments.
    Arity { builtin: &'static str, got: usize },
    /// A type mismatch detected while evaluating (e.g. indexing a scalar).
    Type(String),
    /// The constraint's final value was a list (must be bool or number).
    ResultType,
    /// The per-evaluation instruction budget was exhausted (runaway guard).
    Budget,
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslError::Parse(m) => write!(f, "parse error: {m}"),
            DslError::Forbidden(m) => write!(f, "not allowed: {m}"),
            DslError::UnknownName(n) => write!(f, "unknown name `{n}`"),
            DslError::UnknownField { base, field } => {
                write!(f, "unknown field `{base}.{field}`")
            }
            DslError::Arity { builtin, got } => {
                write!(f, "builtin `{builtin}` called with {got} argument(s)")
            }
            DslError::Type(m) => write!(f, "type error in constraint: {m}"),
            DslError::ResultType => {
                write!(f, "constraint must return a bool or a number, not a list")
            }
            DslError::Budget => write!(f, "constraint exceeded its evaluation step budget"),
        }
    }
}

impl std::error::Error for DslError {}
