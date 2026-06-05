//! The shared, front-end-independent IR a constraint compiles to.
//!
//! Both the Rust (`syn`) and Python (`rustpython`) front-ends lower to this, so
//! `eval.rs` is the single native evaluator. A `Program` is immutable and
//! `Send + Sync`, so it can be shared across rayon workers behind an `Arc`.

use std::sync::Arc;

/// A runtime value. Scalars are unboxed; lists are refcounted so cloning a
/// `Value` during the tree-walk stays cheap.
#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Arc<[Value]>),
}

/// Scalar fields readable from a finished route (`RouteView`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    RouteTravelTime,
    RouteServiceTime,
    RouteWaitingTime,
    RouteSetupTime,
    RouteStartTime,
    RouteEndTime,
    RouteDistance,
    RouteCost,
    /// Travel/time/distance + fixed component of `cost`.
    RouteCostTravel,
    /// Route-span component of `cost` (`span_cost × span`).
    RouteCostSpan,
    /// Custom-constraint penalty component of `cost`.
    RouteCostCustom,
    /// Derived: `end_time - start_time` (same value as `RouteDuration`, but
    /// named to match the per-vehicle `span_cost` semantics).
    RouteSpan,
    /// Derived: `end_time - start_time`.
    RouteDuration,
    /// Derived: number of stops on the route.
    RouteStopCount,
    /// `1` if at least one driver break was scheduled on this route, else `0`.
    RouteHasBreak,
    /// Number of driver breaks scheduled on this route.
    RouteBreakCount,
    /// Total seconds spent on driver breaks on this route.
    RouteBreakDuration,
    VehicleId,
    /// `vehicle.max_tasks.unwrap_or(i64::MAX)`.
    VehicleMaxTasks,
    VehicleFixed,
    VehiclePerHour,
}

impl Field {
    /// Which scalar fields a `Program` reads — used to decide probe mirroring.
    /// Returns `None` for fields the fast insertion probe cannot supply.
    pub fn probe_safe(self) -> bool {
        matches!(
            self,
            Field::RouteTravelTime | Field::RouteDistance | Field::RouteDuration
        )
    }
}

/// Scalar fields readable from the whole candidate solution (`SolutionView`),
/// used only by global (cross-route) constraint programs. A single expression
/// reads either `route.*`/`vehicle.*` (per-route context) or `solution.*`
/// (global context) — never both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolutionField {
    /// Number of non-empty routes (≈ vehicles used).
    VehiclesUsed,
    /// Total route slots, including empty ones.
    RouteCount,
    /// Number of unassigned tasks.
    UnassignedCount,
    /// Summed cost across all routes.
    SolutionCost,
    /// Summed delivery load (dimension 0) across all routes.
    TotalLoad,
    /// Largest per-route delivery load (dimension 0); 0 with no routes.
    MaxRouteLoad,
    /// Mean route duration over non-empty routes; 0 with none.
    AverageDuration,
}

/// List-valued fields (kept separate so they're never read as scalars).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListField {
    RouteJobIds,
    VehicleCapacity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtin {
    Len,
    Abs,
    Min,
    Max,
    Sum,
    Any,
    All,
    Round,
    Int,
    Float,
    Bool,
    /// `list.contains(x)` / `x in list` → Bool.
    Contains,
    /// `index(list, x)` → position of first x in visiting order, or -1 if absent.
    IndexOf,
    /// `before(list, a, b)` → true iff a occurs before b; false if either absent.
    Before,
    /// `first(list)` → list[0], or -1 if empty.
    First,
    /// `last(list)` → last element, or -1 if empty.
    Last,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Const(Value),
    Field(Field),
    /// A `solution.*` scalar — only valid inside a global program.
    SolutionField(SolutionField),
    ListField(ListField),
    Local(u16),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Cmp(CmpOp, Box<Expr>, Box<Expr>),
    Bool(BoolOp, Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Index(Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(Builtin, Vec<Expr>),
    List(Vec<Expr>),
}

#[derive(Clone, Debug)]
pub struct LetBinding {
    pub slot: u16,
    pub expr: Expr,
}

/// A compiled constraint program: ordered `let` bindings then a return expr.
#[derive(Clone, Debug)]
pub struct Program {
    pub body: Vec<LetBinding>,
    pub ret: Expr,
    pub n_locals: u16,
    pub max_steps: u32,
    /// Scalar fields the program reads (for probe-mirroring decisions).
    pub reads: Vec<Field>,
    /// True when the program is a single `field <= const` hard bound on a
    /// probe-safe field — the form that can be mirrored into `eval.rs`.
    pub mirror_bound: Option<(Field, f64)>,
}

/// A compiled global (cross-route) constraint program. Structurally identical
/// to [`Program`] but evaluated against a `SolutionView` instead of a
/// `RouteView` — it reads `solution.*` fields rather than `route.*`/`vehicle.*`.
/// Globals run only at `recompute_summary` (the cold path), so they carry no
/// probe-mirroring metadata.
#[derive(Clone, Debug)]
pub struct GlobalProgram {
    pub body: Vec<LetBinding>,
    pub ret: Expr,
    pub n_locals: u16,
    pub max_steps: u32,
}

/// Default per-evaluation instruction budget (runaway guard).
pub const DEFAULT_MAX_STEPS: u32 = 4096;
