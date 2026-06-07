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
    /// A user-registered custom dimension (P5), read as a whole-route aggregate
    /// (the maximum cumul over the route). The `u32` is the dimension's index in
    /// the registered list, resolved from its name at compile time. The indexed
    /// form `route.<dim>[k]` lowers to [`ListField::CustomDimension`] instead.
    CustomDimension(u32),
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

/// Scalar fields readable from a single arc context (`ArcCtx`), used only by a
/// **dimension transit** program. A transit expression maps the physical arc
/// (`from`/`to` matrix indices, `distance`/`duration`, `arrival`) plus the
/// dimension's own running `cumul_before` to the integer delta to add to the
/// cumul. It reads neither `route.*`/`vehicle.*` nor `solution.*` — its own
/// namespace, so the three contexts never alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArcField {
    /// Matrix index of the arc origin (`arc.from` / `from`).
    ArcFrom,
    /// Matrix index of the arc destination (`arc.to` / `to`).
    ArcTo,
    /// The dimension's cumul value *before* this arc (`cumul` / `cumul_before`).
    ArcCumulBefore,
    /// Arrival time at the destination (`arc.arrival` / `arrival`).
    ArcArrival,
    /// Physical matrix distance of this arc (`arc.distance` / `distance`).
    ArcDistance,
    /// Speed-scaled travel duration of this arc (`arc.duration` / `duration`).
    ArcDuration,
}

/// Scalar fields readable from a matrix-broker cost/policy context
/// (`crate::broker::BrokerVars`), used only by a broker program. Lets users
/// price API requests and express buy/derive/skip policy in the same DSL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerField {
    /// `broker.n` — locations in this solve.
    N,
    /// `broker.batch_size` — cells in the batch being priced.
    BatchSize,
    /// `broker.tier` — provider price tier.
    Tier,
    /// `broker.budget_remaining` — buy budget left.
    BudgetRemaining,
    /// `broker.cells_known` — cells already cached (DB hits).
    CellsKnown,
    /// `broker.crossing_count` — node frequency across runs.
    CrossingCount,
    /// `broker.haversine_km` — straight-line km of the candidate pair.
    HaversineKm,
}

/// List-valued fields (kept separate so they're never read as scalars).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListField {
    RouteJobIds,
    VehicleCapacity,
    /// The full per-position cumul list of a custom dimension (P5). Produced when
    /// `route.<dim>` is indexed (`route.<dim>[k]`); the bare `route.<dim>` reads
    /// the aggregate scalar [`Field::CustomDimension`] instead. The `u32` is the
    /// dimension's index in the registered list.
    CustomDimension(u32),
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
    /// An `arc.*` scalar — only valid inside a dimension-transit program.
    ArcField(ArcField),
    /// A `broker.*` scalar — only valid inside a broker cost/policy program.
    BrokerField(BrokerField),
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

/// A compiled **dimension-transit** program. Structurally identical to
/// [`Program`] but evaluated against an [`crate::dimension::ArcCtx`] instead of a
/// `RouteView`: it reads `arc.*` fields (`distance`, `duration`, `cumul_before`,
/// `from`, `to`, `arrival`) and returns the integer cumul delta for that arc.
/// Runs on the hot route-eval path (once per arc), so it carries the same step
/// budget guard as the constraint programs.
#[derive(Clone, Debug)]
pub struct ArcProgram {
    pub body: Vec<LetBinding>,
    pub ret: Expr,
    pub n_locals: u16,
    pub max_steps: u32,
}

/// Default per-evaluation instruction budget (runaway guard).
pub const DEFAULT_MAX_STEPS: u32 = 4096;
