# Design note: a sandboxed constraint DSL evaluated natively in the solver

> Status: **implemented** in `crates/brooom/src/pyspell/` (features `pyspell` for the
> Rust front-end via `syn`, `pyspell-python` adds the Python front-end via
> rustpython-parser). Both lower to the shared IR in `pyspell/ir.rs`, evaluated by
> `pyspell/eval.rs`; field-only hard bounds are mirrored into `eval.rs::precompute`.
> Tests: `pyspell/*` unit tests + `tests/pyspell_constraints.rs`. This note records the
> design (captured from a reviewed-and-removed Python→AST sandbox, `pybox`).
>
> Precedence/sequencing builtins (`index`, `before`, `first`, `last`) operate
> over `route.job_ids` in visiting order, so "A before B" / "X first/last" are
> expressible per route with no solver change. Cross-route constraints
> (max-vehicles, client-groups, fairness + an arbitrary global hook) live in
> `crate::global_constraint`, applied in `recompute_summary`.

## Motivation

We already support custom constraints as code (`crates/brooom/src/constraint.rs`):
a closure (Rust) or callable (Python) run on every *completed route*, returning
hard `Infeasible` / soft `Penalty(x)` / `Feasible`. Two limits:

1. **Python is slow & GIL-bound.** The PyO3 callback re-acquires the GIL per
   evaluation, so it only runs at `evaluate_route` confirm points — never in the
   inner loop / insertion probe. It also serializes rayon workers.
2. **Arbitrary code is unsafe.** A Python callable can do anything (I/O, imports,
   syscalls). Fine for trusted local use; not for accepting constraints from
   elsewhere.

Goal: let a user write a constraint in **Python or Rust syntax**, compile it once
to a safe native form, and evaluate it **natively in the hot loop** — fast,
deterministic, sandboxed, no Python runtime needed at solve time.

## What we reuse from the `pybox` sandbox model

The prototype was an AST-walking interpreter (no `exec`/`eval`). The transferable
ideas:

- **Deny-by-default whitelist.** Only explicitly allowed AST nodes/operations
  execute; anything else raises a security error. (pybox: `SAFE_BUILTINS` +
  per-node `_exec_*`/`_eval_*` dispatch; unknown node → `PyBoxSecurityError`.)
- **No I/O, no imports, no attribute escape.** The sandbox simply has no nodes
  for `import`, file/socket access, or arbitrary attribute lookup. Outbound
  effects only via an explicit, mediated gateway.
- **Instruction budget + wall-clock guard** (`_tick`): every step increments a
  counter; exceeding `max_iterations` or a timeout aborts. Bounds runaway code.
- **Pure builtins only**: `len, range, int, float, bool, abs, min, max, sum,
  sorted, reversed, enumerate, zip, map, filter, any, all, round, isinstance,
  divmod, ord, chr, hex, bin` — all deterministic, side-effect free.
- **Static manifest pass** before running: declare capabilities up front and
  enforce them, so what a unit may touch is known before any code executes.

For *constraints* the sandbox is even simpler than pybox's general automation
engine: a constraint is a **pure expression over a fixed route schema** returning
a verdict. No I/O is ever needed, so the dangerous nodes simply don't exist in
the grammar.

## Proposed architecture (Rust)

Three layers; both languages converge on one IR, evaluated natively.

```
Python source ─┐                         ┌─ tree-walk (native, no GIL)
               ├─► [front-end] ─► IR ────┤
Rust source ───┘                         └─ or compile to flat bytecode (Vec<Op>)
                                              │
                                              ▼
                                    Arc<CustomConstraintFn>  ← existing hook
```

### 1. Front-ends (parse → shared IR)
- **Python**: parse with a pure-Rust Python parser (e.g. `rustpython-parser`) —
  no CPython, no GIL. Walk a whitelisted subset and lower to IR.
- **Rust**: parse a restricted expression grammar with `syn`, lower the same
  subset to the same IR.
- Whitelisted subset: literals, `Name` (only schema fields + bound locals),
  `BinOp`, `UnaryOp`, `BoolOp`, `Compare`, `Subscript`/index, `IfExp`, list
  literals & simple comprehensions, and `Call` to the pure-builtin set above.
  Everything else → compile-time rejection (deny-by-default).

### 2. IR / typed value model
- `enum Value { Int(i64), Float(f64), Bool(bool), List(Rc<[Value]>) }`
- `enum Expr { Const(Value), Field(RouteField), Local(u16), Bin(Op, Box, Box),
  Cmp(..), Bool(..), Index(Box, Box), Call(Builtin, Vec<Expr>), If(..) }`
- **Fixed route schema** (the only inputs — no arbitrary attribute access):
  `route.duration_s, route.distance_s|_m, route.service_s, route.waiting_s,
  route.end_time, route.cost, route.job_ids (list), route.load (list),
  vehicle.id, vehicle.capacity (list)`. Extendable, but explicit.
- Result contract: a `bool` → feasible/infeasible, or a number → `Penalty`.
  Same as today's `Verdict`.

### 3. Native evaluator
- Tree-walk over the IR (or compile to flat `Vec<Op>` + a small value stack to
  avoid per-eval allocation). Carry an **instruction budget** per evaluation
  (the `_tick` idea) so a pathological constraint can't blow the solve budget.
- Wrap the compiled program as an `Arc<CustomConstraintFn>` and install it via
  the **existing** `constraint::set_constraints` path — so GPU-gating, the
  eval-cache epoch bump, and the per-route contract all work unchanged.

### 4. Run it *in* the optimization, not just at the end
Because evaluation is native (~ns, no FFI), the compiled constraint is cheap
enough to call in the inner loop. Two integration depths:
- **Now (free):** it already runs in `evaluate_route`, which every accepted
  route passes through (insertion confirm, every local-search move, repair).
- **Next:** when the IR only reads fields the fast probe already has
  (e.g. `duration_s`, `load`), mirror it into `eval.rs::precompute` to prune
  candidates *before* the full evaluation — exactly how backhaul was mirrored.
  Constraints that read whole-route aggregates stay at the evaluator.

## Why this beats the current Python callback
- **Native speed** → usable inside the hot loop / probe, not just at confirm.
- **No GIL** → rayon workers don't serialize.
- **Deterministic & sandboxed** → no I/O, no imports; bounded instruction count.
- **No Python at solve time** → compile once, run headless (CLI, server, WASM).

## Honest scope / non-goals
- A **subset** of Python/Rust (pure expressions + simple control flow over the
  route schema), not arbitrary code — that's the point (safety + speed).
- **Per-route only.** Cross-route / global constraints ("≤ N vehicles in zone Z")
  need a separate, solution-level mechanism — still a genuine gap vs Timefold.
- Two front-ends + IR + evaluator is real work (~a few hundred lines each), but
  `rustpython-parser` and `syn` do the parsing; the novel part is the lowering
  whitelist and the native evaluator.

## Suggested build order
1. IR + native evaluator + the fixed route schema, wrapped as a
   `CustomConstraintFn`. (Unit-testable without any parser.)
2. Rust front-end via `syn` (smaller grammar, no new heavy dep beyond `syn`).
3. Python front-end via `rustpython-parser`.
4. `eval.rs` probe mirroring for field-only constraints.
5. Expose in `mpee-py` as `constraints=["<dsl source>"]` alongside the existing
   callable form.
