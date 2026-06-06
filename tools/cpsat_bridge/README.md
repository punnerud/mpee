# cpsat_bridge — brooom → OR-Tools CP-SAT interop bridge (POC)

A minimal, **actually-working** exporter that takes a brooom `Problem` (the same
VROOM-style JSON the solver already accepts) and emits a self-contained,
runnable **OR-Tools CP-SAT Python script** for a small routing/assignment
instance. The generated script **round-trips** its answer: on a feasible solve it
writes a brooom warm-start JSON, so the exact CP-SAT tour can be fed straight back
to `brooom --warm-start`.

This is the honest escape hatch for the rare class of VRP instances that need
**bidirectional domain propagation / global feasibility reasoning** — constraints
a local-search VRP solver (MPEE/brooom) structurally cannot close well. The
*why* and a worked boundary example live in
[`../../crates/brooom/docs/cpsat-boundary.md`](../../crates/brooom/docs/cpsat-boundary.md).

It is a **bridge, not a re-implemented solver.** It does not embed a
constraint-propagation engine in brooom (that would be dishonest and out of
scope). It generates an external model and hands the propagation-hard instance to
a tool built for it.

## Quick start

```bash
# 1. Generate a CP-SAT model from a brooom Problem
python3 export_cpsat.py ../../crates/brooom/examples/oslo_5jobs.json > /tmp/oslo_cpsat.py

# 2. Solve it (needs OR-Tools); it also writes solution_ws.json (warm-start)
pip install ortools
python3 /tmp/oslo_cpsat.py            # writes ./solution_ws.json

# 3. Round-trip the exact tour back into brooom as a warm-start
brooom --warm-start solution_ws.json -i ../../crates/brooom/examples/oslo_5jobs.json
```

The generated script is a standalone Python file with the instance baked in
(travel matrix, demands, services, time windows, skills, groups, PD pairs). It
needs nothing but `ortools`. The warm-start path can be overridden with the
`CPSAT_WS_DIR` env var (default: cwd).

### How the round-trip lines up

brooom's `--warm-start` matches jobs by **`location_index`** (see
`crates/brooom/src/warm_start.rs`), not by job id. The exporter therefore emits
each step's `location_index` aligned with brooom's `resolve_coords` interning
order: depots are interned first (in vehicle order), then jobs in input order, so
the indices the script writes are exactly the ones brooom assigns when it builds
its matrix. For coordinate-only instances (no explicit `index`) this means
depot(s) get the low indices and job *k* (input order) gets index `#depots + k`.

## What was actually run

Executed in this environment (exporter under `python3` 3.14; generated models
solved under `python3.10` with `ortools 9.15`, the interpreter that has OR-Tools
installed here):

* `export_cpsat.py` on `oslo_5jobs.json` → generated script `py_compile`-clean;
  `ortools 9.15` solved it to:

  ```
  status: OPTIMAL
  objective: 1387
  vehicle 1: [12, 15, 17, 11]
  vehicle 2: [13, 16, 14]
  ```

* The propagation stress instance `tw_chain.json` (a chain of tight,
  mutually-constraining time windows) solves to the single feasible visiting
  order, and the emitted `solution_ws.json` round-trips through the built
  `brooom` CLI:

  ```
  status: OPTIMAL
  objective: 1047
  vehicle 1: [12, 13, 11, 14]
  warm-start written: .../solution_ws.json

  $ brooom --warm-start solution_ws.json -i tw_chain.json
  brooom: warm-start loaded — 1 routes, cost=1047.00, unassigned=0
  ```

  Note the order is dictated by the time windows, **not** by nearest-neighbour
  distance — the case the boundary doc is about. brooom accepts the CP-SAT tour
  byte-for-byte (cost matches) instead of having to stumble onto it.

* Per-feature fixtures under `fixtures/` (skills, pickup_delivery, multi_depot,
  multi_tw, multidim_group) each export `py_compile`-clean and solve to OPTIMAL;
  `multi_depot` was additionally round-tripped through `brooom --warm-start`
  (2 routes, cost 630.00, 0 unassigned).

## Scope (a bridge, not feature parity — but wider than the first cut)

Supported subset of the brooom schema:

* **multi-depot** — each vehicle keeps its own start/end node; the
  single-shared-depot assumption is relaxed
* **multi-dimensional capacity** — every `delivery`/`pickup` dim vs
  `vehicle.capacity` (was: `delivery[0]` only)
* **skills** — a vehicle may serve a job only if the job's skills ⊆ the
  vehicle's skills (plus the `allowed_vehicles` allowlist)
* **multiple time windows per job** — modelled as a disjunction: one literal per
  window, exactly one active when the node is served
* **pickup-and-delivery / shipments** — each PD pair is forced onto the same
  vehicle with `arrival[pickup] ≤ arrival[delivery]`
* **priority** — an objective reward for serving high-priority work
* **client groups** — exactly-one (the brooom "one per group" semantic; the RHS
  generalises to k-of-N)
* duration matrix: provided `matrices.car.durations` if present, else built
  haversine-style matching `HaversineMatrix` defaults (13.9 m/s, 1.3 detour)
* jobs are **mandatory** unless they belong to a group

Refused **loudly** (exit code 2, never silently dropped):

* prizes / `disjunction_penalty` / optional jobs (the fat-region case LS handles)
* driver `breaks`, `max_trips` > 1 (multi-trip / reload)
* per-vehicle cost/cap shaping (`span_cost`, `distance_weight`, `max_tasks`,
  `max_travel_time`, `max_distance`), `speed_factor` ≠ 1.0
* `setup` / `release` times
* a provided matrix smaller than the interned location count, or a location with
  neither coordinate nor a matrix-covered index

Still **not** modelled (out of POC scope): brooom's full weighted cost shaping
and the relaxations above — those either belong in native MPEE or are not part of
the propagation-hard class this bridge targets.

## Model shape

Arc-boolean VRP with per-node arrival-time integers, over **job-nodes** (depots
are referenced by their interned location index, not modelled as routable nodes):

* `x[v,i,j]` — vehicle `v` traverses arc `i→j` between job-nodes
* `din[v,k]` / `dout[v,k]` — vehicle `v` enters/leaves the route at job-node `k`
  (the per-vehicle depot arcs, so multi-depot needs no shared node)
* `serve[v,k]` / `served[k]` — `v` serves `k` / `k` served by someone
* `t[k]` — arrival time at job-node `k`

The time variables carry the weight: CP-SAT propagates them bidirectionally
(`t[j] >= t[i] + service_i + travel_ij` on every active arc, plus window
bounds), which simultaneously eliminates subtours (MTZ-on-time) and resolves
tight time-window chains by **inference** rather than stochastic repair. Skills
fix `serve=0` for ineligible pairs; capacity sums per dimension; PD pairs share a
vehicle and order; groups impose exactly-one over `served`. Objective: minimise
total travel seconds minus a priority reward.

## Files

* `export_cpsat.py` — the exporter (pure Python, no deps to generate)
* `tw_chain.json` — propagation stress fixture used by the boundary doc
* `fixtures/` — per-feature fixtures (skills, pickup_delivery, multi_depot,
  multi_tw, multidim_group), each exports clean and solves
* `README.md` — this file
