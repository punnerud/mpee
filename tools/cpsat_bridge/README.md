# cpsat_bridge — brooom → OR-Tools CP-SAT interop bridge (POC)

A minimal, **actually-working** exporter that takes a brooom `Problem` (the same
VROOM-style JSON the solver already accepts) and emits a self-contained,
runnable **OR-Tools CP-SAT Python script** for a small routing/assignment
instance.

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

# 2. Solve it (needs OR-Tools)
pip install ortools
python3 /tmp/oslo_cpsat.py
```

The generated script is a standalone Python file with the instance baked in
(travel matrix, demands, services, time windows). It needs nothing but
`ortools`.

## What was actually run

Both steps above were executed in this environment:

* `export_cpsat.py` on `oslo_5jobs.json` → generated `/tmp/oslo_cpsat.py`
  (262 lines, `python3 -m py_compile` clean).
* `ortools 9.15` installed via pip; the generated model solved to:

  ```
  status: OPTIMAL
  objective (total travel seconds): 1387
  vehicle 1: [12, 15, 17, 11]
  vehicle 2: [14, 16, 13]
  ```

* The bundled propagation stress instance `tw_chain.json` (a chain of tight,
  mutually-constraining time windows) solves to the single feasible visiting
  order:

  ```
  status: OPTIMAL
  objective (total travel seconds): 1047
  vehicle 1: [12, 13, 11, 14]
  ```

  Note the order is dictated by the time windows, **not** by nearest-neighbour
  distance — the case the boundary doc is about.

## Scope (deliberately narrow)

Supported subset of the brooom schema:

* single shared start/end depot (all vehicles)
* single capacity dimension (`delivery[0]` vs `vehicle.capacity[0]`)
* duration matrix: provided `matrices.car.durations` if present, else built
  haversine-style matching `HaversineMatrix` defaults (13.9 m/s, 1.3 detour)
* optional per-job hard time window (first window only)
* jobs are **mandatory** — this is the propagation-hard regime

Refused **loudly** (exit code 2, never silently dropped):

* shipments / pickup-delivery pairing
* multiple distinct depots
* index-only locations without coordinates and without a matching provided
  matrix
* a provided matrix whose size does not match depot+jobs

Not modelled (on purpose — these are exactly the cases local search handles fine
and should stay in brooom): prizes / disjunction penalties / optional jobs,
skills, client-groups, breaks, multi-trip, span/distance cost shaping,
asymmetric per-vehicle speed factors.

## Model shape

Arc-boolean VRP with per-node arrival-time integers:

* `x[v,i,j]` — vehicle `v` traverses arc `i→j`
* `serve[v,k]` — vehicle `v` serves job-node `k` (each job served exactly once)
* `t[k]` — arrival time at job-node `k`

The time variables carry the weight: CP-SAT propagates them bidirectionally
(`t[j] >= t[i] + service_i + travel_ij` on every active arc, plus window
bounds), which simultaneously eliminates subtours (MTZ-on-time) and resolves
tight time-window chains by **inference** rather than stochastic repair.
Objective: minimise total travel seconds.

## Files

* `export_cpsat.py` — the exporter (pure Python, no deps to generate)
* `tw_chain.json` — propagation stress fixture used by the boundary doc
* `README.md` — this file
