# MPEE / brooom — constraint cookbook

One runnable recipe per constraint. Every JSON below is a complete problem you
can save and solve three ways:

```bash
brooom -i problem.json -o solution.json          # CLI
curl -X POST :8088/solve --data-binary @problem.json   # HTTP API (brooom --serve 8088)
```
```python
from mpee import Router
plan = Router("area.pp", "area.ch").solve(open("problem.json").read())   # Python
```

Coordinates are `[lon, lat]`; with a built map cache the matrix is computed for
you. The examples here embed no matrix, so they snap to whatever cache you load
(or pass an explicit `matrices` block to run matrix-only).

Every recipe is backed by a conformance test under
`crates/brooom/tests/` — cited per section, so these are guaranteed correct.

---

## Pick your constraint by problem shape

| You want to… | Use | Recipe |
|---|---|---|
| Limit what a vehicle can carry | `capacity` / `delivery` | [Capacity](#capacity) |
| Restrict arrival to time ranges | `time_windows` | [Time windows](#time-windows) |
| Match jobs to capable vehicles | `skills` | [Skills](#skills) |
| Move goods A→B on one vehicle | shipment `pickup`/`delivery` | [Pickup & delivery](#pickup--delivery) |
| Collect only after delivering | backhaul (pickup-only job) | [Backhaul](#backhaul) |
| Force a rest mid-shift | vehicle `breaks` | [Driver breaks](#driver-breaks) |
| Order two stops on a route | `precedence` | [Precedence](#precedence) |
| Make a stop optional | `prize` (finite) | [Optional jobs](#optional-jobs-prize-collecting) |
| Penalise dropping a stop | `disjunction_penalty` | [Disjunctions](#disjunctions) |
| Serve **one of** a set | `group` | [Client groups](#client-groups-k-of-n) |
| Serve **k of** a set | `group` + `group_cardinality` | [Client groups](#client-groups-k-of-n) |
| Balance workload (soft) | `fairness_weight` | [Fairness](#fairness--balancing) |
| Balance workload (hard) | `balance_spread` | [Fairness](#fairness--balancing) |
| Serve late instead of dropping | `soft_tw` | [Soft time windows](#soft-time-windows) |
| Rank objectives (cars → cost) | `objective` levels | [Lexicographic](#lexicographic-objective) |
| Track fuel/battery/wear | `dimensions` | [Custom dimensions](#custom-dimensions) |
| Any rule you can code | `constraints` (Rust/Python/DSL) | [Custom constraints](#custom-constraints-in-code) |

---

## Capacity
Multi-dimensional (weight + volume + …). A vehicle's `capacity[d]` bounds the
summed `delivery[d]` on its route. Proof: `tests/constraints.rs`,
`tests/integration.rs::capacity_splits_routes`.
```json
{
  "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100, 20]}],
  "jobs": [
    {"id": 1, "location": [10.05,60.0], "delivery": [60, 5]},
    {"id": 2, "location": [10.06,60.0], "delivery": [60, 18]}
  ]
}
```
The second dimension (volume 18+5 > 20) forces two routes even though weight fits.

## Time windows
Multiple windows per stop; the engine picks the first the vehicle can meet.
Proof: `tests/integration.rs::time_windows_force_two_vehicles`.
```json
{
  "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100], "time_window": [0, 100000]}],
  "jobs": [
    {"id": 1, "location": [10.05,60.0], "delivery": [1], "time_windows": [[0, 3600], [7200, 9000]]},
    {"id": 2, "location": [10.06,60.0], "delivery": [1], "time_windows": [[0, 600]]}
  ]
}
```

## Skills
A vehicle may serve a job only if it has **all** the job's skills (integer ids).
Proof: `tests/integration.rs::skills_route_to_correct_vehicle`.
```json
{
  "vehicles": [
    {"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100], "skills": [1]},
    {"id": 2, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100], "skills": [1, 2]}
  ],
  "jobs": [
    {"id": 1, "location": [10.05,60.0], "delivery": [1], "skills": [2]}
  ]
}
```
Job 1 (needs skill 2) can only go on vehicle 2.

## Pickup & delivery
A shipment is a pickup+delivery pair served by one vehicle, pickup first.
Proof: `tests/integration.rs::shipment_pickup_before_delivery_same_vehicle`.
```json
{
  "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [10]}],
  "shipments": [
    {"pickup": {"id": 100, "location": [10.05,60.0]},
     "delivery": {"id": 101, "location": [10.10,60.0]}, "amount": [3]}
  ]
}
```

## Backhaul
A pickup-only job is a backhaul; all deliveries (linehaul) on a route precede
all backhauls. Proof: `tests/constraints.rs::backhaul_*`.
```json
{
  "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100]}],
  "jobs": [
    {"id": 1, "location": [10.05,60.0], "delivery": [10]},
    {"id": 2, "location": [10.06,60.0], "pickup": [10]}
  ]
}
```

## Driver breaks
Mandatory rest within a window; pushes the timeline, not travel.
Proof: `tests/constraints.rs::break_*`.
```json
{
  "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100],
    "breaks": [{"id": 99, "service": 1800, "time_windows": [[14400, 21600]]}]}],
  "jobs": [{"id": 1, "location": [10.05,60.0], "delivery": [1]}]
}
```

## Precedence
`precedence: [[a, b]]` → job `a` before job `b` when both are on the same route.
First-class field, no DSL. Proof: `tests/constraint_parity.rs::native_precedence_orders_a_before_b`.
```json
{
  "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100]}],
  "jobs": [
    {"id": 10, "location": [10.05,60.0], "delivery": [1]},
    {"id": 30, "location": [10.15,60.0], "delivery": [1]}
  ],
  "precedence": [[30, 10]]
}
```

## Optional jobs (prize-collecting)
A finite `prize` makes a job optional, worth that much if served (default = a
huge sentinel ⇒ mandatory). Proof: `tests/global_constraints.rs`.
```json
{
  "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [1]}],
  "jobs": [
    {"id": 1, "location": [10.05,60.0], "delivery": [1], "prize": 50},
    {"id": 2, "location": [10.50,60.0], "delivery": [1], "prize": 50}
  ]
}
```
Capacity 1 ⇒ the farther, equal-prize job is dropped.

## Disjunctions
`disjunction_penalty` = the cost charged when a job is dropped (distinct from
`prize`, the value of serving). Proof: `tests/constraints.rs::disjunction_penalty_*`.
```json
{
  "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [1]}],
  "jobs": [
    {"id": 1, "location": [10.05,60.0], "delivery": [1], "prize": 50},
    {"id": 2, "location": [10.50,60.0], "delivery": [1], "prize": 50, "disjunction_penalty": 100000}
  ]
}
```
Job 2's big drop penalty keeps it served; job 1 is dropped.

## Client groups (k of N)
`group` ids tie jobs together. By default exactly one per group is served; set
`group_cardinality` (a solve option) for k-of-N. Proof:
`tests/constraint_parity.rs::group_cardinality_k_of_n`.
```python
plan = router.solve(problem_json, group_cardinality=(2, 2))   # serve exactly 2 per group
```
```json
{
  "vehicles": [{"id": 1, "start": [10.0,60.0], "end": [10.0,60.0], "capacity": [100]}],
  "jobs": [
    {"id": 1, "location": [10.05,60.0], "delivery": [1], "group": 7, "prize": 100},
    {"id": 2, "location": [10.06,60.0], "delivery": [1], "group": 7, "prize": 100},
    {"id": 3, "location": [10.07,60.0], "delivery": [1], "group": 7, "prize": 100}
  ]
}
```

## Fairness / balancing
Soft nudge via `fairness_weight`, or a **hard** cap on the spread (max − min) of
route duration/load via `balance_spread`. Proof:
`tests/constraint_parity.rs::hard_balance_caps_route_spread`.
```python
plan = router.solve(problem_json, fairness_weight=2.0, fairness_metric="load")   # soft
plan = router.solve(problem_json, balance_spread=0, fairness_metric="load")      # hard: equal loads
```

## Multi-depot, max duration/distance/stops, release times, multi-trip, mixed fleet, setup+cost
All per-vehicle / per-job fields. Proof: `tests/constraints.rs`, `tests/multi_trip.rs`.
```json
{
  "vehicles": [
    {"id": 1, "start": [10.0,60.0], "end": [10.3,60.0], "capacity": [100],
     "max_travel_time": 36000, "max_distance": 200000, "max_tasks": 12,
     "speed_factor": 1.0, "fixed": 100, "per_hour": 3600, "max_trips": 2}
  ],
  "jobs": [
    {"id": 1, "location": [10.05,60.0], "delivery": [1], "release": 3600, "setup": 120}
  ]
}
```
`start ≠ end` = multi-depot; `max_trips > 1` = return to depot and reload.

## Soft time windows
Serve a stop late (or over capacity/duration) for a penalty instead of dropping
it. Auto-on when there are time windows. Proof: `tests/soft_penalty.rs`.
```python
plan = router.solve(problem_json, soft_tw=True)
```
```bash
brooom -i problem.json --soft-tw          # or --no-soft-tw to force off
```

## Lexicographic objective
Rank objectives; each level is pinned as a hard cap for the next. Levels:
`unassigned, vehicles, cost, makespan, distance`. Proof: `tests/lexicographic.rs`.
```python
plan = router.solve(problem_json, objective=["unassigned", "vehicles", "cost"])
```
```json
{ "options": { "objective": { "levels": ["vehicles", "cost"] } }, "vehicles": [...], "jobs": [...] }
```

## Custom dimensions
Track an accumulating quantity (fuel, battery, wear) with a per-arc transit
expression over `distance`/`duration`/`cumul`; hard `min`/`max` + soft bounds.
Proof: `tests/dimensions.rs`.
```python
plan = router.solve(problem_json, dimensions=[
  {"name": "fuel", "transit": "distance / 10", "start": 500, "min": 0,
   "monotonicity": "non_increasing", "soft_min": 50, "soft_weight": 2.0}
])
```

## Custom constraints in code
Any rule, in Rust or Python, parsed to a sandboxed AST and run natively in the
hot loop. Return a bool (hard reject) or a number (soft penalty). Proof:
`tests/custom_constraints.rs`, `tests/pyspell_constraints.rs`.
```python
# Python callable:
def no_night_for_fragile(route):
    return False if route.end_time > 64800 and any(j.fragile for j in route) else None
plan = router.solve(problem_json, constraints=[no_night_for_fragile])

# Or a DSL string (compiled to native IR, no GIL on the path):
plan = router.solve(problem_json, constraints=["route.distance <= 100000",
                                               "sum(1 for j in route if j.fragile) <= 12"])
```
Global (cross-route) rules read a `solution.*` namespace the same way.

---

## Honest boundary vs OR-Tools CP-SAT

MPEE covers the full standard VRP constraint set as turnkey fields/options, plus
code-defined AST constraints in **both** Rust and Python — typically less code
than OR-Tools' callback/dimension boilerplate. What OR-Tools' CP-SAT still does
that MPEE does not: **general constraint programming** with bidirectional
domain propagation for arbitrary logic. For the rare *propagation-hard*
sub-instance (tightly-coupled feasibility, thin feasible region) MPEE ships a
**CP-SAT bridge** (`tools/cpsat_bridge/`) that exports the hard core to CP-SAT,
solves it exactly, and warm-starts back into brooom for full-constraint polish.
See `crates/brooom/docs/cpsat-boundary.md`.

Documented limitations: custom-dimension transits are arc-local (cannot read a
sibling dimension's cumul); overlapping disjunction *sets* (a job in several
groups) is not modelled; lexicographic levels are best-effort (heuristic, not
proven optimal).
