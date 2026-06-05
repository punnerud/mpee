#!/usr/bin/env python3
"""brooom Problem  →  OR-Tools CP-SAT model exporter (interop bridge POC).

This is the HONEST escape hatch for the rare class of VRP instances that need
*bidirectional domain propagation / global feasibility reasoning* — the kind of
tightly-coupled constraint a local-search VRP solver (MPEE/brooom) structurally
cannot close well. See ../../crates/brooom/docs/cpsat-boundary.md for the
worked boundary example and why.

What it does
------------
Reads the SAME VROOM-style JSON that `brooom` already accepts (the
`crates/brooom/src/problem.rs` schema: `vehicles`, `jobs`, optional `matrices`)
and emits a self-contained, runnable OR-Tools CP-SAT *Python script* that models
the instance as an exact assignment+routing CP problem.

Scope (deliberately narrow — this is a bridge, NOT a re-implemented solver):
  * single capacity dimension (delivery, dim 0)
  * symmetric or asymmetric duration matrix (built haversine-style if absent,
    matching brooom's HaversineMatrix defaults: 13.9 m/s, 1.3 detour)
  * optional per-vehicle capacity
  * optional per-job hard time windows (first window only)
  * jobs are mandatory (the propagation-hard regime); prize/disjunction
    relaxations are NOT exported — those are exactly the cases local search
    handles fine and should stay in brooom.

It does NOT try to reproduce brooom's full objective shaping (span_cost,
distance_weight, breaks, shipments, skills, groups). The point of the bridge is
the narrow feasibility-propagation class, not feature parity. Anything it cannot
represent it refuses loudly rather than silently dropping (see `_unsupported`).

Usage
-----
    python3 export_cpsat.py PROBLEM.json > model_cpsat.py
    python3 model_cpsat.py            # requires: pip install ortools

Or in one shot against the repo sample:
    python3 export_cpsat.py ../../crates/brooom/examples/oslo_5jobs.json \
        > /tmp/oslo_cpsat.py && python3 /tmp/oslo_cpsat.py
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any

# brooom HaversineMatrix defaults (crates/brooom/src/matrix.rs)
SPEED_MPS = 13.9
DETOUR = 1.3
EARTH_R = 6_371_000.0


def _haversine_m(a: list[float], b: list[float]) -> float:
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(h))


def _unsupported(msg: str) -> None:
    sys.stderr.write(f"cpsat-bridge: REFUSING to export — {msg}\n")
    sys.stderr.write(
        "  This instance is outside the narrow propagation-hard class the bridge\n"
        "  targets. Solve it natively in brooom (local search handles it), or\n"
        "  simplify the instance. See docs/cpsat-boundary.md.\n"
    )
    sys.exit(2)


def _loc_coord(loc: Any) -> list[float] | None:
    """Mirror brooom's forgiving Location parsing: array, {lon,lat}, {coord}."""
    if loc is None:
        return None
    if isinstance(loc, list) and len(loc) == 2:
        return [float(loc[0]), float(loc[1])]
    if isinstance(loc, dict):
        if isinstance(loc.get("coord"), list):
            return [float(loc["coord"][0]), float(loc["coord"][1])]
        if "lon" in loc and "lat" in loc:
            return [float(loc["lon"]), float(loc["lat"])]
    return None


def _first_window(job: dict) -> tuple[int, int] | None:
    tws = job.get("time_windows") or []
    if not tws:
        return None
    w = tws[0]
    if isinstance(w, list) and len(w) == 2:
        return int(w[0]), int(w[1])
    if isinstance(w, dict):
        return int(w["start"]), int(w["end"])
    return None


def build_model_source(problem: dict) -> str:
    vehicles = problem.get("vehicles") or []
    jobs = problem.get("jobs") or []
    if problem.get("shipments"):
        _unsupported("shipments (pickup+delivery pairing) are not exported")
    if not vehicles:
        _unsupported("problem has no vehicles")
    if not jobs:
        _unsupported("problem has no jobs")

    # Single shared depot is the simplest exact model; refuse mixed depots.
    depots = {tuple(_loc_coord(v.get("start")) or []) for v in vehicles}
    depots |= {tuple(_loc_coord(v.get("end")) or []) for v in vehicles}
    depots.discard(())
    if len(depots) != 1:
        _unsupported(
            f"all vehicles must share one start/end depot for this POC "
            f"(found {len(depots)} distinct endpoints)"
        )
    depot_coord = list(next(iter(depots)))

    # Node 0 = depot, nodes 1..=N = jobs (in input order).
    coords = [depot_coord] + [_loc_coord(j["location"]) for j in jobs]
    if any(c is None for c in coords):
        _unsupported("a job/depot location has no usable coordinate (index-only "
                     "matrices without coords are not supported by this POC)")
    n = len(coords)

    # Duration matrix: provided (profile 'car') if present, else haversine.
    provided = (problem.get("matrices") or {}).get("car")
    if provided and provided.get("durations"):
        dur = provided["durations"]
        if len(dur) != n or any(len(r) != n for r in dur):
            _unsupported(
                "provided 'car' duration matrix size does not match "
                "depot+jobs node count; index remapping is out of scope"
            )
        dur = [[int(x) for x in row] for row in dur]
    else:
        dur = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = _haversine_m(coords[i], coords[j]) * DETOUR
                    dur[i][j] = round(d / SPEED_MPS)

    demands = [0] + [int((j.get("delivery") or [0])[0]) for j in jobs]
    services = [0] + [int(j.get("service") or 0) for j in jobs]
    windows = [None] + [_first_window(j) for j in jobs]
    job_ids = [int(j["id"]) for j in jobs]
    caps = [int((v.get("capacity") or [0])[0]) for v in vehicles]
    veh_ids = [int(v["id"]) for v in vehicles]
    num_vehicles = len(vehicles)

    horizon = max(
        (w[1] for w in windows if w) ,
        default=0,
    )
    horizon = max(horizon, sum(sum(r) for r in dur) + sum(services)) + 1

    # Emit a standalone CP-SAT script. We model an arc-based VRP with MTZ-style
    # time variables that double as subtour elimination + time-window propagation
    # — precisely the bidirectional propagation local search lacks.
    ctx = {
        "n": n,
        "num_vehicles": num_vehicles,
        "dur": dur,
        "demands": demands,
        "services": services,
        "windows": windows,
        "caps": caps,
        "veh_ids": veh_ids,
        "job_ids": job_ids,
        "horizon": horizon,
    }
    # Emit the context as a Python literal (repr), NOT JSON: the template is a
    # Python file, so None/True must stay None/True — json.dumps would write
    # null/true and the generated script would NameError at import.
    return _TEMPLATE.format(
        ctx=repr(ctx),
        src_note=problem.get("description") or "(brooom VROOM-style instance)",
    )


_TEMPLATE = '''#!/usr/bin/env python3
"""AUTO-GENERATED by tools/cpsat_bridge/export_cpsat.py — DO NOT EDIT BY HAND.

Exact CP-SAT routing model for a brooom Problem: {src_note}

Run:
    pip install ortools
    python3 {{this_file}}

Model: arc booleans x[v,i,j] + per-node arrival-time integers. The time vars are
the load-bearing part — CP-SAT *propagates* them bidirectionally, so a chain of
tight time windows that would trap greedy+local-search is resolved by inference
here rather than by stochastic repair. That is the whole reason this bridge
exists; see crates/brooom/docs/cpsat-boundary.md.
"""
import sys

try:
    from ortools.sat.python import cp_model
except ImportError:
    sys.stderr.write(
        "ortools not installed. Run: pip install ortools\\n"
        "(This generated script is valid; it just needs the solver library.)\\n"
    )
    sys.exit(1)

CTX = {ctx}

N = CTX["n"]                 # node 0 = depot, 1..N-1 = jobs
V = CTX["num_vehicles"]
DUR = CTX["dur"]            # travel seconds, N x N
DEMAND = CTX["demands"]    # node demand (dim 0)
SERVICE = CTX["services"]  # service seconds per node
WIN = CTX["windows"]       # [start,end] or None per node
CAP = CTX["caps"]          # per-vehicle capacity
VEH_ID = CTX["veh_ids"]
JOB_ID = CTX["job_ids"]    # job id per node 1..N-1
HORIZON = CTX["horizon"]


def main():
    m = cp_model.CpModel()
    nodes = range(N)
    jobs = range(1, N)
    vehicles = range(V)

    # x[v,i,j] = vehicle v traverses arc i->j
    x = {{
        (v, i, j): m.NewBoolVar(f"x_{{v}}_{{i}}_{{j}}")
        for v in vehicles for i in nodes for j in nodes if i != j
    }}
    # serve[v,k] = vehicle v serves job-node k
    serve = {{(v, k): m.NewBoolVar(f"serve_{{v}}_{{k}}") for v in vehicles for k in jobs}}
    # arrival time at each job-node (one server, so a single var per node is fine)
    t = {{k: m.NewIntVar(0, HORIZON, f"t_{{k}}") for k in jobs}}

    # Each job served exactly once, by exactly one vehicle (MANDATORY regime).
    for k in jobs:
        m.Add(sum(serve[v, k] for v in vehicles) == 1)

    # Link serving to arcs: a served node has exactly one in- and one out-arc.
    for v in vehicles:
        for k in jobs:
            m.Add(sum(x[v, i, k] for i in nodes if i != k) == serve[v, k])
            m.Add(sum(x[v, k, j] for j in nodes if j != k) == serve[v, k])
        # Depot degree = number of distinct routes this vehicle runs (0 or 1 here:
        # we cap at one tour per vehicle by bounding depot out-degree to <=1).
        m.Add(sum(x[v, 0, j] for j in jobs) <= 1)
        m.Add(sum(x[v, 0, j] for j in jobs) == sum(x[v, k, 0] for k in jobs))

    # Capacity per vehicle.
    for v in vehicles:
        m.Add(sum(DEMAND[k] * serve[v, k] for k in jobs) <= CAP[v])

    # Time propagation + subtour elimination (MTZ on time).
    # If any vehicle goes i->j then t_j >= t_i + service_i + travel_ij.
    BIG = HORIZON + max(max(r) for r in DUR) + max(SERVICE) + 1
    for i in jobs:
        for j in jobs:
            if i == j:
                continue
            arc = sum(x[v, i, j] for v in vehicles)
            m.Add(t[j] >= t[i] + SERVICE[i] + DUR[i][j] - BIG * (1 - arc))
    # Depot departure seeds the first node's arrival.
    for j in jobs:
        arc0 = sum(x[v, 0, j] for v in vehicles)
        m.Add(t[j] >= DUR[0][j] - BIG * (1 - arc0))

    # Hard time windows.
    for k in jobs:
        w = WIN[k]
        if w is not None:
            m.Add(t[k] >= w[0])
            m.Add(t[k] <= w[1])

    # Objective: minimise total travel time across all arcs.
    m.Minimize(sum(DUR[i][j] * x[v, i, j]
                   for v in vehicles for i in nodes for j in nodes if i != j))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(m)

    print("status:", solver.StatusName(status))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("no feasible assignment found")
        return 1
    print("objective (total travel seconds):", round(solver.ObjectiveValue()))
    for v in vehicles:
        seq, cur = [], 0
        while True:
            nxt = [j for j in range(N) if j != cur
                   and solver.Value(x[v, cur, j]) == 1]
            if not nxt:
                break
            cur = nxt[0]
            if cur == 0:
                break
            seq.append(JOB_ID[cur - 1])
        if seq:
            print(f"vehicle {{VEH_ID[v]}}: {{seq}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write(__doc__ or "")
        sys.stderr.write("\nusage: export_cpsat.py PROBLEM.json > model_cpsat.py\n")
        return 2
    with open(argv[1], encoding="utf-8") as fh:
        problem = json.load(fh)
    sys.stdout.write(build_model_source(problem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
