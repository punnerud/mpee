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
`crates/brooom/src/problem.rs` schema: `vehicles`, `jobs`, `shipments`, optional
`matrices`) and emits a self-contained, runnable OR-Tools CP-SAT *Python script*
that models the instance as an exact assignment+routing CP problem.

The generated script ALSO round-trips its answer: on a feasible solve it writes a
brooom warm-start JSON (the `load_warm_start` schema — `routes[].vehicle` + a
`steps[]` list of `{type: "job", location_index}` in solved visiting order) so the
exact CP-SAT tour can be fed straight back to `brooom --warm-start`. The location
indices are aligned with brooom's `resolve_coords` interning order (depots first
in vehicle order, then jobs in input order) so the round-trip lands on the right
jobs.

Scope (a bridge, NOT a re-implemented solver — but wider than the first cut):
  * multi-dimensional capacity (all `delivery`/`pickup` dims vs `vehicle.capacity`)
  * symmetric or asymmetric duration matrix (built haversine-style if absent,
    matching brooom's HaversineMatrix defaults: 13.9 m/s, 1.3 detour)
  * per-vehicle capacity (per dimension)
  * skills (a vehicle may only serve a job whose skills ⊆ vehicle skills)
  * multiple hard time windows per job (disjunction: one literal per window,
    exactly one active)
  * pickup-and-delivery shipments (same vehicle + arrival[pickup] ≤ arrival[delivery])
  * multi-depot (per-vehicle start/end nodes; the single-shared-depot assumption
    is relaxed)
  * priority (objective weight that rewards serving high-priority work)
  * client groups (exactly-one / k-of-N over group members)

Still REFUSED loudly (exit 2) rather than silently dropped: prize/disjunction
relaxation (optional jobs), driver breaks, multi-trip, per-vehicle span/distance
cost shaping, speed factors, setup/release times, max_tasks/max_travel/max_distance
caps. Those are exactly the cases local search handles fine and should stay in
brooom — or are simply out of POC scope. See `_unsupported`.

It does NOT try to reproduce brooom's full objective shaping. The point of the
bridge is the propagation-hard feasibility class plus a faithful warm-start, not
feature parity.

Usage
-----
    python3 export_cpsat.py PROBLEM.json > model_cpsat.py
    python3 model_cpsat.py            # requires: pip install ortools
                                      # writes solution_ws.json next to itself
    brooom --warm-start solution_ws.json PROBLEM.json

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
        "  This instance is outside the class the bridge targets. Solve it\n"
        "  natively in brooom (local search handles it), or simplify the\n"
        "  instance. See docs/cpsat-boundary.md.\n"
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


def _loc_index(loc: Any) -> int | None:
    """Explicit matrix index, if the input gave one (`{"index": n}`)."""
    if isinstance(loc, dict) and isinstance(loc.get("index"), int):
        return int(loc["index"])
    return None


def _windows(job: dict) -> list[tuple[int, int]]:
    """ALL hard windows for a job (was: first window only)."""
    out: list[tuple[int, int]] = []
    for w in job.get("time_windows") or []:
        if isinstance(w, list) and len(w) == 2:
            out.append((int(w[0]), int(w[1])))
        elif isinstance(w, dict):
            out.append((int(w["start"]), int(w["end"])))
    return out


def _cap_vec(v: Any) -> list[int]:
    if not v:
        return []
    return [int(x) for x in v]


def _pad(vec: list[int], dims: int) -> list[int]:
    return (vec + [0] * dims)[:dims]


class _Interner:
    """Replicate brooom resolve_coords interning order so the location_index we
    emit in the warm-start matches what brooom assigns when it builds its matrix.

    brooom order (crates/brooom/src/matrix.rs resolve_coords): explicit indices
    are honoured first; coordinate-only points are interned in iteration order —
    vehicle start, vehicle end (per vehicle), then jobs (and shipment halves) in
    input order. Identical coords collapse to one index.
    """

    def __init__(self) -> None:
        self._coords: list[list[float]] = []
        self._by_index: dict[int, list[float]] = {}

    def intern(self, loc: Any) -> int:
        idx = _loc_index(loc)
        coord = _loc_coord(loc)
        if idx is not None:
            if coord is not None:
                self._by_index[idx] = coord
            return idx
        if coord is None:
            _unsupported(
                "a location has neither coordinate nor matrix index "
                "(cannot place it in the routing matrix)"
            )
        for i, existing in enumerate(self._coords):
            if abs(existing[0] - coord[0]) < 1e-7 and abs(existing[1] - coord[1]) < 1e-7:
                return self._reserved(i)
        self._coords.append(coord)
        return self._reserved(len(self._coords) - 1)

    def _reserved(self, free_pos: int) -> int:
        # Free coords land *after* the highest explicit index, matching brooom's
        # two-pass scheme. If any explicit index was given we offset; for the
        # all-coordinate common case max_explicit is -1 → offset 0.
        return self._offset() + free_pos

    def _offset(self) -> int:
        return (max(self._by_index) + 1) if self._by_index else 0

    def coord_for(self, index: int) -> list[float] | None:
        if index in self._by_index:
            return self._by_index[index]
        free_pos = index - self._offset()
        if 0 <= free_pos < len(self._coords):
            return self._coords[free_pos]
        return None


def build_model_source(problem: dict) -> str:
    vehicles = problem.get("vehicles") or []
    jobs = list(problem.get("jobs") or [])
    shipments = problem.get("shipments") or []
    if not vehicles:
        _unsupported("problem has no vehicles")
    if not jobs and not shipments:
        _unsupported("problem has no jobs or shipments")

    # ---- Loud refusals for features still out of scope -------------------
    for j in jobs:
        if j.get("prize") is not None:
            _unsupported("per-job `prize` (prize-collecting / optional jobs) is "
                         "not exported — optional jobs are the fat-region case LS "
                         "handles; keep them in brooom")
        if j.get("disjunction_penalty") is not None:
            _unsupported("`disjunction_penalty` (optional drop) is not exported")
        if j.get("setup"):
            _unsupported("`setup` time is not exported")
        if j.get("release"):
            _unsupported("`release` time is not exported")
    for v in vehicles:
        if v.get("breaks"):
            _unsupported("driver `breaks` are not exported")
        if v.get("max_trips", 1) not in (1, None):
            _unsupported("`max_trips` > 1 (multi-trip / reload) is not exported")
        for shaping in ("span_cost", "distance_weight", "max_tasks",
                        "max_travel_time", "max_distance"):
            if v.get(shaping):
                _unsupported(f"per-vehicle `{shaping}` cost/cap shaping is not exported")
        if v.get("speed_factor", 1.0) not in (1.0, None):
            _unsupported("per-vehicle `speed_factor` != 1.0 is not exported")

    # ---- Build the node set, mirroring brooom's location interning -------
    interner = _Interner()
    # Vehicle endpoints first (start then end), so coordinate-only depots get the
    # low indices brooom would assign. Multi-depot: each vehicle keeps its own
    # start/end node; the single-shared-depot assumption is relaxed.
    veh_start_node: list[int] = []
    veh_end_node: list[int] = []
    for v in vehicles:
        s = v.get("start")
        e = v.get("end")
        if s is None and e is None:
            _unsupported("a vehicle has neither start nor end location")
        s = s if s is not None else e
        e = e if e is not None else s
        veh_start_node.append(interner.intern(s))
        veh_end_node.append(interner.intern(e))

    # Expand shipments into pickup/delivery job-nodes, recording PD pairs.
    expanded: list[dict] = list(jobs)
    pd_pairs: list[tuple[int, int]] = []  # (pickup job-node idx, delivery idx), 0-based into expanded
    for sh in shipments:
        p = dict(sh["pickup"])
        d = dict(sh["delivery"])
        # carry shipment-level skills onto both halves
        sh_sk = sh.get("skills") or []
        if sh_sk:
            p["skills"] = list(p.get("skills") or []) + list(sh_sk)
            d["skills"] = list(d.get("skills") or []) + list(sh_sk)
        if sh.get("priority"):
            p.setdefault("priority", sh["priority"])
            d.setdefault("priority", sh["priority"])
        pi = len(expanded)
        expanded.append(p)
        di = len(expanded)
        expanded.append(d)
        pd_pairs.append((pi, di))

    # Node 0..K-1 are job-nodes (expanded). Each carries its interned loc index.
    job_loc_index = [interner.intern(j["location"]) for j in expanded]
    job_coords = [interner.coord_for(li) for li in job_loc_index]
    if any(c is None for c in job_coords):
        # Coordinate-less but indexed jobs are fine ONLY if a provided matrix
        # covers them; pure-haversine needs coords.
        provided = (problem.get("matrices") or {}).get("car")
        if not (provided and provided.get("durations")):
            _unsupported("a job/depot location has no coordinate and no provided "
                         "matrix covers its index — cannot build a haversine matrix")

    # Stable node numbering for the CP model: 0..K-1 = job-nodes.
    K = len(expanded)

    # ---- Capacity (multi-dimensional) ------------------------------------
    dims = 0
    for j in expanded:
        dims = max(dims, len(j.get("delivery") or []), len(j.get("pickup") or []))
    for v in vehicles:
        dims = max(dims, len(v.get("capacity") or []))
    # net per-node load delta per dim: delivery adds to required load picked up at
    # depot; for capacity feasibility we use the classic "sum of demands ≤ cap"
    # form per dim (delivery treated as load carried; pickup as load gained).
    demand = []  # demand[node][dim] — capacity consumed by serving this node
    for j in expanded:
        deliv = _pad(_cap_vec(j.get("delivery")), dims)
        pick = _pad(_cap_vec(j.get("pickup")), dims)
        # A node consumes max(delivery, pickup) on its dimension for the simple
        # "total assigned load ≤ vehicle capacity" envelope this POC checks.
        demand.append([max(d, p) for d, p in zip(deliv, pick)])
    caps = [_pad(_cap_vec(v.get("capacity")), dims) for v in vehicles]

    # ---- Skills ----------------------------------------------------------
    job_skills = [sorted(set(int(s) for s in (j.get("skills") or []))) for j in expanded]
    veh_skills = [sorted(set(int(s) for s in (v.get("skills") or []))) for v in vehicles]

    # ---- Time windows (multiple per job) ---------------------------------
    job_windows = [_windows(j) for j in expanded]
    services = [int(j.get("service") or 0) for j in expanded]

    # ---- Priority & groups -----------------------------------------------
    priorities = [int(j.get("priority") or 0) for j in expanded]
    groups: dict[int, list[int]] = {}
    for k, j in enumerate(expanded):
        g = j.get("group")
        if g is not None:
            groups.setdefault(int(g), []).append(k)
    # exactly-one per group → those members are optional/disjoint. Jobs NOT in a
    # group remain mandatory.
    grouped_nodes = {k for members in groups.values() for k in members}

    # ---- Allowed-vehicles allowlist --------------------------------------
    allowed = []  # allowed[node] = None (any) | list[vehicle index]
    veh_id_to_idx = {int(v["id"]): vi for vi, v in enumerate(vehicles)}
    for j in expanded:
        av = j.get("allowed_vehicles")
        if av is None:
            allowed.append(None)
        else:
            allowed.append([veh_id_to_idx[int(x)] for x in av if int(x) in veh_id_to_idx])

    veh_ids = [int(v["id"]) for v in vehicles]
    job_ids = [int(j["id"]) for j in expanded]
    num_vehicles = len(vehicles)

    # ---- Distance/duration matrix over ALL distinct location indices ------
    # The matrix is keyed by interned location index. We collect the set of
    # indices actually used (depots + job-nodes) and build a dense matrix sized
    # max_index+1 (provided matrices are already that shape).
    used_indices = set(veh_start_node) | set(veh_end_node) | set(job_loc_index)
    matrix_n = max(used_indices) + 1
    provided = (problem.get("matrices") or {}).get("car")
    if provided and provided.get("durations"):
        dur = provided["durations"]
        if len(dur) < matrix_n or any(len(r) < matrix_n for r in dur):
            _unsupported("provided 'car' duration matrix is smaller than the "
                         "interned location count; index remapping is out of scope")
        dur = [[int(x) for x in row] for row in dur]
    else:
        coord_by_index = {li: interner.coord_for(li) for li in used_indices}
        if any(coord_by_index[li] is None for li in used_indices):
            _unsupported("cannot build a haversine matrix: a used location index "
                         "has no coordinate and no provided matrix")
        dur = [[0] * matrix_n for _ in range(matrix_n)]
        for i in used_indices:
            for j in used_indices:
                if i != j:
                    d = _haversine_m(coord_by_index[i], coord_by_index[j]) * DETOUR
                    dur[i][j] = round(d / SPEED_MPS)

    horizon = max((w[1] for ws in job_windows for w in ws), default=0)
    horizon = max(horizon, sum(sum(r) for r in dur) + sum(services)) + 1

    ctx = {
        "K": K,                       # number of job-nodes
        "num_vehicles": num_vehicles,
        "dur": dur,                   # matrix over interned location indices
        "job_loc_index": job_loc_index,
        "veh_start_node": veh_start_node,
        "veh_end_node": veh_end_node,
        "dims": dims,
        "demand": demand,
        "caps": caps,
        "job_skills": job_skills,
        "veh_skills": veh_skills,
        "job_windows": [[list(w) for w in ws] for ws in job_windows],
        "services": services,
        "priorities": priorities,
        "groups": {str(g): m for g, m in groups.items()},
        "grouped_nodes": sorted(grouped_nodes),
        "allowed": allowed,
        "pd_pairs": [list(p) for p in pd_pairs],
        "veh_ids": veh_ids,
        "job_ids": job_ids,
        "horizon": horizon,
    }
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

On a feasible solve this writes `solution_ws.json` next to the chosen output dir
(default: cwd) in brooom's `load_warm_start` schema, so the exact tour can be fed
back with:  brooom --warm-start solution_ws.json PROBLEM.json

Model: arc booleans x[v,i,j] over job-nodes + per-vehicle depot arcs, plus
per-node arrival-time integers. The time vars are the load-bearing part — CP-SAT
*propagates* them bidirectionally, so a chain of tight time windows that would
trap greedy+local-search is resolved by inference here rather than by stochastic
repair. That is the whole reason this bridge exists; see
crates/brooom/docs/cpsat-boundary.md.
"""
import json
import os
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

K = CTX["K"]                       # job-nodes 0..K-1
V = CTX["num_vehicles"]
DUR = CTX["dur"]                  # travel seconds, indexed by location index
JOB_LOC = CTX["job_loc_index"]   # location index per job-node
VSTART = CTX["veh_start_node"]   # start location index per vehicle
VEND = CTX["veh_end_node"]       # end location index per vehicle
DIMS = CTX["dims"]
DEMAND = CTX["demand"]           # K x DIMS
CAP = CTX["caps"]                # V x DIMS
JOB_SKILLS = CTX["job_skills"]
VEH_SKILLS = CTX["veh_skills"]
WIN = CTX["job_windows"]         # list of [start,end] per job-node
SERVICE = CTX["services"]
PRIORITY = CTX["priorities"]
GROUPS = CTX["groups"]           # group id (str) -> [job-node ...]
GROUPED = set(CTX["grouped_nodes"])
ALLOWED = CTX["allowed"]         # None or [vehicle idx ...] per job-node
PD_PAIRS = CTX["pd_pairs"]       # [pickup-node, delivery-node]
VEH_ID = CTX["veh_ids"]
JOB_ID = CTX["job_ids"]
HORIZON = CTX["horizon"]

# Objective weight rewarding a served high-priority job (subtracted from cost).
PRIORITY_WEIGHT = 1000


def _serves(v, k):
    """Skill + allowlist eligibility of vehicle v for job-node k."""
    if not set(JOB_SKILLS[k]).issubset(set(VEH_SKILLS[v])):
        return False
    if ALLOWED[k] is not None and v not in ALLOWED[k]:
        return False
    return True


def main():
    m = cp_model.CpModel()
    jobs = range(K)
    vehicles = range(V)

    # x[v,i,j] = vehicle v traverses arc i->j between job-nodes.
    x = {{
        (v, i, j): m.NewBoolVar(f"x_{{v}}_{{i}}_{{j}}")
        for v in vehicles for i in jobs for j in jobs if i != j
    }}
    # depot arcs: in[v,k] = v starts route by going depot->k; out[v,k] = k->depot.
    din = {{(v, k): m.NewBoolVar(f"din_{{v}}_{{k}}") for v in vehicles for k in jobs}}
    dout = {{(v, k): m.NewBoolVar(f"dout_{{v}}_{{k}}") for v in vehicles for k in jobs}}
    # serve[v,k] = vehicle v serves job-node k.
    serve = {{(v, k): m.NewBoolVar(f"serve_{{v}}_{{k}}") for v in vehicles for k in jobs}}
    # arrival time at each job-node.
    t = {{k: m.NewIntVar(0, HORIZON, f"t_{{k}}") for k in jobs}}
    # served[k] = job-node k served by some vehicle (used for optional groups).
    served = {{k: m.NewBoolVar(f"served_{{k}}") for k in jobs}}

    # Skill / allowlist ineligibility forces serve=0.
    for v in vehicles:
        for k in jobs:
            if not _serves(v, k):
                m.Add(serve[v, k] == 0)

    # served[k] linkage.
    for k in jobs:
        m.Add(sum(serve[v, k] for v in vehicles) == served[k])

    # Mandatory unless the node is in a group (group handles its own cardinality).
    for k in jobs:
        if k not in GROUPED:
            m.Add(served[k] == 1)

    # Groups: exactly one member served (k-of-N generalisation: == 1 here, the
    # brooom "exactly one per group" semantics). To get k-of-N change the RHS.
    for gid, members in GROUPS.items():
        m.Add(sum(served[k] for k in members) == 1)

    # Flow conservation per vehicle: in-degree == out-degree == serve.
    for v in vehicles:
        for k in jobs:
            inflow = din[v, k] + sum(x[v, i, k] for i in jobs if i != k)
            outflow = dout[v, k] + sum(x[v, k, j] for j in jobs if j != k)
            m.Add(inflow == serve[v, k])
            m.Add(outflow == serve[v, k])
        # one tour per vehicle: depot out-degree <=1 and balanced.
        m.Add(sum(din[v, k] for k in jobs) <= 1)
        m.Add(sum(din[v, k] for k in jobs) == sum(dout[v, k] for k in jobs))

    # Capacity per vehicle, per dimension.
    for v in vehicles:
        for d in range(DIMS):
            m.Add(sum(DEMAND[k][d] * serve[v, k] for k in jobs) <= CAP[v][d])

    # Time propagation + subtour elimination (MTZ on time) over job-node arcs.
    BIG = HORIZON + max((max(r) for r in DUR), default=0) + max(SERVICE, default=0) + 1
    for i in jobs:
        for j in jobs:
            if i == j:
                continue
            arc = sum(x[v, i, j] for v in vehicles)
            li, lj = JOB_LOC[i], JOB_LOC[j]
            m.Add(t[j] >= t[i] + SERVICE[i] + DUR[li][lj] - BIG * (1 - arc))
    # Depot departure seeds the first node's arrival (per vehicle start node).
    for v in vehicles:
        for k in jobs:
            travel = DUR[VSTART[v]][JOB_LOC[k]]
            m.Add(t[k] >= travel - BIG * (1 - din[v, k]))

    # Multiple time windows: exactly one window literal active per served node.
    for k in jobs:
        wins = WIN[k]
        if not wins:
            continue
        lits = []
        for (lo, hi) in wins:
            b = m.NewBoolVar(f"win_{{k}}_{{lo}}_{{hi}}")
            m.Add(t[k] >= lo).OnlyEnforceIf(b)
            m.Add(t[k] <= hi).OnlyEnforceIf(b)
            lits.append(b)
        # exactly one window iff the node is served (0 if unserved).
        m.Add(sum(lits) == served[k])

    # Pickup-and-delivery: same vehicle + arrival[pickup] <= arrival[delivery].
    for (p, d) in PD_PAIRS:
        for v in vehicles:
            m.Add(serve[v, p] == serve[v, d])  # same vehicle
        m.Add(t[p] <= t[d])

    # Objective: minimise travel time; reward served priority (so dropping a
    # high-priority grouped node is discouraged). For all-mandatory instances the
    # priority term is constant and does not change the optimum.
    travel_cost = sum(
        DUR[JOB_LOC[i]][JOB_LOC[j]] * x[v, i, j]
        for v in vehicles for i in jobs for j in jobs if i != j
    )
    depot_cost = sum(
        DUR[VSTART[v]][JOB_LOC[k]] * din[v, k] + DUR[JOB_LOC[k]][VEND[v]] * dout[v, k]
        for v in vehicles for k in jobs
    )
    priority_reward = sum(PRIORITY[k] * PRIORITY_WEIGHT * served[k] for k in jobs)
    m.Minimize(travel_cost + depot_cost - priority_reward)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(m)

    print("status:", solver.StatusName(status))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("no feasible assignment found")
        return 1
    print("objective:", round(solver.ObjectiveValue()))

    # Follow arcs per vehicle to recover visiting order, then emit BOTH a human
    # summary and the brooom warm-start JSON.
    ws_routes = []
    for v in vehicles:
        # find the depot start node, then walk arcs.
        starts = [k for k in jobs if solver.Value(din[v, k]) == 1]
        if not starts:
            continue
        cur = starts[0]
        seq_nodes = [cur]
        guard = 0
        while guard <= K:
            guard += 1
            nxt = [j for j in jobs if j != cur and solver.Value(x[v, cur, j]) == 1]
            if not nxt:
                break
            cur = nxt[0]
            seq_nodes.append(cur)
        seq_ids = [JOB_ID[k] for k in seq_nodes]
        print(f"vehicle {{VEH_ID[v]}}: {{seq_ids}}")
        steps = [
            {{"type": "job", "location_index": JOB_LOC[k]}}
            for k in seq_nodes
        ]
        ws_routes.append({{"vehicle": VEH_ID[v], "steps": steps}})

    out_dir = os.environ.get("CPSAT_WS_DIR") or os.getcwd()
    ws_path = os.path.join(out_dir, "solution_ws.json")
    with open(ws_path, "w", encoding="utf-8") as fh:
        json.dump({{"routes": ws_routes}}, fh, indent=2)
    print("warm-start written:", ws_path)
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
