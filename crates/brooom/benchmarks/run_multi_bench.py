#!/usr/bin/env python3
"""Multi-solver CVRPTW benchmark on Solomon-format instances.

Runs each solver on the same JSON input and reports cost + wall-clock.
Supported solvers (auto-detected; missing ones are skipped):
  - vroom        (subprocess; needs vroom in PATH)
  - brooom       (subprocess; uses ../target/release/brooom)
  - ortools      (in-process; needs `ortools` pip package)
  - pyvrp        (in-process; HGS implementation; needs `pyvrp` pip package)

Usage:
  python3 run_multi_bench.py r1_0100 r1_0250
  python3 run_multi_bench.py --time-limit 60 r1_0500 r1_1000
  python3 run_multi_bench.py --solvers vroom,brooom,pyvrp r1_2000_s1

Output: one row per (instance, solver) printed as a table and saved to
`results/multi_bench.csv`.

All solvers see the same distance matrix (from the JSON), so any quality
gap is solver-only — not routing-engine-dependent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
INST_DIR = BENCH_DIR / "instances"
RES_DIR = BENCH_DIR / "results"
RES_DIR.mkdir(exist_ok=True)
BROOOM = BENCH_DIR.parent / "target" / "release" / "brooom"


# ---------- Vroom -----------------------------------------------------------

def run_vroom(json_path: Path, time_limit_s: int) -> dict:
    out_path = RES_DIR / f"{json_path.stem}.vroom.json"
    t0 = time.perf_counter()
    # Vroom has -t (number of threads) and -x (exploration level). For
    # a fair comparison we use defaults — single-threaded, level 5 (deepest).
    r = subprocess.run(
        ["vroom", "-i", str(json_path), "-o", str(out_path), "-x", "5"],
        capture_output=True, text=True, timeout=time_limit_s + 60,
    )
    t1 = time.perf_counter()
    if r.returncode != 0 or not out_path.exists():
        return {"cost": None, "time_s": t1 - t0, "error": r.stderr[:500]}
    data = json.loads(out_path.read_text())
    return {
        "cost": data["summary"]["cost"],
        "time_s": t1 - t0,
        "routes": data["summary"]["routes"],
        "unassigned": data["summary"].get("unassigned", 0),
    }


# ---------- brooom ----------------------------------------------------------

def run_brooom(json_path: Path, time_limit_s: int) -> dict:
    out_path = RES_DIR / f"{json_path.stem}.brooom.json"
    t0 = time.perf_counter()
    r = subprocess.run(
        [str(BROOOM), "-i", str(json_path), "-o", str(out_path),
         "-l", str(time_limit_s), "-m", "16"],
        capture_output=True, text=True, timeout=time_limit_s + 60,
    )
    t1 = time.perf_counter()
    if r.returncode != 0 or not out_path.exists():
        return {"cost": None, "time_s": t1 - t0, "error": r.stderr[:500]}
    data = json.loads(out_path.read_text())
    return {
        "cost": data["summary"]["cost"],
        "time_s": t1 - t0,
        "routes": data["summary"].get("routes", len(data.get("routes", []))),
        "unassigned": data["summary"].get("unassigned", 0),
    }


def run_brooom_gpu(json_path: Path, time_limit_s: int) -> dict:
    out_path = RES_DIR / f"{json_path.stem}.brooom_gpu.json"
    # Optimal config from N=500/1000/2000 sweeps: multi=16, ils=30, GPU
    # batch with kick=8 repeats=4 for diversity.
    env = {**os.environ, "BROOOM_GPU_REPEATS": "4", "BROOOM_GPU_KICK": "8", "BROOOM_PENALTY_LS": "1"}
    t0 = time.perf_counter()
    r = subprocess.run(
        [str(BROOOM), "-i", str(json_path), "-o", str(out_path), "--gpu",
         "-m", "16", "--ils-iters", "30", "-l", str(time_limit_s)],
        capture_output=True, text=True, timeout=time_limit_s + 60, env=env,
    )
    t1 = time.perf_counter()
    if r.returncode != 0 or not out_path.exists():
        return {"cost": None, "time_s": t1 - t0, "error": r.stderr[:500]}
    data = json.loads(out_path.read_text())
    return {
        "cost": data["summary"]["cost"],
        "time_s": t1 - t0,
        "routes": data["summary"].get("routes", len(data.get("routes", []))),
        "unassigned": data["summary"].get("unassigned", 0),
    }


# ---------- OR-Tools --------------------------------------------------------

def run_ortools(json_path: Path, time_limit_s: int) -> dict:
    """Run Google OR-Tools VRP solver via its constraint programming layer."""
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as e:
        return {"cost": None, "time_s": 0.0, "error": f"missing ortools: {e}"}

    data = json.loads(json_path.read_text())
    matrix = data["matrices"]["car"]["durations"]
    n_loc = len(matrix)
    vehicles = data["vehicles"]
    jobs = data["jobs"]
    n_veh = len(vehicles)
    depot = vehicles[0]["start_index"]
    veh_cap = vehicles[0]["capacity"][0]
    veh_tw = vehicles[0]["time_window"]

    # Build demand and TW arrays indexed by location.
    demand = [0] * n_loc
    service = [0] * n_loc
    tw_start = [veh_tw[0]] * n_loc
    tw_end = [veh_tw[1]] * n_loc
    job_indices: list[int] = []
    for j in jobs:
        li = j["location_index"]
        demand[li] = j["delivery"][0] if j.get("delivery") else 0
        service[li] = j.get("service", 0)
        if j.get("time_windows"):
            tw_start[li], tw_end[li] = j["time_windows"][0]
        job_indices.append(li)

    t0 = time.perf_counter()

    manager = pywrapcp.RoutingIndexManager(n_loc, n_veh, depot)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_idx, to_idx):
        a = manager.IndexToNode(from_idx); b = manager.IndexToNode(to_idx)
        return matrix[a][b]

    transit_cb = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Capacity
    def demand_cb(from_idx):
        return demand[manager.IndexToNode(from_idx)]
    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0, [veh_cap] * n_veh, True, "Capacity",
    )

    # Time + service
    def time_cb(from_idx, to_idx):
        a = manager.IndexToNode(from_idx); b = manager.IndexToNode(to_idx)
        return matrix[a][b] + service[a]
    time_cb_idx = routing.RegisterTransitCallback(time_cb)
    horizon = veh_tw[1]
    routing.AddDimension(time_cb_idx, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for li in range(n_loc):
        idx = manager.NodeToIndex(li)
        if idx == -1: continue
        time_dim.CumulVar(idx).SetRange(tw_start[li], tw_end[li])
    for v in range(n_veh):
        start = routing.Start(v); end = routing.End(v)
        time_dim.CumulVar(start).SetRange(veh_tw[0], veh_tw[1])
        time_dim.CumulVar(end).SetRange(veh_tw[0], veh_tw[1])
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(start))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(end))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s
    params.log_search = False

    solution = routing.SolveWithParameters(params)
    t1 = time.perf_counter()
    if solution is None:
        return {"cost": None, "time_s": t1 - t0, "error": "no feasible solution"}

    total = 0
    n_routes = 0
    for v in range(n_veh):
        idx = routing.Start(v)
        had_stops = False
        while not routing.IsEnd(idx):
            nxt = solution.Value(routing.NextVar(idx))
            if not routing.IsEnd(nxt):
                had_stops = True
            total += routing.GetArcCostForVehicle(idx, nxt, v)
            idx = nxt
        if had_stops:
            n_routes += 1
    return {"cost": total, "time_s": t1 - t0, "routes": n_routes, "unassigned": 0}


# ---------- PyVRP (HGS) -----------------------------------------------------

def run_pyvrp(json_path: Path, time_limit_s: int) -> dict:
    """HGS-CVRP via PyVRP. Builds a Model from the Vroom JSON."""
    try:
        from pyvrp import Model
        from pyvrp.stop import MaxRuntime
    except ImportError as e:
        return {"cost": None, "time_s": 0.0, "error": f"missing pyvrp: {e}"}

    data = json.loads(json_path.read_text())
    matrix = data["matrices"]["car"]["durations"]
    n_loc = len(matrix)
    vehicles = data["vehicles"]
    jobs = data["jobs"]
    depot_idx = vehicles[0]["start_index"]
    veh_cap = vehicles[0]["capacity"][0]
    veh_tw = vehicles[0]["time_window"]
    n_veh = len(vehicles)

    t0 = time.perf_counter()
    m = Model()
    depot = m.add_depot(x=0, y=0, tw_early=veh_tw[0], tw_late=veh_tw[1])
    m.add_vehicle_type(num_available=n_veh, capacity=[veh_cap],
                       start_depot=depot, end_depot=depot,
                       tw_early=veh_tw[0], tw_late=veh_tw[1])
    client_for_loc = {}
    for j in jobs:
        li = j["location_index"]
        tw_s, tw_e = (j["time_windows"][0] if j.get("time_windows") else veh_tw)
        delivery = j["delivery"][0] if j.get("delivery") else 0
        c = m.add_client(x=0, y=0, delivery=[delivery],
                         service_duration=j.get("service", 0),
                         tw_early=tw_s, tw_late=tw_e)
        client_for_loc[li] = c
    # Edges — full matrix, in raw integer units. PyVRP uses duration for TW.
    nodes = [depot] + [client_for_loc[li] for li in sorted(client_for_loc)]
    locs = [depot_idx] + sorted(client_for_loc)
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i == j: continue
            d = matrix[locs[i]][locs[j]]
            m.add_edge(a, b, distance=d, duration=d)

    res = m.solve(stop=MaxRuntime(time_limit_s), display=False)
    t1 = time.perf_counter()
    if not res.is_feasible():
        return {"cost": None, "time_s": t1 - t0, "error": "no feasible solution"}
    sol = res.best
    n_routes = len([r for r in sol.routes() if len(list(r)) > 0])
    return {"cost": int(sol.distance()), "time_s": t1 - t0,
            "routes": n_routes, "unassigned": 0}


# ---------- Main ------------------------------------------------------------

SOLVERS = {
    "vroom": run_vroom,
    "brooom": run_brooom,
    "brooom_gpu": run_brooom_gpu,
    "ortools": run_ortools,
    "pyvrp": run_pyvrp,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instances", nargs="+",
                    help="instance name (no .json) or path to JSON file")
    ap.add_argument("--time-limit", type=int, default=60,
                    help="per-solver time budget in seconds (default 60)")
    ap.add_argument("--solvers", default=",".join(SOLVERS.keys()),
                    help="comma-separated solver names")
    ap.add_argument("--out-csv", default=str(RES_DIR / "multi_bench.csv"))
    args = ap.parse_args()

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    rows = []

    for inst in args.instances:
        if inst.endswith(".json"):
            path = Path(inst).resolve()
        else:
            path = INST_DIR / f"{inst}.json"
        if not path.exists():
            print(f"  ! {inst}: file not found ({path})", file=sys.stderr)
            continue
        print(f"\n=== {path.name} ===")
        results = {}
        for s in solvers:
            if s not in SOLVERS:
                print(f"  ? unknown solver: {s}", file=sys.stderr); continue
            print(f"  running {s} ...", end=" ", flush=True)
            try:
                r = SOLVERS[s](path, args.time_limit)
            except Exception as e:
                r = {"cost": None, "time_s": 0.0, "error": str(e)[:200]}
            results[s] = r
            if r.get("cost") is None:
                print(f"FAIL ({r.get('error', '?')[:80]})")
            else:
                print(f"cost={r['cost']:.0f} t={r['time_s']:.1f}s routes={r.get('routes', '?')}")
            rows.append({
                "instance": path.stem, "solver": s,
                "cost": r.get("cost"), "time_s": round(r["time_s"], 2),
                "routes": r.get("routes"), "unassigned": r.get("unassigned"),
                "error": r.get("error", ""),
            })

        # Per-instance summary table.
        ok = [(s, results[s]) for s in solvers if results[s].get("cost") is not None]
        if ok:
            best_cost = min(r["cost"] for _, r in ok)
            print(f"\n  {'solver':<10} {'cost':>10} {'Δ vs best':>10} {'time(s)':>10}")
            for s, r in ok:
                d = (r["cost"] - best_cost) / best_cost * 100.0 if best_cost else 0.0
                marker = " ←best" if r["cost"] == best_cost else ""
                print(f"  {s:<10} {r['cost']:>10.0f} {d:>+9.2f}% {r['time_s']:>10.1f}{marker}")

    # Write all rows to CSV.
    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "solver", "cost", "time_s",
                                          "routes", "unassigned", "error"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
