#!/usr/bin/env python3
"""Parse Gehring-Homberger / Solomon BKS solution files and emit Vroom-
style solution JSON, mirroring the format of `*.vroom.json` / `*.brooom.json`
under benchmarks/results/.

Input format (sintef.no convention, also rogalski-wmii-uni-lodz-pl repo):

    Instance name : c1_2_1
    Authors       : ...
    Date          : ...
    Reference     : ...
    Solution
    Route 1 : 6 268 980 ...
    Route 2 : ...
    ...

Customer numbers refer to the canonical .txt instance (1..N). Index 0 is
the depot.

Usage:
    python3 bks_to_solution.py <bks.txt> <instance.json> <out.json>
    python3 bks_to_solution.py --batch <bks_dir> <instance_dir> <out_dir>

Note: BKS files contain routes only, NOT timestamps. Output sets
arrival/duration/etc. to None — they can be recomputed if needed by
running Vroom in `--check` mode on the constructed solution.
"""
import json
import re
import sys
from pathlib import Path


# Pattern to match BKS filename: <instance>.<vehicles>_<distance>.txt
FNAME_PATTERN = re.compile(r"^([a-z0-9_]+)\.(\d+)_([\d.]+)\.txt$")


def parse_bks_routes(text: str) -> list[list[int]]:
    """Returns list of routes; each route is a list of customer indices
    (1..N), excluding the depot."""
    lines = text.splitlines()
    routes = []
    for line in lines:
        if not line.startswith("Route"):
            continue
        # "Route 1 : 6 268 980 ..."
        after = line.split(":", 1)[1].strip()
        if not after:
            continue
        cust = [int(x) for x in after.split() if x]
        routes.append(cust)
    return routes


def routes_to_vroom_solution(routes: list[list[int]], instance_path: Path) -> dict:
    """Wrap parsed routes in a Vroom-style output JSON, matching the shape
    of `*.vroom.json` under benchmarks/results/."""
    inst = json.loads(instance_path.read_text())
    matrix = inst["matrices"][next(iter(inst["matrices"]))]
    durations = matrix["durations"]
    distances = matrix.get("distances")

    n = len(durations)
    # Build job lookup: location_index → demand, time_windows, service.
    job_by_loc = {j["location_index"]: j for j in inst["jobs"]}
    cap = inst["vehicles"][0]["capacity"][0] if inst["vehicles"] else 0

    out_routes = []
    total_cost = 0
    total_dist = 0
    total_service = 0
    total_delivery = 0
    for vidx, route in enumerate(routes):
        steps = [{"type": "start", "location_index": 0, "load": [0]}]
        load = 0
        last = 0
        cost = 0
        dist = 0
        service_sum = 0
        for c in route:
            if c < 0 or c >= n:
                continue
            cost += durations[last][c]
            if distances:
                dist += distances[last][c]
            j = job_by_loc.get(c)
            d = (j or {}).get("delivery", [0])[0]
            load += d
            steps.append({
                "type": "job",
                "id": (j or {}).get("id", c),
                "location_index": c,
                "service": (j or {}).get("service", 0),
                "load": [load],
            })
            service_sum += (j or {}).get("service", 0)
            last = c
        # Close route back to depot.
        cost += durations[last][0]
        if distances:
            dist += distances[last][0]
        steps.append({"type": "end", "location_index": 0, "load": [load]})
        out_routes.append({
            "vehicle": vidx + 1,
            "cost": cost,
            "service": service_sum,
            "duration": cost,
            "distance": dist,
            "delivery": [load],
            "steps": steps,
        })
        total_cost += cost
        total_dist += dist
        total_service += service_sum
        total_delivery += load

    summary = {
        "code": 0,
        "cost": total_cost,
        "routes": len(out_routes),
        "unassigned": 0,
        "delivery": [total_delivery],
        "amount": [total_delivery],
        "pickup": [0],
        "setup": 0,
        "service": total_service,
        "duration": total_cost,
        "waiting_time": 0,
        "distance": total_dist,
    }
    return {
        "code": 0,
        "summary": summary,
        "unassigned": [],
        "routes": out_routes,
    }


def convert_one(bks_path: Path, inst_path: Path, out_path: Path):
    routes = parse_bks_routes(bks_path.read_text())
    sol = routes_to_vroom_solution(routes, inst_path)
    out_path.write_text(json.dumps(sol))


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--help":
        print(__doc__)
        sys.exit(0)

    if args[0] == "--batch":
        bks_dir = Path(args[1])
        inst_dir = Path(args[2])
        out_dir = Path(args[3])
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build map: instance_name → latest BKS file (smallest V then D).
        latest = {}
        for f in bks_dir.rglob("*.txt"):
            m = FNAME_PATTERN.match(f.name)
            if not m:
                continue
            inst, V, D = m.group(1), int(m.group(2)), float(m.group(3))
            key = (V, D, str(f))
            if inst not in latest or key[:2] < latest[inst][:2]:
                latest[inst] = key

        n_done = 0
        for inst, (V, D, fpath) in latest.items():
            ipath = inst_dir / f"{inst}.json"
            if not ipath.exists():
                continue
            opath = out_dir / f"{inst}.bks.json"
            try:
                convert_one(Path(fpath), ipath, opath)
                n_done += 1
            except Exception as e:
                print(f"  failed {inst}: {e}")
        print(f"Converted {n_done} BKS solutions → {out_dir}/")
    else:
        bks_path = Path(args[0])
        inst_path = Path(args[1])
        out_path = Path(args[2])
        convert_one(bks_path, inst_path, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
