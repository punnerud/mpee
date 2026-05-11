#!/usr/bin/env python3
"""Generate Solomon-style CVRPTW instances with embedded matrices.

Output is one JSON file per instance, accepted by both Vroom and brooom
without needing an external routing engine — the matrix is computed here
(Euclidean) and emitted under the "matrices" key.

Usage:
    python3 gen_solomon_like.py 100        # single instance with N=100
    python3 gen_solomon_like.py 100 1000   # several sizes
"""
import json
import math
import random
import sys
from pathlib import Path


def euclid_seconds(a, b, speed_mps=13.9, scale_m=10.0):
    # Coords are in scaled units; one unit = `scale_m` meters.
    dx = (a[0] - b[0]) * scale_m
    dy = (a[1] - b[1]) * scale_m
    d = math.hypot(dx, dy)
    return int(round(d / speed_mps)), int(round(d))


def make_instance(n_customers: int, seed: int = 42) -> dict:
    rng = random.Random(seed)

    # Customer 0 is the depot at (0, 0). Customers scattered in a 200×200
    # square (scaled unit ≈ 10 m, so 2 km box).
    box = 100.0
    speed_mps = 13.9
    scale_m = 10.0

    capacity_per_vehicle = 200
    # Solomon-style fleet: ample vehicles, optimum uses fewer.
    n_vehicles = max(3, n_customers // 5)
    horizon = 24 * 3600

    # Coordinates as integer-ish for matrix indexing.
    coords = [(0.0, 0.0)]  # depot at index 0
    for _ in range(n_customers):
        coords.append((rng.uniform(-box, box), rng.uniform(-box, box)))

    n_loc = len(coords)

    # Square matrices, Euclidean.
    durations = [[0] * n_loc for _ in range(n_loc)]
    distances = [[0] * n_loc for _ in range(n_loc)]
    for i in range(n_loc):
        for j in range(n_loc):
            if i == j:
                continue
            t, d = euclid_seconds(coords[i], coords[j], speed_mps, scale_m)
            durations[i][j] = t
            distances[i][j] = d

    # Demands and time windows.
    jobs = []
    for k in range(1, n_loc):
        demand = rng.randint(1, 25)
        service = 90
        # Earliest feasible based on direct trip from depot.
        out = durations[0][k]
        ret = durations[k][0]
        earliest = out + 60
        latest = horizon - ret - service - 60
        if latest <= earliest + 600:
            tw = [0, horizon]
        else:
            window_w = 7200
            start = rng.randint(earliest, max(earliest, latest - window_w))
            tw = [start, start + window_w]
        jobs.append({
            "id": k,
            "location_index": k,
            "service": service,
            "delivery": [demand],
            "time_windows": [tw],
        })

    vehicles = [
        {
            "id": v + 1,
            "profile": "car",
            "start_index": 0,
            "end_index": 0,
            "capacity": [capacity_per_vehicle],
            "time_window": [0, horizon],
        }
        for v in range(n_vehicles)
    ]

    return {
        "vehicles": vehicles,
        "jobs": jobs,
        "matrices": {
            "car": {"durations": durations, "distances": distances}
        },
    }


def main():
    """Usage:
        gen_solomon_like.py 100                  # single, seed=42
        gen_solomon_like.py 100 200              # multiple sizes
        gen_solomon_like.py --seed 7 100         # custom seed
        gen_solomon_like.py --seed 7 --tag run7 100   # custom output tag
    """
    args = list(sys.argv[1:])
    seed = 42
    tag = None
    while args and args[0].startswith("--"):
        if args[0] == "--seed":
            seed = int(args[1]); args = args[2:]
        elif args[0] == "--tag":
            tag = args[1]; args = args[2:]
        else:
            print(f"unknown flag: {args[0]}"); sys.exit(1)
    if not args:
        print(__doc__, file=sys.stderr); sys.exit(1)
    sizes = [int(s) for s in args]
    out_dir = Path(__file__).parent / "instances"
    out_dir.mkdir(exist_ok=True)
    for n in sizes:
        inst = make_instance(n, seed=seed)
        suffix = f"_{tag}" if tag else ""
        path = out_dir / f"r1_{n:04d}{suffix}.json"
        path.write_text(json.dumps(inst))
        total_demand = sum(j["delivery"][0] for j in inst["jobs"])
        total_cap = sum(v["capacity"][0] for v in inst["vehicles"])
        print(
            f"wrote {path} ({n} customers, {len(inst['vehicles'])} vehicles, "
            f"seed={seed}, demand={total_demand}, cap={total_cap}, util={total_demand/total_cap:.0%})"
        )


if __name__ == "__main__":
    main()
