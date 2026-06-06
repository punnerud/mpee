#!/usr/bin/env python3
"""Seeded real-map CVRPTW instance generator.

Draws N random delivery points inside a real region's bounding box from a
deterministic seed, so anyone can regenerate the *exact* same instance from
`(region, seed, n)` — no need to ship the coordinates. The N×N matrix is built
from real road distances via an **OSRM** `/table` endpoint (so all solvers share
one realistic matrix), or from straight-line **haversine** distances for a fully
offline, dependency-free reproduction.

Output is a Vroom-style JSON with the matrix embedded under `"matrices"."car"`,
which `run_multi_bench.py` feeds unchanged to brooom / vroom / OR-Tools / PyVRP.

Reproducibility contract
------------------------
* Coordinates depend ONLY on (region bbox, seed, n) — pure RNG, identical on
  every machine. Demands / time windows / service times likewise.
* The matrix depends on the routing source:
    --matrix haversine        → pure math, byte-identical everywhere (default)
    --matrix osrm --osrm-host  → real roads from that OSRM build (snapshot)
  Either way the generated instance file is self-contained, so a one-line
  "region=sf seed=7 n=100 matrix=osrm@<host>" header fully describes it.

Examples
--------
    # Fully reproducible, offline (haversine), San Francisco, 100 stops:
    python3 gen_realmap.py --region sf --seed 7 --n 100 -o instances_realmap/sf_s7_n100.json

    # Real road matrix via a local OSRM (driving), Oslo, 50 stops:
    python3 gen_realmap.py --region oslo --seed 1 --n 50 \
        --matrix osrm --osrm-host http://127.0.0.1:5000 -o instances_realmap/oslo_s1_n50.json

    # Then benchmark every solver on the same real-map instance:
    python3 run_multi_bench.py --time-limit 10 instances_realmap/sf_s7_n100.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import urllib.request
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

# (min_lon, min_lat, max_lon, max_lat) — modest, dense city boxes so random
# points land on a connected road network. Extend freely; the seed + these
# numbers are the whole reproducibility surface.
REGIONS = {
    "sf":     (-122.515, 37.735, -122.385, 37.810),   # San Francisco peninsula
    "oslo":   (10.680, 59.890, 10.820, 59.960),        # central Oslo
    "london": (-0.205, 51.460, 0.005, 51.560),         # inner London
    "nyc":    (-74.020, 40.700, -73.930, 40.820),      # Manhattan-ish
}


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle metres between (lon, lat) points."""
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(math.sqrt(h))


def haversine_matrix(coords, speed_mps=8.0):
    """N×N (durations_s, distances_m), straight-line. speed≈8 m/s ≈ 29 km/h city."""
    n = len(coords)
    durs = [[0] * n for _ in range(n)]
    dists = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = haversine_m(coords[i], coords[j])
            dists[i][j] = int(round(d))
            durs[i][j] = int(round(d / speed_mps))
    return durs, dists


def osrm_matrix(coords, host):
    """N×N (durations_s, distances_m) from an OSRM /table endpoint.

    coords are (lon, lat). OSRM returns float seconds / metres; we round to int
    to match the integer-matrix convention every solver here expects.
    """
    locs = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in coords)
    url = f"{host.rstrip('/')}/table/v1/driving/{locs}?annotations=duration,distance"
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.loads(resp.read())
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM table failed: {data.get('code')} {data.get('message','')}")
    n = len(coords)
    durs = [[int(round(data["durations"][i][j] or 0)) for j in range(n)] for i in range(n)]
    dists = [[int(round(data["distances"][i][j] or 0)) for j in range(n)] for i in range(n)]
    return durs, dists


def make_instance(region, seed, n, n_vehicles, matrix_source, osrm_host, horizon, tw_frac):
    if region in REGIONS:
        min_lon, min_lat, max_lon, max_lat = REGIONS[region]
    else:
        raise SystemExit(f"unknown region {region!r}; known: {', '.join(REGIONS)} (or add a bbox)")

    rng = random.Random(seed)
    # Point 0 = depot (box centre); points 1..n = customers, seeded-random in box.
    coords = [((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)]
    for _ in range(n):
        coords.append((rng.uniform(min_lon, max_lon), rng.uniform(min_lat, max_lat)))

    if matrix_source == "osrm":
        if not osrm_host:
            raise SystemExit("--matrix osrm needs --osrm-host")
        durs, dists = osrm_matrix(coords, osrm_host)
    else:
        durs, dists = haversine_matrix(coords)

    # Seeded demands / service / time windows (Solomon-flavoured). Each customer
    # gets a window of width `tw_frac × horizon` placed at a random feasible offset.
    jobs = []
    width = int(horizon * tw_frac)
    for i in range(1, n + 1):
        start = rng.randint(0, max(0, horizon - width))
        jobs.append({
            "id": i,
            "location_index": i,
            "delivery": [rng.randint(1, 25)],
            "service": rng.randint(300, 900),
            "time_windows": [[start, start + width]],
        })

    cap = max(50, (sum(j["delivery"][0] for j in jobs) // max(1, n_vehicles)) + 25)
    vehicles = [{
        "id": v,
        "start_index": 0,
        "end_index": 0,
        "capacity": [cap],
        "time_window": [0, horizon],
    } for v in range(n_vehicles)]

    return {
        "meta": {
            "generator": "gen_realmap.py",
            "region": region, "seed": seed, "n": n,
            "vehicles": n_vehicles, "matrix": matrix_source,
            "osrm_host": osrm_host if matrix_source == "osrm" else None,
            "bbox": REGIONS[region], "horizon": horizon, "tw_frac": tw_frac,
        },
        "vehicles": vehicles,
        "jobs": jobs,
        "matrices": {"car": {"durations": durs, "distances": dists}},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="sf", help=f"one of: {', '.join(REGIONS)}")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=100, help="number of customer stops")
    ap.add_argument("--vehicles", type=int, default=25)
    ap.add_argument("--matrix", choices=["haversine", "osrm"], default="haversine")
    ap.add_argument("--osrm-host", default=None, help="e.g. http://127.0.0.1:5000 or https://router.project-osrm.org")
    ap.add_argument("--horizon", type=int, default=43200, help="planning horizon seconds (default 12h)")
    ap.add_argument("--tw-frac", type=float, default=0.25, help="time-window width as a fraction of horizon")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    inst = make_instance(args.region, args.seed, args.n, args.vehicles,
                         args.matrix, args.osrm_host, args.horizon, args.tw_frac)

    out = Path(args.out) if args.out else \
        BENCH_DIR / "instances_realmap" / f"{args.region}_s{args.seed}_n{args.n}_{args.matrix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inst))
    m = inst["meta"]
    print(f"wrote {out}")
    print(f"  region={m['region']} seed={m['seed']} n={m['n']} vehicles={m['vehicles']} "
          f"matrix={m['matrix']}{('@'+m['osrm_host']) if m['osrm_host'] else ''}")
    print(f"  reproduce: python3 gen_realmap.py --region {m['region']} --seed {m['seed']} "
          f"--n {m['n']} --vehicles {m['vehicles']} --matrix {m['matrix']}"
          + (f" --osrm-host {m['osrm_host']}" if m['osrm_host'] else ""))


if __name__ == "__main__":
    main()
