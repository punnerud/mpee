#!/usr/bin/env python3
"""Convert Solomon CVRPTW .txt instances to Vroom-JSON format.

Solomon convention: distance = Euclidean(coords), time = distance, all
in original units. Published BKS values are double-precision Euclidean
rounded to 2 decimals.

To preserve BKS-comparable precision while keeping integer matrices for
Vroom/brooom, we scale all values by 100. So our solver cost = BKS × 100
(approximately, modulo rounding).

Usage:
    python3 solomon_to_vroom.py In/r101.txt instances_solomon/r101.json
    python3 solomon_to_vroom.py --all In instances_solomon
"""
import json
import math
import sys
from pathlib import Path

SCALE = 100  # Scale factor for integer matrix.


def parse_solomon(path: Path):
    """Returns (n_vehicles, capacity, customers) where customers is a list
    of (id, x, y, demand, ready, due, service) tuples. Index 0 = depot."""
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines) and "VEHICLE" not in lines[i].upper():
        i += 1
    # Skip 2 lines: "NUMBER ... CAPACITY" header, then numbers.
    parts = lines[i + 2].split()
    n_vehicles, capacity = int(parts[0]), int(parts[1])

    while i < len(lines) and "CUSTOMER" not in lines[i].upper():
        i += 1
    # Skip 1 header line + 1 blank line.
    customers = []
    for line in lines[i + 2:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            cid, x, y, demand, ready, due, service = [int(p) for p in parts[:7]]
        except ValueError:
            continue
        customers.append((cid, x, y, demand, ready, due, service))
    return n_vehicles, capacity, customers


def solomon_to_vroom(n_vehicles: int, capacity: int, customers: list, scale: int = SCALE) -> dict:
    n = len(customers)
    coords = [(c[1], c[2]) for c in customers]
    # Square matrix.
    durations = [[0] * n for _ in range(n)]
    distances = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            durations[i][j] = int(round(scale * d))
            distances[i][j] = int(round(scale * d))

    depot = customers[0]
    depot_due = depot[5]

    jobs = []
    for c in customers[1:]:
        cid, x, y, demand, ready, due, service = c
        jobs.append({
            "id": cid,
            "location_index": cid,
            "delivery": [demand],
            "service": int(round(scale * service)),
            "time_windows": [[int(round(scale * ready)), int(round(scale * due))]],
        })

    vehicles = []
    for v in range(n_vehicles):
        vehicles.append({
            "id": v,
            "start_index": 0,
            "end_index": 0,
            "capacity": [capacity],
            "time_window": [0, int(round(scale * depot_due))],
        })

    # Persist raw (x, y) coords under a non-standard `_coords` key — Vroom
    # ignores unknown fields. Useful for downstream pattern extraction.
    return {
        "vehicles": vehicles,
        "jobs": jobs,
        "matrices": {"car": {"durations": durations, "distances": distances}},
        "_coords": [list(c) for c in coords],
    }


def convert_one(src: Path, dst: Path):
    nv, cap, cust = parse_solomon(src)
    inst = solomon_to_vroom(nv, cap, cust)
    dst.write_text(json.dumps(inst))
    print(f"  wrote {dst.name}: {len(cust) - 1} customers, {nv} vehicles, cap {cap}")


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--help":
        print(__doc__)
        sys.exit(0)
    if args[0] == "--all":
        in_dir = Path(args[1])
        out_dir = Path(args[2])
        out_dir.mkdir(exist_ok=True, parents=True)
        files = sorted(in_dir.glob("*.txt"))
        print(f"Converting {len(files)} instances from {in_dir} → {out_dir}")
        for src in files:
            convert_one(src, out_dir / f"{src.stem.lower()}.json")
    else:
        convert_one(Path(args[0]), Path(args[1]))


if __name__ == "__main__":
    main()
