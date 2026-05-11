#!/usr/bin/env python3
"""Extract reusable sub-route patterns from BKS solutions.

A sub-route pattern is a sliding window of K consecutive customers within
a BKS route. We compute a canonical, scale/translation-invariant signature
for each window so similar geometric structures hash to similar keys.

Output: JSONL with one line per sub-route pattern:
    {
      "instance": "c1_2_1",
      "vehicle": 5,
      "window_start": 3,        # position in route
      "customers": [42, 17, 89, 7],   # original customer ids
      "signature": [...],       # canonical k-d feature vector
      "context": {              # original-instance context
        "demand_sum": 38,
        "tw_span": [0.12, 0.45],
        "depot_dist": 0.23,     # distance from depot to first stop, normalized
      }
    }

Two sub-routes with similar `signature` (Euclidean distance < ε) are
candidate matches for transfer.

Usage:
    python3 extract_subroutes.py <bks_solutions_dir> <instances_dir> <out.jsonl>
                                 [--window-size K]
"""
import json
import math
import sys
from pathlib import Path


def signature(coords_window: list[tuple[float, float]]) -> list[float]:
    """Translation, rotation and scale invariant signature for a list of K
    points. Returns 2K-2 floats (relative offsets from first point, rotated
    so first edge is along +x, scaled to first-edge unit length)."""
    if len(coords_window) < 2:
        return [0.0]
    p0 = coords_window[0]
    p1 = coords_window[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    edge_len = math.hypot(dx, dy)
    if edge_len < 1e-9:
        return [0.0] * (2 * len(coords_window) - 2)
    # Rotation angle so (p0 → p1) maps to +x axis.
    cos = dx / edge_len
    sin = dy / edge_len
    # Translate, rotate, scale.
    sig = []
    for p in coords_window[2:]:
        rx = p[0] - p0[0]
        ry = p[1] - p0[1]
        # Inverse rotation: rotate by -theta so edge becomes +x.
        nx = (rx * cos + ry * sin) / edge_len
        ny = (-rx * sin + ry * cos) / edge_len
        sig.extend([nx, ny])
    return sig


def extract_from_route(route_steps: list[dict], coords: list[tuple[float, float]],
                       jobs_by_loc: dict, window_size: int = 4) -> list[dict]:
    """Extract sliding-window sub-route patterns from one BKS route."""
    # Pull customer indices (skip depot starts/ends).
    cust_steps = [s for s in route_steps if s["type"] == "job"]
    if len(cust_steps) < window_size:
        return []
    out = []
    for start in range(0, len(cust_steps) - window_size + 1):
        window = cust_steps[start:start + window_size]
        loc_indices = [s["location_index"] for s in window]
        coords_w = [coords[i] for i in loc_indices]
        sig = signature(coords_w)
        # Context features.
        demand_sum = sum(jobs_by_loc.get(i, {}).get("delivery", [0])[0] for i in loc_indices)
        tw_starts = [jobs_by_loc.get(i, {}).get("time_windows", [[0, 1]])[0][0] for i in loc_indices]
        tw_ends = [jobs_by_loc.get(i, {}).get("time_windows", [[0, 1]])[0][1] for i in loc_indices]
        out.append({
            "window_start": start,
            "customers": loc_indices,
            "signature": sig,
            "demand_sum": demand_sum,
            "tw_min": min(tw_starts),
            "tw_max": max(tw_ends),
        })
    return out


def main():
    args = sys.argv[1:]
    bks_dir = Path(args[0])
    inst_dir = Path(args[1])
    out_path = Path(args[2])
    window_size = 4
    if len(args) > 3 and args[3] == "--window-size":
        window_size = int(args[4])

    n_patterns = 0
    n_files = 0
    with out_path.open("w") as fd:
        for bks_file in sorted(bks_dir.glob("*.bks.json")):
            instance = bks_file.stem.replace(".bks", "")
            inst_file = inst_dir / f"{instance}.json"
            if not inst_file.exists():
                continue
            n_files += 1
            inst = json.loads(inst_file.read_text())
            sol = json.loads(bks_file.read_text())

            # Build location-coord lookup. Coords aren't in our converted
            # JSON (matrix-only); reconstruct from depot=0 and dist matrix.
            # For sub-route signature we need real coordinates — load the
            # ORIGINAL Solomon/G-H .txt to get them.
            # Skip for now if we can't get coords.
            # Better: store coords in instance JSON during conversion.
            coords = inst.get("_coords")  # injected during conversion
            if coords is None:
                # Fallback: skip.
                continue
            jobs_by_loc = {j["location_index"]: j for j in inst["jobs"]}

            for r in sol["routes"]:
                patterns = extract_from_route(r["steps"], coords, jobs_by_loc, window_size)
                for p in patterns:
                    p["instance"] = instance
                    p["vehicle"] = r["vehicle"]
                    fd.write(json.dumps(p) + "\n")
                    n_patterns += 1

    print(f"Extracted {n_patterns} sub-route patterns from {n_files} BKS solutions")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
