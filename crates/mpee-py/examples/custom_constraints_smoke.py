#!/usr/bin/env python3
"""Smoke test for the custom-constraints-as-code path in `Router.solve`.

This is the bit that can't be exercised without a real Python build + map cache,
so run it once locally to confirm the callback wiring end to end.

------------------------------------------------------------------------------
SETUP (one time)
------------------------------------------------------------------------------
1. Build the extension into your environment, from `crates/mpee-py`:

       pip install maturin
       maturin develop --release        # or: pip install -e .

2. Get a routable cache for any area (produces a `.pp` and a `.ch`):

       mpee download europe/great-britain/england/greater-london
       mpee build data/greater-london-latest.osm.pbf        # → *.pp and *.ch

------------------------------------------------------------------------------
RUN
------------------------------------------------------------------------------
    python custom_constraints_smoke.py <cache.pp> <cache.ch>
        # or set MPEE_PP / MPEE_CH env vars and run with no args

Expected: the baseline serves all three jobs; the hard constraint drops job 20;
the soft penalty keeps all three. The script exits non-zero if any of these
fail.
"""

import json
import os
import sys

import mpee


def cache_paths():
    if len(sys.argv) >= 3:
        return sys.argv[1], sys.argv[2]
    pp, ch = os.environ.get("MPEE_PP"), os.environ.get("MPEE_CH")
    if pp and ch:
        return pp, ch
    sys.exit(
        "usage: python custom_constraints_smoke.py <cache.pp> <cache.ch>\n"
        "   (or set MPEE_PP and MPEE_CH). See the docstring for how to build a cache."
    )


def build_problem(router):
    """Place a depot + 3 jobs inside the loaded map so they snap to real roads."""
    b = router.bbox()
    clat = (b["min_lat"] + b["max_lat"]) / 2.0
    clon = (b["min_lon"] + b["max_lon"]) / 2.0
    # Small east/west offsets, clamped well inside the box.
    span = min(b["max_lon"] - b["min_lon"], 0.06) / 4.0
    pt = lambda dl: [clon + dl, clat]  # VROOM order is [lon, lat]
    return json.dumps({
        "vehicles": [{"id": 1, "start": pt(0.0), "end": pt(0.0), "capacity": [10]}],
        "jobs": [
            {"id": 10, "location": pt(+span), "delivery": [1]},
            {"id": 20, "location": pt(+2 * span), "delivery": [1]},
            {"id": 30, "location": pt(-span), "delivery": [1]},
        ],
    })


def served_ids(plan):
    return sorted(s["job_id"] for r in plan["routes"] for s in r["stops"])


def main():
    pp, ch = cache_paths()
    router = mpee.Router(pp, ch)
    problem = build_problem(router)

    # 1. Baseline — no custom constraint.
    base = router.solve(problem, time_limit_s=2.0)
    served = served_ids(base)
    print("baseline served:        ", served, "| unassigned:", base["unassigned"])
    assert served == [10, 20, 30], f"expected all three served, got {served}"

    # 2. Hard constraint: reject any route that visits job 20.
    def reject_20(route):
        # route = {vehicle_id, job_ids, cost, duration_s, distance_m, service_s, waiting_s}
        return False if 20 in route["job_ids"] else None

    hard = router.solve(problem, time_limit_s=2.0, constraints=[reject_20])
    served = served_ids(hard)
    print("with reject-job-20:     ", served, "| unassigned:", hard["unassigned"])
    assert 20 not in served, "job 20 should have been rejected"
    assert 20 in hard["unassigned"], "job 20 should be reported unassigned"
    assert 10 in served and 30 in served, "the other jobs should still be served"

    # 3. Soft penalty: discourage long routes but never reject.
    def penalize_long(route):
        return 500.0 if route["distance_m"] > 1_000 else None

    soft = router.solve(problem, time_limit_s=2.0, constraints=[penalize_long])
    served = served_ids(soft)
    print("with soft penalty:      ", served, "| unassigned:", soft["unassigned"])
    assert served == [10, 20, 30], "a soft penalty must not drop jobs"

    # 4. DSL string (compiled to native code, no per-route GIL callback).
    dsl = router.solve(problem, time_limit_s=2.0, constraints=["20 not in route.job_ids"])
    served = served_ids(dsl)
    print("with DSL string:        ", served, "| unassigned:", dsl["unassigned"])
    assert 20 not in served, "DSL constraint should reject job 20"
    assert 10 in served and 30 in served

    print("\nOK — both the callback and the native DSL constraint paths work. ✓")


if __name__ == "__main__":
    main()
