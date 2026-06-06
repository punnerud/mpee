#!/usr/bin/env python3
"""CP-SAT bridge round-trip test: export -> solve (OR-Tools) -> warm-start brooom.

For every fixture it:
  1. runs `export_cpsat.py <fixture>` to generate a CP-SAT model script,
  2. runs that script (needs `ortools`) to solve and emit `solution_ws.json`,
  3. feeds the warm-start back through `brooom --warm-start` and asserts the
     result is feasible (every mandatory job assigned) and parses cleanly.

This is the test the bridge previously lacked — it proves the whole export ->
solve -> warm-start hand-off works end to end, not just that the generator runs.

Usage:
    .bench-venv/bin/python tools/cpsat_bridge/tests/run_roundtrip.py
    BROOOM=/path/to/brooom .bench-venv/bin/python tools/cpsat_bridge/tests/run_roundtrip.py

Exit code 0 = all fixtures passed; 1 = a failure; 77 = ortools not importable
(skipped, not failed) so CI without ortools doesn't go red.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BRIDGE = Path(__file__).resolve().parent.parent
EXPORT = BRIDGE / "export_cpsat.py"
REPO = BRIDGE.parent.parent
BROOOM = os.environ.get("BROOOM", str(REPO / "target" / "release" / "brooom"))
PY = sys.executable

FIXTURES = sorted((BRIDGE / "fixtures").glob("*.json")) + [BRIDGE / "tw_chain.json"]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def roundtrip(fixture: Path, workdir: Path) -> tuple[bool, str]:
    # 1. export -> model script
    model = workdir / "model.py"
    r = run([PY, str(EXPORT), str(fixture)])
    if r.returncode != 0:
        return False, f"export failed: {r.stderr.strip()[:200]}"
    model.write_text(r.stdout)

    # 2. solve -> solution_ws.json (in workdir)
    env = {**os.environ, "CPSAT_WS_DIR": str(workdir)}
    r = run([PY, str(model)], env=env, cwd=str(workdir))
    if r.returncode != 0:
        return False, f"cp-sat solve failed: {r.stderr.strip()[:200]}"
    ws = workdir / "solution_ws.json"
    if not ws.exists():
        return False, "no solution_ws.json emitted"
    ws_doc = json.loads(ws.read_text())
    if not ws_doc.get("routes"):
        return False, "warm-start has no routes"

    # 3. warm-start back into brooom
    out = workdir / "out.json"
    if not Path(BROOOM).exists():
        return False, f"brooom binary not found at {BROOOM} (build --release)"
    r = run([BROOOM, "--warm-start", str(ws), "-i", str(fixture), "-o", str(out)])
    if r.returncode != 0 or not out.exists():
        return False, f"brooom --warm-start failed: {r.stderr.strip()[:200]}"
    sol = json.loads(out.read_text())
    unassigned = len(sol.get("unassigned", []))
    routes = sol.get("summary", {}).get("routes", len(sol.get("routes", [])))
    if routes < 1:
        return False, "warm-start produced no routes"
    # A group/optional fixture legitimately drops members; for the rest, a clean
    # round-trip should leave nothing unassigned.
    fixture_doc = json.loads(fixture.read_text())
    has_groups = any(j.get("group") is not None for j in fixture_doc.get("jobs", []))
    if unassigned != 0 and not has_groups:
        return False, f"{unassigned} unassigned after warm-start (expected 0)"
    return True, f"routes={routes} unassigned={unassigned} cost={sol.get('summary',{}).get('cost')}"


def main() -> int:
    try:
        import ortools  # noqa: F401
    except ImportError:
        print("SKIP: ortools not importable (pip install ortools). Not a failure.")
        return 77

    ok = True
    for fx in FIXTURES:
        with tempfile.TemporaryDirectory() as d:
            passed, msg = roundtrip(fx, Path(d))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {fx.name}: {msg}")
        ok = ok and passed
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
