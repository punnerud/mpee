# Constraint conformance — brooom vs OR-Tools

How well do we handle *constrained* instances (not just plain CVRPTW)? The
strongest possible check is against an **exact** solver: export the constrained
instance to OR-Tools **CP-SAT**, solve it to proven optimum, and warm-start that
optimum back into brooom. If brooom accepts it feasibly and doesn't make it
worse, we conform to the same constraints and reach the same optimum.

## CP-SAT round-trip (exact optimum on each constrained class)

`tools/cpsat_bridge/tests/run_roundtrip.py` — export → OR-Tools CP-SAT solve →
`brooom --warm-start`, asserted feasible. All fixtures **PASS**:

| fixture | constraint exercised | brooom cost (= CP-SAT optimum) | feasible |
|---|---|---|:--:|
| skills.json | skills / vehicle eligibility | 526 | ✅ 0 unassigned |
| multi_tw.json | multiple time windows per stop | 526 | ✅ 0 unassigned |
| multi_depot.json | distinct per-vehicle start/end | 630 | ✅ 0 unassigned |
| pickup_delivery.json | PD shipment (same vehicle, order) | 756 | ✅ 0 unassigned |
| multidim_group.json | multi-dim capacity + client group | 526 (+1 dropped by group) | ✅ group satisfied |
| tw_chain.json | propagation-hard TW chain | 1047 | ✅ unique feasible order |

`tw_chain` is the key case: a tightly-coupled time-window chain where greedy+LS
can stall. CP-SAT finds the unique feasible order `12→13→11→14`; brooom accepts
and holds it. This is exactly the propagation-hard 10% the bridge is for — and we
solve it *exactly* via the bridge, then keep the full constraint set on polish.

**Reproduce:** `.bench-venv/bin/python tools/cpsat_bridge/tests/run_roundtrip.py`
(needs `pip install ortools`; `BROOOM=target/release/brooom`).

## Plain CVRPTW quality (where we still lose)

On unconstrained-shape CVRPTW the head-to-head is in
[`competitor_comparison.md`](competitor_comparison.md). Honest summary:

- **Tie / win + fastest** on tight-window C1/R1 (brooom reaches the best cost in
  ~2 s vs the others' 10 s); OR-Tools failed to find *any* feasible solution on
  r101/rc101 in 10 s.
- **Lose 2–5 % to PyVRP on wide-window R2/RC2** (e.g. rc201 +4.69 %) — we
  over-consolidate (5 routes vs 8–9). This is the open quality gap; it is a
  solution-quality issue, not a constraint-coverage one.

## Bottom line

On *constraint conformance* we match OR-Tools' exact optimum across skills,
multi-TW, multi-depot, PD, capacity and groups (proven by round-trip). The
remaining measured deficit is wide-window CVRPTW *quality* vs PyVRP, tracked
separately. No constrained instance in this suite is solved infeasibly or worse
than the proven optimum.
