# 4-way head-to-head — brooom vs VROOM vs OR-Tools vs PyVRP

Same instance, same embedded matrix, same wall-clock budget through every solver
(`run_multi_bench.py`). Re-run: `python3 run_multi_bench.py --time-limit 10 <instance>`.

## Solomon n=100/200, 10 s budget

| instance | best solver | cost | brooom | vroom | OR-Tools | PyVRP |
|----------|-------------|------|--------|-------|----------|-------|
| c101 (tight TW, clustered)   | tie       | 82901  | 82901 (2.1s) | 82901 | 82901 | 82901 |
| r101 (tight TW, random)      | brooom/PyVRP | 164287 | **164287 (2.0s)** | +0.4% | ✗ infeasible | 164287 |
| rc101 (tight TW, mixed)      | PyVRP     | 163978 | +0.92% | +1.30% | ✗ infeasible | **163978** |
| c201 (wide TW, clustered)    | tie       | 59158  | 59158 (5.7s) | 59158 | 59158 | 59158 |
| r201 (wide TW, random)       | PyVRP     | 114776 | +2.25% | +4.39% | +1.53% | **114776** |
| rc201 (wide TW, mixed)       | PyVRP     | 126558 | +4.69% | +1.74% | +1.43% | **126558** |

(✗ = OR-Tools did not reach a feasible solution within 10 s on the tight-window
R1/RC1 instances.)

## Honest read

**Where we tie / win**
* **Tight-window + clustered (C1) and tight random (R1):** brooom matches the
  best cost, and reaches it **fastest** (~2 s vs the others' full 10 s).
* **Feasibility under pressure:** OR-Tools failed to find *any* feasible solution
  on r101/rc101 in 10 s; brooom and PyVRP both solved them.
* **Speed:** brooom converges in 2–7 s where OR-Tools/PyVRP use the whole budget.

**Where we lose (and why)**
* **Wide-window R2/RC2:** brooom is 2–5 % behind PyVRP. The tell is the route
  count — on rc201 brooom uses **5 routes** while PyVRP/OR-Tools use 8–9. We
  **over-consolidate**: the local search strips empty routes and rarely re-opens
  a vehicle, so on wide windows (where spreading onto more, shorter routes is
  cheaper) we get stuck in a few long routes. This is the next quality lever.
* PyVRP (HGS) is SOTA at small N and wins overall here; that is expected — its
  population genetic search with millions of evaluations beats our ILS at n≤200.

**The fair framing.** brooom's edge is not "best objective at n=100" — it's
*integrated* (matrix + solver + geocoding in one process, no separate matrix
build), *fast to a good answer*, *memory-thrifty at scale* (streamed matrix), and
*feature-rich* (code-defined AST constraints in Rust+Python, soft time windows,
lexicographic objectives, custom dimensions). On large N (n≥1000) brooom's
prior runs beat PyVRP on most seeds; small-N quality parity with HGS is the open
gap, concentrated on wide-window R2/RC2.

## Reproduce on real maps, not just Solomon

```bash
python3 gen_realmap.py --region sf --seed 7 --n 100 --vehicles 25
python3 run_multi_bench.py --time-limit 10 instances_realmap/sf_s7_n100_haversine.json
```

A first real-road sanity check (SF, seed 3, n=20, OSRM matrix, 5 s) had all four
solvers tie at cost 9102 — small instances are easy; the gaps above appear at
n≥100 with tight/wide time windows.
