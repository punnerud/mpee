# Trying to beat PyVRP on small N (N<301) — findings (NOT merged)

Goal: study PyVRP's code + adopt its mechanisms to beat it on small-N Solomon
(esp. wide-window R2/RC2, where we trail HGS-class quality by 2–5%).

## What PyVRP actually does (0.13.4 source, read directly)

- **Its default solver is Iterated Local Search, not the genetic algorithm**
  (`pyvrp/solve.py` builds `IteratedLocalSearch`). So matching it does **not**
  require building HGS.
- **TW-aware granular neighbourhood** (Vidal 2013): proximity =
  `edge_cost − prize + 0.2·max(0,wait) + 1.0·max(0,time_warp)`, symmetrised, K=50.
- **Late-Acceptance Hill-Climbing** acceptance (Burke & Bykov 2017), history 300:
  accept if better than the cost L iters ago *or* the current cost.
- **Exhaustive-on-best**: full LS on each new best.

## What we implemented (branch `feat/beat-pyvrp`, kept, not merged)

All three, faithfully: `Granular::build_tw` (Vidal proximity), LAHC acceptance in
the ILS loop, and exhaustive-on-best. Deterministic A/B vs current `main`
(`-m 8 --ils-iters 30`):

| instance | main | with package | effect |
|----------|------|--------------|--------|
| c101 | 82902 | 82902 | same |
| r101 | 165354 | **164287** | **−0.65% (ties PyVRP)** |
| rc101 | 166209 | 166159 | −0.03% |
| c201 | 59159 | 59159 | same |
| r201 | 117463 | 117752 | **+0.25% (regressed)** |
| rc201 | 132499 | **131594** | **−0.68%** |

N=1000 (deterministic, via cluster decomposition into <300-job sub-problems, so
the small-N path applies to its clusters):

| seed | main | package | effect |
|------|------|---------|--------|
| s6 | 12154 | 12267 | **+0.93% (regressed)** |
| s7 | 12038 | 12019 | −0.16% |
| s8 | 11962 | 11902 | −0.50% |

## Honest conclusion: did NOT ship

The package helps the wide-window targets (r101 now ties PyVRP, rc201 −0.68%) but
**fails the strict gate**: it regresses r201 (+0.25%) and N=1000 seed s6 (+0.93%).
The N≤300 gate cannot protect N=1000 because that instance is solved as <300-job
clusters, which inherit the new path; effects there are mixed (±0.2–0.9%).

A provably-non-regressing version (run greedy *and* LAHC per seed, keep the
better) would cost ~2× small-N compute — which conflicts with the "don't increase
search time" constraint. So at equal time and zero regression risk, these
mechanisms do **not** cleanly beat PyVRP further on small N.

Standing position (unchanged, honest): we beat OR-Tools and are ~2.5–5× faster on
small N; we trail PyVRP's HGS-class ILS by 2–5% on wide-window R2/RC2; we win at
N≥1000. The branch is preserved for future work (e.g. acceptance diversification
across multi-start seeds with cluster-path gating).
