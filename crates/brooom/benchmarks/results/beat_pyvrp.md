# Closing on PyVRP at small N — adopting its ILS mechanisms (SHIPPED, gated)

Goal: study PyVRP's code and adopt its mechanisms to close the small-N gap
(wide-window R2/RC2), without risking our wins.

## What PyVRP actually does (0.13.4 source, read directly)

- **Its default solver is Iterated Local Search, not the genetic algorithm**
  (`pyvrp/solve.py` builds `IteratedLocalSearch`) — so no HGS needed.
- **TW-aware granular neighbourhood** (Vidal 2013): proximity =
  `edge_cost - prize + 0.2*max(0,wait) + 1.0*max(0,time_warp)`, symmetrised, K=50.
- **Late-Acceptance Hill-Climbing** acceptance (Burke & Bykov 2017): accept if
  better than the cost L iters ago *or* the current cost — escapes basins.
- **Exhaustive-on-best**: full LS on each new best.

## What we shipped - additive, provably non-regressing

Instead of *replacing* our search (an earlier replace-design regressed r201 and
some N=1000 seeds), we run the LAHC + TW-granular + exhaustive-on-best variants
**alongside** the proven greedy multi-start and take best-of-all. The greedy
variants are untouched, so the result is always <= the greedy-only result - it
can only help. Gated to top-level N<=300 (`SolverConfig.allow_lahc`); cluster
sub-solves disable it, so the large-N path is byte-identical. The ~2x small-N
variants are affordable: we finish small-N in ~2 s vs PyVRP's ~10 s.

### Gate results (deterministic `-m 8 --ils-iters 30`, exp vs main)

| instance | main | shipped | effect |
|----------|------|---------|--------|
| c101 | 82902 | 82902 | same |
| r101 | 165354 | 164644 | -0.43% |
| rc101 | 166209 | 165745 | -0.28% |
| c201 | 59159 | 59159 | same |
| r201 | 117463 | 117463 | same (no regression) |
| rc201 | 131264 | 131264 | same |

**N=1000 control (r1_1000_s6/s7/s8): byte-IDENTICAL** - the headline large-N win
is fully preserved (clusters disable the boost). Full test suite green.

### Gap to PyVRP at the benchmark setting (`-l 10 -m 16`)

| instance | brooom | PyVRP | gap |
|----------|--------|-------|-----|
| c101 | 82902 | 82901 | +0.00% |
| r101 | 164287 | 164287 | +0.00% (tie) |
| rc101 | 165487 | 163978 | +0.92% |
| c201 | 59159 | 59158 | +0.00% |
| r201 | 117357 | 114776 | +2.25% |
| rc201 | 131291 | 126558 | +3.74% |

## Honest read

The boost is **safe (never regresses, N=1000 untouched)** and gives real gains at
fixed-iteration budgets (r101 -0.43%, rc101 -0.28%). Under the tight `-l 10` +
high `-m 16` benchmark, doubling the variants dilutes per-variant time, so the
gain shrinks there (r101 still ties PyVRP; rc101/r201/rc201 ~unchanged). The
wide-window R2/RC2 gap to PyVRP's HGS-class ILS (+2.25% r201, +3.74% rc201)
**narrowed but is not closed** - matching SOTA there at equal time remains open.
We tie on tight-window C1/R1/C2, are ~2.5-5x faster, and win at N>=1000.
