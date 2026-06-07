# Matching PyVRP on R2/RC2 — population HGS + O(1) local search (2026-06-07)

The route-opening LS below narrowed the wide-window gap but left ~+1.5–3% to
PyVRP. Two compounding changes close it to **near parity**:

1. **O(1) cost-delta local search** (`local_search.rs`): each move's COST delta
   is exact edge arithmetic (the objective is arc-based when `span_cost==0`); only
   FEASIBILITY needs the full walk, confirmed lazily best-first. Cold LS **6–11×
   faster** (r211 303→~35 ms), **byte-identical cost** (max rel diff 0.00e0,
   `tests/incremental_ls.rs`). Gated to the homogeneous simple envelope;
   `BROOOM_NO_FAST_LS` forces the old path. Disabled inside `cluster_decompose`
   so the decomposed large-N path stays unchanged (N=1000 6-seed mean Δ −0.025%).

2. **Population HGS** (`genetic.rs`): giant-tour Order-Crossover + **Split**
   (Prins/Vidal optimal partition of a customer ordering into routes — it decides
   the route COUNT optimally, the global move ILS lacks; Split recovers PyVRP's
   r205 partition at +0.00%). The O(1) LS gives each offspring's education ~30 ms
   instead of ~250 ms ⇒ **~900–2750 generations** per island in 10 s (was tens).
   Run HGS-primary: a short ILS phase (`BROOOM_HGS_SPLIT`, default 0.15) seeds
   island 0 (best-of safety floor), then HGS islands get the bulk of the budget.

**Route-flexibility gate** (`solver.rs`): HGS only helps when the route count is
flexible (wide windows ⇒ many jobs per route). Greedy jobs-per-route cleanly
separates the families (tight R1/RC1 ≈ 4–6, wide R2/RC2 ≈ 17–33); HGS turns on
only at ≥ 8 (`BROOOM_HGS_MIN_ROUTELEN`). Tight-window stays pure ILS ⇒ no
regression there.

## Results — default config, `-l 10 -m 16`, mean of 2 (vs PyVRP)

| instance | vs ILS baseline | vs PyVRP |
|----------|-----------------|----------|
| r205  | −2.53% | +0.70% |
| r207  | −3.11% | **+0.23%** |
| r211  | −2.45% | +0.54% |
| rc206 | −2.85% | +0.37% |
| rc208 | −0.71% | +0.79% |
| r201  | −1.71% | +2.48% |
| rc201 | −1.16% | +4.30% |
| r101 / rc101 / c101 (tight, HGS gated off) | ±0.0% | unchanged |

Standalone HGS multi-seed (3 seeds) is low-variance: r205 +0.79%, r211 +1.05%,
rc208 +0.64%, r207 +0.77%, rc206 +0.14% mean vs PyVRP (rc206 hits +0.00% twice).

### VERIFIED same-harness eval — all 19 R2/RC2, both solvers 10 s (2026-06-07)

`results/pyvrp_eval_2026.txt` (brooom default `-m 16` vs PyVRP `MaxRuntime(10s)`,
same matrix): **mean Δ = +0.06 %**, every instance within −1.3 … +0.9 %, and
brooom **beats** PyVRP on **r202 −0.27, r206 −0.56, r208 −1.32, r209 −0.61,
rc204 −0.02, rc205 ±0.00**. Worst cases r211 +0.93, rc208 +0.79, r205 +0.70.
Tight-window r101/rc101/c101 tie (+0.00…+0.25 %). So brooom now **matches PyVRP**
on R2/RC2 — the +2–13 % gap is closed. (Earlier per-instance "+2–4 %" figures
compared against an optimistic reference, not PyVRP @10 s; same-harness shows
parity, e.g. r201 +0.03 %, rc201 +0.18 %.)

---

# Closing the R2/RC2 gap to PyVRP — route-opening local search (SHIPPED, gated)

## The diagnosis (verified)

PyVRP 0.13 is **Iterated Local Search, not HGS** (no GeneticAlgorithm/Population
in its source). The comparison is **fair**: both minimise the total matrix-arc
sum over the routes (`run_multi_bench` builds PyVRP edges with distance=duration=d
and reports `sol.distance()`; brooom's default cost = total travel time = same).

Root cause of our wide-window R2/RC2 gap: **route under-opening**. brooom packed
stops into too FEW routes (rc208: 3 vs PyVRP's 4; r211: 3 vs 4), raising distance;
more time helped but plateaued at 3 routes. PyVRP's default LS includes
`RelocateWithDepot` (opens routes mid-search); ours stripped empty routes and had
no in-search route-opening move.

## The fix

`perturb_small` (solver.rs) — the deep-LAHC trajectory's perturbation — now, with
small probability (`open_p≈0.08`), lifts a small **cluster (1–3 stops) onto a
fresh unused vehicle**. Opening a cluster (not a lone stop) gives the new route
enough substance to survive local search; LAHC acceptance lets the walk keep the
temporarily-worse opening until redistribution pays off. Gated to top-level small-N
(`allow_lahc`), additive (best-of), cluster sub-solves disabled.

## Results — brooom vs PyVRP, `-l 10 -m 16`

| instance | before | after | PyVRP | routes (b→after) |
|----------|--------|-------|-------|------------------|
| rc208 | +13.27% | **+2.2%** | 77892 | 3 → 5 |
| rc205 | +4.04%  | **+0.85%** | 115755 | → 8 |
| rc201 | +3.74%  | **+1.27%** | 126558 | → 8 |
| r205  | +4.02%  | **+3.22%** | 95415 | → 6 |
| r208  | −0.89%  | **−0.35% (we win)** | 71557 | → 4 |
| r211  | +4.65%  | **+2.6–5%** | 75523 | 3 → 4 |
| r201  | +2.25%  | +2.1% | 114776 | → 9 |

Mean wide-window gap dropped from ~+4–5% (worst +13%) to ~+1.5–2% (worst ~+3%),
and we now **beat PyVRP on r208**.

## No regression (strict gate passed)

- **Tight-window** (`-l 10 -m 16`): c101 +0.00%, r101 +0.00% (tie), c201 +0.00%,
  rc101 +0.28% (slightly better than before) — unchanged/ties preserved.
- **N=1000** (deterministic): r1_1000_s6/s7 **byte-IDENTICAL** — large-N win
  fully preserved (route-opening gated to small-N; clusters disabled).
- Full suite green. Isolated checks: `tests/over_consolidation.rs`
  (`--ignored --test-threads=1`: rc208 5 routes/79614, r211 4 routes/77512).

## Honest standing

R2/RC2 is now **much closer** to PyVRP (≈+1–3%, was +4–13%) and we win some (r208);
PyVRP's HGS-class LS still edges most wide-window instances. We tie tight-window,
are ~2.5–5× faster, and win at N≥1000.

## Consistency — full R2/RC2 set (19 instances), brooom vs PyVRP, `-l 10 -m 16`

Not a cherry-picked few — every wide-window instance, one rep each (brooom under
`-l` is wall-clock nondeterministic, so treat ±~0.5% as noise):

| metric | value |
|--------|-------|
| mean gap | **+1.66%** |
| median   | +1.42% |
| best     | −0.35% (r208 — we win) |
| worst    | +3.85% (r211) |
| win / tie / lose | **1 / 0 / 18** |

Per instance (gap%, brooom routes / PyVRP routes): r201 +1.77 (8/8), r202 +1.35
(6/7), r203 +0.95 (5/6), r204 +1.59 (4/4), r205 +3.22 (6/5), r206 +1.42 (5/5),
r207 +3.04 (4/4), r208 −0.35 (4/3), r209 +1.85 (5/5), r210 +0.81 (6/6), r211
+3.85 (5/4), rc201 +1.21 (8/9), rc202 +2.40 (8/8), rc203 +0.90 (5/5), rc204 +0.82
(4/4), rc205 +0.85 (8/7), rc206 +3.31 (7/7), rc207 +1.12 (5/6), rc208 +1.51 (5/4).

**Honest reading:** we do NOT consistently beat PyVRP — we lose 18/19 (win only
r208). But the gap is now small and consistent (mean +1.66%, worst +3.85%), down
from the pre-fix +2–13%. The residual is partly route-count mismatch (we still
sometimes over- or under-open by one, e.g. r211 5 vs 4, r205 6 vs 5). PyVRP's
HGS-class C++ local search remains the moat; brooom is a close, fast #2 here,
ties tight-window, and wins at N≥1000.
