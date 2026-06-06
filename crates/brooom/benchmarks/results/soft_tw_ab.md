# Penalty-managed soft constraints — A/B validation

Reproducible head-to-head of `--soft-tw` (penalty-managed soft constraints, the
OR-Tools soft-bound mechanism) vs `--no-soft-tw` (feasible-only hard search),
same time budget (`-l 10`), `target/release/brooom`, Solomon n=100 instances in
`crates/brooom/benchmarks/instances_solomon/`.

## 1. Feasible instances — soft must NOT regress

| instance | hard cost | soft cost | Δ%    |
|----------|-----------|-----------|-------|
| c101     | 82901.9   | 82901.9   | +0.00 |
| c201     | 59158.9   | 59158.9   | +0.00 |
| r101     | 165354.1  | 165354.1  | +0.00 |
| r201     | 117463.1  | 117463.1  | +0.00 |
| rc101    | 166209.1  | 166209.1  | +0.00 |
| rc201    | 132499.1  | 132499.1  | +0.00 |

With a fixed high λ (≈1000× the per-second travel cost), no time-warp / load /
duration violation is ever beneficial on a feasible instance, so the soft search
follows the same trajectory as hard — byte-identical cost. This is why AUTO
(soft on whenever a problem has **job** time windows) is safe to ship on by
default: it never hurts a feasible instance.

## 2. Over-constrained instance — soft serves instead of dropping

`r101` with every job time window shrunk to 25% width around its centre
(`/tmp/r101_tight.json` in the session; regenerate by scaling each `time_windows`
entry). Two customers can no longer be reached within their windows under hard TW.

| mode | unassigned | served distance | objective       |
|------|-----------|-----------------|-----------------|
| hard | 2         | 178438          | 2,000,178,438   |
| soft | 0         | 182052          | 375,582,052     |

Hard drops 2 stops (each charged the ~1e9 drop prize). Soft serves all 4, paying
`λ × lateness` (far below the prize) for the two late stops — a 5.3× lower
objective for +2% served distance. This is the intended behaviour: relax the time
window a little to serve a customer rather than abandon them.

## What we did NOT do (and why)

We first tried PyVRP's mechanism of using infeasibility as a *search bridge* to
reach better **hard-feasible** solutions (wander infeasible, return the best
feasible). Measured on Solomon at our ILS budget (8 starts × 30 kicks), that
**regressed** R/RC by +0.4% … +9% — the penalised walk spends its limited
iterations off the feasible manifold and rarely returns an improving feasible
solution. PyVRP wins with that mechanism because it runs orders of magnitude more
iterations with a diverse population; we do not replicate it here. Instead we ship
the OR-Tools soft-bound semantics above (serve-late beats drop), which is a strict
win: zero cost on feasible instances, large cost reduction when over-constrained.
