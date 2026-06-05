# The CP-SAT boundary: what local search cannot close, and the honest escape hatch

> Status: **design note + working interop POC.** The bridge described in §4 is
> implemented and runs: `tools/cpsat_bridge/export_cpsat.py`. This doc is the
> blunt boundary statement — where MPEE/brooom's architecture structurally stops
> and why, and what to do about it. It does **not** propose building a propagation
> engine inside the solver. That would be re-implementing CP-SAT inside a
> local-search VRP solver — dishonest, and explicitly out of scope.

## 1. What MPEE is, architecturally

brooom is a **metaheuristic local-search** VRP solver: greedy/parallel insertion
to build a solution, then ruin-and-recreate + 2-opt/or-opt/relocate moves under
a population (HGS), guided by a **scalarised cost function**. Constraints enter
in three places:

* **Per-route hook** (`src/constraint.rs`) — a closure/`pyspell` program run on
  each *completed route*, returning `Feasible` / `Infeasible` / `Penalty(x)`.
* **Cross-route hook** (`src/global_constraint.rs`) — a closure run on the whole
  candidate solution in `recompute_summary`, returning an additive `Cost`; a hard
  violation is a large-but-finite `HARD = 1e12` penalty so the search keeps a
  gradient.
* **Native dimensions** (`src/dimension.rs`, capacities, time windows) — checked
  incrementally during evaluation.

Every one of these is **evaluative**: given a fully-formed (or partially-formed)
route, score it. None of them is **deductive**: none can look at the variable
domains and *infer* "node A must precede node B" or "this arc is impossible"
before a move is ever proposed. That asymmetry is the whole boundary.

## 2. The class local search structurally cannot close: tightly-coupled feasibility

Local search is excellent when the feasible region is *fat* — many good solutions,
smooth-ish cost landscape, repair-by-local-move works. It degrades sharply when
feasibility is **tightly coupled across many variables** so that the feasible
region is a thin sliver and almost every local move lands outside it. Then:

* greedy insertion builds an *infeasible* or dead-end partial solution,
* and every small repair move (relocate one node, swap two) is *also* infeasible,
  so the search has no gradient to follow — it does a random walk on a measure-zero
  set and stalls.

The canonical trigger is a **chain of mutually-constraining time windows** (also:
precedence DAGs, exact resource balance, "exactly-k" cardinality coupled with
routing). The order is forced by *inference over the windows*, not by distance —
exactly the signal local search throws away.

### 2.1 A concrete worked example (runnable)

`tools/cpsat_bridge/tw_chain.json`: one vehicle, four jobs, all near the depot,
each with a 5-minute service and a **300-second** time window:

| job | window (s)   |
|-----|--------------|
| 12  | [ 600,  900] |
| 13  | [1800, 2100] |
| 11  | [3000, 3300] |
| 14  | [4200, 4500] |

The windows are spaced so that **exactly one visiting order is feasible**:
`12 → 13 → 11 → 14`. Travel times are tiny relative to the gaps, so distance gives
essentially no signal about the order. A nearest-neighbour/greedy builder orders
by proximity and will happily produce, say, `11 → 12 → 13 → 14`, which is
infeasible at the very first window. Now local search must *discover* the unique
permutation by random relocations — with no cost gradient pointing at it, because
the infeasible orders are penalised flatly (`HARD`), not proportionally to "how
close to feasible" they are.

This generalises: with `n` such jobs there are `n!` orders and one feasible one;
the penalty surface is a flat plateau with a single pinhole. Local search finds
pinholes by luck. CP-SAT finds them by **propagation**: from `t[12] <= 900` and
`t[13] >= t[12] + service + travel`, it *deduces* `t[13]`'s reachable interval,
prunes orders that violate it, and never enumerates the bad `n!`.

### 2.2 Why MPEE can't close it natively — and why faking it is the wrong move

You could *try* to make the per-route hook smarter, but the information just is not
there: the hook sees a finished route and says yes/no. It cannot prune the search
space *ahead* of move generation, because there is no variable-domain
representation to prune — the solver works on concrete permutations, not domains.

The only ways to "fix" this inside MPEE are:

1. **Build a propagation engine** (interval/domain store, constraint queue,
   global propagators). That is literally implementing a CP-SAT-class solver
   inside a local-search solver. It is a different solver wearing a trench coat,
   would dwarf and destabilise the existing hot path, and is explicitly out of
   scope. We will not do this.
2. **Heavy problem-specific seeding** (e.g. topologically pre-order by windows).
   This works for *this* example but is a one-off hack per constraint family; it
   does not generalise and quietly rots as instances drift.

Neither is honest as "MPEE now handles propagation." So we draw the line here.

## 3. The escape hatch, in two tiers

MPEE already has the right answer for the *common* case and now has a bridge for
the *rare* case. Pick by where the constraint actually lives:

| Your constraint is…                                              | Use                                  |
|------------------------------------------------------------------|--------------------------------------|
| soft / preference / penalty ("avoid long routes", "prefer X")    | `constraint.rs` / `pyspell` (in-hot-loop, native, sandboxed) |
| local-hard but feasible region is fat (capacity, single TW, skills, precedence within a route) | native dimensions + per-route/global hooks |
| **propagation-hard**: tightly-coupled feasibility, thin region, greedy+LS stalls (TW chains, exact balance, coupled cardinality) | **export to CP-SAT** via `tools/cpsat_bridge` |

The first two tiers are the existing, fast, native machinery and cover the
overwhelming majority of real instances. The third tier is the bridge — used
**rarely and deliberately**, when an instance is genuinely in the propagation-hard
class above.

## 4. The bridge (implemented, runs)

`tools/cpsat_bridge/export_cpsat.py` reads the same VROOM-style JSON brooom
accepts and emits a standalone OR-Tools CP-SAT Python script for a narrow,
clearly-bounded subset (single depot, single capacity dim, optional hard time
windows, mandatory jobs). It refuses *loudly* (exit 2) on anything outside that
subset rather than silently dropping constraints. See the README there for the
exact scope, the model shape, and the commands.

Verified in this environment:

* generator runs on `crates/brooom/examples/oslo_5jobs.json` and produces a
  `py_compile`-clean script;
* with `ortools 9.15` installed, that script solves to `OPTIMAL` (objective 1387,
  two routes);
* the `tw_chain.json` stress fixture from §2.1 solves to the unique feasible
  order `12 → 13 → 11 → 14` — the order MPEE's greedy+LS would have to stumble onto
  by chance.

### What the bridge does NOT claim

* It is **not** feature-parity with brooom: no shipments, multi-depot, skills,
  groups, breaks, multi-trip, prize/disjunction relaxation, or brooom's full
  cost shaping. Those either belong in native MPEE or are simply out of POC scope.
* It is **not** integrated into the solve loop and must not be. It is an offline
  exporter you reach for when an instance is in the narrow hard class. Wiring it
  into the hot path would re-introduce exactly the "solver inside a solver"
  problem we refused in §2.2.
* It does **not** round-trip the answer back into brooom automatically. For the
  POC the value is demonstrating that the *hard sub-instance* is exportable and
  solvable by the right tool; piping the CP-SAT route back in as a warm start is
  a natural next step but is not built here.

## 5. Honest bottom line

This branch closes the gap **partially and honestly**: it does not give MPEE
native CP-SAT modelling power (that would be a new solver), but it (a) states
precisely where local search structurally stops and why, with a runnable
counter-example, and (b) ships a working bridge so the rare propagation-hard
instance has a real, validated path to an exact solver instead of silently
getting a wrong/stalled answer. The 90%+ of instances that are *not*
propagation-hard continue to use the fast native hooks; the bridge is the
deliberate exit for the rest.
