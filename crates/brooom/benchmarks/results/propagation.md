# Native structured propagation — measured effect

`crate::propagate::tighten` runs once before search: tighten time windows (sound
bounds from depot travel + shift), close precedence transitively, and flag
provably-unservable jobs. On by default (`--no-propagate` to disable). All numbers
from `target/release/brooom`.

## Soundness — the solution is unchanged (the hard requirement)

Propagation must never tighten away a feasible option. Solomon n=100, propagate
on vs off (`-l 8`, default soft):

| instance | propagate on | propagate off | same? |
|----------|-------------|---------------|:-----:|
| c101  | 82901.9  | 82901.9  | ✅ |
| r101  | 165354.1 | 165354.1 | ✅ |
| rc101 | 166209.1 | 166209.1 | ✅ |
| r201  | 117463.1 | 117463.1 | ✅ |

Byte-identical cost — the sound bounds prune only provably-infeasible options, so
the optimum found is unchanged. (Tests in `crates/brooom/tests/propagation.rs`
assert this property, plus that tightening only ever narrows a window and never
empties a servable one.)

## Up-front infeasibility detection (with reasons)

Before, an unservable job silently landed in `unassigned` after a full search.
Now it's deduced up front and reported. `brooom --verbose` on an instance with a
skill-impossible job and a window-unreachable job:

```
brooom: propagation — job 88 unservable: no vehicle has the required skills / is on the allowlist
brooom: propagation — job 99 unservable: no time window is reachable within any vehicle's shift
```

Detected for: no eligible vehicle (skills/allowlist), demand exceeding every
vehicle's capacity, and a window unreachable within any shift (hard mode). Skipped
under soft mode (where the engine serves late instead of dropping).

## Precedence transitive closure

`precedence: [[1,2],[2,3]]` ⇒ the pass adds `1→3` so the route-walk enforces the
implied order without the user spelling it out; ordering cycles are reported.

## What it does NOT do (honest)

- **It does not crack small propagation-hard cases that the search already
  solves.** `tw_chain` (4 jobs, forced TW order) solves to the optimum (cost
  1047, 0 unassigned) *with or without* propagation — brooom's multi-start ILS
  already finds it. Propagation's value is detection + pruning on larger/tighter
  instances, not rescuing cases the search handles.
- **It is not general constraint propagation.** Arbitrary DSL/code constraints
  remain black boxes; their general propagation stays with the CP-SAT bridge
  (`docs/cpsat-boundary.md`). This pass covers the *structured* temporal /
  precedence / resource constraints — the vast majority of real VRPs.

## Bottom line

brooom now does native, sound, structured constraint propagation — narrowing the
practical part of OR-Tools' CP-SAT propagation edge — while keeping the bridge for
the rare general-logic case. No feasible solution is ever changed; the wins are
up-front infeasibility proofs and probe pruning.
