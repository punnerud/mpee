# Live layer — design

**Status: design (no code yet).** This is the plan for giving MPEE live
travel times without a traffic-data subscription: the fleet itself, plus a
handful of cheap external probes, becomes the sensor network. It builds
directly on two things that already exist: the matrix broker's temporal
profiles (Stage E1) and the 4-second CH rebuild.

## Why this can work now

The classic objection to live routing in a CH engine is that the hierarchy
is static — weights are baked in at preprocessing. The standard answers
(CCH, time-dependent CH) are heavy machinery. MPEE's edge-difference build
made the whole question easier: **a metro-area hierarchy rebuilds in ~4
seconds.** "Live" becomes: maintain a per-edge delay layer, fold it into the
weights, rebuild every 1–5 minutes. No new query algorithm, no second code
path — live routing *is* routing.

## Signal sources

1. **Own fleet (floating car data).** Every vehicle in operation streams
   position pings. Map-matched (the `/match/v1` HMM service) they become
   observed traversal times over known edge sequences — the highest-quality
   signal, exactly on the roads the fleet actually uses, per direction.
2. **External ETA APIs as sensors.** A routing API that returns live ETAs
   (Google/TomTom/HERE) is an oracle you can *query for routes you never
   intend to drive*. Each response is one linear measurement over the edges
   of that route. This is the matrix broker's buy-few-derive-rest philosophy
   applied to time: buy a handful of well-chosen route ETAs, derive the
   per-segment delays.

## The inference problem: network tomography

One observation (a matched trace or a purchased ETA) gives a **route total**,
not per-edge values. Stack the observations:

```
y = A·x + ε
```

* `y` — observed travel times (one per probe/trace), minus free-flow time,
* `A` — route–edge incidence matrix (which edges, per observation),
* `x` — unknown per-edge(-direction) delay in the current time bucket,
* `ε` — noise (GPS, service stops, signal timing).

This is network tomography, and it is underdetermined by construction —
which is precisely the user-visible intuition that "some observations don't
say exactly *where* on the stretch the delay is". The standard tools make it
solvable:

* **Time bucketing** — solve per 5–15 min bucket; delays are
  piecewise-stable.
* **Priors** — broker Stage E1's learned time-of-day profiles are the prior
  mean; the solver only estimates *deviations* from the expected pattern.
* **Regularisation** — delays are sparse (most edges are free-flowing) and
  spatially correlated (congestion is contiguous): LASSO for sparsity plus a
  graph-Laplacian smoothness term. Both keep the system well-posed with few
  observations.
* **Direction-awareness** — every edge has two unknowns; inbound and
  outbound congestion are independent (and that asymmetry is the valuable
  part).

### Triangulation via overlapping routes

Two probes that share a sub-path and diverge isolate the difference to the
non-shared edges — the "triangulation" effect. With k overlapping routes
through an area, the solvable resolution grows roughly with the number of
distinct edge subsets their pairwise differences induce.

### Choosing what to probe: optimal experiment design

Probes cost money (API calls) or detour time (if a vehicle is re-routed to
sense). So choose them to maximise information: given the current posterior
over `x`, pick the next route whose measurement maximally shrinks the
uncertainty — **D-optimal design** (maximise the determinant gain of the
information matrix A'A under the prior). Practical loop:

1. Identify the edges whose delay uncertainty × usage-by-planned-routes is
   highest (uncertainty only matters where the fleet will drive).
2. Generate candidate probe routes through them (alternative-routes service
   gives diverse candidates for free).
3. Greedily pick probes by marginal information per unit cost until the
   budget is spent.

This is the broker's cost/policy machinery (Stage C) with a different value
function.

## The delay store: smart cross-route cache

Key: `(edge_id, direction, time_bucket)` → `(delay_estimate, variance,
last_update)`. Properties:

* **Cross-route reuse** — a delay learned from one probe applies to every
  route over that edge; the cache is the medium through which one paid
  observation serves thousands of plans.
* **TTL decay** — variance grows back toward the prior as estimates age;
  an edge unseen for an hour reverts to the Stage E1 profile.
* **Streaming out** — an SSE/WebSocket endpoint (`/live/v1/stream`) pushes
  bucket updates so dashboards and re-planners react without polling; the
  same feed is the audit log of what the system believed when.

## Applying the layer

* **Routing**: every 1–5 minutes, `edge_w' = edge_w × (1 + delay_factor)`
  over the affected edges, rebuild the CH (~4 s), atomically swap the mmap.
  Queries in flight finish on the old hierarchy; new queries see live truth.
* **Solving (VRP)**: the matrix broker already serves the solver — the live
  layer is one more correction applied to matrix cells whose paths cross
  delayed edges (path-aware correction via the cell's stored route, or
  cheap re-query of affected cells against the rebuilt CH — both fit the
  broker's derive-don't-buy pipeline).
* **Re-plan triggers**: when a delay estimate moves a planned route's ETA
  beyond its time-window slack, emit an event; brooom's warm-start mode
  (`--warm-start`) makes the re-solve incremental.

## Honest limits

* This measures where probes go: coverage is the fleet's operating area plus
  what the probe budget buys. It is a *fleet-local* live layer, not a
  city-wide traffic product.
* Tomography resolves to the granularity the overlaps allow; isolated
  never-probed residential edges stay at their prior.
* External-API probing must respect the provider's ToS; the design treats
  providers as interchangeable broker backends so any compliant source
  plugs in.

## Staged implementation

1. **L0 — plumbing**: delay store + `/live/v1/report` (manual/fleet ping
   ingest via map matching) + periodic rebuild-and-swap. No inference; raw
   per-edge observations only. *Everything needed already exists as parts.*
2. **L1 — inference**: time buckets, priors from Stage E1, regularised
   least-squares solve per bucket (the matrices are small: only edges with
   any observation enter).
3. **L2 — active probing**: external ETA providers as broker Stage F,
   D-optimal probe selection under budget.
4. **L3 — solver integration**: live-corrected matrix cells + re-plan
   triggers + the streaming endpoint.

Each stage is independently shippable and measurable (L0: rebuild latency +
swap correctness; L1: holdout-probe prediction error; L2: information gain
per krone; L3: realised-vs-planned ETA error on live operations).
