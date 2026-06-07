# The cost-aware matrix broker — *the whole world, offline; pay only when you must*

> **The pitch in one line.** MPEE already routes and solves a downloaded area
> **fully offline**. The matrix broker extends that promise to the cases where you
> *do* reach for a paid/limited API (Google Distance Matrix, a metered OSRM, an
> internal endpoint): it buys **only the handful of cells the solver actually
> reads**, derives the rest, remembers everything in a local DB, and — with
> temporal profiles — learns one representative day of live traffic and **replays
> it offline for every similar day**. Offline by default; a few paid cells only
> when reality demands it; never the same cell twice.

A vehicle-routing problem over *N* stops implies an *N × N* travel-time matrix.
If those numbers come from a per-element-billed API, the naïve cost is **N²
calls** — 160,000 for a 400-stop run, most of them long-range cells the solver
never even consults. That is the bill the broker deletes.

---

## Why it saves money (the four levers)

1. **Buy only the skeleton.** The local search only ever reads each stop's
   *K-nearest* neighbours plus the depot and a few landmark rows — a thin
   *skeleton* of *N × N*. The broker ranks candidates with a **free Haversine
   prior**, buys that skeleton **exactly** (zero quality loss on what the solver
   touches), and **derives** every long-range cell with a landmark min-plus
   bridge `base(i,j) = minₗ d(i,l)+d(l,j)` — an upper bound that's directional
   (asymmetric-safe) and gated by a triangle-inequality sanity check. On a
   synthetic 400-stop world this buys **< 50 %** of N² with no measurable
   solution-quality change.

2. **Never buy the same cell twice.** A persistent cell DB keyed by *quantised
   coordinates* (not matrix index) means a distance learned in one solve is reused
   by **any** later solve over the same place. A warm DB makes the second run buy
   **zero** cells. A per-node frequency counter lets you prune rarely-touched
   one-off addresses (buy hubs precisely, derive the long tail).

3. **Price the buy in your own DSL.** A PySpell expression over `broker.*`
   (`n`, `batch_size`, `tier`, `budget_remaining`, `cells_known`,
   `crossing_count`, `haversine_km`, `departure_hour`, `weekday_class`) sets the
   cost/buy policy. Cap the spend with `--buy-budget` and the broker keeps the
   largest affordable *K-nearest-first* prefix (the search-critical cells survive)
   and demotes the rest to derivation.

4. **Learn the day once, replay it offline (temporal profiles).** Travel time
   depends on *when* you leave. The broker can key the DB by a
   `(weekday-class, hour)` window and store **running statistics** per cell.
   Fetch **one representative workday's** hourly cells, and because the key is a
   weekday *class* (Mon–Fri merged) rather than a date, that profile answers
   **every** weekday at that hour **with zero new calls**. Buy a little live
   congestion once; reuse it forever, offline.

> **Provider-agnostic.** The broker wraps *any* cell source — the offline
> Haversine matrix, an OSRM client, Google Distance Matrix, or your own endpoint
> — through one per-origin batching path. Google is an example, not a dependency.
> Even better: pair a **compressed offline graph** (dijeng CH or a matcodec
> `.mtz`) as the free base with a **thin paid "delay" overlay** on just the
> congested corridors, and let the broker fill everything else from the offline
> base.

---

## Congestion & uncertainty — route *around* the queues

Each temporal cell carries a Welford **mean and standard deviation** over its
observations. The mean is the typical (e.g. rush-hour) travel time; the std-dev
is the **uncertainty** — how much that arc deviates from "normal", a queue/
incident signal. With `--uncertainty-weight W`, the static matrix cell becomes:

```
cell = mean + W · std
```

So arcs that are reliably fast stay cheap, while flaky, queue-prone arcs cost
more — and the solver, **unchanged**, routes around them and secures the
dependable corridors. The broker reports a `hotspots` count (cells whose std-dev
is a large fraction of their mean) so you can see the congestion zones it found.

---

## Use it

```bash
# Buy only what the solver reads; derive the rest. Works with ANY --routing.
brooom -i fleet.json --routing google --google-key "$KEY" --broker

# Reuse across runs with a local DB; cap the spend; price it with a spell.
brooom -i fleet.json --routing google --google-key "$KEY" --broker \
       --matrix-db ./fleet.cells --buy-budget 5000 \
       --cost-policy 'if broker.tier == 0 { broker.batch_size * 5 } else { broker.batch_size * 4 }'

# Temporal: build a workday-08:00 profile, penalise uncertain arcs, reuse offline.
brooom -i fleet.json --routing google --google-key "$KEY" --broker \
       --matrix-db ./fleet.cells --departure workday:08 \
       --uncertainty-weight 1.0 --offline-reuse
```

A departure-aware cost spell that only pays for **live** data in the morning rush
and otherwise leans on the offline profile:

```rust
if broker.weekday_class == 0 && broker.departure_hour >= 7 && broker.departure_hour <= 9 {
    broker.batch_size * 5   // rush hour on a workday: worth fresh numbers
} else {
    broker.batch_size       // off-peak / weekend: cheap, reuse the profile
}
```

### Flags

| Flag | Effect |
|---|---|
| `--broker` | Wrap the chosen provider; buy the skeleton, derive the rest. |
| `--matrix-db <path>` | Persistent cell DB: reuse across runs + frequency counter. |
| `--buy-budget <N>` | Hard spend cap (priced by `--cost-policy`); excess is derived. |
| `--freq-threshold <T>` | Skip buying long-range cells to nodes seen < T times. |
| `--cost-policy <spell>` | PySpell cost/buy policy over `broker.*`. |
| `--departure <class:hour>` | Temporal profile window, e.g. `workday:08`, `weekend:17`. |
| `--uncertainty-weight <W>` | Bake `mean + W·std` so queue-prone arcs cost more. |
| `--offline-reuse` | Serve the chosen window from the DB; buy nothing when warm. |

---

## Honest scope

- The temporal profile produces **one static matrix per departure window**. It
  does *not* model travel time changing *during* a route (that's TDVRP, **E2** —
  deliberately deferred, because true time-dependent arc costs would disable the
  O(1) cost-delta local search that makes brooom match PyVRP). E1 covers the bulk
  of real use: short-to-medium runs within a window, with congestion baked in.
- Uncertainty quality grows with samples per `(cell, weekday-class, hour)`. A
  cold profile has std ≈ 0, so the penalty vanishes and the broker degrades
  gracefully to the plain skeleton behaviour.
- Min-plus derivation is an **upper bound** on the true distance — conservative on
  metric data; the metric gate falls back to the Haversine fill when the bought
  cells look non-metric, so a derived cell never silently under-estimates.

## How it's verified

`cargo test -p brooom --test broker_cost_quality` (9 tests): skeleton buys few /
reproduces bought cells exactly / derives sound upper bounds; warm-DB second run
buys zero; budget cap bites; Welford mean+std and temporal bucketing round-trip
through a flush/reopen; a workday profile learned one day serves another day for
free on the 400-stop derive path; a rush profile bakes a strictly larger matrix
than off-peak; and the uncertainty weight penalises high-variance cells and flags
them as hotspots. The broker is inert unless `--broker` is passed — absent it,
routing is byte-for-byte today's behaviour.
