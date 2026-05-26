# mpee — integration overview

This document ties together the contracts already described in each
crate and explains how **dijeng** (routing) and **brooom** (VRP solver)
are designed to run as **one process with shared memory**.

For the full detail, see:

- [`crates/dijeng/integration.txt`](crates/dijeng/integration.txt) — the
  full routing API surface, threading model, binary format (`RTBL0001`),
  Apple-UMA notes.
- [`crates/brooom/integration.txt`](crates/brooom/integration.txt) — the
  full solver side: where `Granular::from_knn_flat()` plugs in and what
  fallback strategies exist for `evaluate_route`.

---

## What each part does

### dijeng — Contraction-Hierarchies road-network router

Takes an OSM PBF, builds CSR → PP (reordered) → CH (ranked shortcuts)
and serves:

- **Single-pair**: `ch::query(src, dst) → Option<f32>` (88 µs/call).
- **Many-to-many**: `ch::matrix_with_dist(srcs, dsts)` or streaming
  `matrix_with_dist_chunked` with a hard RAM cap.
- **K-NN**: `knn::knn_matrix_flat(graph, customers, k, edge_dist)` —
  returns `Vec<(u32, f32, f32)>` of length `N*K` with `(idx, dur_s, dist_m)`.
- **Snap to road**: `routing::RoutingService::route(src_lat, src_lon, …)`.

Built once; the cache mmaps in ~20 µs regardless of size.

### brooom — Vehicle-Routing-Problem solver

Takes a Vroom-compatible JSON problem (jobs + vehicles + time windows +
capacities) and finds a route assignment that minimises total cost.

It consumes:

- A **distance source** — today either its own Haversine, OSRM over
  HTTP, or a precomputed matrix.
- A **granular K-nearest-neighbour graph** for the local-search moves
  (2-opt, relocate, exchange, swap-star, Or-opt, …).

Those are precisely the two things dijeng produces. The end result of
integration is that brooom drops its Haversine fallback and dijeng
drops the binary file format — everything is pointers in the same
address space.

---

## The integration contract in one screen

```rust
// 1. Load the CH cache mmap'd once per process (≈20 µs regardless of size)
let pp = dijeng::cache_pp::load_mmap("data/greater-london.osm.pbf.pp")?;
let ch = dijeng::cache_ch::load_mmap("data/greater-london.osm.pbf.ch")?;

// 2. Convert VRP customer lat/lon into pp graph IDs via snap
let customers: Vec<u32> = problem.jobs.iter()
    .map(|job| snap_lat_lon(&pp, job.lat, job.lon))
    .collect();

// 3. Build the granular K=160 K-NN — 1.22 s for 50 000 customers, 92 MB output
let knn: Vec<(u32, f32, f32)> = dijeng::knn::knn_matrix_flat(
    &pp.graph,
    &customers,
    160,
    Some(pp.edge_dist.as_slice()),
);

// 4. ZERO-COPY: brooom consumes the SAME Vec directly
let granular = brooom::granular::Granular::from_knn_flat(
    &knn,
    customers.len(),
    160,
);

// 5. Solve. The hot path is now pure array indexing in shared memory.
let solved = brooom::solver::solve_with_matrix(&problem, &granular, &cfg)?;
```

**No files are written, no sockets opened, no copies between the
engines.** The whole transfer is a single `&Vec<(u32, f32, f32)>`
reference across the crate boundary.

---

## Why "on the fly" instead of a full matrix

For N = 50 000 customers, a full N×N matrix with duration + distance as
f32 is ≈ 20 GB. K-NN with K=160 is ≈ 92 MB. That is a ~220× reduction
at this scale.

The VRP solver performs two kinds of distance look-ups:

1. **Hot path (10–1000 M/s)**: "is j among the K-nearest of i?" Answered
   from the K-NN array.
2. **Cold path (~10 k/s)**: "what is the distance depot → customer X?"
   Answered via `ch::query` (88 µs) — still cheap because it happens
   rarely.

The full matrix is never needed if K-NN is large enough (K ≥ 80) and
the depot row is precomputed separately. This is exactly what brooom
and dijeng were optimised to do together.

---

## Status: shipped and running

This integration is **done** — it runs end-to-end today, it is not scaffolding:

- `mpee-cli` has live path-deps on `brooom` + `dijeng`. Its `solve` /
  `pipeline` verbs load a CH cache, snap coords, build the N×N
  duration+distance matrix via dijeng's bucket-MMM, and hand it to
  `brooom::solver::solve_with_matrix` — one process, no IPC, no disk on the
  hot path.
- The same pipeline is exposed to Python via `mpee.Router` (in
  `crates/mpee-py`) and shipped on PyPI as **`mpee`** (`mpee route` /
  `optimize` / `solve`, plus `download` / `build`).
- The snap layer (`dijeng::routing::RoutingService` /
  `matrix_with_distance`) and the in-process cache build
  (`dijeng::build::build_cache`) are wired in.

**Note on the contract above:** the shipped path builds a *precomputed CH
matrix* and calls `solve_with_matrix(&problem, &matrix, &cfg)` (ideal up to a
few thousand stops). brooom builds its granular K-NN neighbourhood from that
matrix inside the solver; the zero-copy granular hand-off sketched above is the
design target for extreme N (50k+).

---

## Apple Silicon: unified memory

On the M-series the CPU and GPU share the same physical RAM. dijeng
can write K-NN data directly into a `wgpu::Buffer` mapped with
`MAP_WRITE | STORAGE`, and brooom's GPU megakernel reads it without a
copy. That saves ~92 MB of peak RAM at N=50k. See
`crates/dijeng/integration.txt` §8 and `crates/brooom/integration.txt`
§5 for the code.

---

## Threading model

Both engines use `rayon` over a shared global thread pool. dijeng
recommends **one `PathScratch` per worker** to avoid allocator
contention (drops from 51k to 14k queries/s otherwise). brooom's
local-search is already `Send + Sync` and is happy with the same pool.
mpee-cli should configure the pool once and let both operate inside
`pool.install(|| { … })`.

---

## Reference implementation

The glue code lives in `crates/mpee-cli/src/main.rs` (the `solve` / `pipeline`
verbs) and in `crates/mpee-py/src/lib.rs` (the `Router` class). Both drive the
same `dijeng` + `brooom` library calls in one process.
