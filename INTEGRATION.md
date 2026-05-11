# mpe-engine — integration overview

This document ties together the contracts already described in each
crate and explains how **dijkstra** (routing) and **brooom** (VRP solver)
are designed to run as **one process with shared memory**.

For the full detail, see:

- [`crates/dijkstra/integration.txt`](crates/dijkstra/integration.txt) — the
  full routing API surface, threading model, binary format (`RTBL0001`),
  Apple-UMA notes.
- [`crates/brooom/integration.txt`](crates/brooom/integration.txt) — the
  full solver side: where `Granular::from_knn_flat()` plugs in and what
  fallback strategies exist for `evaluate_route`.

---

## What each part does

### dijkstra — Contraction-Hierarchies road-network router

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

Those are precisely the two things dijkstra produces. The end result of
integration is that brooom drops its Haversine fallback and dijkstra
drops the binary file format — everything is pointers in the same
address space.

---

## The integration contract in one screen

```rust
// 1. Load the CH cache mmap'd once per process (≈20 µs regardless of size)
let pp = sssp_bench::cache_pp::load_mmap("data/greater-london.osm.pbf.pp")?;
let ch = sssp_bench::cache_ch::load_mmap("data/greater-london.osm.pbf.ch")?;

// 2. Convert VRP customer lat/lon into pp graph IDs via snap
let customers: Vec<u32> = problem.jobs.iter()
    .map(|job| snap_lat_lon(&pp, job.lat, job.lon))
    .collect();

// 3. Build the granular K=160 K-NN — 1.22 s for 50 000 customers, 92 MB output
let knn: Vec<(u32, f32, f32)> = sssp_bench::knn::knn_matrix_flat(
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
and dijkstra were optimised to do together.

---

## What is missing (this commit)

1. **The `mpe-cli` subcommands don't make calls yet** — scaffolding only.
   Once brooom and sssp_bench are added as path dependencies in
   `crates/mpe-cli/Cargo.toml` (commented out today), the rest is roughly
   50 lines of plumbing.
2. **brooom does not have an `MmmMatrixSource` impl** that consumes a
   dijkstra `&ContractionHierarchy`. See
   `crates/brooom/integration.txt` §2 Step 2 for the ~40-line impl.
3. **The snap layer** (lat/lon → pp node ID) lives in
   `dijkstra::routing::RoutingService`, but brooom expects customers to
   already carry node IDs. A small adapter belongs in `mpe-cli`.

Once those three are in place, `mpe pipeline <region> <problem.json>`
runs end-to-end in a single Rust process without touching disk after
the cache load.

---

## Apple Silicon: unified memory

On the M-series the CPU and GPU share the same physical RAM. dijkstra
can write K-NN data directly into a `wgpu::Buffer` mapped with
`MAP_WRITE | STORAGE`, and brooom's GPU megakernel reads it without a
copy. That saves ~92 MB of peak RAM at N=50k. See
`crates/dijkstra/integration.txt` §8 and `crates/brooom/integration.txt`
§5 for the code.

---

## Threading model

Both engines use `rayon` over a shared global thread pool. dijkstra
recommends **one `PathScratch` per worker** to avoid allocator
contention (drops from 51k to 14k queries/s otherwise). brooom's
local-search is already `Send + Sync` and is happy with the same pool.
mpe-cli should configure the pool once and let both operate inside
`pool.install(|| { … })`.

---

## Reference implementation

When this integration is finished, the glue code lives in
`crates/mpe-cli/src/main.rs` (the `solve` and `pipeline` subcommands).
The pattern is already sketched as comments there.
