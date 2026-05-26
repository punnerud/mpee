# dijeng  (a.k.a. dijeng)

> Part of the **[mpee](../../README.md)** workspace. Standalone crate —
> `cargo build` in this directory works on its own. Integrates with
> [`brooom`](../brooom/) (the VRP solver) via the contract in
> [`integration.txt`](integration.txt) and the workspace overview in
> [`INTEGRATION.md`](../../INTEGRATION.md).

An OSRM-competitor routing engine in Rust with Contraction Hierarchies,
bucket-based many-to-many matrix building, granular K-NN neighbours, and
a binary streaming format to feed VRP solvers without paying any
JSON/HTTP overhead.

Designed to be integrated as a path dependency inside a Cargo workspace
alongside a solver (e.g. brooom — a Rust implementation of VROOM) and a
GUI, so routing data flows via direct `&Vec` references in the same
address space instead of IPC or file exchange.

## TL;DR — measured numbers, Apple M3 Pro, Greater London CH (n=1.16M, m_aug=3.9M)

| Operation | Time | Throughput / note |
|---|---:|---:|
| Single p2p `ch::query_with_path_into` | 88-93 µs/call | 11 k calls/s (1 thread) |
| Single p2p in parallel (11 threads, reused scratch) | 17 µs effective | **59 k calls/s** |
| Matrix 5k × 5k dur+dist | 1.13 s | 22 M cells/s |
| Matrix 10k × 10k dur+dist | 4.3 s | 23 M cells/s |
| Matrix 30k × 30k dur+dist | 159 s | 5.7 M cells/s |
| **Matrix 50k × 50k dur+dist (chunked, 500 MB cap)** | **94 s** | **26.6 M cells/s** |
| Matrix 100k × 100k dur+dist | 1098 s | 9.1 M cells/s |
| **K-NN 50k × K=160 (granular)** | **1.22 s** | 41 k srcs/s, 92 MB output |
| FF-ordering 50k coords | 8.0 s | row permutation for streaming |
| CH build (one-off, caches to disk) | 3-4 min | 150 MB cache |
| Cache mmap-load | ~0.02 ms | regardless of size |

Correctness verified against full Dijeng on every benchmark (ε = 1e-3 relative).

## Architecture

```
                        ┌─── data/<file>.osm.pbf ────┐
                        │   (Greater London 119 MB)  │
                        └──────────────┬─────────────┘
                                       │  load_osm_routing_par()
                                       │  (rayon parallel parser)
                                       ▼
                  ┌────────────────────────────────────┐
                  │  CSR cache (SSSPCSR3)              │
                  │  CsrGraph + coords + edge_dist     │
                  │  bench_pp builds this              │
                  └────────────────────┬───────────────┘
                                       │  preprocess() + transpose
                                       ▼
                  ┌────────────────────────────────────┐
                  │  PP cache (SSSPP2C)                │
                  │  BFS-reorder + light/heavy split   │
                  │  forward + reverse edge_dist       │
                  └────────────────────┬───────────────┘
                                       │  ch::build_with_dist()
                                       │  (~3-4 min, 11 threads)
                                       ▼
                  ┌────────────────────────────────────┐
                  │  CH cache (SSSPCH1D)               │
                  │  rank-ordered, dual-channel        │
                  │  duration + distance shortcuts     │
                  └─┬──────────────┬───────────────┬───┘
                    │              │               │
        ch::query   │              │ knn_matrix    │ matrix_with_dist_chunked
        (88 µs)     │              │ (1.22 s/50k)  │ (94 s/50k×50k)
                    ▼              ▼               ▼
              ┌─────────┐    ┌────────────┐  ┌─────────────────┐
              │ Single  │    │ Granular   │  │  Streaming      │
              │ p2p     │    │ K-NN       │  │  binary rows    │
              │ route   │    │ (96 MB)    │  │  (RTBL0001)     │
              └─────────┘    └────────────┘  └────────┬────────┘
                                                      │
                                              ┌───────▼───────┐
                                              │ brooom solver │
                                              │ GUI / app     │
                                              └───────────────┘
```

### Why Contraction Hierarchies

CH (Geisberger et al., 2008) is the industry standard for road-network
routing (OSRM, GraphHopper, RoutingKit). Each vertex is assigned a
`rank`, and the graph is augmented with shortcuts so that a
bidirectional Dijeng that only relaxes "up-edges" finds the shortest
path in microseconds. Our implementation:

* **rank-ordered layout** (`SSSPCH1D`) — the CSR is stored in ascending
  rank order, so hot vertices live first in memory → better cache
  behaviour.
* **dual-channel** — every edge carries both duration (weight for
  optimisation) and distance (passive carrier). Distance matrices end up
  just as fast as duration matrices instead of requiring per-cell path
  unpacking.
* **edge-difference heuristic** for vertex ordering with lazy
  priority-update.

### Why bucket-MMM (Many-to-Many)

Naive approach: call `ch::query(src, dst)` for every (s,t) pair. For
50k×50k = 2.5 billion calls × 88 µs = 60 hours. Unworkable.

Bucket-MMM (Knopp et al.) is O(forward_per_src + backward_per_dst +
bucket_scans). For 50k×50k on London = **94 s in parallel**, ~3000×
faster than the naive p2p loop.

The algorithm:
1. **Forward sweep** — per src, an upward Dijeng in `graph_fwd`,
   accumulating `(s_idx, dur, dist)` in a bucket per settled vertex u.
2. **Backward sweep** — per dst, an upward Dijeng in `graph_bwd`, and
   for every visited u: scan `buckets[u]` and update `out[s_idx, t_idx]`
   with `(d_to_u_fwd + d_to_u_bwd, dist_carried)`.

Parallelisation: forward via rayon `fold` + scatter-merge, backward via
`par_iter` over dsts with disjoint cells in the output matrix.

### Chunked streaming — why and how

For 50k×50k the full output is 25 GB f32. That demands chunking. Our
`matrix_with_dist_chunked(ch, srcs, dsts, K, on_chunk)`:

* Splits srcs into batches of size K.
* Per batch: forward + backward + callback with `K × n_dst` rows.
* Peak RAM per batch: ~K × n_dst × 8 B + ~150 MB working state.
* Cache-friendly: smaller chunk = smaller bucket state = better L3 hit
  rate.

Empirical sweet spot on M3 Pro: chunk=1500 (matches the SLC size). The
**memory budget API** (`plan_for_budget`) selects this automatically
within a byte cap.

### K-NN granular

For VRP solvers that use granular neighbours (Toth-Vigo) — only the K
nearest customers are relevant for local search. `ch::knn_matrix`
returns this **directly in ~1 s for 50k × K=160**, and **eliminates the
need for a full N×N matrix** (10 GB → 96 MB).

The algorithm is plain Dijeng on the uncontracted reordered graph
(`pp.graph`, not the CH-augmented one — shortcuts would skip over
customers), with early termination once K customers are settled.

### Farthest-first row ordering

For streaming to a solver that can work with an incomplete matrix:
`farthest_first_order` returns a permutation whose first two rows are
the approximate diameter pair, and each subsequent row maximises
min-distance to the rows already chosen. This streams the
geometrically diverse rows first, so the solver gets a global skeleton
early.

## Build & run

```bash
# 0. Make sure OSM data exists
mkdir -p data
curl -L -o data/greater-london.osm.pbf \
  https://download.geofabrik.de/europe/united-kingdom/england/greater-london-latest.osm.pbf

# 1. Build CSR + PP cache (~1 min on first run, instant after)
cargo run --release --bin bench_pp -- london car

# 2. Build CH cache (~3-4 min on first run, instant after)
cargo run --release --bin bench_ch -- london car

# 3. Measure performance
cargo run --release --bin bench_matrix    -- london car 10000,30000
cargo run --release --bin bench_knn       -- london 50000 160
cargo run --release --bin bench_latency   -- london 200000

# 4. Start the OSRM-compatible HTTP server (multi-profile)
cargo run --release --bin serve
# /car/route/v1/...  /bicycle/table/v1/...  etc.
```

Other profiles build the same way:

```bash
cargo run --release --bin bench_pp -- london motorcycle
cargo run --release --bin bench_ch -- london motorcycle
```

Cache files then get `.motorcycle.pp` / `.motorcycle.ch` etc. The `car`
profile keeps the unsuffixed names for backward compatibility.

## Benchmarks in detail

### Memory-budget scan (50k × 50k, M3 Pro, 11 threads)

| Budget cap | Plan | Actual peak | Time | Note |
|---:|---|---:|---:|---|
| 200 MB | chunk=112 | 192 MB | 296 s | tight — too-small chunks add batch overhead |
| 500 MB | chunk=576 | 371 MB | **94 s** | **sweet spot** |
| 800 MB | chunk=1039 | 762 MB | 94 s | same perf, more headroom for other apps |
| 1500 MB | chunk=1500 (capped) | 1057 MB | 95 s | saturated at chunk=1500 |

The saturation knee at chunk≈1500 reflects the SLC size on Apple
Silicon (~24-48 MB). On Intel/AMD parts with larger L3 (~70 MB+) the
knee may land at 3000-5000.

### Single-pair latency (`ch::query`)

| Variant | Time/call | Throughput |
|---|---:|---:|
| 1 thread, alloc-per-call (`ch::query`) | 119 µs | 8.4 k calls/s |
| 1 thread, reused scratch | **88 µs** | 11.4 k calls/s |
| 11 threads, alloc-per-call | 70 µs effective | 14 k calls/s (allocator contention) |
| 11 threads, reused scratch (`map_init`) | **17 µs effective** | **59 k calls/s** |
| Path-unpacking, 1 thread, reused scratch | 166 µs (~1100 nodes/path) | 6.0 k calls/s |

Lesson: per-thread `PathScratch::new(n)` is essential. Without it you
lose 4× to allocator contention in parallel workloads.

### Matrix scaling

| Size | Time (random src) | Time (FF ordering) | Output | Peak |
|---:|---:|---:|---:|---:|
| 1k × 1k | 0.27 s | 0.27 s + 0.08 s ff | 8 MB | < 1 GB |
| 5k × 5k | 1.13 s | 1.21 s + 0.5 s ff | 191 MB | < 1 GB |
| 10k × 10k | 4.3 s | 4.3 s + 1.1 s ff | 763 MB | 1.6 GB |
| 20k × 20k | 50 s | 50 s + ~3 s ff | 3.0 GB | 3.9 GB |
| 30k × 30k | 159 s | — | 6.8 GB | 8.2 GB |
| **50k × 50k** | **94 s (chunked)** | **205 s + 8 s ff** | 19 GB (or streamed) | **2.5 GB** |
| 100k × 100k | 1098 s (chunked) | — | 76 GB (streamed) | 4.6 GB |

For 50k+ chunked output is mandatory on a 36 GB machine. On a 64+ GB
Mac Studio in-memory also works, but chunked is still faster thanks to
cache effects.

### K-NN scaling (M3 Pro, London)

For granular VRP solvers: N customers × K=160 nearest, sorted by dur.

| N customers | K | Time | Output | Peak | Throughput |
|---:|---:|---:|---:|---:|---:|
| 50 000 | 160 | **1.22 s** | 92 MB | 220 MB | 41 k srcs/s |

99% of sources reach the full K=160. The remainder are in isolated
components (<160 nodes).

## Comparison with OSRM

Same machine (M3 Pro), London road network:

| Metric | dijeng | OSRM |
|---|---:|---:|
| Preprocessing time | 3-4 min | ~37 s |
| CH cache size | 150 MB | ~140 MB |
| p2p query (internal) | 88 µs | ~30 µs |
| p2p query (over HTTP) | n/a | ~780 µs |
| /table 1000×1000 over HTTP | ~0.4 s (local, internal) | ~0.8 s |
| /table 10k×10k dur+dist | 4.3 s | impractical (M-by-N matrix mode, no chunking) |
| /table 50k×50k dur+dist | 94 s (chunked, 500 MB) | n/a (RAM OOM) |

OSRM is ~3× faster per p2p query, but has no chunked many-to-many for
matrices larger than ~5k×5k without external orchestration. For VRP
fleets of 50k–100k customers, dijeng is the only practical option.

## Files / modules

```
src/
  lib.rs              — module roots
  main.rs             — synthetic SSSP benchmark (older)

  # Foundations
  buffer.rs           — Buffer<T> (owns or mmap-slice)
  graph.rs            — CsrGraph + edge_dist + synthetic graphs
  cache.rs            — CSR cache (SSSPCSR3) + mmap loader
  osm.rs              — PBF parser (rayon-parallel)
  osm_profile.rs      — Profile enum (Car/Motorcycle/Bicycle/Foot)
  preprocess.rs       — BFS reorder + light/heavy split
  cache_pp.rs         — PP cache (SSSPP2C)
  bidir.rs            — bidirectional Dijeng, transpose_with_dist

  # SSSP algorithms (older focus, still usable)
  dijeng.rs         — binary + 4-ary heap
  delta_step.rs       — Δ-stepping
  duan.rs             — Duan-inspired bucket (with caveat — see NB at the bottom)
  auto.rs             — sssp_auto heuristic selector

  # Contraction Hierarchies
  ch.rs               — CH build + query + matrix_with_dist + chunked variant
  cache_ch.rs         — CH cache (SSSPCH1D)
  paged.rs            — PagedMmap (chunked LRU for graphs > RAM)

  # Many-to-many
  knn.rs              — knn_matrix + knn_matrix_flat
  farthest_first.rs   — farthest_first_order
  budget.rs           — MatrixBudget + plan_for_budget

  # I/O and streaming
  binary_table.rs     — RTBL0001 row-streaming + symmetric variant + CRC32
  varint.rs           — LEB128 + zig-zag
  polyline.rs         — Google polyline encoder
  geo_index.rs        — LatLonGrid (snap lat/lon → vertex in ~50 µs)

  # App layer
  routing.rs          — RoutingService (lat/lon facade for apps)
  snap.rs             — older snapping (no longer used)

  bin/
    bench_pp.rs       — PP pipeline bench (also builder)
    bench_ch.rs       — CH build + 1000-query bench
    bench_matrix.rs   — matrix scaling with budget / binary writer
    bench_knn.rs      — K-NN granular bench
    bench_latency.rs  — single-pair latency profile
    bench_london.rs   — SSSP algorithm bench on London (older focus)
    bench_ch_extra.rs — CH on synthetic graphs (RMAT, grid, Rubik, SNAP)
    bench_paged.rs    — PagedMmap demonstration
    bench_petgraph.rs — comparison against petgraph
    bench_osrm.rs     — comparison against an OSRM HTTP server
    parser_bench.rs   — verify parallel PBF parser
    rtbl_inspect.rs   — read an RTBL file and report contents + CRC
    serve.rs          — OSRM-compatible HTTP server (multi-profile)
    intercity.rs      — misc
    ch_test.rs        — misc
    delta_repro.rs    — repro for delta-stepping
    duan_repro.rs     — repro for the duan bug
    bench_extra.rs    — misc
```

## Workspace integration

To use this crate from another Rust binary (e.g. the brooom solver) in
the same workspace:

```toml
# Top-level Cargo.toml
[workspace]
members = ["dijeng", "brooom", "app-gui"]

# brooom/Cargo.toml
[dependencies]
dijeng = { path = "../dijeng" }
rayon = "1"
```

Then brooom can consume K-NN data without a single copy:

```rust
use dijeng::knn::knn_matrix_flat;
use std::sync::Arc;

let granular = Arc::new(knn_matrix_flat(&pp.graph, &customers, 160, Some(&pp.edge_dist)));
// `granular` owns a 96 MB Vec; cloning the Arc gives every brooom thread
// the same RAM block.
```

Apple Silicon UMA bonus: the same `Vec` can be wrapped as an `MTLBuffer`
with `StorageModeShared` for GPU shaders, no copy. See `integration.txt`
section 8 for a concrete recipe.

## Status and further work

Implemented in this session:
- [x] Dual-channel CH (durations + distances)
- [x] Bucket MMM matrix_with_dist + chunked variant (parallel)
- [x] K-NN granular (knn_matrix + flat)
- [x] Farthest-first row ordering
- [x] Memory budget API (plan_for_budget)
- [x] Binary RTBL0001 format (Variant A + Symmetric B + pad64 + CRC32)
- [x] Varint module (LEB128 + zig-zag)
- [x] OSRM-compatible HTTP server (multi-profile)
- [x] Multi-profile preprocessing (car/motorcycle/bicycle/foot)
- [x] Parallel forward + backward MMM sweeps
- [x] Path-unpacking + per-thread PathScratch
- [x] Cache-friendly CSR buckets

Not yet implemented (in priority order for later):
- [ ] Transit Node Routing (TNR) — 1-5 µs/p2p query, ~5-15 min
      preprocessing. Worth it for an interactive app with many p2p calls.
- [ ] u16 dist arrays for thread state — would halve thread state and
      avoid drops under cap on iPhone-class hardware.
- [ ] Parallel CH preprocessing — drop from 3-4 min to <1 min on 11 cores.
- [ ] CCH (Customizable CH) for faster re-weighting (e.g. live traffic).

## Caveats from older code (carried over)

`duan_inspired` (`src/duan.rs`) is **not** a correct implementation of
Duan et al. (STOC 2025); it is a pragmatic simplification with a known
correctness bug on some sparse graphs with high weight variance
(particularly real OSM road networks). `sssp_auto` therefore routes
such graphs to `delta_stepping`, which is proven correct. See the
comment in the older README version and `src/bin/duan_repro.rs` for a
repro.

## License / attribution

OSM data © OpenStreetMap contributors, ODbL.
