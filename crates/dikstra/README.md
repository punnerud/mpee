# sssp_bench  (a.k.a. dikstra)

> Del av **[mpe-engine](../../README.md)** workspace. Selvstendig crate —
> `cargo build` i denne mappen fungerer alene. Integreres med
> [`brooom`](../brooom/) (VRP-solver) via kontrakten i
> [`integration.txt`](integration.txt) og workspace-oversikten i
> [`INTEGRATION.md`](../../INTEGRATION.md).

OSRM-konkurrent ruting-motor i Rust med Contraction Hierarchies, bucket-basert
many-to-many matrise-bygging, K-NN granular naboer, og et binært
strømmings-format for å mate VRP-solvere uten å betale JSON/HTTP-overhead.

Bygget for å integreres som path-dep i en Cargo workspace sammen med en
solver (eks. brooom — en Rust-implementasjon av VROOM) og en GUI, slik at
ruting-data flytter seg via direkte `&Vec`-referanser i samme adresseplass
istedenfor IPC eller fil-utveksling.

## TL;DR — målte tall, Apple M3 Pro, Greater London CH (n=1.16M, m_aug=3.9M)

| Operasjon | Tid | Throughput / merknad |
|---|---:|---:|
| Single p2p `ch::query_with_path_into` | 88-93 µs/kall | 11 k kall/s (1 tråd) |
| Single p2p parallelt (11 tråder, reused scratch) | 17 µs effektivt | **59 k kall/s** |
| Matrise 5k × 5k dur+dist | 1.13 s | 22 M celler/s |
| Matrise 10k × 10k dur+dist | 4.3 s | 23 M celler/s |
| Matrise 30k × 30k dur+dist | 159 s | 5.7 M celler/s |
| **Matrise 50k × 50k dur+dist (chunked, 500 MB cap)** | **94 s** | **26.6 M celler/s** |
| Matrise 100k × 100k dur+dist | 1098 s | 9.1 M celler/s |
| **K-NN 50k × K=160 (granular)** | **1.22 s** | 41 k srcs/s, 92 MB output |
| FF-ordering 50k coords | 8.0 s | rad-permutasjon for streaming |
| CH-bygg (én gang, caches til disk) | 3-4 min | 150 MB cache |
| Cache mmap-load | ~0.02 ms | uansett størrelse |

Korrekthet verifisert mot full Dijkstra på alle benchmarks (ε = 1e-3 relativ).

## Arkitektur

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
                                       │  (~3-4 min, 11 tråder)
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

### Hvorfor Contraction Hierarchies

CH (Geisberger et al., 2008) er industristandard for veinetts-routing
(OSRM, GraphHopper, RoutingKit). Hver knute tildeles en `rank`, og grafen
augmenteres med snarveier slik at en bidireksjonell Dijkstra som kun
relakserer "oppoverkanter" finner korteste vei i mikrosekund. Vår
implementasjon:

* **rank-ordert layout** (`SSSPCH1D`) — CSR-en lagres med stigende
  rank-rekkefølge, så hot vertikser ligger først i minnet → bedre cache.
* **dual-channel** — hver kant bærer både varighet (vekt for optimering)
  og distanse (passiv bæring). Distanse-matriser blir like raske som
  varighets-matriser i stedet for å kreve sti-unpacking per celle.
* **edge-difference heuristikk** for vertex-ordering med lazy
  priority-update.

### Hvorfor bucket-MMM (Many-to-Many)

Naiv tilnærming: kjør `ch::query(src, dst)` for hvert (s,t)-par. For
50k×50k = 2.5 milliarder kall × 88 µs = 60 timer. Uhåndterlig.

Bucket-MMM (Knopp et al.) er O(forward_per_src + backward_per_dst +
bucket_scans). For 50k×50k på London = **94 s parallelt**, ~3000× raskere
enn naive p2p-løkka.

Algoritmen:
1. **Forward sweep** — per src en upward Dijkstra i `graph_fwd`,
   akkumulere `(s_idx, dur, dist)` i en bucket per settled vertex u.
2. **Backward sweep** — per dst en upward Dijkstra i `graph_bwd`, og for
   hver besøkt u: scan `buckets[u]` og oppdater `out[s_idx, t_idx]` med
   `(d_to_u_fwd + d_to_u_bwd, dist_carried)`.

Parallelisering: forward via rayon `fold` + scatter-merge, backward via
`par_iter` over dsts med disjoint celler i output-matrisen.

### Chunked streaming — hvorfor og hvordan

For 50k×50k tar full output 25 GB f32. Det krever chunking. Vår
`matrix_with_dist_chunked(ch, srcs, dsts, K, on_chunk)`:

* Splitter srcs i batcher av størrelse K.
* Per batch: forward + backward + callback med `K × n_dst` rader.
* Peak-RAM per batch: ~K × n_dst × 8 B + ~150 MB working state.
* Cache-vennlig: lavere chunk = lavere bucket-state = bedre L3-hit-rate.

Empirisk sweet spot på M3 Pro: chunk=1500 (matcher SLC-størrelsen).
**Memory budget API** (`plan_for_budget`) velger den automatisk innenfor
en byte-cap.

### K-NN granular

For VRP-solvere som bruker granular naboer (Toth-Vigo) — kun de K nærmeste
kundene relevant for lokal søking. `ch::knn_matrix` returnerer dette
**direkte i ~1 s for 50k×K=160**, og **erstatter behovet for full N×N**
matrise (10 GB → 96 MB).

Algoritmen er plain Dijkstra på den uncontracted re-ordrede grafen
(`pp.graph`, ikke CH-augmentert — shortcuts ville hoppet over kunder), med
tidlig terminering når K kunder er settlet.

### Farthest-first row ordering

For strømming til en solver som kan jobbe med ufullstendig matrise:
`farthest_first_order` returnerer en permutasjon hvor de første to radene
er det tilnærmede diameter-paret, og hver påfølgende rad maksimerer
min-avstand til allerede valgte. Strømmer dermed geometrisk diverse rader
først; solveren får et globalt skjelett tidlig.

## Build & run

```bash
# 0. Sørg for at OSM-data finnes
mkdir -p data
curl -L -o data/greater-london.osm.pbf \
  https://download.geofabrik.de/europe/united-kingdom/england/greater-london-latest.osm.pbf

# 1. Bygg CSR + PP-cache (~1 min første gang, instant siden)
cargo run --release --bin bench_pp -- london car

# 2. Bygg CH-cache (~3-4 min første gang, instant siden)
cargo run --release --bin bench_ch -- london car

# 3. Måle ytelse
cargo run --release --bin bench_matrix    -- london car 10000,30000
cargo run --release --bin bench_knn       -- london 50000 160
cargo run --release --bin bench_latency   -- london 200000

# 4. Start OSRM-kompatibel HTTP-server (multi-profile)
cargo run --release --bin serve
# /car/route/v1/...  /bicycle/table/v1/...  etc.
```

Andre profiler bygges på samme måte:

```bash
cargo run --release --bin bench_pp -- london motorcycle
cargo run --release --bin bench_ch -- london motorcycle
```

Cache-filer får da `.motorcycle.pp` / `.motorcycle.ch` osv. `car`
beholder unsuffixed for bakoverkompatibilitet.

## Benchmarks i detalj

### Memory-budget scan (50k × 50k, M3 Pro, 11 tråder)

| Budget cap | Plan | Actual peak | Tid | Notat |
|---:|---|---:|---:|---|
| 200 MB | chunk=112 | 192 MB | 296 s | trang — for liten chunk gir batch-overhead |
| 500 MB | chunk=576 | 371 MB | **94 s** | **sweet spot** |
| 800 MB | chunk=1039 | 762 MB | 94 s | samme perf, mer headroom for andre apper |
| 1500 MB | chunk=1500 (capped) | 1057 MB | 95 s | saturert ved chunk=1500 |

Saturasjons-knee på chunk≈1500 reflekterer SLC-størrelsen på Apple Silicon
(~24-48 MB). På Intel/AMD med større L3 (~70 MB+) kan knee være 3000-5000.

### Single-pair latency (`ch::query`)

| Variant | Tid/kall | Throughput |
|---|---:|---:|
| 1 tråd, alloc-per-call (`ch::query`) | 119 µs | 8.4 k kall/s |
| 1 tråd, gjenbrukt scratch | **88 µs** | 11.4 k kall/s |
| 11 tråder, alloc-per-call | 70 µs effektivt | 14 k kall/s (allokator-kontensjon) |
| 11 tråder, gjenbrukt scratch (`map_init`) | **17 µs effektivt** | **59 k kall/s** |
| Path-unpacking, 1 tråd, reused scratch | 166 µs (~1100 noder/path) | 6.0 k kall/s |

Lærdom: per-tråd `PathScratch::new(n)` er essensielt. Uten det taper man
4× på allokator-kontensjon i parallelle workloads.

### Matrise-skalering

| Størrelse | Tid (random src) | Tid (FF ordering) | Output | Peak |
|---:|---:|---:|---:|---:|
| 1k × 1k | 0.27 s | 0.27 s + 0.08 s ff | 8 MB | < 1 GB |
| 5k × 5k | 1.13 s | 1.21 s + 0.5 s ff | 191 MB | < 1 GB |
| 10k × 10k | 4.3 s | 4.3 s + 1.1 s ff | 763 MB | 1.6 GB |
| 20k × 20k | 50 s | 50 s + ~3 s ff | 3.0 GB | 3.9 GB |
| 30k × 30k | 159 s | — | 6.8 GB | 8.2 GB |
| **50k × 50k** | **94 s (chunked)** | **205 s + 8 s ff** | 19 GB (eller stream) | **2.5 GB** |
| 100k × 100k | 1098 s (chunked) | — | 76 GB (stream) | 4.6 GB |

For 50k+ er chunked obligatorisk på 36 GB-maskin. På 64+ GB Mac Studio kan
in-memory også fungere, men chunked er fortsatt raskere pga cache-effekter.

### K-NN-skalering (M3 Pro, London)

For granular VRP-solvere: N kunder × K=160 nærmeste, sortert på dur.

| N kunder | K | Tid | Output | Peak | Throughput |
|---:|---:|---:|---:|---:|---:|
| 50 000 | 160 | **1.22 s** | 92 MB | 220 MB | 41 k srcs/s |

99 % av kildene får full K=160. Resten er i isolerte komponenter
(<160 noder).

## Sammenligning med OSRM

På samme maskin (M3 Pro), London veinett:

| Metrikk | sssp_bench | OSRM |
|---|---:|---:|
| Preprocessing-tid | 3-4 min | ~37 s |
| CH cache-størrelse | 150 MB | ~140 MB |
| p2p query (intern) | 88 µs | ~30 µs |
| p2p query (over HTTP) | n/a | ~780 µs |
| /table 1000×1000 over HTTP | ~0.4 s (lokalt, intern) | ~0.8 s |
| /table 10k×10k dur+dist | 4.3 s | impractical (M-by-N matrix-mode, no chunking) |
| /table 50k×50k dur+dist | 94 s (chunked, 500 MB) | n/a (RAM-OOM) |

OSRM er ~3× raskere per p2p-query, men har ingen chunked many-to-many for
matriser større enn ~5k×5k uten ekstern orchestration. For VRP-flåter på
50k-100k kunder er sssp_bench den eneste praktiske veien.

## Filer / moduler

```
src/
  lib.rs              — module roots
  main.rs             — synthetic SSSP benchmark (eldre)

  # Grunnlag
  buffer.rs           — Buffer<T> (eier eller mmap-slice)
  graph.rs            — CsrGraph + edge_dist + synth-grafer
  cache.rs            — CSR-cache (SSSPCSR3) + mmap loader
  osm.rs              — PBF-parser (rayon-parallell)
  osm_profile.rs      — Profile enum (Car/Motorcycle/Bicycle/Foot)
  preprocess.rs       — BFS reorder + light/heavy split
  cache_pp.rs         — PP-cache (SSSPP2C)
  bidir.rs            — bidirectional Dijkstra, transpose_with_dist

  # SSSP-algoritmer (eldre fokus, fortsatt brukbare)
  dijkstra.rs         — binary + 4-ary heap
  delta_step.rs       — Δ-stepping
  duan.rs             — Duan-inspired bucket (med caveat — se NB nederst)
  auto.rs             — sssp_auto heuristisk velger

  # Contraction Hierarchies
  ch.rs               — CH bygg + query + matrix_with_dist + chunked variant
  cache_ch.rs         — CH-cache (SSSPCH1D)
  paged.rs            — PagedMmap (chunked LRU for grafer > RAM)

  # Many-to-many
  knn.rs              — knn_matrix + knn_matrix_flat
  farthest_first.rs   — farthest_first_order
  budget.rs           — MatrixBudget + plan_for_budget

  # I/O og strømming
  binary_table.rs     — RTBL0001 row-streaming + symmetrisk variant + CRC32
  varint.rs           — LEB128 + zig-zag
  polyline.rs         — Google polyline-encoder
  geo_index.rs        — LatLonGrid (snap lat/lon → vertex i ~50 µs)

  # App-laget
  routing.rs          — RoutingService (lat/lon-fasade for app)
  snap.rs             — eldre snapping (brukes ikke lenger)

  bin/
    bench_pp.rs       — PP-pipeline bench (også builder)
    bench_ch.rs       — CH-bygg + 1000-query bench
    bench_matrix.rs   — matrise-skalering med budget / binary writer
    bench_knn.rs      — K-NN granular bench
    bench_latency.rs  — single-pair latency profil
    bench_london.rs   — SSSP-algoritme-bench på London (eldre fokus)
    bench_ch_extra.rs — CH på syntetiske grafer (RMAT, grid, Rubik, SNAP)
    bench_paged.rs    — PagedMmap demonstrasjon
    bench_petgraph.rs — sammenligning mot petgraph
    bench_osrm.rs     — sammenlign mot OSRM HTTP-server
    parser_bench.rs   — verifiser parallell PBF-parser
    rtbl_inspect.rs   — les en RTBL-fil og rapporter innhold + CRC
    serve.rs          — OSRM-kompatibel HTTP-server (multi-profile)
    intercity.rs      — alt
    ch_test.rs        — alt
    delta_repro.rs    — repro for delta-stepping
    duan_repro.rs     — repro for duan-bug
    bench_extra.rs    — alt
```

## Workspace-integrasjon

For å bruke fra en annen Rust-binary (eks. brooom-solver) i samme workspace:

```toml
# Top-level Cargo.toml
[workspace]
members = ["sssp_bench", "brooom", "app-gui"]

# brooom/Cargo.toml
[dependencies]
sssp_bench = { path = "../sssp_bench" }
rayon = "1"
```

Da kan brooom konsumere K-NN-data uten en eneste kopi:

```rust
use sssp_bench::knn::knn_matrix_flat;
use std::sync::Arc;

let granular = Arc::new(knn_matrix_flat(&pp.graph, &customers, 160, Some(&pp.edge_dist)));
// granular eier 96 MB Vec; Arc-cloning gir alle brooom-tråder samme RAM-blokk
```

Apple Silicon UMA bonus: samme `Vec` kan wraps som `MTLBuffer` med
`StorageModeShared` for GPU-shaders uten kopiering. Se `integration.txt`
seksjon 8 for konkret oppskrift.

## Status og videre arbeid

Implementert i denne sesjonen:
- [x] Dual-channel CH (durations + distances)
- [x] Bucket MMM matrix_with_dist + chunked variant (parallel)
- [x] K-NN granular (knn_matrix + flat)
- [x] Farthest-first row ordering
- [x] Memory budget API (plan_for_budget)
- [x] Binary RTBL0001 format (Variant A + Symmetric B + pad64 + CRC32)
- [x] Varint module (LEB128 + zig-zag)
- [x] OSRM-kompatibel HTTP-server (multi-profile)
- [x] Multi-profile preprocessing (car/motorcycle/bicycle/foot)
- [x] Parallel forward + backward MMM sweeps
- [x] Path-unpacking + per-thread PathScratch
- [x] Cache-vennlig CSR-buckets

Ikke-implementert (i prioritert rekkefølge for senere):
- [ ] Transit Node Routing (TNR) — 1-5 µs/p2p query, ~5-15 min
      preprocessing. Verdi for interaktiv app med mange p2p-kall.
- [ ] u16 dist-arrays for trådstate — kunne halvere thread state og slippe
      drypp under cap på iPhone-klassens hardware.
- [ ] Parallell CH-preprocessing — fra 3-4 min til <1 min på 11 kjerner.
- [ ] CCH (Customizable CH) for raskere re-weighting (eks. live traffic).

## Caveats fra eldre kode (videreført)

`duan_inspired` (`src/duan.rs`) er **ikke** en korrekt implementasjon av
Duan et al. (STOC 2025); den er en praktisk forenkling som har en kjent
korrekthets-bug på enkelte tynne grafer med høy vekt-variasjon (særlig
ekte OSM-veinett). `sssp_auto` ruter derfor slike grafer til
`delta_stepping`, som er bevist korrekt. Se kommentaren i den gamle
README-versjonen og `src/bin/duan_repro.rs` for repro.

## Lisens / attribution

OSM-data © OpenStreetMap contributors, ODbL.
