# mpe-engine — integrasjons-oversikt

Dette dokumentet binder sammen kontraktene som allerede er beskrevet i hver
crate, og forklarer hvordan **dikstra** (routing) og **brooom** (VRP-solver)
er ment å kjøre som **én prosess med felles minne**.

For dybdedetaljer, se:

- [`crates/dikstra/integration.txt`](crates/dikstra/integration.txt) — den fulle
  routing-API-flaten, tråding-modell, binærformat (`RTBL0001`), Apple-UMA-notater.
- [`crates/brooom/integration.txt`](crates/brooom/integration.txt) — den fulle
  solver-siden: hvor `Granular::from_knn_flat()` plugges inn, hvilke
  fall-back-strategier som finnes for `evaluate_route`.

---

## Hva er hver del?

### dikstra — Contraction-Hierarchies veinetts-router

Tar en OSM PBF, bygger CSR → PP (omsortert) → CH (rangerte snarveier) og
serverer:

- **Single-pair**: `ch::query(src, dst) → Option<f32>` (88 µs/kall).
- **Many-to-many**: `ch::matrix_with_dist(srcs, dsts)` eller streaming
  `matrix_with_dist_chunked` med RAM-tak.
- **K-NN**: `knn::knn_matrix_flat(graph, customers, k, edge_dist)` —
  returnerer `Vec<(u32, f32, f32)>` av lengde `N*K` med `(idx, dur_s, dist_m)`.
- **Snap til vei**: `routing::RoutingService::route(src_lat, src_lon, …)`.

Bygd én gang, cache mmap'es på ~20 mikrosekunder uansett størrelse.

### brooom — Vehicle-Routing-Problem-løser

Tar et Vroom-kompatibelt JSON-problem (jobs + vehicles + tidsvinduer +
kapasiteter) og finner en rute-tildeling som minimerer total kostnad.

Den bruker:

- En **avstandskilde** — i dag enten egen Haversine, OSRM via HTTP, eller en
  forhåndsbygd matrise.
- Et **granulært K-nærmeste-nabolag** for lokalsøk-flyttene (2-opt,
  relocate, exchange, …).

Disse to inngangene er nettopp det dikstra produserer. Resultatet av
integrasjonen er at brooom slipper sin Haversine-fallback og dikstra
slipper det binære filformatet — alt er pekere i samme adresseplass.

---

## Integrasjonskontrakten i én skjerm

```rust
// 1. Last CH-cachen mmap'd én gang per prosess (≈20 µs uansett størrelse)
let pp = sssp_bench::cache_pp::load_mmap("data/greater-london.osm.pbf.pp")?;
let ch = sssp_bench::cache_ch::load_mmap("data/greater-london.osm.pbf.ch")?;

// 2. Konverter VRP-kundenes lat/lon til pp-graf-id-er via snap
let customers: Vec<u32> = problem.jobs.iter()
    .map(|job| snap_lat_lon(&pp, job.lat, job.lon))
    .collect();

// 3. Bygg granulær K=160 K-NN — 1.22 s for 50 000 kunder, 92 MB output
let knn: Vec<(u32, f32, f32)> = sssp_bench::knn::knn_matrix_flat(
    &pp.graph,
    &customers,
    160,
    Some(pp.edge_dist.as_slice()),
);

// 4. ZERO-COPY: brooom konsumerer SAMME Vec direkte
let granular = brooom::granular::Granular::from_knn_flat(
    &knn,
    customers.len(),
    160,
);

// 5. Solve. Hot path er nå pure array-indexing i delt minne.
let solved = brooom::solver::solve_with_matrix(&problem, &granular, &cfg)?;
```

**Ingen filer skrives, ingen sockets åpnes, ingen kopier mellom motorene.**
Hele forflyttingen er en `&Vec<(u32, f32, f32)>`-referanse på tvers av
crate-grensen.

---

## Hvorfor "on the fly" istedenfor full matrise

For N = 50 000 kunder er en full N×N matrise med duration+distance som f32
≈ 20 GB. K-NN med K=160 er ≈ 92 MB. Reduksjonen er ~220× på dette
størrelsesnivået.

Det er to typer oppslag VRP-løseren gjør:

1. **Hot path (10-1000 M/s)**: "Er j blant K nærmeste til i?" Svar via K-NN-array.
2. **Cold path (~10k/s)**: "Hva er avstanden depot→kunde X?" Svar via
   `ch::query` (88 µs) — fortsatt billig fordi det skjer sjelden.

Den fulle matrisen er aldri nødvendig hvis K-NN er stort nok (K ≥ 80) og
depot-raden er prekomputert separat. Det er nøyaktig hva brooom og dikstra
er optimalisert for å gjøre sammen.

---

## Hva som mangler (denne committen)

1. **`mpe-cli`-kommandoene gjør ikke calls ennå** — bare scaffolding. Når
   brooom og sssp_bench legges inn som path-dep i `crates/mpe-cli/Cargo.toml`
   (utkommentert i dag), er resten ~50 linjer rør.
2. **brooom har ikke en `MmmMatrixSource`-impl** som tar en dikstra `&ContractionHierarchy`.
   Se `crates/brooom/integration.txt` §2 Step 2 for den 40-linjers impl-en.
3. **Snap-laget** (lat/lon → pp-node-id) ligger i `dikstra::routing::RoutingService`,
   men brooom forventer at kundene allerede har node-id-er. Trenger en liten
   adapter i `mpe-cli`.

Når disse tre er på plass kjører `mpe pipeline <region> <problem.json>` ende-til-ende
i én Rust-prosess uten å rør disken etter cache-load.

---

## Apple Silicon: unified memory

På M-serie deler CPU og GPU samme fysiske RAM. dikstra kan skrive K-NN-data
direkte inn i en `wgpu::Buffer` med `MAP_WRITE | STORAGE`, og brooom sin
GPU-megakernel leser det uten kopi. Det sparer ~92 MB peak RAM ved N=50k.
Se `crates/dikstra/integration.txt` §8 og `crates/brooom/integration.txt` §5
for koden.

---

## Trådmodell

Begge motorer bruker `rayon` over en delt global thread pool. dikstra anbefaler
**én `PathScratch` per worker** for å unngå allocator-strid (drop fra 51k til
14k spørringer/s ellers). brooom sitt lokalsøk er allerede `Send + Sync` og
trives med samme pool. Mpe-cli skal sette opp pool-en én gang og la begge
operere innenfor `pool.install(|| { … })`.

---

## Referanseimplementasjon

Når denne integrasjonen ferdigstilles havner glue-koden i
`crates/mpe-cli/src/main.rs` (subkommandoene `solve` og `pipeline`). Mønsteret
er allerede skissert som kommentar-block der.
