# mpe-engine

En åpen, samlet Rust-arbeidsbenk for ruting og kjøretøysruteoptimering —
en alternativ stakk til **OSRM** + **VROOM**, der begge motorene kjører
i samme prosess og deler minne direkte.

| Lag                       | Crate                                  | Alternativ til    |
|---------------------------|----------------------------------------|--------------------|
| Veinetts-router (CH)      | [`crates/dikstra/`](crates/dikstra/)   | OSRM               |
| VRP-løser (GPU + CPU LS)  | [`crates/brooom/`](crates/brooom/)     | VROOM              |
| Felles CLI / orkestrering | [`crates/mpe-cli/`](crates/mpe-cli/)   | (ny)               |

Begge motorene **er** selvstendige Rust-prosjekter (egen `Cargo.toml`, egne
binærer, egne tester) og kan kjøres helt frikoblet. mpe-engine pakker dem
som workspace-medlemmer slik at en tredje crate, `mpe-cli`, kan bruke begge
biblioteks-API-ene i samme adresseplass — ingen IPC, ingen fil-overlevering
på hot path.

---

## Bakgrunn

Både dikstra og brooom ble nylig optimalisert for å kjøre beregninger
**on the fly** istedenfor å prekalkulere hele N×N-matrisen:

- **dikstra** har bucket-basert many-to-many MMM som streamer rader,
  granulær K-NN (K=160 → 92 MB istedenfor 20 GB for 50k kunder), og
  88 µs single-pair CH-spørringer.
- **brooom** sin lokal-søk-løkke leser bare K-nærmeste-naboer på hot path
  og faller tilbake til single-pair-spørringer for ruteevaluering — så
  full matrise er aldri nødvendig.

Sammen betyr det at man kan løse et 50 000-kunders VRP på en bærbar uten
å materialisere noen full avstandsmatrise.

---

## Layout

```
mpe-engine/
├── Cargo.toml                      # Workspace-rot
├── README.md                       # Du er her
├── INTEGRATION.md                  # Hvordan crate-ene snakker sammen
├── .gitignore
└── crates/
    ├── brooom/                     # VRP-løser  (Vroom-alternativ)
    │   ├── Cargo.toml              # Standalone — `cargo build` virker her
    │   ├── README.md               # Solver-detaljer + benchmarks
    │   ├── integration.txt         # API-kontrakt mot ekstern routing-motor
    │   └── src/
    ├── dikstra/                    # CH routing-motor  (OSRM-alternativ)
    │   ├── Cargo.toml              # Standalone — `cargo build` virker her
    │   ├── README.md               # Routing-detaljer + benchmarks
    │   ├── integration.txt         # API-kontrakt mot solver
    │   └── src/
    └── mpe-cli/                    # Tynn felles driver
        ├── Cargo.toml              # Path-dep på begge motorer
        └── src/main.rs             # Subkommandoer: download / build / solve / pipeline
```

Hver motor har sin egen `integration.txt` som beskriver akkurat hvilke
typer og funksjoner som er det støttede grensesnittet. [`INTEGRATION.md`](INTEGRATION.md)
på rot-nivå er ovenfra-perspektivet — hvordan de to passer sammen.

---

## Komme i gang

### Krav
- Rust stabil (testet på 1.76+).
- For brooom GPU-pathen: en wgpu-støttet GPU (Metal på Mac, Vulkan på Linux, DX12 på Win).
- For brooom sin nevrale del: `ort` laster ned ~200 MB ONNX-runtime ved første bygg.

### Bygge hele workspace

```bash
cargo build --release --workspace
```

### Bygge bare én del

```bash
cargo build --release -p brooom
cargo build --release -p sssp_bench
cargo build --release -p mpe-cli
```

Eller kjør crate-en helt selvstendig:

```bash
cd crates/brooom && cargo build --release
cd crates/dikstra && cargo build --release
```

### Kjøre

```bash
# CLI-helpen
cargo run --release -p mpe-cli -- --help

# Direkte VRP-solve med brooom (Vroom-kompatibel JSON)
cargo run --release -p brooom -- -i problem.json -o solution.json

# Bygg en CH-cache for London (én gang, ~3-4 min)
cargo run --release -p sssp_bench --bin bench_pp -- london car
cargo run --release -p sssp_bench --bin bench_ch -- london car
```

---

## Hva er status?

- **dikstra**: full produksjonsklar routing — CH bygges, K-NN på 1.2 s for
  50k kunder, OSRM-kompatibel HTTP-server (`bench_osrm`, `serve`), korrekthet
  verifisert mot full Dijkstra. Se [crates/dikstra/README.md](crates/dikstra/README.md).
- **brooom**: GPU-akselerert VRP, vinner mot PyVRP/Vroom/OR-Tools på Solomon
  R1-1000 (p ≈ 3·10⁻⁸), Vroom-kompatibel I/O. Se [crates/brooom/README.md](crates/brooom/README.md).
- **mpe-cli**: scaffolding. Subkommandoene `download`, `build`, `solve`,
  `pipeline` er definert, men `build/solve/pipeline` returnerer en bail-melding
  som ber deg bruke de underliggende binærene direkte. Path-dep-ene mot brooom
  og sssp_bench er forberedt (utkommentert) i `crates/mpe-cli/Cargo.toml` —
  selve sammenkoblingen er kjent (se [INTEGRATION.md](INTEGRATION.md)) og kan
  legges inn uten arkitektur-endringer.

Begge motorer **kommuniserer ikke ennå** — de er forberedt for det. Når
`mpe-cli` aktiveres som planlagt deler de samme `Vec<(u32, f32, f32)>` for
K-NN-tabellen, uten kopi.

---

## Lisens

MIT for brooom og mpe-cli. Se hver crate for detaljer.
