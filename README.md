# mpe-engine

An open, unified Rust workbench for routing and vehicle-routing
optimisation — an alternative stack to **OSRM** + **VROOM**, where both
engines run in the same process and share memory directly.

| Layer                       | Crate                                  | Alternative to    |
|-----------------------------|----------------------------------------|--------------------|
| Road-network router (CH)    | [`crates/dijkstra/`](crates/dijkstra/) | OSRM               |
| VRP solver (GPU + CPU LS)   | [`crates/brooom/`](crates/brooom/)     | VROOM              |
| Shared CLI / orchestration  | [`crates/mpe-cli/`](crates/mpe-cli/)   | (new)              |

Both engines **are** standalone Rust projects (each with its own
`Cargo.toml`, its own binaries, its own tests) and can be run completely
independently. mpe-engine wraps them as workspace members so a third
crate, `mpe-cli`, can use both library APIs in the same address space —
no IPC, no file hand-off on the hot path.

> Note on the folder name: the routing crate lives in `crates/dijkstra/`
> after Edsger W. Dijkstra. The earlier spelling `dikstra` was a typo.
> The Cargo crate name itself is still `sssp_bench` (the original
> standalone-project name).

---

## Background

Both dijkstra and brooom were recently optimised to compute distances
**on the fly** rather than precomputing the entire N×N matrix:

- **dijkstra** has bucket-based many-to-many MMM that streams rows,
  granular K-NN (K=160 → 92 MB instead of 20 GB for 50k customers), and
  88 µs single-pair CH queries.
- **brooom**'s local-search loop reads only K-nearest neighbours on the
  hot path and falls back to single-pair queries for route evaluation —
  so the full matrix is never required.

Together they make it possible to solve a 50 000-customer VRP on a
laptop without ever materialising a full distance matrix.

---

## Layout

```
mpe-engine/
├── Cargo.toml                      # Workspace root
├── README.md                       # You are here
├── INTEGRATION.md                  # How the crates talk to each other
├── .gitignore
└── crates/
    ├── brooom/                     # VRP solver  (Vroom alternative)
    │   ├── Cargo.toml              # Standalone — `cargo build` works in-place
    │   ├── README.md               # Solver details + benchmarks
    │   ├── integration.txt         # API contract with an external router
    │   └── src/
    ├── dijkstra/                   # CH routing engine (OSRM alternative)
    │   ├── Cargo.toml              # Standalone — `cargo build` works in-place
    │   ├── README.md               # Routing details + benchmarks
    │   ├── integration.txt         # API contract with the solver
    │   └── src/
    └── mpe-cli/                    # Thin shared driver
        ├── Cargo.toml              # Path-deps on both engines
        └── src/main.rs             # Subcommands: download / build / solve / pipeline
```

Each engine has its own `integration.txt` describing exactly which types
and functions form its supported interface. [`INTEGRATION.md`](INTEGRATION.md)
at the workspace root is the bird's-eye view — how the two fit together.

---

## Getting started

### Requirements
- Stable Rust (tested with 1.76+).
- For the brooom GPU path: a wgpu-supported GPU (Metal on Mac, Vulkan on Linux, DX12 on Windows).
- For brooom's neural module: `ort` downloads ~200 MB of ONNX Runtime on first build.

### Build the whole workspace

```bash
cargo build --release --workspace
```

### Build a single crate

```bash
cargo build --release -p brooom
cargo build --release -p sssp_bench
cargo build --release -p mpe-cli
```

Or build a crate completely standalone:

```bash
cd crates/brooom && cargo build --release
cd crates/dijkstra && cargo build --release
```

### Run

```bash
# CLI help
cargo run --release -p mpe-cli -- --help

# Direct VRP solve with brooom (Vroom-compatible JSON)
cargo run --release -p brooom -- -i problem.json -o solution.json

# Build a CH cache for London (one-off, ~3-4 min)
cargo run --release -p sssp_bench --bin bench_pp -- london car
cargo run --release -p sssp_bench --bin bench_ch -- london car
```

---

## Status

- **dijkstra**: production-ready routing — CH build, K-NN in 1.2 s for
  50k customers, OSRM-compatible HTTP server (`bench_osrm`, `serve`),
  correctness verified against full Dijkstra. See
  [crates/dijkstra/README.md](crates/dijkstra/README.md).
- **brooom**: GPU-accelerated VRP, beats PyVRP / Vroom / OR-Tools on
  Solomon R1-1000 (p ≈ 3·10⁻⁸), Vroom-compatible I/O. See
  [crates/brooom/README.md](crates/brooom/README.md).

- **mpe-cli**: scaffolding. The `download`, `build`, `solve`, `pipeline`
  subcommands are defined but `build/solve/pipeline` currently bail out
  with a message asking you to invoke the underlying binaries directly.
  The path dependencies on brooom and sssp_bench are prepared (commented
  out) in `crates/mpe-cli/Cargo.toml` — wiring is documented in
  [INTEGRATION.md](INTEGRATION.md) and can be added without further
  restructuring.

The two engines **do not yet communicate** — they are prepared for it.
Once `mpe-cli` is activated as planned, they will share the same
`Vec<(u32, f32, f32)>` for the K-NN table with no copy.

---

## License

MIT for brooom and mpe-cli. See each crate for details.
