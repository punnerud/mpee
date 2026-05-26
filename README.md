# MPEE — Offline route calculations and optimization

An open, unified Rust engine for routing and vehicle-routing
optimisation — an alternative stack to **OSRM** + **VROOM**, where both
engines run in the same process and share memory directly. Everything runs
**offline** from a downloaded map: no API keys, no per-request billing, no
data leaving your machine. It uses CPU **and** GPU, less memory than the
alternatives, and the whole engine fits under ~50 MB.

In head-to-head tests on a Mac, MPEE produced **shorter routes than
[VROOM](https://github.com/VROOM-Project) at equal runtime**.

## Install (Python / CLI)

The fastest way to use the engine is the `mpee` Python package — a thin CLI
and library over the same Rust core:

```bash
pip install mpee

# 1. Get a map once (OpenStreetMap extract → routable cache):
mpee download europe/great-britain/england/greater-london
mpee build data/greater-london-latest.osm.pbf

# 2. Route from A to B (offline):
mpee route 51.5080,-0.1281 51.5138,-0.0984 --cache data/greater-london-latest.osm.pbf
#   → distance: 2.38 km   duration: 4.4 min

# 3. Optimize a multi-vehicle delivery run over your own stops:
mpee optimize --stops stops.txt --vehicles 5 --capacity 20 \
    --cache data/greater-london-latest.osm.pbf
#   → 50 stops, 3 vehicles used, 60.0 km total (solved in 4.6s)
```

From Python:

```python
import mpee
r = mpee.Router("data/greater-london-latest.osm.pbf.pp", "data/greater-london-latest.osm.pbf.ch")
leg = r.route(51.5080, -0.1281, 51.5138, -0.0984)     # {distance_km, duration_min, ...}
plan = r.optimize(stops, vehicles=5, capacity=20)      # multi-vehicle VRP
```

See [`crates/mpee-py/`](crates/mpee-py/) for the full Python API, and
[pypi.org/project/mpee](https://pypi.org/project/mpee/).

> **Heads-up: there are two CLIs, both named `mpee`.** The commands above are
> the **pip package** (`pip install mpee` → verbs `route` / `optimize` /
> `download` / `build`). The Rust workspace binary
> (`cargo run -p mpee-cli`, i.e. `./target/release/mpee-cli`) is a *different* tool
> with VRP-focused verbs (`gen` / `download` / `build` / `solve` / `pipeline`)
> — it has **no** point-to-point `route`. Use the pip package for the examples
> on this page.

## The Rust workbench

| Layer                       | Crate                                    | Alternative to    |
|-----------------------------|------------------------------------------|--------------------|
| Road-network router (CH)    | [`crates/dijeng/`](crates/dijeng/)   | OSRM               |
| VRP solver (GPU + CPU LS)   | [`crates/brooom/`](crates/brooom/)       | VROOM              |
| Shared CLI / orchestration  | [`crates/mpee-cli/`](crates/mpee-cli/)     | (new)              |
| Live HTTP + map UI          | [`crates/mpee-viz/`](crates/mpee-viz/) | (new)              |

Both engines **are** standalone Rust projects (each with its own
`Cargo.toml`, its own binaries, its own tests) and can be run completely
independently. mpee wraps them as workspace members so a third
crate, `mpee-cli`, can use both library APIs in the same address space —
no IPC, no file hand-off on the hot path.

> Note on the folder name: the routing crate lives in `crates/dijeng/`
> after Edsger W. Dijkstra. The earlier spelling `dikstra` was a typo.
> The Cargo crate name itself is still `dijeng` (the original
> standalone-project name).

---

## Background

Both dijeng and brooom were recently optimised to compute distances
**on the fly** rather than precomputing the entire N×N matrix:

- **dijeng** has bucket-based many-to-many MMM that streams rows,
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
mpee/
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
    ├── dijeng/                   # CH routing engine (OSRM alternative)
    │   ├── Cargo.toml              # Standalone — `cargo build` works in-place
    │   ├── README.md               # Routing details + benchmarks
    │   ├── integration.txt         # API contract with the solver
    │   └── src/
    ├── mpee-cli/                    # Rust CLI binary `mpee` (VRP driver)
    │   ├── Cargo.toml              # Path-deps on both engines
    │   └── src/main.rs             # Verbs: gen / download / build / solve / pipeline
    ├── mpee-py/                     # PyO3 bindings + the `pip install mpee` CLI
    │   └── ...                      # Verbs: route / optimize / download / build
    └── mpee-viz/                    # Live VRP-on-a-map demo server (Leaflet UI)
```

Each engine has its own `integration.txt` describing exactly which types
and functions form its supported interface. [`INTEGRATION.md`](INTEGRATION.md)
at the workspace root is the bird's-eye view — how the two fit together.

---

## Getting started

### Requirements
- Stable Rust (tested with 1.76+).
- For the brooom GPU path: a wgpu-supported GPU (Metal on Mac, Vulkan on Linux, DX12 on Windows).

**ONNX is not pulled by default.** brooom's experimental `neural` module (and
its ~200 MB `ort` runtime) is opt-in — only `--features neural` / `--features
full` builds it. A plain `cargo build --workspace`, `-p mpee-cli`, or
`pip install mpee` downloads no ONNX. Routing/VRP need none of it.

### Build the whole workspace

```bash
cargo build --release --workspace
```

### Build a single crate

```bash
cargo build --release -p brooom
cargo build --release -p dijeng
cargo build --release -p mpee-cli
```

Or build a crate completely standalone:

```bash
cd crates/brooom && cargo build --release
cd crates/dijeng && cargo build --release
```

### Run

```bash
# CLI help
cargo run --release -p mpee-cli -- --help

# Direct VRP solve with brooom (Vroom-compatible JSON)
cargo run --release -p brooom -- -i problem.json -o solution.json

# Build a routable CH cache for ANY region (in-process, ~seconds–minutes):
cargo run --release -p mpee-cli -- build data/greater-london-latest.osm.pbf
```

> Build caches with `mpee build` (or the pip `mpee build`) — **not** with
> `bench_pp` / `bench_ch`. Those are *benchmark* tools: they write the cache and
> then run a long SSSP/Dijkstra benchmark suite (100 s+) that has nothing to do
> with building it. `mpee build` runs the build-only pipeline in-process.

### Solve your own VRP from JSON

`mpee-cli solve` runs a VROOM-style problem against a CH cache. A minimal one
ships at [`examples/problem.json`](examples/problem.json):

```bash
# Build a cache first (above), then solve:
cargo run --release -p mpee-cli -- solve examples/problem.json \
    --ch data/greater-london-latest.osm.pbf.ch \
    --pp data/greater-london-latest.osm.pbf.pp \
    --time-limit-s 5 -o solution.json
#   → jobs 6 (assigned 6 / unassigned 0), 1 vehicle, ≈ 15.2 km
```

Schema (VROOM-native — note coordinates are **`[lon, lat]`** arrays here, unlike
the Python `solve()` which uses `{"coord": [lon, lat]}`):

```jsonc
{
  "vehicles": [
    { "id": 1, "start": [-0.1278, 51.5074], "end": [-0.1278, 51.5074],
      "capacity": [100], "profile": "car" }
  ],
  "jobs": [
    { "id": 101, "location": [-0.0984, 51.5138], "delivery": [10] }
  ]
}
```

Per-vehicle `capacity` / `skills` / `time_window` / `speed_factor` /
`max_travel_time` / `max_distance` and per-job `delivery` / `pickup` / `skills`
/ `time_windows` / `service` / `priority` are all honoured — see the full
constraint table in [`crates/mpee-py/`](crates/mpee-py/) (same engine model).

---

## Status

- **dijeng**: production-ready routing — CH build, K-NN in 1.2 s for
  50k customers, OSRM-compatible HTTP server (`bench_osrm`, `serve`),
  correctness verified against full Dijeng. See
  [crates/dijeng/README.md](crates/dijeng/README.md).
- **brooom**: GPU-accelerated VRP, beats PyVRP / Vroom / OR-Tools on
  Solomon R1-1000 (p ≈ 3·10⁻⁸), Vroom-compatible I/O. See
  [crates/brooom/README.md](crates/brooom/README.md).

- **mpee-cli**: end-to-end VRP pipeline. The `download` / `build` /
  `solve` / `pipeline` subcommands load a CH cache via dijeng, snap
  random / supplied coords to the road graph, build the N×N
  duration+distance matrix with dijeng's bucket-MMM, and hand the
  matrix straight to brooom's solver — all in the same address space,
  no IPC, no disk on the hot path.
- **mpee-viz**: live HTTP server (port 8032 by default) that runs the
  same pipeline and then serves the result over an embedded
  Leaflet-based mobile UI. Designed so a phone on the same network can
  load `http://<laptop-ip>:8032/`, see every route colour-coded on a
  map, tap a stop for `job_id` / `vehicle_id` / `stop_order`, and zoom
  to any vehicle's bounding box. See
  [`crates/mpee-viz/README.md`](crates/mpee-viz/README.md) for build
  instructions and the measured 2 000 / 5 000-job runs over Greater
  London.

### Measured end-to-end runs (Apple M3 Pro, Central London CH)

| N jobs | vehicles | matrix     | solve   | wall     | assigned | distance |
|-------:|---------:|-----------:|--------:|---------:|---------:|---------:|
| 2 000  | 50       | 0.32 s     | 122 s   | ~2 min   | 1 982    | 1 332 km |
| 5 000  | 100      | 4.10 s     | 515 s   | ~9 min   | 4 948    | 2 535 km |

Both runs ship a 99 %+ assignment rate. The 1 % dropped are random
points that snapped to road fragments isolated from the depot (parks,
gated roads, pedestrian-only paths). Real address inputs do not have
this issue.

---

## License

MIT — every crate: `brooom`, `dijeng` (Cargo name `dijeng`), `mpee-cli`,
`mpee-py`, and `mpee-viz`. See each crate's `Cargo.toml` for details.

---

## Naming

- **MPEE** stands for **Morten Punnerud-Engelstad Engine**.
- **dijeng** is the routing sub-engine — short for *Dijkstra engine*.

The iOS/iPadOS/macOS SwiftUI demo (and its C ABI bridge) lives in a separate
repo, `mpee-ios`, so this engine repo stays platform-neutral.
