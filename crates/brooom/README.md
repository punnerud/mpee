# brooom

> Part of the **[mpee](../../README.md)** workspace. Standalone crate —
> `cargo build` inside this directory works on its own. Integrates with
> [`dijeng`](../dijeng/) (CH routing) via the contract in
> [`integration.txt`](integration.txt) and the workspace overview in
> [`INTEGRATION.md`](../../INTEGRATION.md).

**A GPU-accelerated VRP solver in Rust — state-of-the-art on N ≥ 500.**

brooom is an open Rust alternative to Vroom for solving the Vehicle Routing
Problem with Time Windows (VRPTW). It runs on Apple Silicon (Metal),
Linux (Vulkan), and Windows (DX12) via a single WGSL megakernel that keeps
the entire local-search loop on the GPU.

On Solomon-style synthetic N=1000 instances with a 60 s budget, brooom beats
the next-best solver (PyVRP, an HGS-CVRP implementation) on **17 of 20**
random seeds with statistical significance p ≈ 10⁻⁷. On N=2000 brooom
wins **3 of 3** with a mean 1.6 % gap to PyVRP, and on N=50 000 it is the
only tested solver that converges at all on a laptop.

---

## At a glance

| Property              | Value |
|-----------------------|-------|
| Language              | Rust (stable) |
| GPU backend           | wgpu (Metal / Vulkan / DX12) |
| Lines of code         | ~16 000 |
| Solves                | CVRP, VRPTW, PDPTW, multi-depot, backhaul, multi-trip, prize-collecting, client-groups, multi-vehicle, time windows, release times, capacity, skills, driver breaks, precedence, fairness/max-vehicles |
| Input format          | Vroom-compatible JSON (drop-in) |
| Output format         | Vroom-compatible JSON |
| Routing engine        | OSRM via HTTP, custom CH engine (MMM), or precomputed matrix |
| Scale tested          | N = 50 000 customers on a laptop |
| License               | MIT |

---

## Why brooom is different

Most VRP solvers either:
1. Run a strong CPU local-search loop (LKH-style: Vroom, HGS-CVRP, OR-Tools), or
2. Wrap a metaheuristic library in Python (PyVRP, jsprit, etc).

brooom does neither. It runs the **entire local-search inner loop** —
2-opt, relocate, exchange, 2-opt\*, swap-star, Or-opt-2, Or-opt-3, ILS-kick,
regret-3 repair — as a **single WGSL compute kernel**. The CPU only sees the
final solution. This gives:

1. **Sub-millisecond per-iteration cost** at N = 1000–5000, where CPU
   solvers spend 10–100 µs per LS iter.
2. **Population search for free** — the same megakernel runs 64–256
   trajectories in parallel as separate workgroups, all on one dispatch.
3. **Cross-platform** — the same shader runs on Mac, Linux, and Windows
   without code changes.
4. **Scales to N = 50 000** via two paths: (a) cluster-decomposition for
   dense matrices, (b) `coord-mode` where the shader recomputes Euclidean
   distance from f32 coordinates instead of reading a 10 GB matrix.

The GPU-resident state machine is fully described in `src/gpu_population.rs`
(~3 200 lines, including the WGSL shader source).

---

## Benchmarks (Solomon R1-style synthetic, 60 s budget, n=20 seeds, M3 Pro)

| Solver       | Mean cost | Wins | Δ vs brooom |
|--------------|----------:|-----:|------------:|
| **brooom**   | **11 810** | **17/20 (85 %)** | — |
| pyvrp (HGS)  | 11 931 | 3/20 (15 %) | +121 (+1.02 %) |
| vroom        | 11 934 | 0/20 ( 0 %) | +124 (+1.05 %) |
| ortools      | 14 685 | 0/20 ( 0 %) | +2 875 (+24.34 %) |

Statistical tests:
- **Binomial p-value vs uniform 25 %**: p ≈ 2.96·10⁻⁸ (overwhelmingly significant)
- **Head-to-head vs PyVRP**: 17/20 wins, p ≈ 0.0013
- **vs Vroom / OR-Tools**: 20/20 wins, p ≈ 9.5·10⁻⁷

Raw data: `benchmarks/results/n1000_aggregate_n20.csv`.

### Where brooom wins

| N    | brooom vs next-best    | Notes |
|------|------------------------|-------|
| 500  | +1.5 % (vs vroom)      | First N where brooom takes over |
| 1000 | +1.0 % (vs pyvrp) over 20 seeds | Statistically rock-solid |
| 2000 | +1.6 % (vs pyvrp) over 3 seeds | 3/3 wins |
| 5000+| (only solver that scales) | Vroom times out, PyVRP slows down |
| 50K  | unique capability      | Coord-mode GPU + cluster-decompose |

### Where brooom doesn't win

| N    | Best solver | Why |
|------|-------------|-----|
| 100  | pyvrp (−2.0 %) | HGS-CVRP finds tighter optima on small N |
| 250  | pyvrp / brooom (borderline) | Within RNG noise |
| N<200, latency-critical | vroom (~0.4 s) | Vroom is 100× faster on small N |
| PDPTW | (unbenchmarked) | brooom supports it, never measured head-to-head |

For the full N-scaling curve see `benchmarks/results/multi_bench.csv`.

---

## Architecture

```
                 ┌────────────────────────────────────────────────┐
                 │  CLI (main.rs)                                  │
                 │  Vroom-compatible JSON input/output             │
                 └────────────────┬────────────────────────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Solver pipeline (solver.rs)                        │
        │  • Multi-start (M parallel attempts, default M=16) │
        │  • Per-attempt: greedy insertion → CPU LS → polish  │
        │  • Optional cluster-decomposition (auto for N≥500)  │
        │  • Optional GPU megakernel polish (--gpu)           │
        │  • Optional HGS-style population search (--hgs)     │
        └─────────────────────────┬─────────────────────────┘
                                  │
            ┌─────────────────────┼──────────────────────┐
            │                     │                      │
       ┌────▼────┐         ┌─────▼─────┐         ┌──────▼──────┐
       │  CPU    │         │  GPU      │         │  Cluster-   │
       │  LS     │         │  Mega-    │         │  decompose  │
       │  loop   │         │  kernel   │         │  (large N)  │
       │         │         │           │         │             │
       │ rayon   │         │ 64–256    │         │ K-medoids   │
       │ multi-  │         │ trajectories      │ sub-solves  │
       │ start   │         │ parallel  │         │ + boundary  │
       └─────────┘         └───────────┘         └─────────────┘
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │  Distance source (matrix.rs)                       │
        │  • Precomputed JSON matrix                         │
        │  • Haversine from coords                           │
        │  • OSRM HTTP /table                                │
        │  • MMM K-NN (see integration.txt)                  │
        │  • Coord-mode (Euclidean in shader, N=50K)         │
        └────────────────────────────────────────────────────┘
```

### The GPU megakernel

A single WGSL compute kernel that runs the entire LS inner loop:

```
loop {
    // Phase 1: precompute per-route TW + capacity state
    // Phase 2a: best 2-opt per route                  (intra-route)
    // Phase 2b: best granular relocate                (inter-route, O(N×K))
    // Phase 2c: best inter-route exchange             (O(K²) per pair)
    // Phase 2d: best 2-opt*                           (cross-route tail swap)
    // Phase 2e: best swap-star (full Vidal)           (reposition swap)
    // Phase 2f: best Or-opt-2 (segment relocate, L=2)
    // Phase 2g: best Or-opt-3 (segment relocate, L=3)
    // Phase 3: argmin across all operators
    // Phase 4: apply winning move; update state
    if no improving move { ILS-kick + regret-3 repair }
    if converged { break }
}
```

Each workgroup runs the loop for its own trajectory; up to 256 trajectories
proceed in parallel on a single dispatch. The "megakernel" architecture
means there's no GPU↔CPU sync per iteration — total round-trip cost is one
dispatch per `solve_full()` call.

Optional features:
- **Penalty-LS** (`BROOOM_PENALTY_LS=1`): allow controlled TW-violations
  in granular relocate, with PENALTY × violation added to the cost delta.
  Marginal improvement on small N, helps escape local optima on N ≥ 1000.
- **Coord-mode** (auto-enabled for N≥50K when matrix doesn't fit): shader
  computes Euclidean distance from f32 coords on the fly. No matrix upload.

### Key files

| File                            | Purpose                                  |
|---------------------------------|------------------------------------------|
| `src/main.rs`                   | CLI entry point, Vroom-format I/O        |
| `src/solver.rs`                 | Top-level pipeline orchestration         |
| `src/local_search.rs`           | CPU LS loop with all operators           |
| `src/gpu_population.rs`         | GPU megakernel + WGSL shader (3 200 LOC) |
| `src/gpu_polish.rs`             | Batched GPU polish wrapper               |
| `src/cluster_decompose.rs`      | K-medoids + per-cluster sub-solving      |
| `src/granular.rs`               | K-NN neighbourhoods (Toth-Vigo)          |
| `src/hgs.rs`                    | HGS-style population search (experimental)|
| `src/matrix.rs`                 | Matrix, OSRM client, Haversine           |
| `src/problem.rs`                | Vroom-compatible JSON schema             |
| `benchmarks/run_multi_bench.py` | Multi-solver comparison harness          |

---

## Algorithms implemented

### Local-search operators
- **2-opt** (intra-route reversal)
- **Relocate** (inter-route, granular)
- **Exchange** (inter-route swap)
- **2-opt\*** (cross-route tail swap)
- **swap-star** (Vidal-style full-position swap)
- **Or-opt-2 / Or-opt-3** (segment relocate, L ∈ {2, 3})
- **ILS-kick** (random tear-down) + **regret-3 repair**

### Initialization
- **Greedy insertion** (sequential insertion with regret-1 ordering)
- **Solomon I1** insertion (CW saving as fallback)
- **Random restart** (perturbed greedy)

### Acceptance / search strategies
- **Best-improvement** (default)
- **First-improvement** (with `--first-improve`)
- **ILS** (iterated local search with destroy-and-rebuild)
- **Multi-start** (M parallel attempts, best-of-K)
- **HGS-MVP** (SREX crossover + diversity-aware selection — experimental)
- **Penalty-LS** (controlled TW violations, escapes local optima)

### Scaling features
- **Cluster decomposition** — K-medoids on the matrix, sub-solve each
  cluster, stitch boundary routes. Auto-enabled for N ≥ 500. Critical for
  N ≥ 5000 where flat LS is matrix-bound.
- **Coord-mode** — for N ≥ 50K where the N² matrix doesn't fit, the GPU
  shader recomputes distances from f32 coords instead of reading a buffer.
- **Streaming K-NN** — see `integration.txt`. Pre-computed K-NN from an
  external engine (e.g. MMM) eliminates the dense matrix entirely.

---

## Usage

### Quick start

```bash
# Build
cargo build --release

# Solve a Vroom-format problem with default settings (CPU-only)
./target/release/brooom -i problem.json -o solution.json

# Use GPU (requires Apple Silicon, NVIDIA/AMD with Vulkan, or Windows DX12)
./target/release/brooom -i problem.json -o solution.json --gpu

# Recommended production config for N ≥ 1000
BROOOM_GPU_REPEATS=4 BROOOM_GPU_KICK=8 BROOOM_PENALTY_LS=1 \
  ./target/release/brooom -i problem.json -o solution.json \
    --gpu -m 16 -l 60
```

### Common CLI flags

| Flag                    | Default | Notes |
|-------------------------|---------|-------|
| `-i, --input`           | (req)   | Vroom-format JSON input |
| `-o, --output`          | stdout  | JSON output path |
| `--gpu`                 | off     | Enable GPU megakernel polish |
| `-m, --multi-start`     | 8       | Parallel starts; recommend 16 for N ≥ 1000 |
| `-l, --time-limit-s`    | none    | Wall-clock budget |
| `-k, --granular-k`      | auto    | K-NN size; auto: 80 (N≥500), 40 (N≥100), 20 |
| `--ils-iters`           | 30      | ILS iterations per restart |
| `--decompose K`         | auto    | Cluster decomposition (K clusters) |
| `--exact-polish`        | off     | Brute-force solve routes ≤ 14 stops |
| `--osrm-url`            | none    | OSRM /table endpoint for matrix |
| `--mmm-knn PATH`        | none    | Path to MMM K-NN data (see integration.txt) |
| `--objective`           | scalar  | `scalar` or `lexicographic` (N-level) |
| `--objective-levels`    | none    | Comma list, highest priority first (e.g. `vehicles,cost`); implies `--objective lexicographic` |
| `--options PATH`        | none    | JSON file with an `options` object (`objective` + `dimensions`), merged over the input's `options` |
| `--dimensions JSON`     | none    | Inline JSON list of custom accumulator dimensions |
| `--warm-start PATH`     | none    | Vroom-style solution JSON to seed local search (strictly safe) |
| `--broker`              | off     | Cost-aware matrix broker: buy only the cells the solver reads, derive the rest ([docs](docs/matrix-broker.md)) |
| `--matrix-db PATH`      | none    | Broker cell DB: reuse bought cells across runs + frequency counter |
| `--buy-budget N`        | none    | Broker hard spend cap (priced by `--cost-policy`); excess is derived |
| `--cost-policy SPELL`   | none    | Broker cost/buy policy (PySpell over `broker.*`) |
| `--departure CLASS:HOUR`| none    | Broker temporal profile (e.g. `workday:08`) — learn one day, reuse offline |
| `--uncertainty-weight W`| 0       | Broker: bake `mean + W·std` so queue-prone arcs cost more |
| `--offline-reuse`       | off     | Broker: serve the chosen window from the DB; buy nothing when warm |

### Cost-aware matrix broker — pay only for the cells you use

When the matrix comes from a **paid/limited** provider (Google Distance Matrix, a
metered OSRM, your own endpoint), the broker buys only the thin *skeleton* the
local search actually reads, **derives** the long-range rest, **reuses** a local
DB across runs (warm DB → zero buys the second time), and — with temporal
profiles — **learns one representative day of traffic and replays it offline** for
every similar day, baking congestion/uncertainty into the matrix so the solver
routes around the queues. Provider-agnostic; absent `--broker`, routing is
unchanged. **Full story + sales argument: [`docs/matrix-broker.md`](docs/matrix-broker.md).**

### Environment variables

| Var                     | Effect |
|-------------------------|--------|
| `BROOOM_GPU_REPEATS`    | Number of GPU polish rounds (default 1, recommend 4) |
| `BROOOM_GPU_KICK`       | Number of random task removes per ILS-kick (default 3) |
| `BROOOM_PENALTY_LS`     | `=1` to enable penalty-LS in granular relocate |
| `HGS_EDUCATE_KICK`      | Kick count during HGS educate (default 0) |

### Objectives & custom dimensions

Two solve-level knobs shape *what* "best" means and *what* accumulates along a
route. Both are reachable identically from the Rust `SolverConfig` API, the
`--objective` / `--dimensions` CLI flags, the JSON `options` block, and the
Python bindings — same native code behind each.

**Lexicographic objective (true N-level, not a weighted sum).** Optimise in
strict priority order: minimise the first level; among solutions that achieve it,
minimise the second; and so on. Each level pins its achieved value as a hard cap
for the next and warm-starts from the previous level's solution. Levels:
`vehicles`, `unassigned`, `cost`, `makespan`, `distance`.

```bash
# "Use as few vehicles as possible, then cut cost within that fleet size."
brooom -i problem.json --objective lexicographic --objective-levels vehicles,cost
```

```rust
use brooom::{solver::{solve, SolverConfig, ObjectiveMode, LexObjective}};
let cfg = SolverConfig {
    objective_mode: ObjectiveMode::Lexicographic {
        levels: vec![LexObjective::Vehicles, LexObjective::Cost],
    },
    ..Default::default()
};
let sol = solve(&mut problem, Some(&matrix), cfg)?;
```

The default is `Scalar` (single weighted cost), byte-identical to before. It is
*best-effort* lexicographic: each pinned value is the metaheuristic's best, not a
proven optimum.

**Custom accumulator dimensions (OR-Tools-style `RoutingDimension`).** Track a
quantity that accumulates along each route — fuel, a cooling budget, a resource —
via a per-arc transit. Declare hard `min`/`max`, soft (`soft_max`/`soft_min` +
`soft_weight`), and a monotonicity so the bound prunes inside the insertion probe.

```jsonc
// options.dimensions — a draining fuel tank that must not hit empty.
{ "options": { "dimensions": [
  { "name": "fuel", "transit": "distance / 10", "start": 500,
    "min": 0, "monotonicity": "non_increasing" } ] } }
```

```rust
use brooom::dimension::{ArcCtx, CustomDimension, DimensionGuard};
use std::sync::Arc;
let fuel = CustomDimension::new("fuel", Arc::new(|c: &ArcCtx| -(c.distance / 10)))
    .with_start(500).with_min(0).draining();
let _g = DimensionGuard::install(vec![fuel]); // RAII — cleared on drop
```

From a sandboxed string (JSON/CLI/Python) the transit is a pyspell expression over
the arc (`distance`, `duration`, `cumul`), compiled to native code — never
arbitrary code. A `non_increasing`+`min` (or `non_decreasing`+`max`) dimension is
mirrored into the O(1) probe; soft bounds add a penalty instead of rejecting. A
shared cross-route resource (e.g. a depot loading dock) is a solution-level global
— enforced post-hoc, not in-routing.

> For the propagation-hard class a local-search solver can't close well (tightly
> coupled time windows), see the honest [CP-SAT interop bridge](../../tools/cpsat_bridge/):
> export the instance to OR-Tools CP-SAT and round-trip the tour back as a
> `--warm-start`.

---

## Practical guidance: which solver should I use?

| Use case                       | Recommended         | Why |
|--------------------------------|---------------------|-----|
| N = 100, latency < 1 s         | **vroom**            | 0.4 s, 99 % of PyVRP quality |
| N = 100, highest quality       | **pyvrp**            | HGS-CVRP optimal on small N |
| N = 100, many instances        | **brooom + xargs -P 16** | 8.5× throughput |
| N = 250–500                    | pyvrp or brooom     | Borderline |
| N = 500 – 2 000                | **brooom_gpu**       | Best quality + GPU advantage |
| N = 1 000 (production)         | **brooom_gpu**       | 17/20 wins vs PyVRP, p < 10⁻⁷ |
| N ≥ 5 000                      | **brooom_gpu**       | Only solver that scales here |
| N = 50 000 (laptop)            | **brooom_gpu coord-mode** | Unique capability |
| PDPTW                          | (unbenchmarked)     | brooom supports it natively |
| Multi-depot                    | supported           | distinct per-vehicle start/end (Vroom-style) |
| Backhaul / driver breaks       | supported           | enforced in the route evaluator |

---

## Project structure

```
brooom/
├── src/                    # solver source (16K LOC Rust)
├── benchmarks/             # multi-solver benchmark harness
│   ├── run_multi_bench.py  # vroom / brooom / pyvrp / ortools
│   ├── gen_solomon_like.py # synthetic instance generator
│   ├── instances/          # 100+ test instances (N=50 to N=50K)
│   └── results/            # CSV outputs
├── examples/               # standalone Rust examples
│   ├── gpu_megakernel_demo.rs
│   ├── gpu_scaling_synthetic.rs
│   └── gpu_population_roundtrip.rs
├── tests/                  # integration tests
├── integration.txt         # how to integrate with MMM / external CH engines
├── PROSJEKTER.md           # development roadmap
├── next1.txt              # detailed measurement notes (latest findings)
└── README.md               # you are here
```

---

## What's experimental, what's stable

**Stable / production-ready:**
- All standard LS operators (2-opt, relocate, exchange, swap-star, Or-opt, 2-opt\*)
- CPU multi-start + ILS pipeline
- GPU megakernel polish (`--gpu`)
- Cluster decomposition for large N
- Vroom-compatible JSON I/O
- OSRM matrix client

**Experimental / partially built:**
- HGS-MVP (`--hgs`) — runs without crashing but doesn't beat strong CPU baseline
- Penalty-LS (`BROOOM_PENALTY_LS=1`) — works on relocate only, marginal effect alone
- Pattern-DB / NN refiner — implemented but not in production path
- Distillation pipeline (PyTorch → ONNX → Rust) — research code

**Not implemented:**
- Cross-route / global constraints in code (the custom-constraint hook is
  per-route — see `constraint.rs`; soft *per-route* penalties are supported)
- Multiple shifts per vehicle (one time-window per vehicle, not several)
- Exact branch-and-cut (only brute-force `--exact-polish` for routes ≤ 14)

---

## Limitations and honest assessment

brooom is **state-of-the-art on a specific operating point**: N ≥ 500
Solomon R1-style synthetic instances with a 30–120 s budget on Apple
Silicon. It is **not** general state-of-the-art on CVRPTW. Areas where
brooom has measured weaknesses or has not been validated:

1. **N ≤ 100** — PyVRP wins 3/3 by ~2 %. The asymptote is algorithmic, not
   time-bound: 300 s budget gives same cost as 60 s. To close this gap
   brooom would need true HGS with infeasible sub-population.
2. **Latency for small N** — Vroom solves N=100 in 0.4 s with 99 % of
   PyVRP's quality. For latency-sensitive use cases (interactive routing,
   sub-second response), brooom's 30 s warmup is a poor fit.
3. **PDPTW / multi-depot / heterogeneous fleet / backhaul / breaks** —
   supported and feasibility-tested (`tests/constraints.rs`), but solution
   quality is never benchmarked head-to-head against other solvers.
4. **C / RC / R2 instance series** (clustered, mixed, long time-windows) —
   only R1-style tested; no claim on other geographies.
5. **Real road-network distances** — only tested on synthetic Euclidean.
   For production, integrate via MMM (`integration.txt`).
6. **Solomon canonical benchmarks** — the 56 well-known instances haven't
   been run head-to-head; only synthetic R1-style.

For a fully-validated SOTA claim on CVRPTW we'd need the 56 Solomon
canonical instances and the 300 Gehring-Homberger instances, against the
best published HGS-CVRPTW results (not just PyVRP).

---

## Citation / acknowledgements

brooom builds on standard VRP literature:
- Toth & Vigo (2014) — *Vehicle Routing: Problems, Methods, and Applications*
- Vidal et al. (2014) — *A Hybrid Genetic Algorithm with Adaptive Diversity Management*
- Helsgaun (2017) — *LKH-3* (LKH-style operators in the LS loop)
- Nagata & Bräysy (2009) — *Edge Assembly Crossover* (SREX foundation)

The novel contributions are (a) the **monolithic WGSL megakernel** that keeps
the entire LS inner loop on the GPU, and (b) the **multi-trajectory population
mode** that exploits this via batched workgroups. Both are described in
`src/gpu_population.rs`.

---

## License

MIT. See `LICENSE`.

---

## Status

Active development. Latest measurements: `next1.txt` (treated as a living log).
Benchmark CSVs: `benchmarks/results/`. Production config: see Usage section above.
