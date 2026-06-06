# Benchmarks — reproducible, real, head-to-head

Everything here runs the **same instances** through **every solver** so quality
gaps are solver-only (one shared distance matrix per instance). No self-reported
numbers — re-run it yourself.

## Solvers compared

| solver  | how it runs | notes |
|---------|-------------|-------|
| brooom  | `target/release/brooom` (this repo) | integrated matrix + VRP, soft-TW auto-on for time-windowed instances |
| vroom   | `vroom` in PATH | the classic OSRM+VROOM optimiser |
| OR-Tools| `ortools` pip pkg (in-process) | CP routing layer, GLS metaheuristic |
| PyVRP   | `pyvrp` pip pkg (in-process) | HGS — current SOTA on CVRPTW |

Install the Python ones into a venv (3.12 has wheels):

```bash
python3.12 -m venv .bench-venv
.bench-venv/bin/pip install pyvrp ortools
```

## Running

```bash
# Solomon-format instances (matrix embedded in the JSON):
.bench-venv/bin/python run_multi_bench.py --time-limit 10 \
    instances_solomon/c101.json instances_solomon/r201.json
```

## Reproducible REAL-MAP instances (`gen_realmap.py`)

Grok's review noted the early numbers were synthetic (Solomon). `gen_realmap.py`
fixes that: it draws N delivery points inside a real city's bounding box **from a
seed**, so anyone reproduces the *exact* instance from `(region, seed, n)` —
without shipping coordinates. The N×N matrix is built from **real OSRM road
distances**, or from haversine for a fully-offline, byte-identical reproduction.

```bash
# Fully offline + byte-reproducible (haversine), San Francisco, 100 stops:
python3 gen_realmap.py --region sf --seed 7 --n 100 --vehicles 25

# Real road matrix via OSRM (local or the public demo), Oslo:
python3 gen_realmap.py --region oslo --seed 1 --n 50 \
    --matrix osrm --osrm-host https://router.project-osrm.org

# Benchmark every solver on the same real-map instance:
python3 run_multi_bench.py --time-limit 10 instances_realmap/sf_s7_n100_haversine.json
```

Regions: `sf`, `oslo`, `london`, `nyc` (add a bbox in `REGIONS` for more). The
seed fixes coordinates, demands, service times and time windows; the generated
file is self-contained and carries a `meta` block with the exact reproduce
command.

### Using OSRM as the routing/matrix backend for brooom

brooom can build its matrix from OSRM instead of its own engine — useful to
compare *like for like* against the OSRM+VROOM stack:

```bash
brooom -i coords_problem.json --routing osrm --osrm-host http://127.0.0.1:5000
```

(Default is brooom's own integrated, streamed matrix — no separate build step.)

## Latest results

* `results/multi_bench.csv` — raw per-(instance, solver) rows from the last run.
* `results/competitor_comparison.md` — the curated 4-way head-to-head + honest
  read of where we win and lose.
* `results/soft_tw_ab.md` — soft-TW on/off A/B (no regression on feasible
  instances; serves stops late instead of dropping on over-constrained ones).
