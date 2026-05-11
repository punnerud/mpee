# mpe-py

> Part of the **[mpe-engine](../../README.md)** workspace.

PyO3 bindings that expose mpe-engine's in-process VRP pipeline
(brooom + sssp_bench) as a Python extension module. Drop-in for a
Flask / FastAPI / Streamlit front-end without dropping the
shared-memory architecture: Python and Rust live in the same process,
so the JSON bytes the browser fetches come straight out of the same
`Arc<String>` that the Rust solver thread just populated.

The headline use-case is the **macOS Application Firewall**: every
fresh `mpe-serve` binary requires per-build approval before non-
loopback connections work. Python (Apple-signed system binary) is
already allowed by default — so `python3 app.py` binds `0.0.0.0:8032`
without prompts.

---

## Architecture

```
                ┌──── Python process (Flask + mpe_py) ────────────────┐
                │                                                     │
                │  import mpe_py                                       │
                │  eng = mpe_py.Engine()                               │
                │  eng.start_solve(region="london", n_jobs=500, ...)   │
                │             │                                        │
                │             ▼  std::thread::spawn (inside Rust)      │
                │  ┌────────────────────────────────────────────────┐  │
                │  │ Rust solver thread:                            │  │
                │  │   sssp_bench::load_mmap → matrix_with_distance │  │
                │  │   brooom::solve_with_matrix (iterative)        │  │
                │  │   publishes Arc<String> after every chunk      │  │
                │  └────────────────────────────────────────────────┘  │
                │             │                                        │
                │             ▼  engine.get_dataset_json() (no copy)   │
                │  Flask routes: /api/status /api/dataset /            │
                │             │                                        │
                └─────────────┼────────────────────────────────────────┘
                              ▼
                         📱 phone on the same network
```

The boundary between Python and Rust is two thin methods:
`get_status_json()` and `get_dataset_json()`. Both read from the
shared `Arc<RwLock<AppState>>`. The JSON string is **already
serialised** in Rust — Python just returns its bytes verbatim.

---

## Build & run

```bash
# From the repo root (or anywhere — venv lives in this crate dir):
cd crates/mpe-py

# Once: set up a Python venv and install build tools.
python3 -m venv venv
source venv/bin/activate
pip install maturin flask

# Builds the Rust extension and installs it into the venv.
maturin develop --release

# Starts the Flask server (defaults to London N=500, 0.0.0.0:8032).
python3 python/app.py
```

The first `maturin develop --release` takes about 30–60 s (full
optimised LTO build of brooom + sssp_bench + bindings). Subsequent
incremental builds are seconds.

On macOS this **avoids the per-binary Application Firewall prompt**
that `mpe-serve` triggers — Python is already trusted by the firewall.

---

## Python API

```python
import mpe_py
import json, time

eng = mpe_py.Engine()
eng.start_solve(
    region="london",      # "london" / "oslo" / "manhattan" / "paris"
    n_jobs=500,
    n_vehicles=20,
    capacity=200,
    seed=7,
    ch="data/greater-london.osm.pbf.ch",
    pp="data/greater-london.osm.pbf.pp",
    time_limit_s=45.0,
    multi_start=1,
)

while not eng.is_done():
    status = json.loads(eng.get_status_json())
    print(status["state"], status["phase"], status["message"])
    ds = eng.get_dataset_json()
    if ds is not None:
        # Same JSON shape as mpe-serve's /api/dataset bundle.
        bundle = json.loads(ds)
        print(f"iter {bundle['iter']}: cost={bundle['cost']:.0f}")
    time.sleep(1)

print("final dataset_iter:", eng.dataset_iter())
```

### Method reference

| Method | Returns | Notes |
|--------|---------|-------|
| `Engine()` | `Engine` | Construct an idle engine. |
| `start_solve(...)` | `None` | Spawn the background solver thread. |
| `get_status_json()` | `str` | Status (state, phase, message, progress, elapsed_s, dataset_iter, config). |
| `get_dataset_json()` | `Optional[str]` | Latest dataset, or `None` until first iter. |
| `dataset_iter()` | `int` | Current published iteration counter. |
| `is_done()` | `bool` | True once the solver finished (or failed). |
| `state()` | `str` | One of `idle` / `solving` / `evolving` / `done` / `failed`. |

`start_solve(...)` arguments mirror the `mpe-serve` CLI:

```python
start_solve(
    region: str,
    n_jobs: int,
    n_vehicles: int,
    capacity: int,
    seed: int,
    ch: str,
    pp: str,
    time_limit_s: float = 45.0,
    multi_start: int = 1,
)
```

---

## Endpoints exposed by `python/app.py`

Identical to mpe-serve so the embedded `index.html` works unchanged:

| Method | Path           | Description |
|--------|----------------|-------------|
| GET    | `/`            | Embedded Leaflet HTML (mobile UI). |
| GET    | `/api/status`  | Live status JSON (polled by the UI). |
| GET    | `/api/dataset` | 200 + dataset, or 202 + status while pre-first-iter. |
| GET    | `/api/health`  | `{"ok":true}` |

CORS is wide-open. The UI polls `/api/status` every ~2 s and re-renders
the map whenever `dataset_iter` advances.

---

## Why not just disable the firewall?

Doing `socketfilterfw --setglobalstate off` works but turns off all
incoming protection. Adding `--add ./target/release/mpe-serve` works
per-binary but every `cargo build` changes the binary's code signature,
so the rule expires and the firewall blocks the next launch silently.
Going through Python sidesteps that entirely.
