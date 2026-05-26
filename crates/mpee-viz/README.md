# mpee-viz

> Part of the **[mpee](../../README.md)** workspace.

Live HTTP server that solves a VRP **in this process** and serves the
result over HTTP. The solver, the routing matrix, the snap layer, and
the HTTP handlers all share the same `&Problem` / `&Solution` /
`&Matrix` references — no IPC, no disk on the hot path, and no
re-serialisation between solving and rendering.

Designed for opening a phone on the same Wi-Fi as the laptop running
the engine and watching how many thousand routes lay out on the map.

---

## Architecture

```
                ┌──────── mpee-viz (single Rust process) ──────────────┐
                │                                                       │
                │  dijeng::cache_pp/ch::load_mmap   (mmap ~20 µs)   │
                │  dijeng::routing::matrix_with_distance            │
                │       │                                               │
                │       ▼   Vec<f32> in shared address space            │
                │  one f32→i32 pass into brooom::Matrix                 │
                │       │                                               │
                │       ▼                                               │
                │  brooom::solver::solve_with_matrix                    │
                │       │                                               │
                │       ▼   &Problem + &Solution + &Matrix held in Arc  │
                │  build_dataset(...) → Dataset                         │
                │       │                                               │
                │       ▼   one serde_json::to_string                   │
                │  tiny_http server on 0.0.0.0:8032                     │
                │                                                       │
                └──────┬────────────────────────────────────────────────┘
                       │  GET /            → embedded Leaflet HTML
                       │  GET /api/dataset → cached JSON bundle
                       │  GET /api/health
                       ▼
                    📱 phone on the same network
```

There is **one** allocation crossing a process boundary: the JSON
string sent to the browser. Everything before that — the matrix the
solver consumed, the routes the renderer iterates — lives in the same
RAM, same address space.

---

## Build & run

```bash
# Once: build the release binary (~1 min, ort already cached after first brooom build).
cargo build --release -p mpee-viz

# Then: solve + serve in one command.
./target/release/mpee-viz \
  --region london \
  --n-jobs 5000 --n-vehicles 100 --capacity 350 --seed 1 \
  --ch data/greater-london.osm.pbf.ch \
  --pp data/greater-london.osm.pbf.pp \
  --time-limit-s 20 --multi-start 2 \
  --port 8032
```

The server logs the URL once it binds. Point a phone on the same Wi-Fi
network at `http://<laptop-ip>:8032/` — it will fetch `/api/dataset`
and draw the result on a Leaflet map.

To find your laptop IP on macOS: `ipconfig getifaddr en0`.

---

## What the map shows

- **Yellow dot** — the depot.
- **Coloured dots** — every assigned stop, coloured by vehicle. Hue is
  picked deterministically via golden-ratio stride so neighbouring
  routes don't share a colour.
- **Coloured lines** — the visiting order of each route, straight-line
  segments from depot → stop 1 → stop 2 → … → depot. (Real
  road-following polylines are a follow-up — they require running
  `dijeng::ch::query_with_path` for each consecutive pair, which is
  cheap individually but adds payload weight to the JSON bundle.)
- **Red dots** (toggleable) — unassigned jobs, if any.

Tapping a stop opens a popup with the job ID, the vehicle that serves
it, its position in the route, and the cumulative load after the stop.
Tapping a vehicle row in the footer zooms to that vehicle's bounding
box. The footer buttons toggle stops / routes / unassigned globally
and let you turn all routes on or off at once.

---

## Measured runs (Apple M3 Pro, Central London CH)

These are the same datasets that mpee-cli was tested on, served live.

### N=2000 jobs, 50 vehicles, capacity=300, seed=7

```
matrix 2100×2100         0.32 s   (13.6 M cells/s)
unreachable cells        73 170 / 4 410 000   (1.7 %)
dropped after snap       18 jobs (snapped to isolated fragments)
brooom solve            122 s   (multi_start=4, 20 s budget/variant)
─────────────────────────────────────────────────────
assigned                 1 982 / 2 000   (99.1 %)
routes used              37 / 50
total drive time         42.5 h
total drive distance     1 332 km
JSON payload sent        ~280 kB
```

### N=5000 jobs, 100 vehicles, capacity=350, seed=1

```
matrix 5200×5200         4.10 s   (6.6 M cells/s)
unreachable cells        486 455 / 27 040 000   (1.8 %)
dropped after snap       52 jobs
brooom solve            515 s   (multi_start=2, 20 s budget/variant)
─────────────────────────────────────────────────────
assigned                 4 948 / 5 000   (99.0 %)
routes used              78 / 100
total drive time         80.2 h
total drive distance     2 535 km
JSON payload sent        ~700 kB
```

Both cases fit comfortably in a mobile-Safari render cycle. The map
remains interactive on iPhone with 5 000 markers thanks to Leaflet's
canvas-renderer (`preferCanvas: true`).

---

## API endpoints

| Method | Path           | Description                                  |
|--------|----------------|----------------------------------------------|
| GET    | `/`            | Embedded Leaflet HTML (the mobile UI)        |
| GET    | `/api/dataset` | The cached JSON bundle (full problem + sol)  |
| GET    | `/api/health`  | `{"ok": true}`                               |

CORS is wide-open (`Access-Control-Allow-Origin: *`) so a Python/Flask
front-end on a different port can call the API directly. Example:

```python
import requests, flask
app = flask.Flask(__name__)

@app.get("/")
def index():
    data = requests.get("http://localhost:8032/api/dataset").json()
    return flask.render_template_string(MY_HTML, data=data)

app.run(host="0.0.0.0", port=8033)
```

---

## What still costs disk / CPU

- The **first** request after startup is cheap: the dataset is built
  once at boot and held in `Arc<String>`. Every subsequent request is
  a memcpy of that string.
- The HTML is `include_str!`-ed at compile time — no template engine,
  no file reads at runtime.
- Solving is the only slow step. With the defaults above:
  - 2 000 jobs ≈ 2 min wall (mostly brooom)
  - 5 000 jobs ≈ 9 min wall (mostly brooom)
- Resolving the laptop's local IP for the phone is the only manual step.

---

## Limitations / follow-ups

1. **Real polylines** — currently straight lines. To draw road-following
   geometry, call `dijeng::ch::query_with_path_into` for each
   consecutive (i, j) in every route, batch the resulting node-id sequences
   into the dataset payload, and let Leaflet draw the multi-segment
   polylines. ~50 lines of code; mainly bumps the JSON payload from
   ~700 kB to a few MB at N=5 000.
2. **Live re-solve** — today the server solves once at startup. A POST
   endpoint that accepts new parameters and re-runs `solve_in_process`
   would re-render with the same hot caches without restart.
3. **Per-route tile caching** — at very high zoom mobile Safari still
   does fine, but a vector-tile route layer would let us push to N≥20 000.
