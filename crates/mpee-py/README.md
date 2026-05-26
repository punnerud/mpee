# MPEE — Offline route calculations and optimization

`pip install mpee`

A self-contained routing + delivery-optimization engine you drive from
Python or a command line. Download a map **once**, then compute everything
**offline** — no API keys, no per-request billing, no data leaving your
machine:

- 🧭 **Point-to-point routes** — driving distance + time between two coordinates.
- 🚚 **Multi-vehicle optimization (VRP)** — best routes for *N* vehicles over your own stops, with capacities.
- 🗺️ **Bring your own map** — any OpenStreetMap extract (a city, a country).
- ⚡ **Fast & small** — a Rust engine (CH routing + `brooom` VRP solver) using CPU **and** GPU, with a footprint under ~50 MB.

The engine is built from two Rust crates — `dijeng` (contraction-hierarchy
routing) and `brooom` (the VRP solver). Together they solve a
50,000-customer VRP on a laptop **without ever materialising a full
distance matrix**, and in head-to-head tests on a Mac they produced
**shorter routes than [VROOM](https://github.com/VROOM-Project) at equal
runtime**, using less memory.

---

## Install

```bash
pip install mpee
```

Wheels are prebuilt for Linux / macOS / Windows (Python 3.8+). No Rust
toolchain needed to install.

---

## 1. Get a map (once)

```bash
# Download an OpenStreetMap extract from Geofabrik …
mpee download europe/great-britain/england/greater-london

# … and preprocess it into a routable cache (.pp + .ch).
# Seconds for a city, a few minutes for a whole country.
mpee build data/greater-london-latest.osm.pbf          # car (default)
# mpee build data/greater-london-latest.osm.pbf bicycle  # or a profile (car|bicycle|foot)
```

After this you are fully offline. The cache is reusable forever — a re-run of
`mpee build` reuses it instantly (pass `--force` to rebuild, `--quiet` to
hush progress).

## 2. Route from A to B

```bash
mpee route 51.5080,-0.1281 51.5138,-0.0984 --cache data/greater-london-latest.osm.pbf
```
```
distance: 2.38 km
duration: 4.4 min
from (snapped): (51.50753, -0.12802)
to   (snapped): (51.51328, -0.09844)
```

## 3. Optimize a delivery run over many stops

```bash
# stops.txt: one "lat,lon" per line (or a JSON [[lat,lon], …])
mpee optimize --stops stops.txt --vehicles 5 --capacity 20 \
    --cache data/greater-london-latest.osm.pbf
```
```
stops: 50  vehicles used: 3/5  unassigned: 0
total: 60.0 km, 115 min  (solved in 4.6s)
  vehicle 1: 14 stops, 17.2 km, 33 min
  vehicle 2: 16 stops, 19.3 km, 37 min
  vehicle 4: 20 stops, 23.5 km, 45 min
```

Tune the fleet with `--vehicles` and `--capacity`; bound the search with
`--time SECONDS`; pin the depot with `--depot LAT,LON` (defaults to the
centroid of the stops).

---

## Use it from Python

```python
import mpee

# The .pp/.ch cache comes from a one-time `mpee download` + `mpee build`
# (see "Get a map" above) — or build it straight from Python:
#   mpee.Router.build("data/greater-london-latest.osm.pbf")   # → .pp + .ch
r = mpee.Router("data/greater-london-latest.osm.pbf.pp",
                "data/greater-london-latest.osm.pbf.ch")

# Distance + time between two points (set geometry=True for the path).
leg = r.route(51.5080, -0.1281, 51.5138, -0.0984)
print(f"{leg['distance_km']:.2f} km, {leg['duration_min']:.1f} min")

# Optimize 50 deliveries across 5 vehicles.
stops = [(51.51, -0.12), (51.49, -0.10), ...]   # your (lat, lon) list
plan = r.optimize(stops, vehicles=5, capacity=20, time_limit_s=5.0)
print(plan["total_distance_km"], "km over", plan["vehicles_used"], "vehicles")
for route in plan["routes"]:
    for stop in route["stops"]:
        print(route["vehicle_id"], stop["order"], stop["lat"], stop["lon"])

# Other helpers:
r.snap(51.50, -0.12)          # nearest routable road node
r.table(stops)                # full N×N duration + distance table
r.bbox()                      # coverage of the loaded map
```

`Router.build("map.osm.pbf", profile="car")` builds a cache from Python too
(`profile` is `car` | `bicycle` | `foot`). It **reuses an existing cache** and
returns instantly (`{"cached": True}`) unless you pass `force=True`, and it
prints build progress by default — pass `progress=False` to silence it when
driving the library from code:

```python
info = mpee.Router.build("data/region-latest.osm.pbf", progress=False)
# {'pp_path': ..., 'ch_path': ..., 'nodes': ..., 'build_secs': ..., 'cached': False}
```

(The `mpee build` CLI mirrors this: `--quiet` silences progress, `--force`
rebuilds an existing cache.)

## Real fleets: per-vehicle & per-stop constraints

For mixed fleets and constrained jobs, use `Router.solve(problem)` with a
[VROOM](https://github.com/VROOM-Project)-style problem (JSON). It exposes the
engine's full model — vehicle types, capacities, skills, time windows and
distinct start/end depots:

```python
import json, mpee
r = mpee.Router("data/greater-london-latest.osm.pbf.pp", "data/greater-london-latest.osm.pbf.ch")

problem = {
  "vehicles": [
    # A big van and a faster motorcycle from the same depot ([lon, lat]):
    {"id": 1, "start": {"coord": [-0.1278, 51.5074]}, "end": {"coord": [-0.1278, 51.5074]},
     "capacity": [200], "speed_factor": 1.0},
    {"id": 2, "start": {"coord": [-0.1278, 51.5074]}, "end": {"coord": [-0.0984, 51.5138]},  # ends elsewhere
     "capacity": [40], "speed_factor": 1.6, "skills": [7]},          # only this vehicle has skill 7
  ],
  "jobs": [
    {"id": 101, "location": {"coord": [-0.0837, 51.4954]}, "delivery": [5], "skills": [7]},  # → must use the skilled vehicle
    {"id": 102, "location": {"coord": [-0.1196, 51.5098]}, "delivery": [30],                 # a heavy package (weight 30)
     "time_windows": [{"start": 0, "end": 3600}]},                                           # deliver within the first hour
    {"id": 103, "location": {"coord": [-0.0890, 51.5161]}, "delivery": [12], "priority": 100},# keep this one if capacity is tight
    # ... up to tens of thousands of stops ...
  ],
}

plan = r.solve(json.dumps(problem), time_limit_s=5.0)
for route in plan["routes"]:
    print("vehicle", route["vehicle_id"], "→",
          [s["job_id"] for s in route["stops"]], f"{route['distance_km']:.1f} km")
print("unassigned:", plan["unassigned"])   # over-capacity / outside every window / unreachable
```

| Per **vehicle** | Per **stop** |
|---|---|
| `capacity` (multi-dimensional) | `delivery` / `pickup` (multi-dim package sizes / weights) |
| `skills` — which jobs it may serve | `skills` — required vehicle capability |
| `speed_factor` — e.g. a motorcycle at `1.6` | `time_windows` — allowed arrival times |
| `time_window` — the driver's shift | `service` — time spent at the stop |
| `max_travel_time` / `max_distance` | `priority` — which jobs to keep when demand > capacity |
| distinct `start` / `end` locations | |

`solve()` serves every job it feasibly can and returns the rest in
`unassigned` (over capacity, outside all time windows, or no road to them) —
it never invents an impossible route.

## Plan a work week (multi-day, multiple depots)

Model a week by giving each driver **one vehicle per day**, each bound to that
day's shift window; pin a delivery to a weekday with a matching time window.

```python
DAY = lambda k: {"start": k*86400 + 8*3600, "end": k*86400 + 18*3600}   # 08:00–18:00 on day k (Mon=0 … Fri=4)
depots = [[-0.1278, 51.5074], [-0.1300, 51.5230]]                       # two start/end points ([lon, lat])

vehicles = []
for depot in depots:
    for _driver in range(5):       # 5 drivers per depot → 10 drivers
        for k in range(5):         # Mon–Fri → one vehicle per driver per day
            vehicles.append({"id": len(vehicles) + 1,
                             "start": {"coord": depot}, "end": {"coord": depot},
                             "capacity": [1000], "time_window": DAY(k)})

# Each order is scheduled on a weekday by giving it that day's time window.
jobs = [{"id": o["id"], "location": {"coord": o["coord"]}, "delivery": [o["weight"]],
         "time_windows": [DAY(o["day"])]} for o in my_orders]

plan = r.solve(json.dumps({"vehicles": vehicles, "jobs": jobs}), time_limit_s=5.0)
```

Every driver starts and ends at their own depot, deliveries land on their
scheduled day, and each route stays inside the 8-hour shift. Drop a job's
`time_windows` to let the optimizer place it on whatever day is cheapest.

---

## Optional: serve caches on your LAN

A side feature for sharing prebuilt caches with another device (e.g. a
phone) so it can route without rebuilding:

```bash
mpee serve --data-dir data
```

---

## How it works

`pip install mpee` ships a compiled Rust extension (`mpee._mpee`) plus a
thin Python CLI. All routing and optimization run **in-process** — the map
cache is memory-mapped, so opening it is near-instant and peak RAM stays
low. Nothing is sent to a server.

**MPEE** stands for **Morten Punnerud-Engelstad Engine**. MIT licensed.
Source: [github.com/punnerud/mpee](https://github.com/punnerud/mpee).
