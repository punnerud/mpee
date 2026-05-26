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
mpee build data/greater-london-latest.osm.pbf
```

After this you are fully offline. The cache is reusable forever (until you
want fresher map data).

## 2. Route from A to B

```bash
mpee route 51.5080,-0.1281 51.5138,-0.0984 --cache data/greater-london.osm.pbf
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
    --cache data/greater-london.osm.pbf
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

r = mpee.Router("data/greater-london.osm.pbf.pp",
                "data/greater-london.osm.pbf.ch")

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
(`profile` is `car` | `bicycle` | `foot`).

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
