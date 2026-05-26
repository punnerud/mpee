"""mpee + Flask — Python serves the HTTP layer, Rust does the solving.

Why this exists: macOS Application Firewall asks per-binary permission
for non-loopback listeners. Python (Apple-signed) is already allowed,
so Flask can bind 0.0.0.0:8032 freely without prompting for every
mpee-viz rebuild. The Rust solver runs inside this same Python
process via `import mpee`, so the dataset lives in the same RAM
that brooom's solve_with_matrix just finished writing.

Run:
    cd crates/mpee-py
    python3 -m venv venv
    source venv/bin/activate
    pip install maturin flask
    maturin develop --release
    python3 python/app.py

Then point a phone at http://<laptop-ip>:8032/.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from flask import Flask, Response, send_file

import mpee  # the Rust extension built by `maturin develop`


HERE = pathlib.Path(__file__).resolve().parent
INDEX_PATH = HERE / "index.html"
REPO_ROOT = HERE.parent.parent.parent  # crates/mpee-py/python -> repo root


def main() -> int:
    p = argparse.ArgumentParser(description="Run the mpee Flask UI on top of the Rust solver.")
    p.add_argument("--region", default="london", choices=["london", "oslo", "manhattan", "paris"])
    p.add_argument("--n-jobs", type=int, default=500)
    p.add_argument("--n-vehicles", type=int, default=20)
    p.add_argument("--capacity", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ch", default=str(REPO_ROOT / "data" / "greater-london.osm.pbf.ch"))
    p.add_argument("--pp", default=str(REPO_ROOT / "data" / "greater-london.osm.pbf.pp"))
    p.add_argument("--time-limit-s", type=float, default=45.0)
    p.add_argument("--multi-start", type=int, default=1)
    p.add_argument(
        "--radius-km", type=float, default=0.0,
        help="If > 0, place jobs uniformly inside a disk of this radius "
             "around the city-centre depot (instead of the rectangular bbox).",
    )
    p.add_argument(
        "--max-travel-time-min", type=int, default=0,
        help="Cap per-vehicle route travel time (minutes). 0 = unbounded.",
    )
    p.add_argument(
        "--max-distance-km", type=float, default=0.0,
        help="Cap per-vehicle route distance (km). 0 = unbounded.",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8032)
    args = p.parse_args()

    if not os.path.exists(args.ch):
        print(f"error: CH cache not found at {args.ch}", file=sys.stderr)
        return 2
    if not os.path.exists(args.pp):
        print(f"error: PP cache not found at {args.pp}", file=sys.stderr)
        return 2

    # Spin up the Rust engine and kick off the solve in a background
    # thread (the Rust side handles the threading). Python keeps a
    # single reference to the Engine; the dataset stays in Rust RAM.
    engine = mpee.Engine()
    engine.start_solve(
        region=args.region,
        n_jobs=args.n_jobs,
        n_vehicles=args.n_vehicles,
        capacity=args.capacity,
        seed=args.seed,
        ch=args.ch,
        pp=args.pp,
        time_limit_s=args.time_limit_s,
        multi_start=args.multi_start,
        radius_km=args.radius_km,
        max_travel_time_s=args.max_travel_time_min * 60,
        max_distance_m=int(args.max_distance_km * 1000),
    )
    print(
        f"mpee started solving: region={args.region} n_jobs={args.n_jobs} "
        f"n_vehicles={args.n_vehicles} budget={args.time_limit_s}s",
        file=sys.stderr,
    )

    app = Flask(__name__)

    # CORS open — same as mpee-viz, so a separate UI host can hit us too.
    @app.after_request
    def cors(resp: Response) -> Response:
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.get("/")
    @app.get("/index.html")
    def index() -> Response:
        return send_file(INDEX_PATH, mimetype="text/html")

    @app.get("/api/health")
    def health() -> Response:
        return Response('{"ok":true}', mimetype="application/json")

    @app.get("/api/status")
    def status() -> Response:
        return Response(engine.get_status_json(), mimetype="application/json",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/dataset")
    def dataset() -> Response:
        body = engine.get_dataset_json()
        if body is None:
            # No iteration has completed yet — return the status with 202
            # so the browser knows to keep polling.
            return Response(engine.get_status_json(), mimetype="application/json",
                            status=202, headers={"Cache-Control": "no-store"})
        return Response(body, mimetype="application/json",
                        headers={"Cache-Control": "no-store"})

    print(f"flask listening on http://{args.host}:{args.port}/", file=sys.stderr)
    if args.host == "0.0.0.0":
        print(f"on a phone, point at  http://<laptop-ip>:{args.port}/", file=sys.stderr)
    # threaded=True so the long-polling browsers don't queue behind each other.
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
