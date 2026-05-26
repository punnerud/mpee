"""`mpee` — offline route calculations and optimization from the shell.

Subcommands:
    route      point-to-point driving route (distance + time)
    optimize   multi-vehicle delivery optimization (VRP) over your own stops
    download   fetch an OSM extract from Geofabrik
    build      preprocess a .osm.pbf into a routable .pp + .ch cache
    serve      (optional) serve prebuilt caches over the local network

Everything runs offline against a local cache — no API keys, no network
except the one-time `download`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _latlon(s: str) -> tuple[float, float]:
    """Parse a "lat,lon" string into a (lat, lon) float tuple."""
    try:
        lat, lon = (float(x) for x in s.replace(" ", "").split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected LAT,LON, got {s!r}")
    return (lat, lon)


def _resolve_cache(args) -> tuple[str, str]:
    """Return (pp_path, ch_path) from --cache PREFIX or explicit --pp/--ch."""
    if args.pp and args.ch:
        return args.pp, args.ch
    if args.cache:
        return f"{args.cache}.pp", f"{args.cache}.ch"
    raise SystemExit("error: give --cache PREFIX (expects PREFIX.pp + PREFIX.ch) "
                     "or both --pp and --ch")


def _load_stops(path: str) -> list[tuple[float, float]]:
    """Read stops from a file: JSON array of [lat, lon], or one 'lat,lon'
    per line (blank lines and # comments ignored)."""
    text = Path(path).read_text()
    text_stripped = text.lstrip()
    if text_stripped.startswith("["):
        return [(float(a), float(b)) for a, b in json.loads(text)]
    stops = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        lat, lon = (float(x) for x in line.split(","))
        stops.append((lat, lon))
    return stops


def _add_cache_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache", metavar="PREFIX",
                   help="cache prefix; loads PREFIX.pp + PREFIX.ch")
    p.add_argument("--pp", help="explicit .pp cache path")
    p.add_argument("--ch", help="explicit .ch cache path")


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def cmd_route(args) -> int:
    from . import Router
    pp, ch = _resolve_cache(args)
    r = Router(pp, ch)
    leg = r.route(args.frm[0], args.frm[1], args.to[0], args.to[1],
                  geometry=args.geometry)
    if args.json:
        print(json.dumps(leg))
        return 0
    print(f"distance: {leg['distance_km']:.2f} km")
    print(f"duration: {leg['duration_min']:.1f} min")
    print(f"from (snapped): {leg['source_snapped']}")
    print(f"to   (snapped): {leg['destination_snapped']}")
    if args.geometry:
        print(f"geometry: {len(leg['geometry'])} points")
    return 0


def cmd_optimize(args) -> int:
    from . import Router
    pp, ch = _resolve_cache(args)
    stops = _load_stops(args.stops)
    depot = _latlon(args.depot) if args.depot else None
    r = Router(pp, ch)
    plan = r.optimize(stops, vehicles=args.vehicles, capacity=args.capacity,
                      depot=depot, time_limit_s=args.time)
    if args.json:
        print(json.dumps(plan))
        return 0
    print(f"stops: {len(stops)}  vehicles used: {plan['vehicles_used']}/{args.vehicles}"
          f"  unassigned: {len(plan['unassigned'])}")
    print(f"total: {plan['total_distance_km']:.1f} km, "
          f"{plan['total_duration_min']:.0f} min  (solved in {plan['solve_s']:.1f}s)")
    for rt in plan["routes"]:
        print(f"  vehicle {rt['vehicle_id']}: {rt['n_stops']} stops, "
              f"{rt['distance_km']:.1f} km, {rt['duration_min']:.0f} min")
    if plan["unassigned"]:
        from collections import Counter
        reasons = Counter(d["reason"] for d in plan.get("unassigned_detail", []))
        if reasons:
            print("  unassigned by reason: " +
                  ", ".join(f"{r}={c}" for r, c in reasons.items()))
    return 0


def cmd_download(args) -> int:
    import urllib.request

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.slug.strip("/")
    url = f"https://download.geofabrik.de/{slug}-latest.osm.pbf"
    dest = out_dir / f"{slug.rsplit('/', 1)[-1]}-latest.osm.pbf"
    print(f"downloading {url}\n  -> {dest}")

    def _progress(block, block_size, total):
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            sys.stdout.write(f"\r  {pct:3d}%  ({total/1e6:.0f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _progress)
    print("\ndone. Next:  mpee build", dest)
    return 0


def cmd_build(args) -> int:
    from . import Router
    info = Router.build(args.pbf, profile=args.profile,
                        progress=not args.quiet, force=args.force)
    if info["cached"]:
        print("reused existing cache (pass --force to rebuild)")
    else:
        print(f"done in {info['build_secs']:.1f}s  "
              f"({info['nodes']:,} nodes, {info['edges']:,} edges)")
    print(f"  pp: {info['pp_path']}")
    print(f"  ch: {info['ch_path']}")
    print(f"route with:  mpee route LAT,LON LAT,LON --cache {info['ch_path'][:-3]}")
    return 0


def cmd_serve(args) -> int:
    from . import server
    return server.serve(data_dir=args.data_dir, port=args.port)


# --------------------------------------------------------------------------
# argument parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mpee",
        description="MPEE — offline route calculations and optimization.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("route", help="point-to-point driving route")
    pr.add_argument("frm", type=_latlon, metavar="FROM_LAT,LON")
    pr.add_argument("to", type=_latlon, metavar="TO_LAT,LON")
    pr.add_argument("--geometry", action="store_true", help="include path geometry")
    pr.add_argument("--json", action="store_true", help="emit JSON")
    _add_cache_args(pr)
    pr.set_defaults(func=cmd_route)

    po = sub.add_parser("optimize", help="multi-vehicle delivery optimization (VRP)")
    po.add_argument("--stops", required=True,
                    help="file with stops: JSON [[lat,lon],...] or 'lat,lon' per line")
    po.add_argument("--vehicles", type=int, default=1)
    po.add_argument("--capacity", type=int, default=1_000_000)
    po.add_argument("--depot", help="LAT,LON (default: centroid of stops)")
    po.add_argument("--time", type=float, default=5.0, help="solve time budget (s)")
    po.add_argument("--json", action="store_true", help="emit JSON")
    _add_cache_args(po)
    po.set_defaults(func=cmd_optimize)

    pd = sub.add_parser("download", help="fetch an OSM extract from Geofabrik")
    pd.add_argument("slug", help="Geofabrik path, e.g. europe/great-britain/england/greater-london")
    pd.add_argument("--out-dir", default="data")
    pd.set_defaults(func=cmd_download)

    pb = sub.add_parser("build", help="preprocess a .osm.pbf into a routable cache")
    pb.add_argument("pbf", help="path to the .osm.pbf")
    # Profile is an optional positional (matches bench_pp/bench_ch), so both
    # `mpee build x.osm.pbf` and `mpee build x.osm.pbf bicycle` work.
    pb.add_argument("profile", nargs="?", default="car",
                    choices=["car", "bicycle", "foot"], help="routing profile (default: car)")
    pb.add_argument("--quiet", "-q", action="store_true", help="suppress build progress output")
    pb.add_argument("--force", action="store_true", help="rebuild even if a cache already exists")
    pb.set_defaults(func=cmd_build)

    ps = sub.add_parser("serve", help="serve prebuilt caches over the local network")
    ps.add_argument("--data-dir", default="data")
    ps.add_argument("--port", type=int, default=8033)
    ps.set_defaults(func=cmd_serve)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
