"""`mpee` — offline route calculations and optimization from the shell.

Subcommands:
    route      point-to-point driving route (distance + time)
    optimize   multi-vehicle delivery optimization (VRP) over your own stops
    reverse    reverse-geocode: nearest street name to a LAT,LON point
    geocode    forward-geocode: look up a street by name -> its LAT,LON
    crossing   intersection search: where two named streets cross
    download   fetch an OSM extract from Geofabrik
    build      preprocess a .osm.pbf into a routable .pp + .ch cache

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


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    import math
    r = 6_371_000.0
    la1, la2 = math.radians(a[0]), math.radians(b[0])
    dlat = math.radians(b[0] - a[0])
    dlon = math.radians(b[1] - a[1])
    h = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def _resolve_cache(args) -> tuple[str, str]:
    """Return (pp_path, ch_path) from --cache PREFIX or explicit --pp/--ch."""
    if args.pp and args.ch:
        return args.pp, args.ch
    if args.cache:
        return f"{args.cache}.pp", f"{args.cache}.ch"
    raise SystemExit("error: give --cache PREFIX (expects PREFIX.pp + PREFIX.ch) "
                     "or both --pp and --ch")


def _resolve_pp(args) -> str:
    """Return just the .pp path (geocoding needs no .ch): --cache PREFIX or --pp."""
    if getattr(args, "pp", None):
        return args.pp
    if args.cache:
        return f"{args.cache}.pp"
    raise SystemExit("error: give --cache PREFIX (uses PREFIX.pp + PREFIX.names) or --pp")


def _open_router(pp: str, ch: str | None = None):
    """Open a Router, turning load failures into a clean `error:` line instead
    of a raw traceback. `ch=None` opens a geocoding-only Router (skips .ch)."""
    from . import Router
    try:
        return Router(pp, ch) if ch is not None else Router(pp)
    except (RuntimeError, OSError, ValueError) as e:
        raise SystemExit(f"error: {e}")


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
                   help="cache prefix = the .osm.pbf path; loads PREFIX.pp + "
                        "PREFIX.ch (+ PREFIX.names for geocoding)")
    p.add_argument("--pp", help="explicit .pp cache path")
    p.add_argument("--ch", help="explicit .ch cache path")


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def cmd_route(args) -> int:
    pp, ch = _resolve_cache(args)
    r = _open_router(pp, ch)
    leg = r.route(args.frm[0], args.frm[1], args.to[0], args.to[1],
                  geometry=args.geometry)
    if args.json:
        print(json.dumps(leg))
        return 0
    src, dst = leg["source_snapped"], leg["destination_snapped"]
    snap_from = _haversine_m(args.frm, src)
    snap_to = _haversine_m(args.to, dst)
    print(f"distance: {leg['distance_km']:.2f} km")
    print(f"duration: {leg['duration_min']:.1f} min")
    print(f"from (snapped): ({src[0]:.5f}, {src[1]:.5f})  [{snap_from:.0f} m from input]")
    print(f"to   (snapped): ({dst[0]:.5f}, {dst[1]:.5f})  [{snap_to:.0f} m from input]")
    if snap_from > 500 or snap_to > 500:
        print("WARN: a point snapped >500 m — check LAT,LON order and cache coverage")
    if args.geometry:
        print(f"geometry: {len(leg['geometry'])} points")
    return 0


def cmd_optimize(args) -> int:
    pp, ch = _resolve_cache(args)
    stops = _load_stops(args.stops)
    depot = _latlon(args.depot) if args.depot else None
    r = _open_router(pp, ch)
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


def cmd_reverse(args) -> int:
    r = _open_router(_resolve_pp(args))   # geocoding-only: no .ch loaded
    if not r.has_names():
        raise SystemExit(
            f"error: no .names sidecar next to {pp} — rebuild the cache "
            "(`mpee build`) with this version to enable geocoding")
    name = r.reverse(args.point[0], args.point[1])
    if args.json:
        print(json.dumps({"name": name}))
        return 0
    print(name if name else "(no street name on the nearest road)")
    return 0


def cmd_geocode(args) -> int:
    r = _open_router(_resolve_pp(args))   # geocoding-only: no .ch loaded
    if not r.has_names():
        raise SystemExit(
            f"error: no .names sidecar next to {pp} — rebuild the cache "
            "(`mpee build`) with this version to enable geocoding")
    near = _latlon(args.near) if args.near else None
    hit = r.geocode(args.query, near=near)
    if hit is None:
        raise SystemExit(f"no street matching {args.query!r} found in this area")
    if args.json:
        print(json.dumps(hit))
        return 0
    print(hit["name"])
    print(f"{hit['lat']:.6f},{hit['lon']:.6f}")
    return 0


def cmd_crossing(args) -> int:
    r = _open_router(_resolve_pp(args))   # geocoding-only: no .ch loaded
    if not r.has_names():
        raise SystemExit(
            f"error: no .names sidecar next to {pp} — rebuild the cache "
            "(`mpee build`) with this version to enable geocoding")
    near = _latlon(args.near) if args.near else None
    hits = r.intersection(args.a, args.b, near=near, radius_km=args.radius_km)
    if not hits:
        raise SystemExit(
            f"no intersection of {args.a!r} and {args.b!r} found "
            "(unknown street, or they share no node)")
    if args.json:
        print(json.dumps(hits))
        return 0
    print(f"{args.a} × {args.b}: {len(hits)} match(es)")
    for h in hits:
        print(f"{h['lat']:.6f},{h['lon']:.6f}")
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

    prv = sub.add_parser("reverse", help="reverse-geocode: nearest street name to a point")
    prv.add_argument("point", type=_latlon, metavar="LAT,LON")
    prv.add_argument("--json", action="store_true", help="emit JSON")
    _add_cache_args(prv)
    prv.set_defaults(func=cmd_reverse)

    pg = sub.add_parser("geocode", help="forward-geocode: street name -> LAT,LON")
    pg.add_argument("query", help="street name (case-insensitive; substring matches)")
    pg.add_argument("--near", metavar="LAT,LON",
                    help="on a multi-city cache, return the match nearest this point")
    pg.add_argument("--json", action="store_true", help="emit JSON")
    _add_cache_args(pg)
    pg.set_defaults(func=cmd_geocode)

    pc = sub.add_parser("crossing", help="intersection search: where two streets cross")
    pc.add_argument("a", help="first street name")
    pc.add_argument("b", help="second street name")
    pc.add_argument("--near", metavar="LAT,LON",
                    help="sort crossings nearest-first to this point (disambiguates cities)")
    pc.add_argument("--radius-km", type=float, default=None,
                    help="with --near, keep only crossings within this many km")
    pc.add_argument("--json", action="store_true", help="emit JSON")
    _add_cache_args(pc)
    pc.set_defaults(func=cmd_crossing)

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

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
