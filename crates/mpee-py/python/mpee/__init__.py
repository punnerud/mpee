"""MPEE — offline route calculations and optimization.

A self-contained Rust routing + VRP engine (dijeng + brooom) you drive
from Python. Download a map once, build a cache, then compute routes and
optimize multi-vehicle delivery plans entirely offline — no network, no
external service, no API keys.

Quick start::

    import mpee

    # Open a prebuilt cache (build one with `mpee build <map.osm.pbf>`).
    r = mpee.Router("greater-london.osm.pbf.pp", "greater-london.osm.pbf.ch")

    # Point-to-point driving route.
    leg = r.route(51.5080, -0.1281, 51.5138, -0.0984)
    print(leg["distance_km"], "km,", leg["duration_min"], "min")

    # Optimize a 50-stop, 5-vehicle delivery plan.
    plan = r.optimize(stops, vehicles=5, capacity=20, time_limit_s=5.0)
    print(plan["total_distance_km"], "km over", plan["vehicles_used"], "vehicles")

See the ``mpee`` command-line tool for the same operations from a shell.
"""

from ._mpee import Engine, Router  # noqa: F401

try:  # populated from package metadata when installed
    from importlib.metadata import version as _version

    __version__ = _version("mpee")
except Exception:  # pragma: no cover - source checkouts
    __version__ = "0.0.0"

__all__ = ["Router", "Engine", "__version__"]
