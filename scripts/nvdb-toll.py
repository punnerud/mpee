#!/usr/bin/env python3
"""Bygg en takstabell for norske bomstasjoner fra NVDB.

OSM vet *hvor* bomstasjonene er, men nesten aldri hva de koster — 6 % av de
norske portalene har en `charge`-tagg. NVDB (Statens vegvesen, objekttype 45)
har takst for liten og stor bil, rushtidstakst, innkrevningsretning og
timesregel for alle sammen.

Vi kobler de to geografisk: hver NVDB-stasjon matches til nærmeste
`barrier=toll_booth` / `highway=toll_gantry` i datasettet. Resultatet skrives
som `toll.tariff.tsv`, som ruteren leser ved oppstart.

    python3 scripts/nvdb-toll.py <datasett-mappe>
"""
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

API = ("https://nvdbapiles.atlas.vegvesen.no/vegobjekter/api/v4/vegobjekter/45"
       "?antall=1000&inkluder=egenskaper,lokasjon,geometri&srid=4326")
# NVDB-koordinater er innen noen få meter av veien; OSM-portalen kan stå litt
# ved siden av. 300 m er romslig nok til å ta høyde for det, og trangt nok til
# at to ulike stasjoner ikke forveksles.
MAX_MATCH_M = 300.0


def fetch(cache: Path):
    if cache.exists():
        return json.loads(cache.read_text())
    out, url = [], API
    while url:
        req = urllib.request.Request(
            url, headers={"Accept": "application/json", "X-Client": "mpee-planet"})
        d = json.load(urllib.request.urlopen(req, timeout=60))
        objs = d.get("objekter", [])
        out.extend(objs)
        nxt = d.get("metadata", {}).get("neste", {}).get("href")
        if not objs or not nxt:
            break
        url = nxt
        time.sleep(0.2)
    cache.write_text(json.dumps(out))
    return out


def haversine(a, b, c, d):
    r = 6_371_000.0
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = math.radians(c - a), math.radians(d - b)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def main(datadir: Path):
    stations = fetch(datadir / "nvdb-bomstasjoner.json")
    gates = []
    for line in (datadir / "toll.meta").read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        if len(f) >= 7:
            gates.append({"osm": int(f[0]), "lat": int(f[1]) / 1e7,
                          "lon": int(f[2]) / 1e7, "name": f[3]})
    # Bucket by 0.01° so the match is a local scan, not 463 × 1056 comparisons.
    cells = {}
    for g in gates:
        cells.setdefault((round(g["lat"], 2), round(g["lon"], 2)), []).append(g)

    rows, used, missed = [], set(), []
    for o in stations:
        e = {p["navn"]: p.get("verdi") for p in o["egenskaper"]}
        # WKT is "POINT Z (lat lon alt)" — and sometimes "POINT(lat lon)".
        m = re.match(r"POINT\s*Z?\s*\(\s*([-\d.]+)\s+([-\d.]+)",
                     o.get("geometri", {}).get("wkt", ""))
        if not m:
            missed.append((e.get("Navn bomstasjon"), "no geometry"))
            continue
        lat, lon = float(m.group(1)), float(m.group(2))
        best, bd = None, 1e9
        for dla in (-0.02, -0.01, 0, 0.01, 0.02):
            for dlo in (-0.02, -0.01, 0, 0.01, 0.02):
                for g in cells.get((round(lat + dla, 2), round(lon + dlo, 2)), []):
                    if g["osm"] in used:
                        continue
                    d = haversine(lat, lon, g["lat"], g["lon"])
                    if d < bd:
                        bd, best = d, g
        if not best or bd > MAX_MATCH_M:
            missed.append((e.get("Navn bomstasjon"), f"{bd:.0f} m to nearest gate"))
            continue
        used.add(best["osm"])
        rows.append([
            str(best["osm"]),
            str(e.get("Navn bomstasjon") or best["name"] or ""),
            "NOK",
            str(e.get("Takst liten bil") or 0),
            str(e.get("Innkrevningsretning") or ""),
            str(e.get("Navn bompengeanlegg") or ""),
            str(e.get("Timesregel, varighet") or 0),
            str(e.get("Rushtidstakst liten bil") or e.get("Takst liten bil") or 0),
        ])

    out = datadir / "toll.tariff.tsv"
    with out.open("w", encoding="utf-8") as f:
        f.write("# osm_id\tname\tcurrency\tcar\tdirection\tscheme\t"
                "hour_rule_min\trush_car\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"{len(rows)}/{len(stations)} NVDB stations matched to OSM gates "
          f"within {MAX_MATCH_M:.0f} m → {out}")
    for name, why in missed[:10]:
        print(f"  unmatched: {name} ({why})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
