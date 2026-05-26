# mpee cache server

> **This is not the `mpee` Python library.** The pip package (`pip install
> mpee`, source in [`crates/mpee-py/`](../crates/mpee-py/)) is the routing/VRP
> library + CLI. This `python/` folder is a small stdlib-only HTTP server that
> shares prebuilt caches over the LAN. The packaged `mpee serve` does the same
> thing and is the recommended way — this standalone script predates it.

Serves prebuilt CH (Contraction Hierarchy) caches over local Wi-Fi to the
iPhone app, skipping the 5–30 minute on-device CH build.

## Quick start

```bash
cd python
./setup.sh          # one-time: creates venv (Python 3.12)
./run.sh            # starts the server, prints the URL
```

The server prints something like:

```
mpee cache server
  data dir: /Users/.../mpee/data
  listening on http://192.168.1.218:8033/
  available regions:
    - greater-london.osm.pbf (223 MB)
    - greater-london.osm.pbf.bicycle (207 MB)
    - england.osm.pbf (3.18 GB)

Enter http://192.168.1.218:8033/ in the iPhone app's 'Local server' field.
```

In the iPhone app:

1. Select the **Local server** tab in the source picker (default).
2. Paste the printed URL into the text field.
3. Tap the refresh icon → the picker fills with available regions.
4. Pick a region → **Download prebuilt cache**.
5. The phone streams `.pp` + `.ch` directly into `mpe_load_ch`, ready
   for routing in seconds. No CH build runs on the device.

## Endpoints

| Path                  | Returns                                         |
|-----------------------|-------------------------------------------------|
| `GET /`               | HTML status page with the URL and region list   |
| `GET /regions`        | JSON: `[{name, pp_file, ch_file, pp_size, ch_size, total_size}]` |
| `GET /cache/<file>`   | Raw byte stream of the named cache file         |

The server is stdlib-only (no pip deps). The venv exists so we can pin
a Python 3.12 interpreter and add real deps later (FastAPI, etc.)
without breaking the script.

## Generating new caches

The server lists every `<name>.pp` + `<name>.ch` pair it finds in
`../data/`. To add a region, build it on the Mac with `bench_pp` and
`bench_ch` (see `crates/dijeng/README.md`) and the server picks it up
automatically — no restart required for `/regions` listing, but the
phone may have cached an older listing.

## Why a server at all?

The on-device CH build (`crates/dijeng/src/ch.rs`) is the slowest
phase by far — 5–10× slower on iPhone than on M3 due to single-thread
CH-contraction. For small bbox extracts (Oslo central, ~100 km²) the
build finishes in seconds and the server is unnecessary. For full
country/region caches, downloading a prebuilt `.ch` is dramatically
faster than rebuilding it.

In the long run this should be a hosted CDN endpoint serving daily-
updated caches. For now the local Python server lets development
iterate without rebuilding on every iPhone test.
