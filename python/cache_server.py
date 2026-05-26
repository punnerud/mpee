"""Serve prebuilt CH caches over local Wi-Fi to the mpee iPhone app.

Run from the project root:

    cd python
    ./setup.sh         # one-time: create venv
    ./run.sh           # start the server (prints the URL to enter on the phone)

Then point the iPhone app's "Local server" source at the printed URL.
The phone downloads `*.pp` + `*.ch` for the chosen region and skips the
~10 minute CH build that on-device construction would otherwise need.

Endpoints
---------
- ``GET /``            — small HTML status page with copy-pastable URL.
- ``GET /regions``     — JSON ``[{name, pp_size, ch_size, total_size}, …]``.
- ``GET /cache/<name>``— raw byte stream of a single cache file.

Threaded request handling so a long download doesn't block ``/regions``
while the phone is browsing. Range requests are intentionally NOT
implemented — iOS retries the whole file on a dropped connection,
which is fine on local Wi-Fi.
"""

from __future__ import annotations

import http.server
import json
import os
import socket
import socketserver
import sys
from pathlib import Path
from urllib.parse import unquote

DATA_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()
PORT = int(os.environ.get("MPE_CACHE_PORT", "8033"))
CHUNK_BYTES = 256 * 1024  # 256 KiB — keeps wfile.write latency low on slow links


def list_regions() -> list[dict]:
    """Find paired ``<name>.pp`` + ``<name>.ch`` files.

    A region is "available" iff both halves are present. The pair is
    keyed by the basename without the trailing extension — same as the
    on-disk convention used by ``bench_pp`` / ``bench_ch``.
    """
    regions: list[dict] = []
    for ch in sorted(DATA_DIR.glob("*.ch")):
        base = ch.name[:-3]  # strip ".ch"
        pp = DATA_DIR / f"{base}.pp"
        if not pp.exists():
            continue
        pp_size = pp.stat().st_size
        ch_size = ch.stat().st_size
        regions.append(
            {
                "name": base,
                "pp_file": pp.name,
                "ch_file": ch.name,
                "pp_size": pp_size,
                "ch_size": ch_size,
                "total_size": pp_size + ch_size,
            }
        )
    return regions


class CacheHandler(http.server.BaseHTTPRequestHandler):
    # http.server's default request log is noisy; we override it so the
    # console only shows our own messages.
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[{self.address_string()}] {fmt % args}\n")

    # ----- routes ----------------------------------------------------------

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._serve_status_page()
            return
        if path == "/regions":
            self._serve_regions()
            return
        if path.startswith("/cache/"):
            self._serve_cache_file(unquote(path[len("/cache/"):]))
            return
        self.send_error(404, "unknown path")

    def _serve_status_page(self) -> None:
        regions = list_regions()
        rows = "\n".join(
            f"<tr><td>{r['name']}</td><td>{fmt_mb(r['pp_size'])}</td>"
            f"<td>{fmt_mb(r['ch_size'])}</td><td>{fmt_mb(r['total_size'])}</td></tr>"
            for r in regions
        )
        html = f"""<!doctype html>
<title>mpee cache server</title>
<style>body{{font-family:-apple-system,sans-serif;margin:2em;max-width:48em}}
table{{border-collapse:collapse}}td,th{{padding:.25em .75em;border:1px solid #ccc;text-align:left}}
code{{background:#f4f4f4;padding:.1em .4em;border-radius:3px}}</style>
<h1>mpee cache server</h1>
<p>Listening on <code>{get_local_ip()}:{PORT}</code> — enter this URL in the
iPhone app's "Local server" source field.</p>
<p>Data directory: <code>{DATA_DIR}</code></p>
<h2>Available regions ({len(regions)})</h2>
<table><tr><th>Region</th><th>.pp</th><th>.ch</th><th>Total</th></tr>{rows}</table>
<h2>API</h2>
<ul>
  <li><code>GET /regions</code> — JSON listing</li>
  <li><code>GET /cache/&lt;filename&gt;</code> — raw bytes</li>
</ul>
"""
        self._send_bytes(html.encode(), "text/html; charset=utf-8")

    def _serve_regions(self) -> None:
        payload = json.dumps(list_regions()).encode()
        self._send_bytes(payload, "application/json")

    def _serve_cache_file(self, name: str) -> None:
        # Defence against path traversal: reject anything with separators
        # or relative parts. Only flat filenames inside DATA_DIR are
        # allowed.
        if "/" in name or ".." in name or name.startswith("."):
            self.send_error(400, "bad filename")
            return
        path = DATA_DIR / name
        if not path.is_file():
            self.send_error(404, f"no such cache file: {name}")
            return
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        # Allow the iPhone app to read the filename back if it wants;
        # not load-bearing but handy when debugging with curl.
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.end_headers()
        sent = 0
        with path.open("rb") as f:
            while True:
                chunk = f.read(CHUNK_BYTES)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Phone gave up — log once and abort cleanly.
                    sys.stderr.write(
                        f"[abort] client disconnected at {sent}/{size} bytes\n"
                    )
                    return
                sent += len(chunk)

    # ----- helpers ---------------------------------------------------------

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def fmt_mb(n: int) -> str:
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 ** 3):.2f} GB"
    return f"{n / (1024 ** 2):.1f} MB"


def get_local_ip() -> str:
    """Best-effort: the IP the phone would dial.

    Opens a UDP socket to a public address and reads the local end of
    the kernel's chosen route. No packets are actually sent — the
    socket is closed before any data leaves.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    # Daemon threads so Ctrl-C exits immediately even mid-download.
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    if not DATA_DIR.is_dir():
        sys.exit(f"data directory not found: {DATA_DIR}")
    regions = list_regions()
    ip = get_local_ip()
    print(f"\nmpee cache server")
    print(f"  data dir: {DATA_DIR}")
    print(f"  listening on http://{ip}:{PORT}/")
    print(f"  also reachable as http://localhost:{PORT}/")
    if regions:
        print(f"  available regions:")
        for r in regions:
            print(f"    - {r['name']} ({fmt_mb(r['total_size'])})")
    else:
        print(f"  WARNING: no .pp/.ch pairs found in {DATA_DIR}")
    print(f"\nEnter http://{ip}:{PORT}/ in the iPhone app's 'Local server' field.\n")
    with ThreadingServer(("0.0.0.0", PORT), CacheHandler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")


if __name__ == "__main__":
    main()
