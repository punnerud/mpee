"""Optional: serve prebuilt CH/PP caches over the local network.

A side feature — the focus of mpee is the offline CLI/library. This small
stdlib-only HTTP server hands prebuilt ``<name>.pp`` + ``<name>.ch`` pairs
from a data directory to another machine (e.g. a phone) so it can route
without rebuilding the cache itself.
"""

from __future__ import annotations

import http.server
import json
import socket
import socketserver
from pathlib import Path
from urllib.parse import unquote

_CHUNK = 256 * 1024


def _list_regions(data_dir: Path) -> list[dict]:
    regions = []
    for ch in sorted(data_dir.glob("*.ch")):
        base = ch.name[:-3]
        pp = data_dir / f"{base}.pp"
        if not pp.exists():
            continue
        regions.append({
            "name": base,
            "pp_file": pp.name, "ch_file": ch.name,
            "pp_size": pp.stat().st_size, "ch_size": ch.stat().st_size,
            "total_size": pp.stat().st_size + ch.stat().st_size,
        })
    return regions


def _local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def serve(data_dir: str = "data", port: int = 8033) -> int:
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"error: data dir {root} does not exist")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # quieter
            pass

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/regions":
                self._send_json(_list_regions(root))
                return
            if self.path.startswith("/cache/"):
                name = unquote(self.path[len("/cache/"):])
                f = (root / name).resolve()
                if root not in f.parents or not f.is_file():
                    self._send_json({"error": "not found"}, 404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(f.stat().st_size))
                self.end_headers()
                with f.open("rb") as fh:
                    while chunk := fh.read(_CHUNK):
                        self.wfile.write(chunk)
                return
            regions = _list_regions(root)
            rows = "".join(f"<li>{r['name']} ({r['total_size']/1e6:.0f} MB)</li>"
                           for r in regions) or "<li>(none — build one with `mpee build`)</li>"
            html = f"<h1>mpee cache server</h1><p>data: {root}</p><ul>{rows}</ul>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

    class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    url = f"http://{_local_ip()}:{port}/"
    print(f"mpee cache server\n  data dir: {root}\n  listening on {url}")
    for r in _list_regions(root):
        print(f"    - {r['name']} ({r['total_size']/1e6:.0f} MB)")
    with Threaded(("0.0.0.0", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0
