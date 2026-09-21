"""Serve the real Pages export with a deterministic model-worker endpoint.

Chromium nested workers bypass Playwright's script-routing interception. Serve
this protocol fixture at the HTTP boundary so no model/CDN access is required.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

WEB_ROOT = Path(__file__).resolve().parents[1]


class PagesHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        if urlsplit(path).path == "/nlp-semantic-worker.js":
            return str(WEB_ROOT / "e2e/fixtures/semantic-worker.js")
        return super().translate_path(path)


if __name__ == "__main__":
    handler = partial(PagesHandler, directory=str(WEB_ROOT / "out"))
    with ThreadingHTTPServer(("127.0.0.1", 4173), handler) as server:
        server.serve_forever()
