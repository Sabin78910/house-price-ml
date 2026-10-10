"""Serve predictions over HTTP: POST /predict with {"features": [8 numbers]}."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from house_price.model import load_model
from house_price.predict import drift_warnings, explain, parse_features
from house_price.web import CSP, CSS, JS, PAGE


def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            assets = {
                "/": (PAGE, "text/html"),
                "/index.html": (PAGE, "text/html"),
                "/app.js": (JS, "text/javascript"),
                "/app.css": (CSS, "text/css"),
            }
            if self.path not in assets:
                self._send(404, {"error": "not found"})
                return
            text, ctype = assets[self.path]
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length))
                values = data.get("features") if isinstance(data, dict) else None
                if not isinstance(values, list) or any(
                    isinstance(v, bool) or not isinstance(v, int | float) for v in values
                ):
                    raise ValueError('body must be {"features": [numbers]}')
                features = parse_features(",".join(repr(float(v)) for v in values))
            except (ValueError, OverflowError) as exc:  # JSON errors are ValueErrors
                self._send(400, {"error": str(exc)})
                return
            price = float(model.predict([features])[0])
            half = getattr(model, "interval_halfwidth_", 0.0)
            self._send(
                200,
                {
                    "price": price,
                    "interval_low": price - half,
                    "interval_high": price + half,
                    "factors": [{"feature": i, "effect": e} for i, e in explain(model, features)],
                    "warnings": [
                        {"feature": i, "z_score": z} for i, z in drift_warnings(model, features)
                    ],
                },
            )

        def log_message(self, format, *args) -> None:  # noqa: A002
            pass

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/model.joblib", help="saved model path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        model = load_model(args.model)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model))
    print(f"serving on http://{args.host}:{args.port}/predict")
    server.serve_forever()


if __name__ == "__main__":
    main()
