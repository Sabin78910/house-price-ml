"""Serve predictions over HTTP: POST /predict with {"features": [8 numbers]}."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib

from house_price.predict import parse_features


def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
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
                {"price": price, "interval_low": price - half, "interval_high": price + half},
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
    server = ThreadingHTTPServer((args.host, args.port), make_handler(joblib.load(args.model)))
    print(f"serving on http://{args.host}:{args.port}/predict")
    server.serve_forever()


if __name__ == "__main__":
    main()
