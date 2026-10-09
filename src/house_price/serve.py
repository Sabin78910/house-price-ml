"""Serve predictions over HTTP: POST /predict with {"features": [8 numbers]}."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib

from house_price.predict import drift_warnings, explain, parse_features

N_FEATURES = 8

_INPUTS = "".join(
    f'<label>Feature {i}<input type="number" step="any" name="f{i}" value="0" required></label>'
    for i in range(N_FEATURES)
)

PAGE = (
    """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>House price demo</title>
<style>
body{font-family:system-ui,sans-serif;background:#f4f6fb;color:#1c2333;margin:0;padding:2rem}
main{max-width:34rem;margin:auto}
form{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;background:#fff;padding:1.2rem;
border-radius:12px;box-shadow:0 2px 8px #0001}
label{display:flex;flex-direction:column;font-size:.85rem;gap:.25rem}
input{padding:.5rem;border:1px solid #c5ccdb;border-radius:6px;font-size:1rem}
button{grid-column:1/-1;padding:.7rem;border:0;border-radius:8px;background:#3b5bdb;
color:#fff;font-size:1rem;cursor:pointer}
#result{margin-top:1.2rem;background:#fff;padding:1.2rem;border-radius:12px;
box-shadow:0 2px 8px #0001}
#price{font-size:2rem;font-weight:700}
.bar{position:relative;height:10px;background:#dbe4ff;border-radius:5px;margin:.8rem 0 .3rem}
.bar i{position:absolute;top:-3px;width:4px;height:16px;background:#3b5bdb;border-radius:2px}
</style></head><body><main>
<h1>House price demo</h1>
<form id="f" action="/predict" method="post">"""
    + _INPUTS
    + """<button type="submit">Predict</button></form>
<section id="result" hidden>
<div id="price"></div>
<div class="bar" aria-label="90% range"><i id="marker"></i></div>
<div id="range"></div>
<h2>Top factors</h2><ol id="factors"></ol>
<p id="error" role="alert"></p>
</section>
<script>
const form=document.getElementById("f");
form.addEventListener("submit",async e=>{
  e.preventDefault();
  const features=[...form.querySelectorAll("input")].map(i=>Number(i.value));
  const r=await fetch("/predict",{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify({features})});
  const d=await r.json();
  document.getElementById("result").hidden=false;
  document.getElementById("error").textContent=d.error||"";
  if(d.error)return;
  const f=x=>x.toFixed(2);
  document.getElementById("price").textContent=f(d.price);
  document.getElementById("range").textContent=
    "90% range: "+f(d.interval_low)+" to "+f(d.interval_high);
  document.getElementById("marker").style.left="calc(50% - 2px)";
  document.getElementById("factors").innerHTML=d.factors.map(
    x=>"<li>Feature "+x.feature+": "+(x.effect>=0?"+":"")+f(x.effect)+"</li>").join("");
});
</script></main></body></html>"""
)


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
            if self.path not in ("/", "/index.html"):
                self._send(404, {"error": "not found"})
                return
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
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
    server = ThreadingHTTPServer((args.host, args.port), make_handler(joblib.load(args.model)))
    print(f"serving on http://{args.host}:{args.port}/predict")
    server.serve_forever()


if __name__ == "__main__":
    main()
