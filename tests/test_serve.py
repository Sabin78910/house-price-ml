import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from house_price.model import train
from house_price.serve import make_handler


@pytest.fixture(scope="module")
def base_url():
    model, _ = train(offline=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(model))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def post(url, body):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


def test_predict_ok(base_url):
    status, body = post(base_url + "/predict", {"features": [0.0] * 8})
    assert status == 200
    assert body["interval_low"] < body["price"] < body["interval_high"]


@pytest.mark.parametrize(
    "body",
    [
        {"features": [1, 2, 3]},
        {"features": ["a"] * 8},
        {"features": [True] * 8},
        {"features": [1e999] * 8},
        {"nope": 1},
        [1, 2],
        b"not json",
    ],
)
def test_predict_bad_input(base_url, body):
    status, resp = post(base_url + "/predict", body)
    assert status == 400
    assert "error" in resp


def test_unknown_path_404(base_url):
    status, _ = post(base_url + "/other", {"features": [0] * 8})
    assert status == 404


def test_predict_drift_warnings(base_url):
    _, normal = post(base_url + "/predict", {"features": [0.0] * 8})
    assert normal["warnings"] == []
    _, far = post(base_url + "/predict", {"features": [1000.0] + [0.0] * 7})
    assert far["warnings"][0]["feature"] == 0


def test_index_serves_form(base_url):
    with urllib.request.urlopen(base_url + "/") as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/html")
        html = resp.read().decode()
    assert "<form" in html
    assert "/predict" in html
    assert html.count('type="number"') == 8


def test_predict_returns_top_factors(base_url):
    _, body = post(base_url + "/predict", {"features": [1.0] * 8})
    assert len(body["factors"]) == 3
    assert set(body["factors"][0]) == {"feature", "effect"}


def fetch(url):
    with urllib.request.urlopen(url) as resp:
        return resp, resp.read().decode()


def test_index_has_strict_csp_and_no_inline_code(base_url):
    resp, html = fetch(base_url + "/")
    csp = resp.headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp
    assert "script-src 'self'" in csp
    assert "style-src 'self'" in csp
    assert "unsafe-inline" not in csp
    assert "<style" not in html
    assert " style=" not in html
    assert "<script src=" in html
    assert "<script>" not in html
    assert "onclick" not in html


def test_index_has_sliders_region_picker_and_chart_slots(base_url):
    _, html = fetch(base_url + "/")
    assert html.count('type="range"') == 8
    assert html.count('type="number"') == 8
    assert html.count("data-lat=") == 9
    for slot in ('id="range-bar"', 'id="waterfall"', 'id="price"'):
        assert slot in html


@pytest.mark.parametrize(
    ("path", "ctype", "needle"),
    [
        ("/app.js", "text/javascript", "requestAnimationFrame"),
        ("/app.css", "text/css", "prefers-reduced-motion"),
    ],
)
def test_static_assets(base_url, path, ctype, needle):
    resp, body = fetch(base_url + path)
    assert resp.headers["Content-Type"].startswith(ctype)
    assert needle in body
    assert "Content-Security-Policy" in resp.headers


def test_js_avoids_innerhtml(base_url):
    _, js = fetch(base_url + "/app.js")
    assert "innerHTML" not in js
