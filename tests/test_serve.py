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
