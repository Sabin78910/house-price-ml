import joblib
import pytest

from house_price import model as model_mod
from house_price.model import load_model, save_model
from house_price.predict import main as predict_main


@pytest.fixture
def saved(tmp_path):
    m, _ = model_mod.train(offline=True)
    path = tmp_path / "model.joblib"
    save_model(m, path)
    return path


def test_save_writes_checksum(saved):
    assert saved.with_name("model.joblib.sha256").read_text().strip()


def test_valid_load(saved, capsys):
    assert load_model(saved) is not None
    assert capsys.readouterr().err == ""


def test_tampered_rejected(saved):
    with saved.open("ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_model(saved)


def test_missing_checksum_warns(saved, capsys):
    saved.with_name("model.joblib.sha256").unlink()
    assert load_model(saved) is not None
    assert "warning" in capsys.readouterr().err


def test_predict_cli_exits_nonzero_on_tamper(saved):
    with saved.open("ab") as f:
        f.write(b"x")
    with pytest.raises(SystemExit) as exc:
        predict_main(["--model", str(saved), "--features", "0,0,0,0,0,0,0,0"])
    assert exc.value.code != 0


def test_serve_refuses_tampered(saved):
    from house_price.serve import main as serve_main

    saved.write_bytes(b"junk")
    with pytest.raises(SystemExit) as exc:
        serve_main(["--model", str(saved), "--port", "0"])
    assert exc.value.code != 0
    assert joblib is not None
