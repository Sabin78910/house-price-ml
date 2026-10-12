from house_price.model import build_model, load_data, train


def test_offline_training_is_accurate():
    _, metrics = train(offline=True)
    assert metrics["r2"] > 0.8


def test_model_predicts_shape():
    import numpy as np

    model = build_model().fit(np.random.rand(50, 8), np.random.rand(50))
    assert model.predict(np.random.rand(3, 8)).shape == (3,)


def test_metrics_include_cv_r2():
    _, metrics = train(offline=True)
    assert {"cv_r2_mean", "cv_r2_std"} <= metrics.keys()
    assert metrics["cv_r2_mean"] > 0.8
    assert metrics["cv_r2_std"] >= 0


def test_importance_report_has_8_entries(tmp_path):
    import json
    import sys

    from house_price.model import main

    sys.argv = ["model", "--offline", "--out", str(tmp_path)]
    main()
    report = json.loads((tmp_path / "importance.json").read_text())
    assert len(report) == 8
    assert all(isinstance(v, float) for v in report.values())


def test_training_is_reproducible_with_same_seed():
    import numpy as np

    x, _ = load_data(offline=True)
    model_a, metrics_a = train(offline=True, seed=7)
    model_b, metrics_b = train(offline=True, seed=7)
    np.testing.assert_array_equal(model_a.predict(x), model_b.predict(x))
    assert metrics_a == metrics_b


def test_interval_coverage_on_test_split():
    import numpy as np
    from sklearn.model_selection import train_test_split

    model, _ = train(offline=True)
    x, y = load_data(offline=True)
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    half = model.interval_halfwidth_
    assert half > 0
    coverage = np.mean(np.abs(y_test - model.predict(x_test)) <= half)
    assert 0.85 <= coverage <= 0.95


def test_metrics_report_interval_coverage(tmp_path):
    import json
    import sys

    from house_price.model import main

    _, metrics = train(offline=True)
    assert 0.85 <= metrics["interval_coverage"] <= 0.95
    assert metrics == train(offline=True)[1]

    sys.argv = ["model", "--offline", "--out", str(tmp_path)]
    main()
    written = json.loads((tmp_path / "metrics.json").read_text())
    assert written["interval_coverage"] == metrics["interval_coverage"]


def test_metrics_json_has_required_keys(tmp_path):
    import json
    import sys

    from house_price.model import main

    sys.argv = ["model", "--offline", "--out", str(tmp_path)]
    main()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert {"mae", "rmse", "r2", "cv_r2_mean"} <= metrics.keys()
    assert metrics["rmse"] >= metrics["mae"] > 0


def test_r2_not_below_committed_baseline():
    import json
    from pathlib import Path

    baseline = json.loads((Path(__file__).parent / "baseline.json").read_text())
    _, metrics = train(offline=True)
    assert metrics["r2"] >= baseline["r2"] - 0.02


def test_error_by_band_known_values():
    import numpy as np

    from house_price.model import error_by_band

    y_pred = np.arange(9, dtype=float)
    # low band over-predicted by 1, mid exact, high under-predicted by 2
    y_true = y_pred + np.array([-1, -1, -1, 0, 0, 0, 2, 2, 2])
    result = error_by_band(y_true, y_pred)
    assert list(result) == ["low", "mid", "high"]
    assert result["low"] == {"mae": 1.0, "bias": -1.0, "n": 3}
    assert result["mid"] == {"mae": 0.0, "bias": 0.0, "n": 3}
    assert result["high"] == {"mae": 2.0, "bias": 2.0, "n": 3}


def test_error_by_band_file_written_offline(tmp_path):
    import json
    import sys

    from house_price.model import main

    sys.argv = ["model", "--offline", "--out", str(tmp_path)]
    main()
    written = json.loads((tmp_path / "error_by_price_band.json").read_text())
    assert list(written) == ["low", "mid", "high"]
    assert sum(b["n"] for b in written.values()) == 400  # 20% of 2000
    sys.argv = ["model", "--offline", "--out", str(tmp_path / "b")]
    main()
    assert json.loads((tmp_path / "b" / "error_by_price_band.json").read_text()) == written


def test_relative_error_metrics_hand_computed():
    import numpy as np

    from house_price.model import relative_error_metrics

    y_true = np.array([100.0, 200.0, 400.0, 1000.0])
    y_pred = np.array([105.0, 150.0, 400.0, 1500.0])  # errors: 5%, 25%, 0%, 50%
    m = relative_error_metrics(y_true, y_pred)
    assert m["mdape"] == 0.15  # median of 0, .05, .25, .5
    assert m["within_10pct"] == 0.5
    assert m["within_20pct"] == 0.5


def test_relative_error_metrics_ignore_zero_targets():
    import numpy as np

    from house_price.model import relative_error_metrics

    m = relative_error_metrics(np.array([0.0, 100.0]), np.array([5.0, 110.0]))
    assert m == {"mdape": 0.1, "within_10pct": 1.0, "within_20pct": 1.0}
    all_zero = relative_error_metrics(np.zeros(2), np.ones(2))
    assert all_zero == {"mdape": 0.0, "within_10pct": 0.0, "within_20pct": 0.0}


def test_train_reports_relative_error_metrics():
    _, metrics = train(offline=True)
    assert metrics["mdape"] >= 0
    assert 0 <= metrics["within_10pct"] <= metrics["within_20pct"] <= 1


def test_error_by_region_cells_and_small_cells():
    import numpy as np

    from house_price.model import error_by_region

    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=(n, 8))
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(size=n)
    result = error_by_region(x, y_true, y_pred, y_pred - 1.0, y_pred + 1.0)
    assert 1 <= len(result) <= 9
    assert sum(c["n"] for c in result.values()) == n
    for cell in result.values():
        assert set(cell) == {"n", "mae", "bias", "coverage"}
        assert 0.0 <= cell["coverage"] <= 1.0

    # tiny sample: every cell has < 5 rows, so only n is reported
    small = error_by_region(x[:12], y_true[:12], y_pred[:12], y_pred[:12] - 1, y_pred[:12] + 1)
    assert sum(c["n"] for c in small.values()) == 12
    assert all(c == {"n": c["n"]} for c in small.values())


def test_error_by_region_values_hand_computed():
    import numpy as np

    from house_price.model import error_by_region

    x = np.zeros((6, 8))  # one region: all lat/lon identical
    y_true = np.array([1.0, 2, 3, 4, 5, 10])
    y_pred = np.zeros(6)
    result = error_by_region(x, y_true, y_pred, y_pred - 4, y_pred + 4)
    (cell,) = result.values()
    assert cell == {"n": 6, "mae": 25 / 6, "bias": 25 / 6, "coverage": 4 / 6}


def test_error_by_region_file_written_offline(tmp_path):
    import json
    import sys

    from house_price.model import main

    sys.argv = ["model", "--offline", "--out", str(tmp_path)]
    main()
    written = json.loads((tmp_path / "error_by_region.json").read_text())
    assert sum(c["n"] for c in written.values()) == 400
    assert all(c.get("coverage", 0.0) <= 1.0 for c in written.values())


def test_error_by_band_interval_known_values():
    import numpy as np

    from house_price.model import error_by_band

    y_pred = np.arange(1, 10, dtype=float)
    y_true = y_pred.copy()
    y_true[[0, 1, 5, 8]] += 10  # two misses in low, one in mid, one in high
    result = error_by_band(y_true, y_pred, y_pred - 1, y_pred + 1)
    assert result["low"]["coverage"] == 1 / 3
    assert result["mid"]["coverage"] == 2 / 3
    assert result["high"]["coverage"] == 2 / 3
    assert result["low"]["mean_rel_width"] == 1.0  # width 2 / mean pred 2
    assert result["high"]["mean_rel_width"] == 2 / 8.0


def test_error_by_band_coverage_matches_global_offline():
    import json
    import sys
    import tempfile
    from pathlib import Path

    from house_price.model import main

    with tempfile.TemporaryDirectory() as d:
        sys.argv = ["model", "--offline", "--out", d]
        main()
        bands = json.loads((Path(d) / "error_by_price_band.json").read_text())
        metrics = json.loads((Path(d) / "metrics.json").read_text())
    n = sum(b["n"] for b in bands.values())
    assert all(0.0 <= b["coverage"] <= 1.0 and b["mean_rel_width"] > 0 for b in bands.values())
    weighted = sum(b["coverage"] * b["n"] for b in bands.values()) / n
    assert abs(weighted - metrics["interval_coverage"]) < 1e-9


def _run_main(out, *extra):
    import sys

    from house_price.model import main

    sys.argv = ["model", "--offline", "--out", str(out), *extra]
    main()


def _seed_better_model(out):
    import json

    out.mkdir(exist_ok=True)
    (out / "model.joblib").write_bytes(b"old model")
    (out / "model.joblib.sha256").write_text("old checksum\n")
    (out / "metrics.json").write_text(json.dumps({"r2": 5.0}))


def test_history_appended_each_run(tmp_path):
    import json

    _run_main(tmp_path)
    _run_main(tmp_path)
    lines = (tmp_path / "metrics_history.jsonl").read_text().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert {"timestamp", "seed", "offline", "r2", "mae", "rmse"} <= entry.keys()
    assert entry["offline"] is True
    assert entry["timestamp"].endswith("+00:00")


def test_worse_model_refused(tmp_path, capsys):
    import json

    import pytest

    _seed_better_model(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run_main(tmp_path)
    assert exc.value.code != 0
    assert "refusing" in capsys.readouterr().err
    assert (tmp_path / "model.joblib").read_bytes() == b"old model"
    assert (tmp_path / "model.joblib.sha256").read_text() == "old checksum\n"
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"r2": 5.0}
    assert len((tmp_path / "metrics_history.jsonl").read_text().splitlines()) == 1


def test_better_model_saved(tmp_path):
    import json

    _run_main(tmp_path)
    first = json.loads((tmp_path / "metrics.json").read_text())
    _run_main(tmp_path)  # equal score is accepted
    assert json.loads((tmp_path / "metrics.json").read_text()) == first
    (tmp_path / "metrics.json").write_text(json.dumps({"r2": -1.0}))
    _run_main(tmp_path)
    assert json.loads((tmp_path / "metrics.json").read_text())["r2"] > 0.8


def test_force_overrides_guard(tmp_path):
    import json

    _seed_better_model(tmp_path)
    _run_main(tmp_path, "--force")
    assert json.loads((tmp_path / "metrics.json").read_text())["r2"] < 5.0
    assert (tmp_path / "model.joblib").read_bytes() != b"old model"
