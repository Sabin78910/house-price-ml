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
