from house_price.model import build_model, train


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
