import joblib
import pytest

from house_price.model import train
from house_price.predict import explain, main, parse_features


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    model, _ = train(offline=True)
    path = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    joblib.dump(model, path)
    return path


def test_parse_features_ok():
    assert parse_features("1,2,3,4,5,6,7,8.5") == [1, 2, 3, 4, 5, 6, 7, 8.5]


@pytest.mark.parametrize("raw", ["1,2,3", "1,2,3,4,5,6,7,8,9", "a,b,c,d,e,f,g,h"])
def test_parse_features_invalid(raw):
    with pytest.raises(ValueError):
        parse_features(raw)


def test_main_prints_prediction(model_path, capsys):
    main(["--model", str(model_path), "--features", "0,0,0,0,0,0,0,0"])
    float(capsys.readouterr().out.split()[0])


def test_main_bad_count_exits(model_path):
    with pytest.raises(SystemExit):
        main(["--model", str(model_path), "--features", "1,2"])


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "NaN"])
def test_parse_features_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        parse_features(f"1,2,3,4,5,6,7,{bad}")


@pytest.mark.parametrize("raw", ["1,2,3", "1,2,3,4,5,6,7,8,9"])
def test_parse_features_count_message(raw):
    with pytest.raises(ValueError, match="expected 8 features"):
        parse_features(raw)


def test_main_non_finite_exits(model_path):
    with pytest.raises(SystemExit):
        main(["--model", str(model_path), "--features", "1,2,3,4,5,6,7,nan"])


def test_main_prints_interval(model_path, capsys):
    main(["--model", str(model_path), "--features", "0,0,0,0,0,0,0,0"])
    out = capsys.readouterr().out
    assert "90% interval" in out


def test_train_stores_medians():
    model, _ = train(offline=True)
    assert len(model.train_medians_) == 8


def test_explain_top3_sorted_and_deterministic(model_path):
    model = joblib.load(model_path)
    feats = [3.0, -2.0, 1.0, 0.5, -1.5, 2.0, 0.0, 1.0]
    a = explain(model, feats)
    assert a == explain(model, feats)
    assert len(a) == 3
    effects = [abs(e) for _, e in a]
    assert effects == sorted(effects, reverse=True)
    assert all(0 <= i < 8 for i, _ in a)


def test_explain_feature_at_median_has_zero_effect(model_path):
    model = joblib.load(model_path)
    feats = [float(m) for m in model.train_medians_]
    assert all(e == 0 for _, e in explain(model, feats))


def test_main_explain_flag(model_path, capsys):
    main(["--model", str(model_path), "--features", "3,-2,1,0.5,-1.5,2,0,1", "--explain"])
    out = capsys.readouterr().out
    assert out.count("feature_") == 3


def test_main_no_explain_by_default(model_path, capsys):
    main(["--model", str(model_path), "--features", "3,-2,1,0.5,-1.5,2,0,1"])
    assert "feature_" not in capsys.readouterr().out
