import joblib
import pytest

from house_price.model import train
from house_price.predict import drift_warnings, explain, main, parse_features


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


def test_train_saves_feature_stats():
    model, _ = train(offline=True)
    assert len(model.train_means_) == 8
    assert len(model.train_stds_) == 8


def test_drift_warnings_flags_far_inputs(model_path):
    model = joblib.load(model_path)
    typical = [float(m) for m in model.train_means_]
    assert drift_warnings(model, typical) == []
    far = list(typical)
    far[2] += 4 * float(model.train_stds_[2])
    flagged = drift_warnings(model, far)
    assert [i for i, _ in flagged] == [2]
    assert flagged[0][1] > 3


def test_drift_warnings_boundary_and_missing_stats(model_path):
    model = joblib.load(model_path)
    edge = [float(m) for m in model.train_means_]
    edge[0] += 3 * float(model.train_stds_[0])
    assert drift_warnings(model, edge) == []

    class Bare:
        pass

    assert drift_warnings(Bare(), edge) == []


def test_main_prints_drift_warning(model_path, capsys):
    main(["--model", str(model_path), "--features", "1000,0,0,0,0,0,0,0"])
    assert "feature_0" in capsys.readouterr().err


GOOD = "0,0,0,0,0,0,0,0"
HEADER = ",".join(f"feature_{i}" for i in range(8))


def _run_csv(model_path, tmp_path, text):
    src = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    src.write_text(text)
    code = None
    try:
        main(["--model", str(model_path), "--csv", str(src), "--output", str(out)])
    except SystemExit as exc:
        code = exc.code
    return code, out.read_text().splitlines() if out.exists() else []


def test_csv_valid_file(model_path, tmp_path):
    code, lines = _run_csv(model_path, tmp_path, f"{GOOD}\n1,1,1,1,1,1,1,1\n")
    assert code is None
    assert len(lines) == 3
    assert lines[0].split(",")[-3:] == ["prediction", "lower", "upper"]
    assert len(lines[1].split(",")) == 11


def test_csv_header_and_blank_lines(model_path, tmp_path):
    code, lines = _run_csv(model_path, tmp_path, f"\n{HEADER}\n\n{GOOD}\n\n")
    assert code is None
    assert len(lines) == 2


def test_csv_bad_row_reported(model_path, tmp_path, capsys):
    code, lines = _run_csv(model_path, tmp_path, f"{GOOD}\n1,2,3\n{GOOD}\n")
    assert code != 0
    assert len(lines) == 3
    assert "line 2:" in capsys.readouterr().err


def test_csv_empty_file(model_path, tmp_path):
    code, lines = _run_csv(model_path, tmp_path, "")
    assert code is None
    assert len(lines) == 1


def test_csv_drift_warning_has_line(model_path, tmp_path, capsys):
    code, lines = _run_csv(model_path, tmp_path, f"{GOOD}\n1e9,0,0,0,0,0,0,0\n")
    assert code is None
    assert len(lines) == 3
    assert "line 2:" in capsys.readouterr().err


def test_csv_stdout_default(model_path, tmp_path, capsys):
    src = tmp_path / "in.csv"
    src.write_text(f"{GOOD}\n")
    main(["--model", str(model_path), "--csv", str(src)])
    assert capsys.readouterr().out.splitlines()[0].endswith("prediction,lower,upper")


def test_features_and_csv_exclusive(model_path, tmp_path):
    with pytest.raises(SystemExit):
        main(["--model", str(model_path), "--features", GOOD, "--csv", "x.csv"])
    with pytest.raises(SystemExit):
        main(["--model", str(model_path)])
