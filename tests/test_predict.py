import joblib
import pytest

from house_price.model import train
from house_price.predict import main, parse_features


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
    float(capsys.readouterr().out.strip())


def test_main_bad_count_exits(model_path):
    with pytest.raises(SystemExit):
        main(["--model", str(model_path), "--features", "1,2"])
