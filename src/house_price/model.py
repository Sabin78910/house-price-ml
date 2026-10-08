"""Train and evaluate a house price regression model on the California housing dataset.

Uses a synthetic dataset when `offline=True` so tests and CI never need a download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(offline: bool = False, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if offline:
        x, y = make_regression(n_samples=2000, n_features=8, noise=10.0, random_state=seed)
        return x, y
    data = fetch_california_housing()
    return data.data, data.target


def build_model(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", HistGradientBoostingRegressor(max_iter=300, random_state=seed)),
        ]
    )


def train(offline: bool = False, seed: int = 42) -> tuple[Pipeline, dict[str, float]]:
    x, y = load_data(offline, seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    model = build_model(seed).fit(x_train, y_train)
    pred = model.predict(x_test)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_r2 = cross_val_score(build_model(seed), x, y, cv=cv, scoring="r2")
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="use synthetic data")
    parser.add_argument("--out", default="artifacts", help="output directory")
    args = parser.parse_args()

    model, metrics = train(offline=args.offline)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
