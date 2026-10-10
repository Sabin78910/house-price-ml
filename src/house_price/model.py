"""Train and evaluate a house price regression model on the California housing dataset.

Uses a synthetic dataset when `offline=True` so tests and CI never need a download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INTERVAL_ALPHA = 0.1  # 90% prediction interval
LAT_COL, LON_COL = 6, 7  # California housing feature order
MIN_REGION_ROWS = 5


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


def conformal_quantile(
    model: Pipeline, x_cal: np.ndarray, y_cal: np.ndarray, alpha: float
) -> float:
    """Split conformal half-width: finite-sample corrected (1 - alpha) quantile of |residuals|."""
    scores = np.abs(y_cal - model.predict(x_cal))
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def error_by_band(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    """MAE and mean signed error (true - pred) per tercile band of predicted value."""
    order = np.argsort(y_pred, kind="stable")
    result = {}
    for name, idx in zip(("low", "mid", "high"), np.array_split(order, 3), strict=True):
        err = y_true[idx] - y_pred[idx]
        result[name] = {
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "n": int(len(idx)),
        }
    return result


def error_by_region(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    min_n: int = MIN_REGION_ROWS,
) -> dict[str, dict[str, float]]:
    """MAE, mean signed error (true - pred) and interval coverage per latitude x longitude tercile.

    Terciles are computed on the given rows (latitude and longitude are columns 6 and 7).
    Cells with fewer than `min_n` rows report only `n`; empty cells are omitted.
    """

    def tercile(values: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.quantile(values, [1 / 3, 2 / 3]), values, side="left")

    lat, lon = tercile(x_test[:, LAT_COL]), tercile(x_test[:, LON_COL])
    result = {}
    for i, lat_name in enumerate(("south", "mid_lat", "north")):
        for j, lon_name in enumerate(("west", "mid_lon", "east")):
            idx = (lat == i) & (lon == j)
            n = int(idx.sum())
            if n == 0:
                continue
            if n < min_n:
                result[f"{lat_name}_{lon_name}"] = {"n": n}
                continue
            err = y_true[idx] - y_pred[idx]
            result[f"{lat_name}_{lon_name}"] = {
                "n": n,
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "coverage": float(
                    np.mean((y_true[idx] >= lower[idx]) & (y_true[idx] <= upper[idx]))
                ),
            }
    return result


def relative_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Median absolute percentage error and share of predictions within 10%/20% (fractions).

    Rows with a target of 0 have no defined percentage error and are excluded; if every
    target is 0, all three metrics are reported as 0.0.
    """
    mask = y_true != 0
    if not mask.any():
        return {"mdape": 0.0, "within_10pct": 0.0, "within_20pct": 0.0}
    ape = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
    return {
        "mdape": float(np.median(ape)),
        "within_10pct": float(np.mean(ape <= 0.10)),
        "within_20pct": float(np.mean(ape <= 0.20)),
    }


def importance_report(offline: bool = False, seed: int = 42) -> dict[str, float]:
    """Permutation importance (mean R2 drop) of a model evaluated on the test split."""
    x, y = load_data(offline, seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    model = build_model(seed).fit(x_train, y_train)
    result = permutation_importance(model, x_test, y_test, n_repeats=5, random_state=seed)
    return {f"feature_{i}": float(v) for i, v in enumerate(result.importances_mean)}


def train(offline: bool = False, seed: int = 42) -> tuple[Pipeline, dict[str, float]]:
    x, y = load_data(offline, seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x_train, y_train, test_size=0.25, random_state=seed
    )
    model = build_model(seed).fit(x_fit, y_fit)
    # Half-width of the 90% prediction interval, calibrated on the held-out split.
    model.interval_halfwidth_ = conformal_quantile(model, x_cal, y_cal, INTERVAL_ALPHA)
    model.train_medians_ = np.median(x_fit, axis=0)
    model.train_means_ = x_fit.mean(axis=0)
    model.train_stds_ = x_fit.std(axis=0)
    pred = model.predict(x_test)
    model.error_by_band_ = error_by_band(y_test, pred)
    hw = model.interval_halfwidth_
    model.error_by_region_ = error_by_region(x_test, y_test, pred, pred - hw, pred + hw)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_r2 = cross_val_score(build_model(seed), x, y, cv=cv, scoring="r2")
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "interval_coverage": float(np.mean(np.abs(y_test - pred) <= model.interval_halfwidth_)),
        **relative_error_metrics(y_test, pred),
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
    }
    return model, metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_model(model: Pipeline, path: Path) -> None:
    """Dump the model and write `<path>.sha256` next to it."""
    path = Path(path)
    joblib.dump(model, path)
    path.with_name(path.name + ".sha256").write_text(_sha256(path) + "\n")


def load_model(path: str | Path) -> Pipeline:
    """Load a model after verifying its SHA-256 checksum (warns if the checksum file is absent)."""
    path = Path(path)
    sum_path = path.with_name(path.name + ".sha256")
    if sum_path.exists():
        if _sha256(path) != sum_path.read_text().strip():
            raise ValueError(f"checksum mismatch for {path}: refusing to load")
    else:
        print(f"warning: {sum_path} not found; loading {path} unverified", file=sys.stderr)
    return joblib.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="use synthetic data")
    parser.add_argument("--out", default="artifacts", help="output directory")
    args = parser.parse_args()

    model, metrics = train(offline=args.offline)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_model(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "error_by_price_band.json").write_text(json.dumps(model.error_by_band_, indent=2))
    (out / "error_by_region.json").write_text(json.dumps(model.error_by_region_, indent=2))
    report = importance_report(offline=args.offline)
    (out / "importance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
