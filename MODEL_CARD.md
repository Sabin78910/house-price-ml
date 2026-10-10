# Model Card: House Price Regressor

## Overview
Scikit-learn pipeline: `StandardScaler` + `HistGradientBoostingRegressor` (`max_iter=300`), predicting median house value.
Training is deterministic: all randomness is controlled by `seed` (default 42); training twice with the same seed gives identical predictions (see `tests/test_model.py`).

## Data
- Online: California housing dataset (`sklearn.datasets.fetch_california_housing`, derived from the 1990 US census).
- Offline (`--offline`, used by tests/CI): synthetic data from `make_regression` (2000 samples, 8 features, noise 10). Not representative of real housing.
- Split: 80/20 train/test, plus 5-fold shuffled cross-validation.

## Features
- California: the 8 dataset features (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude).
- Offline: 8 anonymous numeric features (`feature_0`..`feature_7`).
- Permutation importance is written to `importance.json`.

## Metrics
From `metrics.json` produced by `PYTHONPATH=src python -m house_price.model --offline` (synthetic data, seed 42):

| Metric | Value |
|---|---|
| r2 (test) | 0.950 |
| mae (test) | 28.45 |
| rmse (test) | 36.77 |
| interval_coverage (test, 90% interval) | 0.925 |
| cv_r2_mean | 0.953 |
| cv_r2_std | 0.005 |

`error_by_price_band.json` reports test-set `mae`, `bias` (mean of true − predicted) and `n` for low/mid/high terciles of predicted price. Positive bias means under-prediction in that band; a large `high` bias or MAE signals weak extremes.

Regenerate for real data by running without `--offline`; values will differ.

## Limitations
- Offline metrics reflect synthetic data only and say nothing about real-world accuracy.
- California data is from 1990 and block-group level; it does not generalise to other regions or current prices.
- Target values are capped in the original dataset; no fairness or bias analysis has been done.
- No hyperparameter tuning.
- Split-conformal 90% prediction intervals (`interval_halfwidth_`) are provided; `interval_coverage` is the empirical coverage on the test split and may deviate from 0.90 on small samples or under distribution shift.
