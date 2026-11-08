# House Price ML

A regression service that predicts California median house values and says how uncertain each prediction is. It trains a scikit-learn pipeline (`StandardScaler` + `HistGradientBoostingRegressor`) and wraps every prediction in a 90% interval from normalised split-conformal calibration, so the range widens with the predicted price. The model is served through a CLI, a batch CSV mode and a small HTTP API, and is checksum-verified before every load.

[![CI](https://github.com/Sabin78910/house-price-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Sabin78910/house-price-ml/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

- **Calibrated uncertainty:** each prediction comes with a 90% interval `ŷ ± ŷ·q`, where `q` is a finite-sample-corrected quantile of `|y − ŷ| / ŷ` on a held-out calibration split. The lower bound is clipped at 0.
- **Error analysis beyond one score:** R², MAE, RMSE, interval coverage, median absolute percentage error, the share of predictions within 10% and 20%, 5-fold CV R², error and coverage per predicted-price tercile, error per latitude × longitude cell, and permutation importance.
- **Promotion guard:** every run appends to `metrics_history.jsonl`. A new model that scores a lower R² than the saved one is refused (non-zero exit) unless `--force` is passed.
- **Integrity check:** training writes a SHA-256 file next to `model.joblib`. `predict` and `serve` refuse to load a model whose checksum does not match.
- **Three ways to predict:** a single-row CLI with optional top-3 feature effects, a batch CSV mode that reports bad rows by line number, and a JSON HTTP endpoint with a static demo page.
- **Input checks and drift warnings:** inputs must be exactly 8 finite numbers. A warning is raised for any feature more than 3 standard deviations from its training mean.
- **Offline mode:** `--offline` trains on synthetic data, so tests and CI never download anything.

## How it works

1. **Load data.** `fetch_california_housing` (8 features, 1990 US census), or with `--offline` a synthetic set from `make_regression` (2000 rows, 8 features, noise 10) shifted to positive prices.
2. **Split.** 80/20 train/test. The training part is split again 75/25 into a fit set and a conformal calibration set.
3. **Fit.** `StandardScaler` then `HistGradientBoostingRegressor(max_iter=300)`. All randomness comes from one seed (default 42).
4. **Calibrate.** Compute the relative half-width `q` on the calibration set and store it on the model. Store training medians, means and standard deviations for explanations and drift checks.
5. **Evaluate.** Score the test split, run 5-fold shuffled cross-validation on the full data and compute permutation importance.
6. **Save or refuse.** Append the run to the history file. Then save the model, its checksum and the JSON reports, or exit non-zero if the existing model has a higher R².
7. **Predict.** Verify the checksum, validate the input, predict, attach the interval, and add drift warnings and (optionally) feature effects.

## Results

These figures come from `MODEL_CARD.md`. They were produced by `python -m house_price.model --offline` on **synthetic data** (seed 42) and say nothing about real-world accuracy.

| Metric (test split unless noted) | Value |
|---|---|
| R² | 0.950 |
| MAE | 28.45 |
| RMSE | 36.77 |
| 90% interval coverage | 0.925 |
| CV R² mean (5-fold) | 0.953 |
| CV R² std | 0.005 |

The interval coverage figure was recorded before intervals became price-relative, so rerun training to refresh it. `tests/baseline.json` stores the R² baseline (0.95) that CI checks against.

No real-data metrics are committed. To produce them (this downloads the California dataset):

```bash
PYTHONPATH=src .venv/bin/python -m house_price.model --out artifacts
cat artifacts/metrics.json
```

The weekly training workflow runs the same command and posts `metrics.json` to the GitHub Actions run summary.

## Getting started

Prerequisites: Python 3.11 or later (CI uses 3.12).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

# Train
PYTHONPATH=src .venv/bin/python -m house_price.model             # California data (downloads)
PYTHONPATH=src .venv/bin/python -m house_price.model --offline   # synthetic data, no network

# Predict one row (8 comma-separated numbers), with the top 3 feature effects
PYTHONPATH=src .venv/bin/python -m house_price.predict --features "0,0,0,0,0,0,0,0" --explain

# Batch prediction: output adds prediction,lower,upper columns
PYTHONPATH=src .venv/bin/python -m house_price.predict --csv input.csv --output out.csv

# HTTP API and demo page on http://127.0.0.1:8000/
PYTHONPATH=src .venv/bin/python -m house_price.serve
curl -X POST http://127.0.0.1:8000/predict -d '{"features": [0,0,0,0,0,0,0,0]}'

# Tests and lint
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Training writes to `artifacts/`: `model.joblib`, `model.joblib.sha256`, `metrics.json`, `metrics_history.jsonl`, `error_by_price_band.json`, `error_by_region.json` and `importance.json`.

`POST /predict` returns `price`, `interval_low`, `interval_high`, the top-3 `factors` and any drift `warnings`. Malformed input gets a `400` with an error message.

## Project structure

| Path | Purpose |
|---|---|
| `src/house_price/model.py` | Data loading, training, conformal calibration, evaluation reports, checksummed save/load, training CLI |
| `src/house_price/predict.py` | Input parsing, single and batch CSV prediction, drift warnings, feature effects |
| `src/house_price/serve.py` | HTTP server (standard library `ThreadingHTTPServer`) for `POST /predict` and the demo page |
| `src/house_price/web.py` | Demo page HTML, CSS and JS, served as separate files under a strict Content-Security-Policy |
| `tests/` | pytest suite and `baseline.json` (R² quality gate) |
| `MODEL_CARD.md` | Model card: data, features, metrics, limitations, integrity |
| `.github/workflows/` | CI, weekly training and CodeQL workflows |
| `requirements.txt`, `requirements-dev.txt`, `pyproject.toml` | Runtime and dev dependencies, ruff and pytest configuration |

## Testing and CI

The pytest suite runs entirely on synthetic data. It covers:

- **Model:** accuracy, reproducibility with a fixed seed, and an R² regression gate (at most 0.02 below `tests/baseline.json`).
- **Metrics:** hand-computed checks for the per-band, per-region and relative-error metrics.
- **Intervals:** coverage, scaling with price and clipping at zero.
- **Training runs:** history logging, the promotion guard and `--force`.
- **Integrity:** tampered models are rejected by both `predict` and `serve`, and a missing checksum gives a warning.
- **CLI:** input validation and batch CSV handling (header, blank and bad rows).
- **HTTP server:** error responses, the CSP header, and no inline scripts or `innerHTML` on the demo page.

Workflows in `.github/workflows/`:

| Workflow | Trigger | Runs |
|---|---|---|
| `ci.yml` | Push and PR to `main`, manual | `ruff check`, `ruff format --check`, `pytest -q`, `pip-audit` on `requirements.txt`, offline training with the model uploaded as a build artefact |
| `train.yml` | Sundays 04:00 UTC, manual | Retrains on the California data, posts `metrics.json` to the run summary, uploads artefacts |
| `codeql.yml` | Push and PR to `main`, manual | CodeQL security analysis for Python |

## Design decisions and limitations

- **Offline data in CI.** Synthetic data keeps CI fast, deterministic and free of network access. The cost is that the committed metrics describe synthetic data only.
- **Normalised conformal intervals.** These give price-proportional ranges with a marginal (not per-house) 90% coverage guarantee. Coverage can drift from 0.90 on small samples or under distribution shift.
- **Simple feature effects.** Each feature is set to its training median and the change in prediction is reported. This is cheap and deterministic, but it ignores interactions and is not a SHAP-style attribution.
- **Limited integrity guarantee.** The SHA-256 check guards against corruption or accidental replacement. It does not stop someone who can replace both the model and its checksum. `joblib.load` should only be used on trusted files.
- **Data scope.** The California data is from 1990 at block-group level, with capped target values. No hyperparameter tuning or fairness analysis has been done. The demo page sliders use standardised values from −3 to 3, which suit the synthetic model rather than raw California units.

## License

MIT. See [LICENSE](LICENSE).

## Author

**Sabin Khanal**: [Portfolio](https://sabinkhanal13.com.np) · [LinkedIn](https://www.linkedin.com/in/sabin-khanal-950163326) · [GitHub](https://github.com/Sabin78910)
