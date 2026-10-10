# House Price ML

Gradient-boosting regression model that predicts California house prices (scikit-learn pipeline). Tests use synthetic data, so CI never needs a download.

![CI](https://github.com/Sabin78910/house-price-ml/actions/workflows/ci.yml/badge.svg)

## Run (Mac Terminal / VS Code)
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
PYTHONPATH=src .venv/bin/python -m house_price.model          # real data
PYTHONPATH=src .venv/bin/python -m house_price.model --offline # synthetic
.venv/bin/pytest
```

## API / Output
Outputs `artifacts/model.joblib` (plus `model.joblib.sha256`, verified by `predict` and `serve` before loading; a mismatch is refused, a missing checksum file only warns) and `artifacts/metrics.json` (R², MAE, RMSE, interval coverage, CV R², plus `mdape` — median absolute percentage error — and `within_10pct`/`within_20pct` — share of test predictions within 10%/20% of the true price, all as fractions; zero-valued targets are excluded from these three). Also writes `artifacts/error_by_price_band.json` (per predicted-price tercile: `n`, `mae`, `bias`, plus 90% interval `coverage` as a fraction and `mean_rel_width`, the mean interval width divided by the mean prediction) and `artifacts/error_by_region.json` (test-split MAE, mean signed error and 90% interval coverage per latitude-tercile × longitude-tercile cell, up to 9; cells with fewer than 5 test rows report only `n`).

Batch prediction: `PYTHONPATH=src .venv/bin/python -m house_price.predict --csv input.csv --output out.csv` (8 numeric columns, header optional; output adds `prediction,lower,upper`; bad rows are reported as `line N: reason` on stderr and the exit code is non-zero).

## Automation (runs on GitHub, no laptop needed)
| Workflow | Trigger | What it does |
|---|---|---|
| CI | push / PR | lint, tests, pip-audit, trains offline model artifact |
| Weekly training | Sundays + manual | retrains on real data, posts metrics to the run summary |
| CodeQL | push / PR / weekly | security analysis |
| Dependabot | weekly | dependency update PRs |
