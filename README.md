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
Outputs `artifacts/model.joblib` (plus `model.joblib.sha256`, verified by `predict` and `serve` before loading; a mismatch is refused, a missing checksum file only warns) and `artifacts/metrics.json` (R², MAE).

Batch prediction: `PYTHONPATH=src .venv/bin/python -m house_price.predict --csv input.csv --output out.csv` (8 numeric columns, header optional; output adds `prediction,lower,upper`; bad rows are reported as `line N: reason` on stderr and the exit code is non-zero).

## Automation (runs on GitHub, no laptop needed)
| Workflow | Trigger | What it does |
|---|---|---|
| CI | push / PR | lint, tests, pip-audit, trains offline model artifact |
| Weekly training | Sundays + manual | retrains on real data, posts metrics to the run summary |
| CodeQL | push / PR / weekly | security analysis |
| Dependabot | weekly | dependency update PRs |
