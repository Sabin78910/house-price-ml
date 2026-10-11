# House Price ML
Gradient boosting price model with honest prediction ranges and factor explanations.

## Stack
Python 3.12 · scikit-learn pipeline · pandas · pytest · ruff. Tests use synthetic data, so CI never downloads.

## Commands
- Test: `pytest` · Lint: `ruff check .`
- Train: `python -m src.train` · Serve: `python -m src.serve`

## Architecture
- `src/features.py` feature engineering · `src/train.py` pipeline and metrics · `src/intervals.py` conformal
  prediction ranges · `src/serve.py` HTTP demo.
- Everything that touches the model goes through the sklearn Pipeline, so preprocessing can never leak.

## Model rules
- Report a prediction range, never a bare number. State the coverage of that range.
- Keep the quality gate: a trained model that misses the metric threshold must fail the build.
- Record metrics per run; never silently overwrite a model that scored better.

## Testing rules
- Tests use synthetic data and must run offline in under a minute.
- Run `pytest` and `ruff check .` before finishing.

## Security and git
- Never commit secrets, keystores, .env files or email addresses.
- Keep the Content-Security-Policy strict; add a host to connect-src only when a feature needs it.
- `main` is protected: branch, open a PR, merge when CI is green. One issue per PR, under ~300 lines.

## Done when
Lint clean, tests pass, CI green, PR opened with a three line summary.
