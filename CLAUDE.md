# House Price ML
Purpose: House price regression model (scikit-learn).

## Commands
- Test: `.venv/bin/pytest -q`
- Lint: `.venv/bin/ruff check . && .venv/bin/ruff format --check .`
- Build: `PYTHONPATH=src .venv/bin/python -m house_price.model --offline`

## Architecture
src/house_price/ — keep training deterministic (fixed seeds); tests use offline synthetic data

## Rules
- Read only the files you need; do not scan the whole repo.
- Every behavior change needs a test. Run tests and lint before finishing.
- No new dependencies, permissions, or signing/secrets changes without asking.
- Never commit secrets, keystores, .env files.
- Keep PRs under ~300 changed lines; one issue per PR.
- Be concise: diffs plus a 3-line summary.
- If tests still fail after 3 attempts, stop and report the blocker.

## Definition of done
Lint clean, tests pass, CI green, short summary, PR opened as draft.
