"""Predict a house price from a saved model and 8 comma-separated feature values."""

from __future__ import annotations

import argparse
import math

import joblib

N_FEATURES = 8


def parse_features(raw: str) -> list[float]:
    try:
        values = [float(v) for v in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"features must be numbers: {exc}") from exc
    if len(values) != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} features, got {len(values)}")
    bad = [i for i, v in enumerate(values) if not math.isfinite(v)]
    if bad:
        raise ValueError(f"features must be finite (no NaN/inf); bad positions: {bad}")
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/model.joblib", help="saved model path")
    parser.add_argument("--features", required=True, help="8 comma-separated numbers")
    args = parser.parse_args(argv)

    try:
        features = parse_features(args.features)
    except ValueError as exc:
        parser.error(str(exc))
    model = joblib.load(args.model)
    price = float(model.predict([features])[0])
    half = getattr(model, "interval_halfwidth_", None)
    if half is None:
        print(price)
    else:
        print(f"{price} (90% interval: {price - half} to {price + half})")


if __name__ == "__main__":
    main()
