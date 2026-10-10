"""Predict a house price from a saved model and 8 comma-separated feature values."""

from __future__ import annotations

import argparse
import math
import sys

from house_price.model import load_model

N_FEATURES = 8
DRIFT_Z = 3.0  # warn when an input is further than this many training std devs from the mean


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


def explain(model, features: list[float], top: int = 3) -> list[tuple[int, float]]:
    """Top features by effect: prediction change when a feature is set to its training median."""
    base = float(model.predict([features])[0])
    effects = []
    for i, median in enumerate(model.train_medians_):
        alt = list(features)
        alt[i] = float(median)
        effects.append((i, base - float(model.predict([alt])[0])))
    effects.sort(key=lambda e: (-abs(e[1]), e[0]))
    return effects[:top]


def drift_warnings(model, features: list[float]) -> list[tuple[int, float]]:
    """(feature index, z-score) for inputs more than DRIFT_Z training std devs from the mean."""
    means = getattr(model, "train_means_", None)
    stds = getattr(model, "train_stds_", None)
    if means is None or stds is None:
        return []
    return [
        (i, abs(v - m) / s)
        for i, (v, m, s) in enumerate(zip(features, means, stds, strict=True))
        if s > 0 and abs(v - m) / s > DRIFT_Z
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/model.joblib", help="saved model path")
    parser.add_argument("--features", required=True, help="8 comma-separated numbers")
    parser.add_argument("--explain", action="store_true", help="show top 3 feature effects")
    args = parser.parse_args(argv)

    try:
        features = parse_features(args.features)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        model = load_model(args.model)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    price = float(model.predict([features])[0])
    half = getattr(model, "interval_halfwidth_", None)
    if half is None:
        print(price)
    else:
        print(f"{price} (90% interval: {price - half} to {price + half})")
    for i, z in drift_warnings(model, features):
        print(f"warning: feature_{i} is {z:.1f} std devs from training mean", file=sys.stderr)
    if args.explain:
        for i, effect in explain(model, features):
            print(f"feature_{i}: {effect:+.4f}")


if __name__ == "__main__":
    main()
