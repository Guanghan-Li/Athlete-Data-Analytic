"""Predict normalized performance scores for a single training session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a session using the trained performance regression model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "performance_model.pkl",
        help="Path to the trained performance model.",
    )
    parser.add_argument(
        "--max-day",
        type=Path,
        default=PROCESSED_DIR / "max_day.json",
        help="Match-day benchmark JSON emitted by train_model.py.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Optional JSON file containing the metrics to score. If omitted, --metrics must be provided.",
    )
    parser.add_argument(
        "--metrics",
        nargs=6,
        type=float,
        metavar=("average_pl", "average_v", "average_a", "average_dec", "average_distance", "average_hsd"),
        help="Six numeric values describing the session. Ignored if --input-json is supplied.",
    )
    return parser.parse_args()


def normalize(value: float, max_value: float) -> float:
    value = max(value, 0)
    max_value = max(max_value, 1)
    return np.log1p(value) / np.log1p(max_value)


def load_inputs(args: argparse.Namespace) -> Dict[str, float]:
    if args.input_json:
        with args.input_json.open("r", encoding="utf-8") as source:
            return json.load(source)
    if args.metrics:
        keys = [
            "average_pl",
            "average_v",
            "average_a",
            "average_dec",
            "average_distance",
            "average_hsd",
        ]
        return dict(zip(keys, args.metrics))
    raise ValueError("Provide either --input-json or --metrics.")


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found at {args.model}. Run train_model.py first.")
    if not args.max_day.exists():
        raise FileNotFoundError(f"Benchmark max_day.json not found at {args.max_day}. Run train_model.py first.")

    model = joblib.load(args.model)
    max_day = json.loads(args.max_day.read_text(encoding="utf-8"))
    inputs = load_inputs(args)

    features = np.array(
        [
            [
                normalize(inputs["average_pl"], max_day["average_pl"]),
                normalize(inputs["average_v"], max_day["average_v"]),
                normalize(inputs["average_a"], max_day["average_a"]),
                normalize(inputs["average_dec"], max_day["average_dec"]),
                normalize(inputs["average_distance"], max_day["average_distance"]),
                normalize(inputs["average_hsd"], max_day["average_hsd"]),
            ]
        ]
    )

    score = float(model.predict(features)[0])
    clamped_score = max(0.0, min(score, 1.0))
    print(json.dumps({"predicted_score": round(clamped_score, 3)}, indent=2))


if __name__ == "__main__":
    main()
