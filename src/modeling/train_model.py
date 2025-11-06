"""Train a regression model that scores practice sessions against match benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a linear regression model that maps session averages to a normalized performance score."
    )
    parser.add_argument(
        "--team-summary",
        type=Path,
        default=PROCESSED_DIR / "team_summary.json",
        help="Aggregated team metrics produced by team_metrics.py.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=MODELS_DIR / "performance_model.pkl",
        help="Where to persist the trained scikit-learn model.",
    )
    parser.add_argument(
        "--max-day-output",
        type=Path,
        default=PROCESSED_DIR / "max_day.json",
        help="Where to persist the match-day benchmark values.",
    )
    return parser.parse_args()


def normalize(value: float, max_value: float) -> float:
    value = max(value, 0)
    max_value = max(max_value, 1)
    return np.log1p(value) / np.log1p(max_value)


def load_team_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Team summary not found at {path}. Run team_metrics.py first.")
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def main() -> None:
    args = parse_args()
    teams = load_team_summary(args.team_summary)

    match_days = [entry for entry in teams.values() if entry["match_label"] and entry["match_label"].count("-") == 2]
    practices = [entry for entry in teams.values() if entry["match_label"] and entry["match_label"].count("-") != 2]

    if len(match_days) < 3:
        print("⚠️ Warning: Fewer than three match days available. Model quality may be limited.")

    if not match_days:
        raise ValueError("No match days detected. Cannot compute benchmark values.")

    max_day = {
        "average_pl": max(day["average_pl"] for day in match_days),
        "average_v": max(day["average_v"] for day in match_days),
        "average_a": max(day["average_a"] for day in match_days),
        "average_dec": min(day["average_dec"] for day in match_days),
        "average_distance": max(day["average_distance"] for day in match_days),
        "average_hsd": max(day["average_hsd"] for day in match_days),
    }

    X_match = [
        [
            normalize(day["average_pl"], max_day["average_pl"]),
            normalize(day["average_v"], max_day["average_v"]),
            normalize(day["average_a"], max_day["average_a"]),
            normalize(day["average_dec"], max_day["average_dec"]),
            normalize(day["average_distance"], max_day["average_distance"]),
            normalize(day["average_hsd"], max_day["average_hsd"]),
        ]
        for day in match_days
    ]
    y_match = [1.0] * len(match_days)

    X_practice = [
        [
            normalize(day["average_pl"], max_day["average_pl"]),
            normalize(day["average_v"], max_day["average_v"]),
            normalize(day["average_a"], max_day["average_a"]),
            normalize(day["average_dec"], max_day["average_dec"]),
            normalize(day["average_distance"], max_day["average_distance"]),
            normalize(day["average_hsd"], max_day["average_hsd"]),
        ]
        for day in practices
    ]

    X_match = np.array(X_match)
    X_practice = np.array(X_practice)
    X_practice = np.nan_to_num(X_practice, nan=0.0)

    if len(X_practice) > 0:
        kmeans = KMeans(n_clusters=min(3, len(X_practice)), n_init=10, random_state=42)
        practice_labels = kmeans.fit_predict(X_practice)
        practice_scores = np.interp(practice_labels, (practice_labels.min(), practice_labels.max()), (0.3, 0.7))
        X_train = np.vstack([X_match, X_practice])
        y_train = np.concatenate([y_match, practice_scores])
    else:
        X_train = X_match
        y_train = np.array(y_match)

    model = LinearRegression()
    model.fit(X_train, y_train)

    args.max_day_output.parent.mkdir(parents=True, exist_ok=True)
    args.max_day_output.write_text(json.dumps(max_day, indent=4), encoding="utf-8")

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    print(f"Model saved to {args.model_output}")
    print(f"Match-day benchmarks saved to {args.max_day_output}")


if __name__ == "__main__":
    main()
