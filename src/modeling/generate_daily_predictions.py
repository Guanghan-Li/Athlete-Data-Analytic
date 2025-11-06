"""Generate per-athlete and team-level prediction summaries for recent activities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the most recent activities using the trained performance model and emit structured summaries."
    )
    parser.add_argument(
        "--recent-activities",
        type=Path,
        default=PROCESSED_DIR / "recent_activities_with_names.json",
        help="JSON produced by fetch_recent_activities.py.",
    )
    parser.add_argument(
        "--max-day",
        type=Path,
        default=PROCESSED_DIR / "max_day.json",
        help="Benchmark metrics produced by train_model.py.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "performance_model.pkl",
        help="Trained regression model created by train_model.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "predicted_performance.json",
        help="Destination JSON for the prediction report.",
    )
    return parser.parse_args()


def normalize(x: float, max_x: float) -> float:
    x = max(x, 0)
    max_x = max(max_x, 1)
    return min(max(np.log1p(x) / np.log1p(max_x), 0), 1)


def load_inputs(args: argparse.Namespace) -> Tuple[Dict, Dict, joblib]:
    if not args.recent_activities.exists():
        raise FileNotFoundError(
            f"Recent activities file not found at {args.recent_activities}. Run fetch_recent_activities.py first."
        )
    if not args.max_day.exists():
        raise FileNotFoundError(f"max_day.json not found at {args.max_day}. Run train_model.py first.")
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not present at {args.model}. Run train_model.py first.")

    recent = json.loads(args.recent_activities.read_text(encoding="utf-8"))
    max_day = json.loads(args.max_day.read_text(encoding="utf-8"))
    model = joblib.load(args.model)
    return recent, max_day, model


def predict(model, max_day: Dict, metrics: Dict[str, float]) -> float:
    features = np.array(
        [
            [
                normalize(metrics["average_pl"], max_day["average_pl"]),
                normalize(metrics["average_v"], max_day["average_v"]),
                normalize(metrics["average_a"], max_day["average_a"]),
                normalize(metrics["average_dec"], max_day["average_dec"]),
                normalize(metrics["average_distance"], max_day["average_distance"]),
                normalize(metrics["average_hsd"], max_day["average_hsd"]),
            ]
        ]
    )
    prediction = float(model.predict(features)[0])
    return round(min(max(prediction, 0), 1), 3)


def main() -> None:
    args = parse_args()
    recent_activities, max_day, model = load_inputs(args)

    predictions = {}
    for session_id, session in recent_activities.items():
        roles = {"F": [], "D": [], "M": [], "GK": []}
        individuals = {}

        for athlete_id, data in session.get("athletes", {}).items():
            metrics = {
                "average_pl": data.get("max_pl", 0),
                "average_v": data.get("max_v", 0),
                "average_a": data.get("max_a", 0),
                "average_dec": data.get("max_dec", 0),
                "average_distance": data.get("total_distance", 0),
                "average_hsd": data.get("high_speed_distance", 0),
            }
            score = predict(model, max_day, metrics)
            individuals[athlete_id] = {
                "first_name": data.get("first_name"),
                "last_name": data.get("last_name"),
                "position": data.get("position"),
                "predicted_score": score,
            }
            role = data.get("position")
            if role in roles:
                roles[role].append(score)

        position_averages = {
            role: round(sum(scores) / len(scores), 3) if scores else 0 for role, scores in roles.items()
        }

        top_groups = {
            "Top 3 Defenders": sorted(roles["D"], reverse=True)[:3],
            "Top 2 Forwards": sorted(roles["F"], reverse=True)[:2],
            "All Midfielders": roles["M"],
            "Top Goalkeeper": sorted(roles["GK"], reverse=True)[:1],
        }
        team_scores = [score for group in top_groups.values() for score in group if isinstance(score, (int, float))]

        predictions[session_id] = {
            "label": session.get("label"),
            "individual_predictions": individuals,
            "position_averages": position_averages,
            "team_predictions_by_position": top_groups,
            "final_grouped_score": round(sum(team_scores) / len(team_scores), 3) if team_scores else 0,
            "overall_team_score": (
                round(sum(pred["predicted_score"] for pred in individuals.values()) / len(individuals), 3)
                if individuals
                else 0
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=4), encoding="utf-8")
    print(f"Prediction report saved to {args.output}")


if __name__ == "__main__":
    main()
