"""Aggregate individual athlete metrics into team-level summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert athlete_metrics.json into per-session team averages for model training."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "athlete_metrics.json",
        help="Path to the athlete metrics JSON produced by fetch_activity_metrics.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "team_summary.json",
        help="Destination for the aggregated team metric summary.",
    )
    parser.add_argument(
        "--squad-size",
        type=int,
        default=11,
        help="Number of top athletes (by player load) to average for each session.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Athlete metrics file not found at {path}. Run fetch_activity_metrics.py first.")
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.input)

    summary = {}
    for session_id, session_data in metrics.items():
        athletes = [
            {"athlete_id": athlete_id, **values} for athlete_id, values in session_data.get("athletes", {}).items()
        ]
        if not athletes:
            continue

        top_athletes = sorted(athletes, key=lambda entry: entry.get("max_pl", 0), reverse=True)[: args.squad_size]
        if not top_athletes:
            continue

        size = len(top_athletes)
        summary[session_id] = {
            "match_label": session_data.get("label"),
            "average_pl": round(sum(a.get("max_pl", 0) for a in top_athletes) / size, 2),
            "average_v": round(sum(a.get("max_v", 0) for a in top_athletes) / size, 2),
            "average_a": round(sum(a.get("max_a", 0) for a in top_athletes) / size, 2),
            "average_dec": round(sum(a.get("max_dec", 0) for a in top_athletes) / size, 2),
            "average_distance": round(sum(a.get("total_distance", 0) for a in top_athletes) / size, 2),
            "average_hsd": round(sum(a.get("high_speed_distance", 0) for a in top_athletes) / size, 2),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, indent=4)

    print(f"Team summary saved to {args.output}")


if __name__ == "__main__":
    main()
