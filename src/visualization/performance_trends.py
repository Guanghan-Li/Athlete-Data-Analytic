"""Visualize predicted performance trends while keeping sensitive details private."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
DEFAULT_DATA_PATH = SAMPLE_DIR / "predictions_sample.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot player- and role-level prediction trends from a structured JSON file. "
            "Names are anonymized by default to keep sensitive details private."
        )
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to predictions JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--show-names",
        action="store_true",
        help="Display the provided first/last names in plots (defaults to anonymized labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="If provided, save plots to this directory instead of opening an interactive window.",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def player_label(player_id: str, player_data: Dict, show_names: bool) -> str:
    if show_names:
        name_parts: List[str] = []
        if player_data.get("first_name"):
            name_parts.append(player_data["first_name"])
        if player_data.get("last_name"):
            name_parts.append(player_data["last_name"])
        if name_parts:
            return " ".join(name_parts)
    if player_data.get("alias"):
        return str(player_data["alias"])
    return f"Player {player_id}"


def transform_predictions(predictions: Dict, show_names: bool) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    player_rows: List[Dict] = []
    overall_rows: List[Dict] = []
    role_scores: Dict[str, List[Dict]] = defaultdict(list)

    for date_label, activity_data in sorted(predictions.items(), key=lambda x: x[0]):
        overall_rows.append(
            {
                "date": date_label,
                "final_grouped_score": activity_data.get("final_grouped_score"),
                "overall_team_score": activity_data.get("overall_team_score"),
            }
        )

        role_aggregates: Dict[str, List[float]] = defaultdict(list)
        for player_id, data in activity_data.get("individual_predictions", {}).items():
            predicted_score = data.get("predicted_score")
            if predicted_score is None:
                continue
            player_rows.append(
                {
                    "date": date_label,
                    "player_id": player_id,
                    "position": data.get("position", "UNK"),
                    "predicted_score": predicted_score,
                    "player_label": player_label(player_id, data, show_names),
                }
            )
            role = data.get("position")
            if role:
                role_aggregates[role].append(predicted_score)

        for role, scores in role_aggregates.items():
            role_scores[role].append(
                {
                    "date": date_label,
                    "avg_score": round(mean(scores), 2),
                }
            )

    df_players = pd.DataFrame(player_rows)
    df_overall = pd.DataFrame(overall_rows)
    df_roles = {role: pd.DataFrame(entries) for role, entries in role_scores.items()}
    return df_players, df_overall, df_roles


def plot_player_scores(df_players: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, group in df_players.groupby("player_label"):
        ax.plot(group["date"], group["predicted_score"], marker="o", label=label)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Individual Player Prediction Trend")
    ax.legend(title="Player", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig


def plot_role_scores(df_roles: Dict[str, pd.DataFrame]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for role, df in df_roles.items():
        if df.empty:
            continue
        ax.plot(df["date"], df["avg_score"], marker="o", label=role)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Predicted Score")
    ax.set_title("Role-Based Prediction Trend")
    ax.legend(title="Role")
    ax.grid(True)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig


def plot_team_scores(df_overall: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_overall["date"], df_overall["final_grouped_score"], marker="o", label="Final Grouped Score")
    ax.plot(df_overall["date"], df_overall["overall_team_score"], marker="s", label="Overall Team Score")
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.set_title("Overall Team Performance Trend")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    return fig


def save_or_show(figures: Iterable[Tuple[str, plt.Figure]], output_dir: Path | None) -> None:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in figures:
            figure.savefig(output_dir / f"{name}.png", bbox_inches="tight")
            plt.close(figure)
        return

    # Display figures interactively when no output directory is provided.
    plt.show()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.input_json)
    df_players, df_overall, df_roles = transform_predictions(predictions, show_names=args.show_names)

    figures: List[Tuple[str, plt.Figure]] = []
    if not df_players.empty:
        figures.append(("player_prediction_trend", plot_player_scores(df_players)))
    if any(not df.empty for df in df_roles.values()):
        figures.append(("role_prediction_trend", plot_role_scores(df_roles)))
    if not df_overall.empty:
        figures.append(("team_prediction_trend", plot_team_scores(df_overall)))

    if not figures:
        raise ValueError("No plot data found in the provided predictions file.")

    save_or_show(figures, args.output_dir)


if __name__ == "__main__":
    main()
