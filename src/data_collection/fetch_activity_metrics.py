"""Pull per-athlete session metrics from the Catapult API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch per-athlete Catapult metrics for each activity listed in games_with_athletes.json."
    )
    parser.add_argument(
        "--games-file",
        type=Path,
        default=RAW_DIR / "games_with_athletes.json",
        help="Location of the games_with_athletes.json produced by build_game_catalog.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "athlete_metrics.json",
        help="Where to store the aggregated athlete metrics.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional Catapult API token. Falls back to the API_KEY env var or .env file.",
    )
    return parser.parse_args()


def resolve_api_key(cli_value: Optional[str]) -> str:
    load_dotenv()
    api_key = cli_value or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API key not found. Provide --api-key or set API_KEY in the environment/.env file.")
    return api_key


def fetch_athlete_data(
    session_id: str,
    label: str,
    athlete: Dict[str, str],
    api_key: str,
) -> Optional[Tuple[str, str, str, float, float, float, float, float, float, str]]:
    """Fetch Catapult sensor metrics for a single athlete within a session."""
    athlete_id = athlete.get("athlete_id") or athlete.get("athelete_id")
    position = athlete.get("position", "Unknown")

    if not athlete_id:
        print(f"Skipping athlete with missing ID in activity {session_id}")
        return None

    url = f"https://connect-us.catapultsports.com/api/v6/activities/{session_id}/athletes/{athlete_id}/sensor"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            athlete_data = response.json()
            if isinstance(athlete_data, list):
                athlete_data = athlete_data[0] if athlete_data else {}

            timeline = athlete_data.get("data", [])
            max_pl = max((entry.get("pl", 0) for entry in timeline), default=0)
            max_v = max((entry.get("v", 0) for entry in timeline), default=0)
            max_a = max((entry.get("a", 0) for entry in timeline), default=0)
            max_dec = min((entry.get("a", 0) for entry in timeline if entry.get("a", 0) < 0), default=0)
            final_o = max((entry.get("o", 0) for entry in timeline), default=0)
            total_distance = max(final_o, 0)
            hsd_threshold = 5.0
            high_speed_distance = sum(entry.get("v", 0) * 0.01 for entry in timeline if entry.get("v", 0) > hsd_threshold)

            return (
                session_id,
                label,
                athlete_id,
                max_pl,
                max_v,
                max_a,
                max_dec,
                total_distance,
                high_speed_distance,
                position,
            )

        if response.status_code == 401:
            print(f"Unauthorized: Check API token. Athlete {athlete_id} in activity {session_id}")
        else:
            print(f"Failed to fetch sensor data for athlete {athlete_id} in game {session_id}: {response.status_code}")
    except requests.RequestException as exc:
        print(f"Error fetching sensor data for athlete {athlete_id} in game {session_id}: {exc}")

    return None


def round_floats(payload: Dict) -> Dict:
    """Recursively round all float values in a nested structure to two decimals."""
    if isinstance(payload, dict):
        return {key: round_floats(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [round_floats(item) for item in payload]
    if isinstance(payload, float):
        return round(payload, 2)
    return payload


def load_games(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"games_with_athletes.json not found at {path}. Run build_game_catalog.py first to generate it."
        )
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)
    games_with_athletes = load_games(args.games_file)

    athlete_metrics: Dict[str, Dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for game in games_with_athletes:
            session_id = game["activity_id"]
            label = game.get("label", "Unknown")
            athlete_metrics[session_id] = {"label": label, "athletes": {}}
            for athlete in game.get("athletes", []):
                futures.append(
                    executor.submit(fetch_athlete_data, session_id, label, athlete, api_key)
                )

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if not result:
                continue
            (
                session_id,
                label,
                athlete_id,
                max_pl,
                max_v,
                max_a,
                max_dec,
                total_distance,
                high_speed_distance,
                position,
            ) = result
            athlete_metrics[session_id]["athletes"][athlete_id] = {
                "max_pl": max_pl,
                "max_v": max_v,
                "max_a": max_a,
                "max_dec": max_dec,
                "total_distance": total_distance,
                "high_speed_distance": high_speed_distance,
                "position": position,
            }

    athlete_metrics = round_floats(athlete_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(athlete_metrics, destination, indent=4)

    print(f"Data saved to {args.output} with rounded float values.")


if __name__ == "__main__":
    main()
