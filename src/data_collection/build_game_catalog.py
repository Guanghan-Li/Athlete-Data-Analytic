"""Build a labeled activity catalogue with attached athlete rosters."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download activities from the Catapult API and enrich them with athlete roster metadata."
    )
    parser.add_argument(
        "--activities-output",
        type=Path,
        default=RAW_DIR / "all_activities.json",
        help="Where to store the full activities payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DIR / "games_with_athletes.json",
        help="Destination for the labeled activities that include athlete information.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional Catapult API token. Falls back to the API_KEY env var or .env file.",
    )
    parser.add_argument(
        "--include-training",
        action="store_true",
        help="Keep unlabeled training sessions instead of filtering them out.",
    )
    return parser.parse_args()


def resolve_api_key(cli_value: Optional[str]) -> str:
    load_dotenv()
    api_key = cli_value or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API key not found. Provide --api-key or set API_KEY in the environment/.env file.")
    return api_key


def call_api(url: str, api_key: str) -> requests.Response:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response


def classify_game(name: str) -> str:
    """Assign match-day style labels based on the activity name."""
    match_date_pattern = r"(\d{1,2}-\d{1,2}-\d{2,4})"
    md_minus_pattern = r"MD\s?-?(\d+)"

    match_date = re.search(match_date_pattern, name)
    if not match_date:
        return "0"

    game_date = match_date.group(1)
    if "MD V" in name:
        return game_date

    md_minus_match = re.search(md_minus_pattern, name)
    if md_minus_match:
        days_before = md_minus_match.group(1)
        return f"{game_date}-{days_before}"

    return "0"


def enrich_rosters(activities: Iterable[Dict], api_key: str, include_training: bool) -> List[Dict]:
    catalog: List[Dict] = []
    last_match_day: Optional[str] = None

    for activity in activities:
        label = classify_game(activity["name"])
        if label == "0" and not include_training:
            continue

        # ensure MD-X activities inherit latest match label
        if label != "0" and label.count("-") == 2:
            last_match_day = label
        elif "MD -" in activity["name"] and last_match_day:
            day_offset = label.split("-")[-1]
            label = f"{last_match_day}-{day_offset}"

        roster_url = f"https://connect-us.catapultsports.com/api/v6/activities/{activity['id']}/athletes"
        try:
            roster_response = call_api(roster_url, api_key)
        except requests.HTTPError as exc:
            print(f"Failed to fetch athletes for activity {activity['id']}: {exc}")
            continue

        athletes_payload = roster_response.json()
        athletes = [
            {
                "athlete_id": athlete["id"],
                "first_name": athlete.get("first_name"),
                "last_name": athlete.get("last_name"),
                "position": athlete.get("position"),
                "position_id": athlete.get("position_id"),
            }
            for athlete in athletes_payload
        ]

        catalog.append(
            {
                "activity_id": activity["id"],
                "name": activity["name"],
                "start_time": activity.get("start_time"),
                "end_time": activity.get("end_time"),
                "modified_at": activity.get("modified_at"),
                "label": label,
                "athletes": athletes,
            }
        )

    return catalog


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)

    activities_url = "https://connect-us.catapultsports.com/api/v6/activities"
    raw_response = call_api(activities_url, api_key)
    activities = raw_response.json()

    args.activities_output.parent.mkdir(parents=True, exist_ok=True)
    args.activities_output.write_text(json.dumps(activities, indent=4), encoding="utf-8")

    catalog = enrich_rosters(activities, api_key, include_training=args.include_training)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=4), encoding="utf-8")

    print(f"Saved {len(catalog)} labeled activities with athlete rosters to {args.output}")


if __name__ == "__main__":
    main()
