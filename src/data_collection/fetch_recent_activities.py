"""Collect the latest activities with enriched athlete metrics and roster names."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the most recent activities and capture per-athlete Catapult metrics with roster names."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of most recent activities to retrieve.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "recent_activities_with_names.json",
        help="Destination file for the enriched activity metrics.",
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


def call_api(url: str, api_key: str) -> requests.Response:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response


def fetch_athlete_metric(session_id: str, athlete_id: str, api_key: str) -> Dict:
    url = f"https://connect-us.catapultsports.com/api/v6/activities/{session_id}/athletes/{athlete_id}/sensor"
    response = call_api(url, api_key)
    payload = response.json()
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    timeline = payload.get("data", [])
    max_pl = max((entry.get("pl", 0) for entry in timeline), default=0)
    max_v = max((entry.get("v", 0) for entry in timeline), default=0)
    max_a = max((entry.get("a", 0) for entry in timeline), default=0)
    max_dec = min((entry.get("a", 0) for entry in timeline if entry.get("a", 0) < 0), default=0)
    final_o = max((entry.get("o", 0) for entry in timeline), default=0)
    total_distance = max(final_o, 0)
    hsd_threshold = 5.0
    high_speed_distance = sum(entry.get("v", 0) * 0.01 for entry in timeline if entry.get("v", 0) > hsd_threshold)

    return {
        "max_pl": round(max_pl, 2),
        "max_v": round(max_v, 2),
        "max_a": round(max_a, 2),
        "max_dec": round(max_dec, 2),
        "total_distance": round(total_distance, 2),
        "high_speed_distance": round(high_speed_distance, 2),
    }


def fetch_athlete_identity(athlete_id: str, api_key: str) -> Dict[str, Optional[str]]:
    url = f"https://connect-us.catapultsports.com/api/v6/athletes/{athlete_id}"
    try:
        response = call_api(url, api_key)
    except requests.HTTPError:
        return {"first_name": None, "last_name": None}
    payload = response.json()
    return {
        "first_name": payload.get("first_name"),
        "last_name": payload.get("last_name"),
    }


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)

    activities_url = "https://connect-us.catapultsports.com/api/v6/activities"
    activities = call_api(activities_url, api_key).json()

    sorted_activities = sorted(activities, key=lambda item: item.get("modified_at", 0), reverse=True)
    selected = sorted_activities[: args.limit]

    enriched: Dict[str, Dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for activity in selected:
            session_id = activity["id"]
            roster_url = f"https://connect-us.catapultsports.com/api/v6/activities/{session_id}/athletes"
            try:
                roster = call_api(roster_url, api_key).json()
            except requests.HTTPError as exc:
                print(f"Failed to load roster for {session_id}: {exc}")
                continue

            enriched[session_id] = {"label": activity.get("name", "Unknown"), "athletes": {}}

            futures = {
                executor.submit(fetch_athlete_metric, session_id, athlete["id"], api_key): athlete for athlete in roster
            }

            for future in concurrent.futures.as_completed(futures):
                athlete = futures[future]
                try:
                    metrics = future.result()
                except requests.HTTPError as exc:
                    print(f"Failed to fetch metrics for athlete {athlete['id']} in {session_id}: {exc}")
                    continue

                identity = fetch_athlete_identity(athlete["id"], api_key)
                enriched[session_id]["athletes"][athlete["id"]] = {
                    **identity,
                    "position": athlete.get("position"),
                    **metrics,
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(enriched, indent=4), encoding="utf-8")
    print(f"Saved recent activity snapshot to {args.output}")


if __name__ == "__main__":
    main()
