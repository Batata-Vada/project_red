"""Fetch SAY-backend's paginated Project Red v2 event export, atomically.

Requires SAY_BACKEND_URL and SAY_PROJECT_RED_EXPORT_TOKEN in the environment.
Only a *successful* complete export replaces the previous JSONL file.
"""

import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "say_practice_events.jsonl"
ENDPOINT = "/api/internal/project-red/events"
PAGE_SIZE = 200
SCHEMA_VERSION = 2
FEATURE_CONTRACT = "say-practice-v1"

HISTORY_FEATURES = {
    "last_problem_rating",
    "mean_solve_time_minutes",
    "time_std_minutes",
    "timed_solved_count",
    "last_problem_solve_time_minutes",
}
OUTCOMES = {"SOLVED", "FAILED", "UNKNOWN"}


def valid_rating(value):
    return type(value) is int and 800 <= value <= 3500


def validate_history(history):
    """Backend supplies this as-of-session-start history; None is valid."""
    if history is None:
        return
    if not isinstance(history, dict) or set(history) != HISTORY_FEATURES:
        raise ValueError("Historical feature contract mismatch")

    for name in HISTORY_FEATURES:
        value = history[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid historical feature: {name}")

    if not valid_rating(history["last_problem_rating"]):
        raise ValueError("Invalid last problem rating")
    count = history["timed_solved_count"]
    if type(count) is not int or not 5 <= count <= 30:
        raise ValueError("Invalid timed solve count")
    if not 1 <= history["mean_solve_time_minutes"] <= 240:
        raise ValueError("Invalid mean solve time")
    if not 0 <= history["time_std_minutes"] <= 240:
        raise ValueError("Invalid solve time standard deviation")
    if not 1 <= history["last_problem_solve_time_minutes"] <= 240:
        raise ValueError("Invalid last solve time")


def validate_event(event, previous_id):
    """Validate one event and return its strictly increasing event_id."""
    if not isinstance(event, dict):
        raise ValueError("Event must be an object")
    event_id = event.get("event_id")
    if type(event_id) is not int or event_id <= previous_id:
        raise ValueError("Invalid or out-of-order event ID")

    for name in ("user_key", "problem_external_id"):
        value = event.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid {name}")

    rating = event.get("problem_rating")
    if rating is not None and not valid_rating(rating):
        raise ValueError("Invalid problem rating")
    duration = event.get("duration_seconds")
    if type(duration) is not int or not 60 <= duration <= 14400:
        raise ValueError("Invalid duration_seconds")
    tags = event.get("tags")
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ValueError("Invalid tags")

    try:
        start = datetime.fromisoformat(event["started_at"])
        finish = datetime.fromisoformat(event["finished_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid session timestamps") from exc
    if start.tzinfo is None or finish.tzinfo is None or finish <= start:
        raise ValueError("Timestamps must be timezone-aware and ordered")

    bool_fields = (
        "official_solved", "official_attempt_observed",
        "final_failure_observed", "previously_solved", "full_history_coverage",
    )
    for name in bool_fields:
        if type(event.get(name)) is not bool:
            raise ValueError(f"Missing or invalid {name}")

    solved = event["official_solved"]
    attempted = event["official_attempt_observed"]
    failure = event["final_failure_observed"]
    coverage = event["full_history_coverage"]
    outcome = event.get("outcome_status")
    if outcome not in OUTCOMES:
        raise ValueError("Invalid outcome_status")
    if solved and not attempted:
        raise ValueError("A verified solve requires an observed attempt")
    if outcome == "SOLVED" and not solved:
        raise ValueError("SOLVED outcome lacks verified solve")
    if outcome == "FAILED" and not (failure and coverage and attempted and not solved):
        raise ValueError("FAILED outcome lacks finalized evidence and full coverage")
    if outcome == "UNKNOWN" and solved:
        raise ValueError("UNKNOWN contradicts verified solve")

    if "history_features" not in event:
        raise ValueError("Missing history_features")
    validate_history(event["history_features"])
    return event_id


def validate_response(payload):
    if not isinstance(payload, dict):
        raise ValueError("Invalid API response")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported export schema version")
    if payload.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError("SAY-backend feature contract mismatch")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) > PAGE_SIZE:
        raise ValueError("Invalid export event page")
    if "next_after_id" not in payload:
        raise ValueError("Missing next_after_id")
    return events


def fetch_events():
    base_url = os.environ["SAY_BACKEND_URL"].rstrip("/")
    token = os.environ["SAY_PROJECT_RED_EXPORT_TOKEN"]
    if not token.strip():
        raise ValueError("SAY_PROJECT_RED_EXPORT_TOKEN must not be empty")
    url = urlsplit(base_url)
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("SAY_BACKEND_URL must be a plain HTTP(S) origin")
    # Production backends should be accessed over HTTPS. HTTP is useful locally.

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    after_id = 0
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=OUTPUT.parent,
        prefix=".say_events_", suffix=".tmp", delete=False,
    ) as temp:
        temp_path = Path(temp.name)

    try:
        with requests.Session() as session, temp_path.open("w", encoding="utf-8") as file:
            while True:
                response = session.get(
                    f"{base_url}{ENDPOINT}",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"after_id": after_id, "limit": PAGE_SIZE},
                    timeout=(5, 30),
                )
                response.raise_for_status()
                events_payload = response.json()
                events = validate_response(events_payload)
                last_id = after_id
                for event in events:
                    last_id = validate_event(event, last_id)
                    file.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
                    total += 1

                cursor = events_payload["next_after_id"]
                if cursor is None:
                    break
                if not events or type(cursor) is not int or cursor != last_id:
                    raise ValueError("Invalid pagination cursor")
                after_id = cursor

        if(total == 0 and OUTPUT.exists() and OUTPUT.stat().st_size > 0):
            raise RuntimeError("Backend retunred 0 events")

        os.replace(temp_path, OUTPUT)
    finally:
        temp_path.unlink(missing_ok=True)

    print("\nPROJECT RED — EVENT EXPORT")
    print("-" * 36)
    if total == 0:
        print("No eligible real practice events yet.")
    else:
        print(f"Fetched {total} real practice events")
    print(f"Saved: {OUTPUT}")
    return total


if __name__ == "__main__":
    fetch_events()
