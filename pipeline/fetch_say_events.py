
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "say_practice_events.jsonl"

ENDPOINT = "/api/internal/project-red/events"
PAGE_SIZE = 200


def validate_event(event, previous_id):
    if not isinstance(event, dict):
        raise ValueError("Event must be an object")

    event_id = event.get("event_id")

    if type(event_id) is not int or event_id <= previous_id:
        raise ValueError("Invalid or duplicate event ID")

    if not isinstance(event.get("user_key"), str):
        raise ValueError("Missing user key")

    if not isinstance(event.get("problem_external_id"), str):
        raise ValueError("Missing problem ID")

    rating = event.get("problem_rating")

    if rating is not None and (
        type(rating) is not int or not 800 <= rating <= 3500
    ):
        raise ValueError("Invalid problem rating")

    duration = event.get("duration_seconds")

    if type(duration) is not int or not 60 <= duration <= 14400:
        raise ValueError("Invalid practice duration")

    for field in (
        "official_solved",
        "official_attempt_observed",
        "previously_solved",
    ):
        if type(event.get(field)) is not bool:
            raise ValueError(f"Invalid {field}")

    tags = event.get("tags")

    if not isinstance(tags, list) or not all(
        isinstance(tag, str) for tag in tags
    ):
        raise ValueError("Invalid tags")

    start = datetime.fromisoformat(event["started_at"])
    finish = datetime.fromisoformat(event["finished_at"])

    if (
        start.tzinfo is None
        or finish.tzinfo is None
        or finish <= start
    ):
        raise ValueError("Invalid session timestamps")

    return event_id


def fetch_events():
    base_url = os.environ["SAY_BACKEND_URL"].rstrip("/")
    token = os.environ["SAY_PROJECT_RED_EXPORT_TOKEN"]

    if not token:
        raise ValueError("Export token cannot be empty")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    after_id = 0

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=OUTPUT.parent,
        prefix=".say_events_",
        suffix=".tmp",
        delete=False,
    ) as temp:
        temp_path = Path(temp.name)

    try:
        with requests.Session() as session:
            with temp_path.open("w", encoding="utf-8") as file:
                while True:
                    response = session.get(
                        f"{base_url}{ENDPOINT}",
                        headers={
                            "Authorization": f"Bearer {token}"
                        },
                        params={
                            "after_id": after_id,
                            "limit": PAGE_SIZE,
                        },
                        timeout=(5, 30),
                    )

                    response.raise_for_status()
                    payload = response.json()

                    if not isinstance(payload, dict):
                        raise ValueError("Invalid API response")

                    if payload.get("schema_version") != 2:
                     raise ValueError("Unsupported schema version")

                    if payload.get("feature_contract") != "say-practice-v1":
                        raise ValueError("Feature contract mismatch")
                    
                    events = payload.get("events")

                    if not isinstance(events, list):
                        raise ValueError("Invalid events list")

                    last_id = after_id

                    for event in events:
                        last_id = validate_event(event, last_id)

                        file.write(
                            json.dumps(event, separators=(",", ":"))
                            + "\n"
                        )
                        total += 1

                    cursor = payload.get("next_after_id")

                    if cursor is None:
                        break

                    if not events or cursor != last_id:
                        raise ValueError("Invalid pagination cursor")

                    after_id = cursor

        if total == 0:
            raise RuntimeError(
                "No eligible practice events were exported"
            )

        # Replace the previous dataset only after
        # the entire export succeeds.
        os.replace(temp_path, OUTPUT)

    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Fetched {total} real practice events")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    fetch_events()
