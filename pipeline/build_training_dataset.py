"""Build a chronological binary-outcome dataset from SAY v2 export.

No heuristic negatives: only SOLVED and finalized FAILED become labels.
The backend's history_features are used directly for train/serve alignment.
"""

import csv
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

# When invoked as a script, sibling import works from any working directory.
try:
    from .fetch_say_events import validate_event
except ImportError:
    from fetch_say_events import validate_event

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "say_practice_events.jsonl"
OUTPUT_FILE = ROOT / "data" / "training_rows.csv"

FEATURES = [
    "candidate_rating",
    "last_problem_rating",
    "mean_solve_time_minutes",
    "time_std_minutes",
    "timed_solved_count",
    "last_problem_solve_time_minutes",
]
COLUMNS = ["user_key", "event_id", "started_at", *FEATURES, "label"]


def load_rows():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fetch SAY events first: {INPUT_FILE}")

    rows = []
    stats = Counter()
    previous_id = 0
    exclusions = []

    with INPUT_FILE.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                previous_id = validate_event(event, previous_id)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid event at JSONL line {line_number}: {exc}") from exc
            stats["total_events"] += 1

            reason = None
            if event["problem_rating"] is None:
                reason = "unrated_target"
            elif event["previously_solved"]:
                reason = "previously_solved"
            elif event["outcome_status"] == "UNKNOWN":
                reason = "unknown_outcome"
            elif event["history_features"] is None:
                reason = "insufficient_history"
            if reason is not None:
                stats[reason] += 1
                exclusions.append((
                    event["event_id"], event["problem_external_id"], reason
                ))
                continue
            outcome = event["outcome_status"]
            history = event["history_features"]

            row = {
                "user_key": event["user_key"],
                "event_id": event["event_id"],
                "started_at": event["started_at"],
                "candidate_rating": event["problem_rating"],
                **history,
                "label": 1 if outcome == "SOLVED" else 0,
            }
            rows.append(row)

    rows.sort(key=lambda row: (datetime.fromisoformat(row["started_at"]), row["event_id"]))
    stats["exclusion_details"] = exclusions
    return rows, stats


def build_dataset():
    rows, stats = load_rows()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Atomic CSV replacement: malformed source doesn't destroy a good dataset.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=OUTPUT_FILE.parent,
        prefix=".training_rows_", suffix=".tmp", delete=False,
    ) as temp:
        temp_path = Path(temp.name)
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, OUTPUT_FILE)
    finally:
        temp_path.unlink(missing_ok=True)

    positive = sum(row["label"] for row in rows)
    print("\nPROJECT RED — REAL DATASET")
    print("-" * 36)
    print(f"Exported events: {stats['total_events']}")
    print(f"Training rows: {len(rows)}")
    print(f"Verified solves: {positive}")
    print(f"Verified failures: {len(rows) - positive}")
    print("Excluded:")
    for key in ("unrated_target", "previously_solved", "unknown_outcome", "insufficient_history"):
        print(f"  {key}: {stats[key]}")
    if not rows:
        print("No eligible real training rows yet.")
        print("Excluded sessions (first failed check per session):")
        for event_id, external_id, reason in stats["exclusion_details"]:
            print(f"  session {event_id} ({external_id}): {reason}")
        print(
            "Five distinct, officially verified timed solves observed before "
            "a later target session are required for historical features. "
            "Those earlier solves do not automatically become training targets."
        )
    elif not 0 < positive < len(rows):
        print("NOTE: Real data currently contains only one outcome class.")
    print(f"Saved: {OUTPUT_FILE}")
    return len(rows)


if __name__ == "__main__":
    build_dataset()
