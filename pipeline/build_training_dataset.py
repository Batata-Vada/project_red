
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "say_practice_events.jsonl"
OUTPUT_FILE = ROOT / "data" / "training_rows.csv"

MIN_HISTORY = 5
MAX_HISTORY = 150
MAX_SOLVES = 30

FEATURES = [
    "candidate_rating",
    "last_problem_rating",
    "mean_solve_time_minutes",
    "time_std_minutes",
    "timed_solved_count",
    "last_problem_solve_time_minutes",
]

COLUMNS = [
    "user_key",
    "event_id",
    "started_at",
    *FEATURES,
    "label",
]


def valid_rating(value):
    return type(value) is int and 800 <= value <= 3500


def load_events():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Run fetch_say_events.py first: {INPUT_FILE}"
        )

    grouped = defaultdict(list)
    seen_ids = set()

    with INPUT_FILE.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            event = json.loads(line)

            event_id = event["event_id"]

            if type(event_id) is not int or event_id in seen_ids:
                raise ValueError(
                    f"Duplicate or invalid event ID on line {line_number}"
                )

            seen_ids.add(event_id)

            user_key = event["user_key"]

            if not isinstance(user_key, str) or not user_key:
                raise ValueError("Invalid user key")

            start = datetime.fromisoformat(event["started_at"])
            finish = datetime.fromisoformat(event["finished_at"])

            if (
                start.tzinfo is None
                or finish.tzinfo is None
                or finish <= start
            ):
                raise ValueError("Invalid session timestamps")

            duration = event["duration_seconds"]

            if type(duration) is not int or not 60 <= duration <= 14400:
                raise ValueError("Invalid session duration")

            for field in (
                "official_solved",
                "official_attempt_observed",
                "previously_solved",
            ):
                if type(event[field]) is not bool:
                    raise ValueError(f"Invalid {field}")

            if (
                event["official_solved"]
                and not event["official_attempt_observed"]
            ):
                raise ValueError(
                    "A verified solve must have an observed attempt"
                )

            event["_start"] = start
            event["_finish"] = finish

            grouped[user_key].append(event)

    return grouped


# def historical_features(completed_sessions):
#     # Match the current backend's 150-session history window.
#     recent = completed_sessions[-MAX_HISTORY:]

#     solved = []
#     seen_problems = set()

#     # Most recently finished sessions first.
#     for event in reversed(recent):
#         problem = event["problem_external_id"]

#         if problem in seen_problems:
#             continue

#         if not event["official_solved"]:
#             continue

#         seen_problems.add(problem)

#         if not valid_rating(event["problem_rating"]):
#             continue

#         solved.append(event)

#         if len(solved) == MAX_SOLVES:
#             break

#     if len(solved) < MIN_HISTORY:
#         return None

#     durations = [
#         event["duration_seconds"] / 60.0
#         for event in solved
#     ]

#     return {
#         "last_problem_rating": solved[0]["problem_rating"],
#         "mean_solve_time_minutes": fmean(durations),
#         "time_std_minutes": pstdev(durations),
#         "timed_solved_count": len(solved),
#         "last_problem_solve_time_minutes": durations[0],
#     }


def build_user_rows(events, stats):
    rows = []

    for event in events:

        if not valid_rating(event["problem_rating"]):
            stats["unrated_target"] += 1
            continue

        if event["previously_solved"]:
            stats["previously_solved"] += 1
            continue

        outcome = event["outcome_status"]

        if outcome not in ("SOLVED", "FAILED"):
            stats["unknown_outcome"] += 1
            continue

        history = event["history_features"]

        if history is None:
            stats["insufficient_history"] += 1
            continue

        row = {
            "user_key": event["user_key"],
            "event_id": event["event_id"],
            "started_at": event["started_at"],
            "candidate_rating": event["problem_rating"],
            "last_problem_rating": history["last_problem_rating"],
            "mean_solve_time_minutes":
                history["mean_solve_time_minutes"],
            "time_std_minutes": history["time_std_minutes"],
            "timed_solved_count": history["timed_solved_count"],
            "last_problem_solve_time_minutes":
                history["last_problem_solve_time_minutes"],
            "label": 1 if outcome == "SOLVED" else 0,
        }

        rows.append(row)
        stats["included"] += 1

    return rows

def build_dataset():
    grouped = load_events()

    rows = []
    stats = Counter()

    for events in grouped.values():
        user_rows = build_user_rows(events, stats)
        rows.extend(user_rows)

    # Global chronological order for later train/test splitting.
    rows.sort(key=lambda row: (
        datetime.fromisoformat(row["started_at"]),
        row["event_id"],
    ))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(row["label"] for row in rows)
    negatives = len(rows) - positives

    print("\nPROJECT RED — DATASET REPORT")
    print("-" * 36)
    print(f"Users with exported events: {len(grouped)}")
    print(f"Exported events: {sum(map(len, grouped.values()))}")
    print(f"Training rows: {len(rows)}")
    print(f"Observed solves: {positives}")
    print(f"Observed non-solves: {negatives}")

    print("\nExcluded events:")
    for reason in (
        "unrated_target",
        "previously_solved",
        "unknown_outcome",
        "insufficient_history",
    ):
        print(f"  {reason}: {stats[reason]}")

    if not rows:
        print(
            "\nNo training rows yet. More verified practice "
            "history may be needed."
        )
    elif positives == 0 or negatives == 0:
        print(
            "\nWARNING: Only one outcome class is present. "
            "Binary classification training is not ready."
        )

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset()
