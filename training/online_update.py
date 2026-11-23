import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "training_rows.csv"

BOOTSTRAP = (
    ROOT / "artifacts" / "bootstrap_v1" / "learner.joblib"
)

OUTPUT_DIR = ROOT / "artifacts" / "online_v1"
CHECKPOINT_PATH = OUTPUT_DIR / "learner.joblib"
INFERENCE_PATH = OUTPUT_DIR / "model.json"

FEATURES = [
    "candidate_rating",
    "last_problem_rating",
    "mean_solve_time_minutes",
    "time_std_minutes",
    "timed_solved_count",
    "last_problem_solve_time_minutes",
]


def load_rows():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run build_training_dataset.py first"
        )

    rows = {}

    with DATA_PATH.open(
        newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "event_id", "user_key", "started_at",
            "label", *FEATURES,
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected training CSV schema")

        for raw in reader:
            event_id = int(raw["event_id"])

            if event_id <= 0 or event_id in rows:
                raise ValueError("Invalid/duplicate event ID")

            timestamp = datetime.fromisoformat(
                raw["started_at"]
            )

            if timestamp.tzinfo is None:
                raise ValueError("Timezone is required")

            if not raw["user_key"].strip():
                raise ValueError("Missing user key")

            values = [
                float(raw[name]) for name in FEATURES
            ]

            if not all(math.isfinite(v) for v in values):
                raise ValueError("Non-finite features")

            if not 800 <= values[0] <= 3500:
                raise ValueError("Invalid candidate rating")

            label = int(raw["label"])

            if label not in (0, 1):
                raise ValueError("Invalid outcome label")

            record = {
                "event_id": event_id,
                "user_key": raw["user_key"],
                "started_at": timestamp.isoformat(),
                "features": values,
                "label": label,
            }

            signature = hashlib.sha256(
                json.dumps(
                    record, sort_keys=True
                ).encode("utf-8")
            ).hexdigest()

            rows[str(event_id)] = {
                **record,
                "timestamp": timestamp,
                "signature": signature,
            }

    return sorted(
        rows.values(),
        key=lambda r: (r["timestamp"], r["event_id"]),
    )


def save_atomic(path, save_function):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        save_function(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def online_update():
    rows = load_rows()

    if not rows:
        print("No eligible real training rows yet.")
        print("Synthetic model remains unchanged.")
        return

    source = (
        CHECKPOINT_PATH
        if CHECKPOINT_PATH.exists()
        else BOOTSTRAP
    )

    checkpoint = joblib.load(source)

    if checkpoint["features"] != FEATURES:
        raise ValueError("Training feature order mismatch")

    model = checkpoint["model"]
    scaler = checkpoint["scaler"]

    processed = checkpoint.get("processed_events", {})

    # The CSV is a full rebuilt dataset.
    # Do not silently ignore removed or corrected labels.
    current_ids = {str(r["event_id"]) for r in rows}

    if set(processed) - current_ids:
        raise RuntimeError(
            "Previously trained events disappeared. "
            "Rebuild from the bootstrap checkpoint."
        )

    for row in rows:
        old_signature = processed.get(str(row["event_id"]))

        if (
            old_signature is not None
            and old_signature != row["signature"]
        ):
            raise RuntimeError(
                "A previously trained event changed. "
                "Rebuild from the bootstrap checkpoint."
            )

    new_rows = [
        row for row in rows
        if str(row["event_id"]) not in processed
    ]

    if not new_rows:
        print("No new real events to learn.")
        return

    X = np.asarray(
        [row["features"] for row in new_rows],
        dtype=np.float64,
    )

    y = np.asarray(
        [row["label"] for row in new_rows],
        dtype=np.int64,
    )

    # Reuse the scaler fitted during pretraining.
    X_scaled = scaler.transform(X)

    # Update the existing model.
    model.partial_fit(
        X_scaled,
        y,
        classes=np.array([0, 1]),
    )

    for row in new_rows:
        processed[str(row["event_id"])] = (
            row["signature"]
        )

    checkpoint["processed_events"] = processed
    checkpoint["real_events_seen"] = len(processed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_atomic(
        CHECKPOINT_PATH,
        lambda path: joblib.dump(checkpoint, path),
    )

    artifact = {
        "model_version": "project-red-online-v1",
        "source": "synthetic_pretraining_plus_real_events",
        "features": FEATURES,
        "x_mean": scaler.mean_.tolist(),
        "x_scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "real_events_seen": len(processed),
    }

    def save_json(path):
        path.write_text(
            json.dumps(artifact, indent=2),
            encoding="utf-8",
        )

    save_atomic(INFERENCE_PATH, save_json)

    print("\nPROJECT RED — ONLINE UPDATE")
    print("-" * 36)
    print(f"New real events: {len(new_rows)}")
    print(f"Total learned events: {len(processed)}")
    print(f"Solved outcomes: {int(y.sum())}")
    print(f"Non-solved outcomes: {len(y) - int(y.sum())}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    online_update()
