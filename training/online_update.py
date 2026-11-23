"""Incrementally fit Project Red on verified SAY training rows.

Reuses the synthetic scaler, skips already learned IDs, and refuses silent
corrections/removals. Use --rebuild to replay the current validated CSV from
bootstrap when earlier records changed. It never writes into SAY-backend.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

try:
    from .model_contract import FEATURES, make_artifact
except ImportError:
    from model_contract import FEATURES, make_artifact

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training_rows.csv"
BOOTSTRAP = ROOT / "artifacts" / "bootstrap_v1" / "learner.joblib"
OUTPUT_DIR = ROOT / "artifacts" / "online_v1"
CHECKPOINT_PATH = OUTPUT_DIR / "learner.joblib"
INFERENCE_PATH = OUTPUT_DIR / "model.json"

COLUMNS = ["user_key", "event_id", "started_at", *FEATURES, "label"]


def load_rows():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Build the training dataset first: {DATA_PATH}")
    rows = []
    seen = set()
    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Unexpected training CSV schema or column order")

        for line_number, raw in enumerate(reader, 2):
            try:
                event_id = int(raw["event_id"])
                label = int(raw["label"])
                timestamp = datetime.fromisoformat(raw["started_at"])
                features = [float(raw[name]) for name in FEATURES]
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid training row {line_number}") from exc

            if event_id <= 0 or event_id in seen:
                raise ValueError(f"Duplicate/invalid event ID at row {line_number}")
            seen.add(event_id)
            if not raw["user_key"] or not raw["user_key"].strip():
                raise ValueError(f"Missing user key at row {line_number}")
            if timestamp.tzinfo is None:
                raise ValueError(f"Timezone missing at row {line_number}")
            if label not in (0, 1):
                raise ValueError(f"Invalid label at row {line_number}")
            if not all(math.isfinite(value) for value in features):
                raise ValueError(f"Non-finite feature at row {line_number}")
            if not (800 <= features[0] <= 3500 and 800 <= features[1] <= 3500):
                raise ValueError(f"Rating outside supported range at row {line_number}")
            if not (1 <= features[2] <= 240 and 0 <= features[3] <= 240
                    and 5 <= features[4] <= 30 and features[4].is_integer()
                    and 1 <= features[5] <= 240):
                raise ValueError(f"Invalid historical feature at row {line_number}")

            record = {
                "event_id": event_id,
                "user_key": raw["user_key"],
                "started_at": timestamp.isoformat(),
                "features": features,
                "label": label,
            }
            record["signature"] = hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest()
            record["timestamp"] = timestamp
            rows.append(record)
    rows.sort(key=lambda row: (row["timestamp"], row["event_id"]))
    return rows


def save_atomic(path, save_function):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        save_function(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_from_checkpoint(checkpoint):
    return make_artifact(
        checkpoint["model"], checkpoint["scaler"],
        version="project-red-online-v1",
        source="synthetic_pretraining_plus_real_events",
        real_events_seen=len(checkpoint.get("processed_events", {})),
    )


def online_update(*, rebuild=False):
    rows = load_rows()
    if not rows:
        if CHECKPOINT_PATH.exists():
            raise RuntimeError(
                "Previously trained events are missing. Do not use the stale online model; "
                "review source data before rebuilding."
            )
        print("No eligible real training rows yet; bootstrap model remains unchanged.")
        return 0

    source = BOOTSTRAP if rebuild or not CHECKPOINT_PATH.exists() else CHECKPOINT_PATH
    if not source.exists():
        raise FileNotFoundError(f"Pretrain first: {source}")
    # Only load checkpoint files generated locally by trusted Project Red code.
    checkpoint = joblib.load(source)
    if checkpoint.get("features") != FEATURES:
        raise ValueError("Checkpoint feature order mismatch")
    if not hasattr(checkpoint.get("model"), "partial_fit"):
        raise ValueError("Checkpoint model does not support incremental learning")

    processed = dict(checkpoint.get("processed_events", {}))
    ids = {str(row["event_id"]) for row in rows}
    if set(processed) - ids:
        raise RuntimeError("Trained events disappeared. Review changes, then use --rebuild.")
    for row in rows:
        id_string = str(row["event_id"])
        if id_string in processed and processed[id_string] != row["signature"]:
            raise RuntimeError(
                f"Previously trained event {id_string} changed. Review data, then use --rebuild."
            )
    new_rows = [row for row in rows if str(row["event_id"]) not in processed]

    if new_rows:
        X = np.asarray([row["features"] for row in new_rows], dtype=np.float64)
        y = np.asarray([row["label"] for row in new_rows], dtype=np.int64)
        X_scaled = checkpoint["scaler"].transform(X)  # Fitted synthetic scaler stays fixed.
        checkpoint["model"].partial_fit(X_scaled, y, classes=np.array([0, 1]))
        for row in new_rows:
            processed[str(row["event_id"])] = row["signature"]
        checkpoint["processed_events"] = processed
        checkpoint["real_events_seen"] = len(processed)

    # Validate model before publishing. If SGD makes difficulty coefficient
    # positive, the existing deployable model remains untouched.
    artifact = artifact_from_checkpoint(checkpoint)
    if new_rows:
        save_atomic(CHECKPOINT_PATH, lambda path: joblib.dump(checkpoint, path))
    # Repair a missing/stale JSON even when no new events appeared.
    save_atomic(
        INFERENCE_PATH,
        lambda path: path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8"),
    )

    print("\nPROJECT RED — ONLINE UPDATE")
    print("-" * 36)
    print(f"New verified events: {len(new_rows)}")
    print(f"Total learned events: {len(processed)}")
    print(f"Updated inference artifact: {INFERENCE_PATH}")
    if not new_rows:
        print("No new events; model weights left unchanged.")
    return len(new_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Relearn all current verified rows starting from the original bootstrap checkpoint",
    )
    args = parser.parse_args()
    online_update(rebuild=args.rebuild)
