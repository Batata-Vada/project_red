
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "bootstrap_v1"

FEATURES = [
    "candidate_rating",
    "last_problem_rating",
    "mean_solve_time_minutes",
    "time_std_minutes",
    "timed_solved_count",
    "last_problem_solve_time_minutes",
]

SEED = 42
SAMPLES = 20000


def generate_synthetic_data():
    rng = np.random.default_rng(SEED)

    # Hidden simulated user ability.
    ability = rng.uniform(900, 3000, SAMPLES)

    solved_count = rng.integers(
        5, 31, size=SAMPLES
    )

    last_rating = np.clip(
        np.round(
            (ability + rng.normal(0, 160, SAMPLES) - 50)
            / 100
        ) * 100,
        800,
        3500,
    )

    mean_time = np.clip(
        27
        + (last_rating - ability) / 25
        - 0.1 * (solved_count - 10)
        + rng.normal(0, 10, SAMPLES),
        5,
        90,
    )

    time_std = np.clip(
        mean_time * rng.uniform(0.1, 0.65, SAMPLES),
        1,
        70,
    )

    last_time = np.clip(
        rng.normal(mean_time, time_std),
        1,
        240,
    )

    candidate_rating = (
        rng.integers(8, 36, size=SAMPLES) * 100
    )

    # Artificial probability rule for pretraining.
    logit = (
        (ability + 120 - candidate_rating) / 210
        + 0.025 * (solved_count - 10)
        - 0.02 * (mean_time - 25)
    )

    probability = 1 / (1 + np.exp(-logit))

    # Simulated observed outcomes.
    labels = rng.binomial(1, probability)

    X = np.column_stack([
        candidate_rating,
        last_rating,
        mean_time,
        time_std,
        solved_count,
        last_time,
    ])

    return X, labels


def pretrain():
    X, y = generate_synthetic_data()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=SEED,
            stratify=y,
        )
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.001,
        random_state=SEED,
        max_iter=2000,
        tol=1e-4,
    )

    model.fit(X_train, y_train)

    predictions = model.predict_proba(X_test)[:, 1]

    print("PROJECT RED — SYNTHETIC PRETRAINING")
    print("------------------------------------")
    print("Training examples:", len(X_train))
    print("Test examples:", len(X_test))

    print(
        "Synthetic Brier score:",
        brier_score_loss(y_test, predictions)
    )

    print(
        "Synthetic ROC AUC:",
        roc_auc_score(y_test, predictions)
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Keep the fitted sklearn objects for future partial_fit.
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": FEATURES,
        },
        OUTPUT / "learner.joblib",
    )

    # Framework-independent inference artifact.
    artifact = {
        "model_version": "project-red-bootstrap-v1",
        "source": "synthetic_pretraining",
        "features": FEATURES,
        "x_mean": scaler.mean_.tolist(),
        "x_scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }

    with (OUTPUT / "model.json").open("w") as file:
        json.dump(artifact, file, indent=2)

    print("Pretraining completed.")
    print("Model saved to:", OUTPUT)


if __name__ == "__main__":
    pretrain()

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
