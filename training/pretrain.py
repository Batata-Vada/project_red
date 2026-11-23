"""Synthetic-only bootstrap training for Project Red.

Do NOT treat synthetic validation metrics as real-world calibration evidence.
"""

import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .model_contract import FEATURES, make_artifact
except ImportError:
    from model_contract import FEATURES, make_artifact

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "bootstrap_v1"
SEED = 42
SAMPLES = 2000


def generate_synthetic_data(samples=SAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    ability = rng.uniform(900, 3000, samples)
    solved_count = rng.integers(5, 31, size=samples)
    last_rating = np.clip(
        np.round((ability + rng.normal(0, 160, samples) - 50) / 100) * 100,
        800, 3500,
    )
    mean_time = np.clip(
        27 + (last_rating - ability) / 25
        - 0.1 * (solved_count - 10) + rng.normal(0, 10, samples),
        5, 90,
    )
    time_std = np.clip(mean_time * rng.uniform(0.1, 0.65, samples), 1, 70)
    last_time = np.clip(rng.normal(mean_time, time_std), 1, 240)
    candidate_rating = rng.integers(8, 36, size=samples) * 100

    # Assumption used to generate bootstrap labels, NOT an empirical CF law.
    logit = (
        (ability + 120 - candidate_rating) / 210
        + 0.025 * (solved_count - 10)
        - 0.02 * (mean_time - 25)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    labels = rng.binomial(1, probability)
    X = np.column_stack([
        candidate_rating, last_rating, mean_time, time_std, solved_count, last_time,
    ]).astype(np.float64)
    return X, labels


def atomic_save(path, save_callback):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        save_callback(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pretrain():
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=0.001,
        random_state=SEED, max_iter=2000, tol=1e-4,
    )
    model.fit(X_train_scaled, y_train)
    predicted = model.predict_proba(X_test_scaled)[:, 1]

    artifact = make_artifact(
        model, scaler, version="project-red-bootstrap-v1",
        source="synthetic_pretraining",
    )
    checkpoint = {"model": model, "scaler": scaler, "features": FEATURES.copy()}

    # Refuse to invalidate an already updated learner by overwriting its base.
    if (ROOT / "artifacts" / "online_v1" / "learner.joblib").exists():
        raise RuntimeError(
            "Online checkpoint exists. Back it up and reset online_v1 before "
            "regenerating bootstrap_v1 (or keep the existing bootstrap checkpoint)."
        )

    atomic_save(OUTPUT / "learner.joblib", lambda p: joblib.dump(checkpoint, p))
    atomic_save(
        OUTPUT / "model.json",
        lambda p: p.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8"),
    )

    print("PROJECT RED — SYNTHETIC PRETRAINING")
    print("-" * 38)
    print(f"Train examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Synthetic Brier score: {brier_score_loss(y_test, predicted):.4f}")
    print(f"Synthetic ROC AUC: {roc_auc_score(y_test, predicted):.4f}")
    print(f"Saved bootstrap checkpoint and inference JSON: {OUTPUT}")
    return artifact


if __name__ == "__main__":
    pretrain()
