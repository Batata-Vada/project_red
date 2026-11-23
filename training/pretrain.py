
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
