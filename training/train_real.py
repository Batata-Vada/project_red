"""Train a deployable probability model exclusively on verified SAY outcomes.

The legacy online learner is deliberately not used as initialization. Before
publishing, evaluate on later sessions and entirely unseen pseudonymous users.
"""
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from training import model_contract
from training.online_update import load_rows

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "real_v1" / "model.json"
MIN_ROWS = 200
MIN_USERS = 10
MIN_TRAIN_ROWS = 100
MIN_EVAL_ROWS = 20
SEED = 20261124


class InsufficientEvidence(RuntimeError):
    """The private dataset cannot yet support an independently validated model."""


def _has_both_classes(rows):
    return {row["label"] for row in rows} == {0, 1}


def _xy(rows):
    return (
        np.asarray([row["features"] for row in rows], dtype=np.float64),
        np.asarray([row["label"] for row in rows], dtype=np.int64),
    )


def _metrics(model, scaler, rows, baseline_rate):
    x, y = _xy(rows)
    p = model.predict_proba(scaler.transform(x))[:, 1]
    return {
        "brier": float(brier_score_loss(y, p)),
        "baseline_brier": float(brier_score_loss(y, np.full(len(y), baseline_rate))),
        "auc": float(roc_auc_score(y, p)),
    }


def validate_cohorts(rows):
    if len(rows) < MIN_ROWS:
        raise InsufficientEvidence(
            f"Need {MIN_ROWS} eligible real outcomes; currently have {len(rows)}."
        )
    users = sorted({row["user_key"] for row in rows})
    if len(users) < MIN_USERS:
        raise InsufficientEvidence(
            f"Need {MIN_USERS} distinct pseudonymous users; currently have {len(users)}."
        )
    if not _has_both_classes(rows):
        raise InsufficientEvidence("Both confirmed successes and failures are required.")

    rng = np.random.default_rng(SEED)
    rng.shuffle(users)
    holdout_count = max(2, int(round(len(users) * 0.2)))
    unseen_users = set(users[:holdout_count])
    unseen = [row for row in rows if row["user_key"] in unseen_users]
    remaining = sorted(
        (row for row in rows if row["user_key"] not in unseen_users),
        key=lambda row: (row["timestamp"], row["event_id"]),
    )
    if len(remaining) < MIN_TRAIN_ROWS + MIN_EVAL_ROWS:
        raise InsufficientEvidence("Too few events remain after user-level holdout.")
    cutoff_index = int(len(remaining) * 0.8)
    cutoff = remaining[cutoff_index]["timestamp"]
    train = [row for row in remaining if row["timestamp"] < cutoff]
    later = [row for row in remaining if row["timestamp"] >= cutoff]

    for name, cohort, minimum in (
        ("training", train, MIN_TRAIN_ROWS),
        ("later-session validation", later, MIN_EVAL_ROWS),
        ("unseen-user validation", unseen, MIN_EVAL_ROWS),
    ):
        if len(cohort) < minimum or not _has_both_classes(cohort):
            raise InsufficientEvidence(
                f"{name} requires {minimum} events and both outcome classes."
            )
    return train, later, unseen, len(users)


def train_real():
    rows = load_rows()
    train, later, unseen, user_count = validate_cohorts(rows)

    train_x, train_y = _xy(train)
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEED)
    model.fit(scaler.transform(train_x), train_y)
    if model.coef_[0][0] >= 0:
        raise InsufficientEvidence("Learned difficulty relationship is not monotonic.")

    baseline_rate = float(train_y.mean())
    later_metrics = _metrics(model, scaler, later, baseline_rate)
    unseen_metrics = _metrics(model, scaler, unseen, baseline_rate)
    for name, metrics in (
        ("later sessions", later_metrics), ("unseen users", unseen_metrics)
    ):
        if metrics["brier"] >= metrics["baseline_brier"] or metrics["auc"] <= 0.5:
            raise InsufficientEvidence(
                f"Real-data model did not beat the held-out baseline for {name}."
            )

    all_x, all_y = _xy(rows)
    final_scaler = StandardScaler().fit(all_x)
    final_model = LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEED)
    final_model.fit(final_scaler.transform(all_x), all_y)
    if final_model.coef_[0][0] >= 0:
        raise InsufficientEvidence("Final real-data model is not difficulty-monotonic.")

    artifact = model_contract.make_artifact(
        final_model, final_scaler, version="project-red-real-v1",
        source="real_verified_practice_outcomes", real_events_seen=len(rows),
    )
    artifact["validation"] = {
        "passed": True,
        "training_rows": len(rows),
        "training_users": user_count,
        "chronological_train_rows": len(train),
        "time_holdout_rows": len(later),
        "user_holdout_rows": len(unseen),
        "time_holdout_brier": later_metrics["brier"],
        "time_holdout_baseline_brier": later_metrics["baseline_brier"],
        "time_holdout_auc": later_metrics["auc"],
        "user_holdout_brier": unseen_metrics["brier"],
        "user_holdout_baseline_brier": unseen_metrics["baseline_brier"],
        "user_holdout_auc": unseen_metrics["auc"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    digest = hashlib.sha256(
        "".join(row["signature"] for row in rows).encode("ascii")
    ).hexdigest()
    artifact["training_dataset_sha256"] = digest
    model_contract.validate_artifact(artifact)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=OUTPUT.parent, prefix=".real_v1_",
        suffix=".tmp", delete=False,
    ) as temporary:
        temp_path = Path(temporary.name)
    try:
        temp_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_path, OUTPUT)
    finally:
        temp_path.unlink(missing_ok=True)

    print(
        f"Validated real-only model: {len(rows)} outcomes, {user_count} users. "
        f"Later Brier {later_metrics['brier']:.4f} vs {later_metrics['baseline_brier']:.4f}; "
        f"unseen Brier {unseen_metrics['brier']:.4f} vs {unseen_metrics['baseline_brier']:.4f}.",
        flush=True,
    )
    print(f"Published {OUTPUT}", flush=True)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        train_real()
    except InsufficientEvidence as exc:
        print(f"Real model not yet deployable: {exc}", flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
