# Project Red — real-data recommendation training

SAY-backend owns authentication, Codeforces submission verification, timed
practice, event export and model serving. Project Red owns private ingestion,
historical dataset construction, training and validation.

## What the five-solve milestone means

The Practice page shows `5 / 5 verified` once the user's recent, distinct,
officially accepted and rated timed solves meet the **inference history**
requirement. This does **not** mean that the next ML model is already trained:
a training sample requires an additional, later practice session with historical
features available when it started and a verified outcome.

Existing Codeforces submissions imported after the fact cannot be pretended
to have been observed earlier. An unsuccessful submission needs a final verdict
and trustworthy synchronization coverage to be used as a negative example.

## Run the real training pipeline

From `SAY-backend`, the all-in-one launcher runs the authenticated Project
Red worker as a sibling process:

```bash
.venv/bin/python scripts/run_local.py
```

Alternatively, in `project_red`, set `SAY_BACKEND_URL` and
`SAY_PROJECT_RED_EXPORT_TOKEN` privately and run:

```bash
.venv/bin/python -m pipeline.auto_update --once
```

The exporter uses schema v2 and the stable `say-practice-v1` feature contract.
The worker makes a complete export, constructs chronological rows and attempts
**real-only** training once at least 200 eligible outcomes exist. It requires
at least 10 distinct pseudonymous users, both success/failure classes, a
chronological later-session holdout and a disjoint unseen-user holdout.
The evaluation model must outperform the constant-rate Brier baseline and have
ROC AUC above 0.5 on both holdouts. These are preliminary safety gates, not
proof that deployment improves learning or is well calibrated.

Only a passing real-only model is written to
`artifacts/real_v1/model.json`. It records the number of real observations,
training user count and held-out evaluation metrics. If any gate fails, the
previous artifact remains untouched. Deployment should still require human
review of calibration by difficulty band and actual recommendation outcomes.

The older `bootstrap_v1` and `online_v1` artifacts were trained with synthetic
initialization. They are **not** suitable to activate as a real-only deployed
recommender, even if they contain some real updates. The automatic worker no
longer publishes those checkpoints for live serving.

## Serving in SAY-backend

Set the backend's private `PROJECT_RED_MODEL_PATH` to the absolute path of the
validated `artifacts/real_v1/model.json`. The local launcher uses the sibling
path by default when that setting is not explicitly configured.
Keep `PROJECT_RED_ENABLED=false` until the model is independently validated
and you deliberately enable it. The backend rejects synthetic-sourced models and
artifacts without successful real-data evaluation metadata.

The Practice page uses the authenticated
`GET /api/me/practice-sessions/readiness` endpoint to show historical
readiness and whether a validated model is available. Only a recommendation
response with `recommendation_engine: "project_red"` means ML inference was
actually used. Other responses safely use SAY's regular recommender.

## Privacy

Raw user events and training CSV files are ignored from Git going forward.
They contain pseudonymous but potentially identifiable behavioral histories.
Previously committed data may still exist in Git history: audit access and
rotate leaked secrets if any; adding `.gitignore` does not remove history.
