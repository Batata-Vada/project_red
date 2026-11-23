# Project Red automatic training

The SAY integration collects **verified** Codeforces submissions in SAY-backend.
The Project Red pipeline performs a complete authenticated export, validates
events, regenerates `data/training_rows.csv`, and incrementally updates
`artifacts/online_v1/{learner.joblib,model.json}` whenever eligible data exists.

The easiest local startup is the SAY-backend sibling launcher:

```bash
cd ../SAY-backend
.venv/bin/python scripts/run_local.py
```

It reads `SAY-backend/.env` and securely passes the export token to the
Project Red worker. No export token is sent to the browser or committed.

To run only this pipeline independently, set `SAY_BACKEND_URL` and
`SAY_PROJECT_RED_EXPORT_TOKEN` and execute:

```bash
.venv/bin/python -m pipeline.auto_update --interval-seconds 180
```

The optional `--once` flag runs one full export/dataset/training cycle.
Keep one worker active per machine; a file lock prevents concurrent cycles.

An empty training CSV is expected until the SAY backend has verified enough
**historically observed** timed solves. The pipeline keeps all existing
filtering, pseudonymization, and outcome rules. Changed old outcomes trigger
a full replay from the bootstrap checkpoint, never double counting.
Data collection does not automatically declare a model calibrated or ready
for deployment. `PROJECT_RED_ENABLED` remains an explicit backend setting;
a live model should be evaluated on held-out real outcomes before enabling.
