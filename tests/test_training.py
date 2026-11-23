"""Offline training tests with small simulated rows; no real SAY data."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from training import pretrain, online_update, model_contract


class TrainingTests(unittest.TestCase):
    def _prepare_bootstrap(self, root):
        with patch.object(pretrain, "ROOT", root), patch.object(pretrain, "OUTPUT", root / "artifacts/bootstrap_v1"):
            pretrain.pretrain()

    def _online_paths(self, root):
        return (
            patch.object(online_update, "DATA_PATH", root / "data/training_rows.csv"),
            patch.object(online_update, "BOOTSTRAP", root / "artifacts/bootstrap_v1/learner.joblib"),
            patch.object(online_update, "OUTPUT_DIR", root / "artifacts/online_v1"),
            patch.object(online_update, "CHECKPOINT_PATH", root / "artifacts/online_v1/learner.joblib"),
            patch.object(online_update, "INFERENCE_PATH", root / "artifacts/online_v1/model.json"),
        )

    def test_pretrain_and_incremental_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare_bootstrap(root)
            baseline = model_contract.read_artifact(root / "artifacts/bootstrap_v1/model.json")
            self.assertLess(baseline["coef"][0], 0)
            input_file = root / "data/training_rows.csv"
            input_file.parent.mkdir(parents=True)
            with input_file.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=online_update.COLUMNS)
                writer.writeheader()
                writer.writerow({
                    "event_id": 1,
                    "user_key": "synthetic-test-user",
                    "started_at": "2026-01-01T12:00:00+00:00",
                    "candidate_rating": 1600,
                    "last_problem_rating": 1500,
                    "mean_solve_time_minutes": 25,
                    "time_std_minutes": 8,
                    "timed_solved_count": 8,
                    "last_problem_solve_time_minutes": 22,
                    "label": 1,
                })

            from contextlib import ExitStack
            with ExitStack() as stack:
                for context in self._online_paths(root):
                    stack.enter_context(context)
                self.assertEqual(online_update.online_update(), 1)
                artifact = model_contract.read_artifact(root / "artifacts/online_v1/model.json")
                self.assertEqual(artifact["real_events_seen"], 1)
                checkpoint = joblib.load(root / "artifacts/online_v1/learner.joblib")
                before = checkpoint["model"].coef_.copy()
                self.assertEqual(online_update.online_update(), 0)
                after = joblib.load(root / "artifacts/online_v1/learner.joblib")["model"].coef_.copy()
                self.assertTrue((before == after).all())
                # Correction triggers manual review, not double-counted training.
                updated = input_file.read_text().replace(",1\n", ",0\n")
                input_file.write_text(updated)
                with self.assertRaises(RuntimeError):
                    online_update.online_update()

    def test_duplicate_event_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "rows.csv"
            lines = [",".join(online_update.COLUMNS),
                     "test,1,2026-01-01T00:00:00+00:00,1600,1500,25,8,8,22,1"]
            p.write_text("\n".join(lines + [lines[-1]]) + "\n")
            with patch.object(online_update, "DATA_PATH", p):
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    online_update.load_rows()


if __name__ == "__main__":
    unittest.main()
