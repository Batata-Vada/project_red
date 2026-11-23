"""Offline tests. All events here are simulated fixtures, not real SAY data."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline import build_training_dataset as dataset
from pipeline import fetch_say_events as fetch


HISTORY = {
    "last_problem_rating": 1500,
    "mean_solve_time_minutes": 25.0,
    "time_std_minutes": 8.0,
    "timed_solved_count": 8,
    "last_problem_solve_time_minutes": 22.0,
}


def event(event_id, outcome="SOLVED", *, history=HISTORY):
    solved = outcome == "SOLVED"
    failed = outcome == "FAILED"
    return {
        "event_id": event_id,
        "user_key": "test-only-pseudonym",
        "problem_external_id": f"1000{event_id}A",
        "problem_rating": 1600,
        "tags": ["implementation"],
        "started_at": "2026-01-01T10:00:00+00:00",
        "finished_at": "2026-01-01T10:20:00+00:00",
        "duration_seconds": 1200,
        "official_solved": solved,
        "official_attempt_observed": solved or failed,
        "final_failure_observed": failed,
        "previously_solved": False,
        "full_history_coverage": failed,
        "outcome_status": outcome,
        "history_features": history,
        "recommendation_run_id": None,
    }


class PipelineTests(unittest.TestCase):
    def test_v2_event_contract(self):
        self.assertEqual(fetch.validate_event(event(5), 0), 5)
        with self.assertRaises(ValueError):
            fetch.validate_event(event(5), 5)
        e = event(6, "FAILED")
        e["full_history_coverage"] = False
        with self.assertRaises(ValueError):
            fetch.validate_event(e, 5)
        e = event(7, "UNKNOWN")
        self.assertEqual(fetch.validate_event(e, 5), 7)
        e = event(8, "SOLVED", history=None)
        self.assertEqual(fetch.validate_event(e, 7), 8)

    def test_response_contract(self):
        payload = {
            "schema_version": 2,
            "feature_contract": "say-practice-v1",
            "events": [],
            "next_after_id": None,
        }
        self.assertEqual(fetch.validate_response(payload), [])
        payload["schema_version"] = 1
        with self.assertRaises(ValueError):
            fetch.validate_response(payload)

    def test_builder_labels_and_unknown_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src, dst = root / "events.jsonl", root / "training.csv"
            events = [event(1, "SOLVED"), event(2, "FAILED"), event(3, "UNKNOWN")]
            src.write_text("".join(json.dumps(e) + "\n" for e in events))
            with patch.object(dataset, "INPUT_FILE", src), patch.object(dataset, "OUTPUT_FILE", dst):
                self.assertEqual(dataset.build_dataset(), 2)
            with dst.open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r["label"] for r in rows], ["1", "0"])
            self.assertEqual(rows[0]["last_problem_rating"], "1500")

    def test_builder_empty_file_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src, dst = root / "events.jsonl", root / "training.csv"
            src.write_text("")
            with patch.object(dataset, "INPUT_FILE", src), patch.object(dataset, "OUTPUT_FILE", dst):
                self.assertEqual(dataset.build_dataset(), 0)
                existing = dst.read_text()
                bad_event = event(5, "FAILED")
                bad_event["official_solved"] = True
                src.write_text(json.dumps(bad_event) + "\n")
                with self.assertRaises(ValueError):
                    dataset.build_dataset()
                self.assertEqual(dst.read_text(), existing)

    def test_fetch_empty_success_and_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("old-data\n")
            response = MagicMock()
            response.json.return_value = {
                "schema_version": 2,
                "feature_contract": "say-practice-v1",
                "events": [], "next_after_id": None,
            }
            mock_session = MagicMock()
            mock_session.get.return_value = response
            with patch.dict(os.environ, {
                "SAY_BACKEND_URL": "http://127.0.0.1:5001",
                "SAY_PROJECT_RED_EXPORT_TOKEN": "test-token",
            }), patch.object(fetch, "OUTPUT", path), patch.object(fetch.requests, "Session") as session_ctor:
                session_ctor.return_value.__enter__.return_value = mock_session
                # An unexpectedly empty export must not erase previously collected events.
                with self.assertRaisesRegex(RuntimeError, "0 events"):
                    fetch.fetch_events()
                self.assertEqual(path.read_text(), "old-data\n")
                path.write_text("old-data\n")
                response.raise_for_status.side_effect = RuntimeError("Simulated API error")
                with self.assertRaises(RuntimeError):
                    fetch.fetch_events()
                self.assertEqual(path.read_text(), "old-data\n")


if __name__ == "__main__":
    unittest.main()
