"""Verify automated training orchestration without fetching live Codeforces data."""
import unittest
from unittest.mock import patch

from pipeline import auto_update


class AutoUpdateTests(unittest.TestCase):
    def test_no_training_until_history_is_eligible(self):
        with patch.object(auto_update, "single_pipeline") as lock, \
             patch.object(auto_update.fetch_say_events, "fetch_events", return_value=4), \
             patch.object(auto_update.build_training_dataset, "build_dataset", return_value=0), \
             patch.object(auto_update.online_update, "online_update") as train:
            lock.return_value.__enter__.return_value = None
            self.assertEqual(
                auto_update.run_cycle(),
                {"events": 4, "rows": 0, "newly_learned": 0},
            )
            train.assert_not_called()

    def test_new_training_rows_are_processed(self):
        with patch.object(auto_update, "single_pipeline") as lock, \
             patch.object(auto_update.fetch_say_events, "fetch_events", return_value=6), \
             patch.object(auto_update.build_training_dataset, "build_dataset", return_value=2), \
             patch.object(auto_update.online_update, "online_update", return_value=2), \
             patch.object(auto_update.model_contract, "read_artifact", return_value={"real_events_seen": 2}):
            lock.return_value.__enter__.return_value = None
            self.assertEqual(auto_update.run_cycle()["newly_learned"], 2)

    def test_official_rejudge_replays_updated_dataset(self):
        with patch.object(auto_update, "single_pipeline") as lock, \
             patch.object(auto_update.fetch_say_events, "fetch_events", return_value=7), \
             patch.object(auto_update.build_training_dataset, "build_dataset", return_value=3), \
             patch.object(auto_update.online_update, "online_update", side_effect=[
                 RuntimeError("Previously trained event 5 changed. Review data, then use --rebuild."),
                 3,
             ]) as train, \
             patch.object(auto_update.model_contract, "read_artifact", return_value={"real_events_seen": 3}):
            lock.return_value.__enter__.return_value = None
            self.assertEqual(auto_update.run_cycle()["rows"], 3)
            self.assertEqual(train.call_args_list[1].kwargs, {"rebuild": True})

    def test_export_failure_does_not_train(self):
        with patch.object(auto_update, "single_pipeline") as lock, \
             patch.object(auto_update.fetch_say_events, "fetch_events", side_effect=RuntimeError("offline")), \
             patch.object(auto_update.build_training_dataset, "build_dataset") as builder:
            lock.return_value.__enter__.return_value = None
            with self.assertRaises(RuntimeError):
                auto_update.run_cycle()
            builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
