"""Exercise automatic real-only training orchestration using mocked components."""
import unittest
from unittest.mock import patch

from pipeline import auto_update


class AutoUpdateTests(unittest.TestCase):
    def test_five_solves_do_not_imply_training_rows(self):
        with patch.object(auto_update, "single_pipeline") as lock, \
             patch.object(auto_update.fetch_say_events, "fetch_events", return_value=5), \
             patch.object(auto_update.build_training_dataset, "build_dataset", return_value=0), \
             patch.object(auto_update.train_real, "train_real") as training:
            lock.return_value.__enter__.return_value = None
            self.assertEqual(
                auto_update.run_cycle(),
                {"events": 5, "rows": 0, "newly_learned": 0},
            )
            training.assert_not_called()

    def test_twelve_outcomes_do_not_activate_synthetic_checkpoint(self):
        with patch.object(auto_update, "single_pipeline") as lock, \
             patch.object(auto_update.fetch_say_events, "fetch_events", return_value=17), \
             patch.object(auto_update.build_training_dataset, "build_dataset", return_value=12), \
             patch.object(auto_update.train_real, "train_real") as training:
            lock.return_value.__enter__.return_value = None
            self.assertEqual(auto_update.run_cycle()["newly_learned"], 0)
            training.assert_not_called()

    def test_validated_real_model_is_published(self):
        with patch.object(auto_update, "single_pipeline") as lock, \
             patch.object(auto_update.fetch_say_events, "fetch_events", return_value=350), \
             patch.object(auto_update.build_training_dataset, "build_dataset", return_value=250), \
             patch.object(auto_update.train_real, "train_real", return_value={
                 "validation": {"passed": True},
             }) as training:
            lock.return_value.__enter__.return_value = None
            self.assertEqual(auto_update.run_cycle()["newly_learned"], 250)
            training.assert_called_once_with()

    def test_insufficient_evidence_preserves_existing_model(self):
        with patch.object(auto_update, "single_pipeline") as lock, \
             patch.object(auto_update.fetch_say_events, "fetch_events", return_value=350), \
             patch.object(auto_update.build_training_dataset, "build_dataset", return_value=250), \
             patch.object(auto_update.train_real, "train_real", side_effect=
                          auto_update.train_real.InsufficientEvidence("not enough users")):
            lock.return_value.__enter__.return_value = None
            self.assertEqual(auto_update.run_cycle()["newly_learned"], 0)

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
