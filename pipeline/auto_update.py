"""Periodically refresh verified SAY events, training data, and the online checkpoint."""
import argparse
import fcntl
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

from pipeline import build_training_dataset, fetch_say_events
from training import train_real

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "data" / ".auto-update.lock"


@contextmanager
def single_pipeline():
    """Prevent overlapping scheduled runs on the same Linux host."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Project Red auto-update process is running") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_cycle():
    """Replace source data only after a complete valid backend export."""
    with single_pipeline():
        events = fetch_say_events.fetch_events()
        rows = build_training_dataset.build_dataset()
        if rows == 0:
            print("Project Red: no eligible training rows; model was not updated.", flush=True)
            return {"events": events, "rows": 0, "newly_learned": 0}

        if rows < train_real.MIN_ROWS:
            print(
                f"Project Red: {events} events and {rows} eligible outcomes; "
                f"real-only deployment requires at least {train_real.MIN_ROWS} "
                "eligible outcomes and independent validation.",
                flush=True,
            )
            return {"events": events, "rows": rows, "newly_learned": 0}

        try:
            trained = train_real.train_real()
        except train_real.InsufficientEvidence as exc:
            logger.info("Real-only model not yet deployable: %s", exc)
            return {"events": events, "rows": rows, "newly_learned": 0}
        if trained["validation"]["passed"] is not True:
            raise RuntimeError("Project Red real-model validation failed")
        print(
            f"Project Red: validated real-only model with {rows} eligible events.",
            flush=True,
        )
        return {"events": events, "rows": rows, "newly_learned": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one complete pipeline cycle")
    parser.add_argument(
        "--interval-seconds", type=int,
        default=int(os.getenv("PROJECT_RED_PIPELINE_INTERVAL_SECONDS", "180")),
        help="Interval between complete export/training cycles (default: 180 seconds)",
    )
    args = parser.parse_args()
    if not 60 <= args.interval_seconds <= 86400:
        parser.error("--interval-seconds must be between 60 and 86400")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not os.getenv("SAY_BACKEND_URL") or not os.getenv("SAY_PROJECT_RED_EXPORT_TOKEN"):
        parser.error("Set SAY_BACKEND_URL and SAY_PROJECT_RED_EXPORT_TOKEN before running")

    print(f"Project Red pipeline started; interval={args.interval_seconds}s", flush=True)
    if args.once:
        run_cycle()
        return

    while True:
        try:
            run_cycle()
        except Exception:
            logger.exception("Project Red update failed; existing verified artifacts are unchanged")
        try:
            time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            print("Project Red pipeline stopped", flush=True)
            break


if __name__ == "__main__":
    main()
