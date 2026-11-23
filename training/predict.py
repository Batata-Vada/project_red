"""Local smoke test: convert solve probabilities to SAY-style center and width.

This mirrors the backend's inference policy but doesn't call SAY-backend.
"""

import argparse
import json
from pathlib import Path

try:
    from .model_contract import read_artifact, solve_probability, predict_range
except ImportError:
    from model_contract import read_artifact, solve_probability, predict_range

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "artifacts" / "bootstrap_v1" / "model.json"
ONLINE = ROOT / "artifacts" / "online_v1" / "model.json"

# An illustrative profile, not a real SAY user.
EXAMPLE_PROFILE = {
    "last_problem_rating": 1500,
    "mean_solve_time_minutes": 25,
    "time_std_minutes": 8,
    "timed_solved_count": 15,
    "last_problem_solve_time_minutes": 22,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=None,
                        help="Explicit model.json path (otherwise online if present, else bootstrap)")
    parser.add_argument("--profile", type=Path, default=None,
                        help="Optional JSON file containing the five historical features")
    args = parser.parse_args()
    model_path = args.artifact or (ONLINE if ONLINE.exists() else BOOTSTRAP)
    artifact = read_artifact(model_path)
    profile = EXAMPLE_PROFILE.copy()
    if args.profile:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        if not isinstance(profile, dict):
            raise ValueError("Profile JSON must be an object")

    print(f"Model: {model_path}")
    print("\nRATING  SUCCESS PROBABILITY")
    for rating in range(800, 3501, 100):
        print(f"{rating:<7} {solve_probability(profile, rating, artifact):.1%}")

    result = predict_range(profile, artifact)
    print("\nSAY-compatible center and width (local illustration):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
