"""Local checkpoint/inference parity check; needs synthetic pretraining first."""

from pathlib import Path

import joblib
import numpy as np

try:
    from .model_contract import FEATURES, read_artifact, solve_probability, predict_range
except ImportError:
    from model_contract import FEATURES, read_artifact, solve_probability, predict_range

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "bootstrap_v1"
PROFILE = {
    "last_problem_rating": 1500,
    "mean_solve_time_minutes": 25.0,
    "time_std_minutes": 8.0,
    "timed_solved_count": 15,
    "last_problem_solve_time_minutes": 22.0,
}


def main():
    checkpoint = joblib.load(BASE / "learner.joblib")  # Trusted local file only.
    artifact = read_artifact(BASE / "model.json")
    assert checkpoint["features"] == FEATURES
    features = checkpoint["features"]
    probs = []
    for rating in (1000, 1200, 1400, 1600, 1800, 2000):
        sample = {"candidate_rating": rating, **PROFILE}
        X = np.array([[sample[name] for name in features]], dtype=np.float64)
        sklearn_prob = float(checkpoint["model"].predict_proba(
            checkpoint["scaler"].transform(X)
        )[0, 1])
        json_prob = solve_probability(PROFILE, rating, artifact)
        assert abs(json_prob - sklearn_prob) < 1e-9, "JSON and sklearn disagree"
        probs.append(json_prob)
        print(f"Rating {rating}: {json_prob:.2%} estimated success (synthetic model)")

    assert all(a > b for a, b in zip(probs, probs[1:])), "Difficulty curve isn't decreasing"
    result = predict_range(PROFILE, artifact)
    assert result["width"] == result["upper_bound"] - result["lower_bound"]
    print("\nLocal prediction checks PASSED")
    print(f"Center: {result['center']}; width: {result['width']}")


if __name__ == "__main__":
    main()
