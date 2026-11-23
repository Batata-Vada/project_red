"""Portable logistic-regression artifact matching SAY-backend recommendations/project_red.py.

Contains no Flask or sklearn dependencies; model.json is sufficient for inference.
The window conversion intentionally mirrors the backend's existing algorithm.
"""

import json
import math
from pathlib import Path

FEATURES = [
    "candidate_rating",
    "last_problem_rating",
    "mean_solve_time_minutes",
    "time_std_minutes",
    "timed_solved_count",
    "last_problem_solve_time_minutes",
]
FEATURE_CONTRACT = "say-practice-v1"
ARTIFACT_SCHEMA_VERSION = 1
MODEL_TYPE = "logistic-regression"
DEFAULT_GRID = [800, 3500, 100]
DEFAULT_TARGET = 0.60
DEFAULT_BOUNDS = [0.45, 0.75]


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_artifact(artifact):
    """Verify the same hard requirements as SAY-backend's _load_artifact()."""
    if not isinstance(artifact, dict):
        raise ValueError("Artifact must be a JSON object")
    if artifact.get("artifact_schema_version") != 1:
        raise ValueError("Unsupported artifact schema")
    if artifact.get("model_type") != MODEL_TYPE:
        raise ValueError("Unsupported model type")
    if artifact.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError("Feature contract mismatch")
    if artifact.get("features") != FEATURES:
        raise ValueError("Feature order mismatch")
    version = artifact.get("model_version")
    if not isinstance(version, str) or not 1 <= len(version) <= 32:
        raise ValueError("Invalid model version")
    for key in ("x_mean", "x_scale", "coef"):
        values = artifact.get(key)
        if not isinstance(values, list) or len(values) != len(FEATURES):
            raise ValueError(f"Invalid {key} length")
        if not all(finite_number(v) for v in values):
            raise ValueError(f"Non-finite {key}")
    if not finite_number(artifact.get("intercept")):
        raise ValueError("Invalid intercept")
    if any(scale <= 0 for scale in artifact["x_scale"]):
        raise ValueError("Nonpositive scale")
    if artifact["coef"][0] >= 0:
        raise ValueError("Probability must decrease with candidate difficulty")

    target = artifact.get("target_probability", DEFAULT_TARGET)
    bounds = artifact.get("acceptable_probability_range", DEFAULT_BOUNDS)
    grid = artifact.get("rating_grid", DEFAULT_GRID)
    if not finite_number(target) or not 0 < target < 1:
        raise ValueError("Invalid target probability")
    if not isinstance(bounds, list) or len(bounds) != 2 or not all(finite_number(x) for x in bounds):
        raise ValueError("Invalid probability bounds")
    if not 0 < bounds[0] < target < bounds[1] < 1:
        raise ValueError("Probability bounds must surround target")
    if (
        not isinstance(grid, list)
        or len(grid) != 3
        or not all(type(x) is int for x in grid)
        or not 800 <= grid[0] < grid[1] <= 3500
        or grid[2] <= 0 or grid[2] % 100 != 0
    ):
        raise ValueError("Invalid rating grid")
    return artifact


def make_artifact(model, scaler, *, version, source, real_events_seen=None):
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "feature_contract": FEATURE_CONTRACT,
        "model_version": version,
        "source": source,
        "features": FEATURES.copy(),
        "x_mean": scaler.mean_.tolist(),
        "x_scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "target_probability": DEFAULT_TARGET,
        "acceptable_probability_range": DEFAULT_BOUNDS.copy(),
        "rating_grid": DEFAULT_GRID.copy(),
    }
    if real_events_seen is not None:
        artifact["real_events_seen"] = real_events_seen
    return validate_artifact(artifact)


def read_artifact(path):
    with Path(path).open(encoding="utf-8") as file:
        return validate_artifact(json.load(file))


def solve_probability(profile, candidate_rating, artifact):
    values = {**profile, "candidate_rating": candidate_rating}
    z = artifact["intercept"]
    for name, mean, scale, coefficient in zip(
        artifact["features"], artifact["x_mean"], artifact["x_scale"], artifact["coef"]
    ):
        value = values[name]
        if not finite_number(value):
            raise ValueError(f"Invalid numeric feature: {name}")
        z += coefficient * ((value - mean) / scale)
    if not math.isfinite(z):
        raise ValueError("Non-finite prediction")
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def predict_range(profile, artifact):
    """Match SAY-backend's rating-window policy (not a midpoint heuristic)."""
    start, stop, step = artifact.get("rating_grid", DEFAULT_GRID)
    target = artifact.get("target_probability", DEFAULT_TARGET)
    low, high = artifact.get("acceptable_probability_range", DEFAULT_BOUNDS)
    options = [(rating, solve_probability(profile, rating, artifact))
               for rating in range(start, stop + 1, step)]
    suitable = [(rating, p) for rating, p in options if low <= p <= high]
    if not suitable:
        raise ValueError("No calibrated difficulty window for the user")
    center = min(suitable, key=lambda item: (abs(item[1] - target), item[0]))[0]
    lower = max(start, min(rating for rating, _ in suitable))
    upper = min(stop, max(rating for rating, _ in suitable))
    if lower == upper:
        lower = max(start, lower - step)
        upper = min(stop, upper + step)
    if lower >= upper:
        raise ValueError("Empty difficulty window")
    return {
        "center": center,
        "width": upper - lower,
        "lower_bound": lower,
        "upper_bound": upper,
        "model_version": artifact["model_version"],
    }
