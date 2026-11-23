"""Add difficulty-aware last-solve times and adjust Project Red recommendations.

Uses a bounded, skewed mixture of solve-speed multipliers rather than a normal draw.
The output is synthetic training data, not observed user performance.
"""
import numpy as np
import pandas as pd

SOURCE = './project_red_center_width.csv'
DEST = './project_red_v5_data_set.csv'
rng = np.random.default_rng(20261017)
df = pd.read_csv(SOURCE)

rating = df['user_rating'].to_numpy(dtype=float)
last = df['last_problem_rating'].to_numpy(dtype=float)
mean = df['mean_solve_time_minutes'].to_numpy(dtype=float)
std = df['time_std_minutes'].to_numpy(dtype=float)
experience = df['problems_solved'].to_numpy(dtype=float)
n = len(df)

# Harder-than-usual tasks take longer; experience modestly reduces time.
gap = last - rating
difficulty_factor = np.clip(np.exp(gap / 1000), 0.62, 2.15)
experience_factor = np.clip((100 / np.maximum(experience, 10)) ** 0.065, 0.78, 1.18)
expected = mean * difficulty_factor * experience_factor

# Typical performance has asymmetric variation; no Gaussian sampling.
# Relative dispersion comes from the user's recorded time standard deviation.
cv = np.clip(std / np.maximum(mean, 1), 0.06, 1.0)
noise = rng.triangular(-0.9, 0.0, 1.3, size=n)
time = expected * (1 + noise * cv * 0.75)

# Explicit, rare exceptional sessions: quick insight, stalled solution, extreme struggle.
u = rng.random(n)
fast = u < 0.018
slow = (u >= 0.018) & (u < 0.052)
extreme = (u >= 0.052) & (u < 0.060)
time[fast] *= rng.uniform(0.16, 0.38, fast.sum())
time[slow] *= rng.uniform(2.3, 4.5, slow.sum())
time[extreme] *= rng.uniform(5.0, 8.0, extreme.sum())
time = np.clip(time, 1.0, 480.0)

# Response to performance vs the difficulty-adjusted expectation.
# Fast completion raises recommended center, narrows range; slow does opposite.
relative_speed = np.log(np.maximum(expected, 1) / time)
relative_speed = np.clip(relative_speed, -1.8, 1.8)
# Less confidence in one observation for highly variable users.
confidence = np.clip(0.85 - 0.35 * cv, 0.45, 0.83)
center_shift = 105 * confidence * relative_speed
width_multiplier = np.clip(np.exp(-0.23 * confidence * relative_speed), 0.72, 1.45)

# Preserve original generated signal while adjusting for new information.
new_center = np.clip(df['center'].to_numpy(float) + center_shift, 800, 3500)
new_width = np.clip(df['width'].to_numpy(float) * width_multiplier, 120, 1200)

# Keep original dataset's sensible lower recommendation floor (200).
new_lower = np.maximum(200, new_center - new_width / 2)
new_upper = new_center + new_width / 2
# If lower bound hit floor, adjust width and center to remain consistent.
new_width = new_upper - new_lower
new_center = (new_lower + new_upper) / 2

idx = df.columns.get_loc('mean_solve_time_minutes')
df.insert(idx, 'last_problem_solve_time_minutes', np.round(time, 2))
df.insert(idx + 1, 'last_solve_time_outlier', np.select([fast, slow, extreme], ['unusually_fast', 'unusually_slow', 'extremely_slow'], default='typical'))
df['center'] = np.round(new_center, 2)
df['width'] = np.round(new_width, 2)
df['lower_bound'] = np.round(new_lower, 2)
df['upper_bound'] = np.round(new_upper, 2)
# Recompute width and center from rounded bounds for exact two-decimal consistency.
df['width'] = np.round(df['upper_bound'] - df['lower_bound'], 2)
df['center'] = np.round((df['lower_bound'] + df['upper_bound']) / 2, 2)
df.to_csv(DEST, index=False)

print('Saved', DEST)
print('Shape:', df.shape)
print('Outliers:', df['last_solve_time_outlier'].value_counts().to_dict())
print('Solve time percentiles:', df['last_problem_solve_time_minutes'].quantile([0, .01, .5, .95, .99, 1]).to_dict())
print('Center changes:', int((np.abs(df['center'] - pd.read_csv(SOURCE)['center']) > .01).sum()))
print('Width changes:', int((np.abs(df['width'] - pd.read_csv(SOURCE)['width']) > .01).sum()))
print(df.head(5).to_string(index=False))
