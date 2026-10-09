"""Prepare the Zenodo pulling-force recordings and compare baseline regressors.

The force target remains in raw sensor counts because the archive does not
include the load-cell calibration factors. See the root README.md for the calibration caveat.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "pulling-force"
OUT = ROOT / "data" / "processed"


def main():
    trajectory = np.loadtxt(DATA / "XY_trajectory.csv", delimiter=",")
    t_position = trajectory[:, 0]
    # The two end modules move together in the recorded coordinate frame.
    position_counts = (np.abs(trajectory[:, 1]) + np.abs(trajectory[:, 2])) / 2
    position_fraction = position_counts / position_counts.max()

    frames = []
    for path in sorted(DATA.glob("P*.csv")):
        match = re.fullmatch(r"P(\d+)_([1-4])\.csv", path.name)
        if not match:
            continue
        setpoint_kpa, repeat = map(int, match.groups())
        raw = np.loadtxt(path, delimiter=",")
        t_force = raw[:, 0]

        # The paper defines pulling force as the mean horizontal force from
        # modules 1 and 3. Their sensor axes have opposite signs in the raw
        # recordings, so sign correction is applied before averaging.
        pulling_force_counts = (raw[:, 1] - raw[:, 5]) / 2
        pressure_kpa = raw[:, 7]

        valid = (t_force >= t_position.min()) & (t_force <= t_position.max())
        if not valid.any():
            continue

        frames.append(pd.DataFrame({
            "pressure_kpa": pressure_kpa[valid],
            "displacement_fraction": np.interp(
                t_force[valid], t_position, position_fraction
            ),
            "pulling_force_sensor_counts": pulling_force_counts[valid],
            "trial": path.stem,
            "pressure_setpoint_kpa": setpoint_kpa,
            "repeat": repeat,
        }))

    data = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "pulling_force_ml.csv", index=False)

    # Interpolation check: reserve complete pressure levels, rather than
    # randomly splitting adjacent samples from the same repeated trajectories.
    held_out = {75, 125, 175}
    test = data["pressure_setpoint_kpa"].isin(held_out).to_numpy()
    x = data[["pressure_kpa", "displacement_fraction"]].to_numpy()
    y = data["pulling_force_sensor_counts"].to_numpy()

    models = {
        "Linear regression": LinearRegression(),
        "Quadratic regression": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
    }
    scores = []
    prediction_tables = []
    for name, model in models.items():
        model.fit(x[~test], y[~test])
        prediction = model.predict(x[test])
        prediction_tables.append(pd.DataFrame({
            "model": name,
            "measured_sensor_counts": y[test],
            "predicted_sensor_counts": prediction,
            "pressure_kpa": data.loc[test, "pressure_kpa"].to_numpy(),
            "displacement_fraction": data.loc[test, "displacement_fraction"].to_numpy(),
            "trial": data.loc[test, "trial"].to_numpy(),
        }))
        scores.append({
            "model": name,
            "test_pressure_levels_kpa": ", ".join(map(str, sorted(held_out))),
            "test_rows": int(test.sum()),
            "mae_sensor_counts": mean_absolute_error(y[test], prediction),
            "rmse_sensor_counts": mean_squared_error(y[test], prediction) ** 0.5,
            "r2": r2_score(y[test], prediction),
        })

    pd.DataFrame(scores).to_csv(OUT / "baseline_metrics.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        OUT / "held_out_predictions.csv", index=False
    )
    print(f"Saved {len(data):,} aligned rows from {data['trial'].nunique()} trials.")
    print(pd.DataFrame(scores).to_string(index=False))
    print("Target unit: raw sensor counts; calibration to newtons is not provided.")


if __name__ == "__main__":
    main()

