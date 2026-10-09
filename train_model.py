"""Compare regressors by holding out one whole pressure level at a time.

After grouped cross-validation, the best model is refit on the full prepared
dataset and saved as soft_actuator_response_model.joblib.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "pulling_force_ml.csv"
OUT = ROOT / "data" / "processed"
FEATURES = ["pressure_kpa", "displacement_fraction"]
TARGET = "pulling_force_sensor_counts"
GROUP = "pressure_setpoint_kpa"


def candidate_models():
    return {
        "Linear regression": LinearRegression(),
        "Quadratic regression": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            LinearRegression(),
        ),
        "Quadratic ridge": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    x = data[FEATURES].to_numpy(dtype=float)
    y = data[TARGET].to_numpy(dtype=float)
    groups = data[GROUP].to_numpy()
    splitter = LeaveOneGroupOut()

    metric_rows = []
    all_predictions = []
    best_name = None
    best_rmse = float("inf")
    best_model = None
    best_oof = None

    for name, model in candidate_models().items():
        pred = np.full(len(data), np.nan)
        for train_idx, test_idx in splitter.split(x, y, groups):
            fold_model = candidate_models()[name]
            fold_model.fit(x[train_idx], y[train_idx])
            pred[test_idx] = fold_model.predict(x[test_idx])

        rmse = mean_squared_error(y, pred) ** 0.5
        metric_rows.append({
            "model": name,
            "evaluation": "leave-one-pressure-level-out",
            "pressure_levels": int(pd.Series(groups).nunique()),
            "rows": int(len(data)),
            "mae_sensor_counts": mean_absolute_error(y, pred),
            "rmse_sensor_counts": rmse,
            "r2": r2_score(y, pred),
        })
        all_predictions.append(pd.DataFrame({
            "model": name,
            "pressure_setpoint_kpa": groups,
            "measured_sensor_counts": y,
            "predicted_sensor_counts": pred,
        }))
        if rmse < best_rmse:
            best_name, best_rmse, best_model, best_oof = name, rmse, model, pred

    metrics = pd.DataFrame(metric_rows).sort_values("rmse_sensor_counts")
    metrics.to_csv(OUT / "pressure_level_cv_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUT / "pressure_level_cv_predictions.csv", index=False
    )
    by_level = []
    for level in sorted(pd.Series(groups).unique()):
        mask = groups == level
        by_level.append({
            "held_out_pressure_setpoint_kpa": int(level),
            "rows": int(mask.sum()),
            "mae_sensor_counts": mean_absolute_error(y[mask], best_oof[mask]),
            "rmse_sensor_counts": mean_squared_error(y[mask], best_oof[mask]) ** 0.5,
            "r2": r2_score(y[mask], best_oof[mask]),
        })
    pd.DataFrame(by_level).to_csv(OUT / "selected_model_by_pressure_level.csv", index=False)

    best_model.fit(x, y)
    model_path = ROOT / "soft_actuator_response_model.joblib"
    joblib.dump(best_model, model_path)
    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "target": TARGET,
        "target_units": "raw sensor counts (not calibrated to newtons)",
        "displacement_definition": "normalized 0–1 from the supplied position trajectory",
        "training_rows": int(len(data)),
        "pressure_levels_kpa": sorted(map(int, pd.Series(groups).unique())),
        "evaluation": "leave-one-pressure-level-out; one full pressure level is held out per fold",
        "selected_by": "lowest pooled RMSE across pressure-level folds",
        "model_file": model_path.name,
    }
    (ROOT / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False))
    print(f"Selected: {best_name} (pooled LOPO RMSE {best_rmse:.2f} counts)")
    print(f"Saved fitted model: {model_path}")
    print("Prediction target remains raw sensor counts pending force calibration.")


if __name__ == "__main__":
    main()
