"""Predict raw pulling-response sensor counts at a pressure and displacement."""

import argparse
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "soft_actuator_response_model.joblib"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure-kpa", type=float, required=True)
    parser.add_argument(
        "--displacement-fraction", type=float, required=True,
        help="Normalized displacement from 0 to 1",
    )
    args = parser.parse_args()
    if not 0 <= args.displacement_fraction <= 1:
        parser.error("--displacement-fraction must be between 0 and 1")
    if not 0 <= args.pressure_kpa <= 200:
        parser.error("--pressure-kpa must be within the measured 0–200 kPa range")

    model = joblib.load(MODEL_PATH)
    prediction = model.predict([[args.pressure_kpa, args.displacement_fraction]])[0]
    print(f"Predicted pulling-response sensor signal: {prediction:.1f} raw counts")
    print("This is not a force value in newtons; calibration is not included.")


if __name__ == "__main__":
    main()
