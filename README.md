# Soft Robotic Materials Response Learning with Machine Learning

This project predicts the pulling response of a soft pneumatic actuator from measured pressure and normalized displacement.

## Repository contents

- `data/processed/pulling_force_ml.csv`: prepared modeling dataset (37,073 samples).
- `data/processed/pressure_level_cv_metrics.csv`: pooled model comparison results.
- `data/processed/selected_model_by_pressure_level.csv`: detailed validation metrics for the selected model at each held-out pressure.
- `data/processed/pressure_level_cv_predictions.csv`: out-of-fold predictions from leave-one-pressure-level-out validation.
- `data/processed/baseline_metrics.csv` and `held_out_predictions.csv`: earlier three-level baseline comparison.
- `soft_actuator_response_model.joblib` and `model_metadata.json`: fitted model and metadata.
- `prepare_and_train.py`, `train_model.py`, and `predict_response.py`: reproducible preparation, training, and prediction scripts.
- `app.py`: optional Streamlit explorer for interactive predictions.
- `presentation/soft-actuator-ml-project.pptx`: course presentation.

## Run the interactive explorer

Install dependencies with `python -m pip install -r requirements.txt`, then run:

```bash
python -m streamlit run app.py
```

## Rebuild the data and model

The raw Zenodo archive is not included. Download **Raw Data - Pulling Force.zip** from the [Zenodo record](https://zenodo.org/records/3627901) and extract its files to `data/pulling-force/`. Then run:

```bash
python prepare_and_train.py
python train_model.py
```

For one prediction, run:

```bash
python predict_response.py --pressure-kpa 100 --displacement-fraction 0.5
```

## Model and validation

Inputs are pressure (kPa) and normalized displacement (0–1). The target is the sign-corrected mean of two opposing horizontal load-cell readings in raw sensor counts. Four regressors were compared: linear regression, quadratic regression, quadratic Ridge, and random forest. Validation held out one complete pressure setpoint per fold across all nine settings.

Quadratic Ridge achieved pooled leave-one-pressure-level-out R² = 0.901, MAE = 135.5 counts, and RMSE = 168.6 counts. Its RMSE was effectively tied with quadratic regression (difference 0.03 counts). Scores varied by pressure level, with weak results at 0 and 50 kPa. This evaluates interpolation for one actuator and setup; transfer to other actuator designs has not been established.

**Units limitation:** The archive does not include load-cell calibration factors, so predictions are raw sensor counts, not force in newtons. Position samples were aligned to force samples by interpolation over their shared time range. Verify these processing assumptions against the original calibration/plotting workflow before interpreting outputs as calibrated force.

## Source

Joshi, S. and Paik, J. (2019). “Multi-DoF Force Characterization of Soft Actuators.” *IEEE Robotics and Automation Letters*, 4(4), 3679–3686. DOI: [10.1109/LRA.2019.2927936](https://doi.org/10.1109/LRA.2019.2927936).

Dataset: [Zenodo record 3627901](https://doi.org/10.5281/zenodo.3627901).
