# Soft Robotic Materials Response Learning with Machine Learning

A reproducible regression project for learning the pulling response of a soft pneumatic actuator from pressure and normalized displacement.

## Project contents

- `train_model.py` compares four regressors with leave-one-pressure-level-out validation, then fits and saves the selected model.
- `predict_response.py` loads the saved model and predicts a response for a pressure/displacement pair.
- `prepare_and_train.py` rebuilds the prepared table from the original Zenodo recordings and produces an initial baseline comparison.
- `data/processed/` contains the processed modeling table and validation outputs.
- `soft_actuator_response_model.joblib` and `model_metadata.json` are the fitted model and its metadata.
- `presentation/soft-actuator-ml-project.pptx` contains the course presentation.

## Run a prediction

Install dependencies from `requirements.txt`, then run:

```bash
python predict_response.py --pressure-kpa 100 --displacement-fraction 0.5
```

Pressure must be within the measured range (0–200 kPa); normalized displacement must be from 0 to 1.

## Rebuild from the source recordings

Download **Raw Data - Pulling Force.zip** from the [Zenodo record](https://zenodo.org/records/3627901), extract it under `data/pulling-force/`, then run:

```bash
python prepare_and_train.py
python train_model.py
```

The raw recordings are not included in this repository. The Zenodo dataset is by Joshi and Paik, *Multi-DoF Force Characterization of Soft Actuators*, IEEE Robotics and Automation Letters 4(4), 3679–3686 (2019), DOI: [10.1109/LRA.2019.2927936](https://doi.org/10.1109/LRA.2019.2927936). Dataset DOI: [10.5281/zenodo.3627901](https://doi.org/10.5281/zenodo.3627901).

## Model and evaluation

Inputs are pressure (kPa) and normalized displacement. The target is the sign-corrected mean of two opposing horizontal load-cell readings, in raw sensor counts. Four regressors are compared: linear regression, quadratic regression, quadratic ridge, and random forest. Validation holds out one complete pressure setpoint at a time across all nine levels.

Quadratic ridge achieved pooled leave-one-pressure-level-out R² = 0.901, MAE = 135.5 counts, and RMSE = 168.6 counts. Its RMSE was effectively tied with quadratic regression (difference 0.03 counts). Results vary by pressure level, including weak performance at 0 and 50 kPa. This result applies to the recorded actuator/setup and does not show transfer to other designs.

**Units limitation:** The archive does not contain the load-cell calibration factors needed to convert sensor counts to newtons. Position samples are aligned to force samples by interpolation over their shared time range. Verify this processing against the authors' calibration and plotting workflow before interpreting outputs as calibrated force.
