# Soft actuator pulling-response ML project

This project trains and compares regression models for the pulling response of a soft pneumatic actuator.

## Run it

1. Download **Raw Data - Pulling Force.zip** from the [Zenodo record](https://zenodo.org/records/3627901).
2. Extract it to `data/pulling-force/`, keeping the original file names.
3. Install the packages listed in `requirements.txt`.
4. Run `python prepare_and_train.py` to rebuild the prepared table and initial three-model comparison.
5. Run `python train_model.py` to compare models with leave-one-pressure-level-out validation and save the selected model.
6. Request a prediction with `python predict_response.py --pressure-kpa 100 --displacement-fraction 0.5`.

The scripts create prepared data, initial comparison results, cross-validation metrics, per-pressure-level metrics, and predictions in `data/processed/`. `train_model.py` saves `soft_actuator_response_model.joblib` and `model_metadata.json`.

## Current model setup

- Inputs: measured pressure in kPa and displacement normalized from 0 to 1.
- Target: sign-corrected mean of the two opposing horizontal load-cell readings, in raw sensor counts.
- Evaluation: leave one complete pressure level out per fold, repeated for all nine settings.
- Model comparison: linear regression, quadratic regression, quadratic ridge, and random forest.

The archive does not contain the load-cell calibration factors, so the target is not reported in newtons. Position samples are aligned to force samples by interpolation over their shared time range. Verify these processing assumptions against the authors' original calibration/plotting workflow before treating predictions as calibrated forces.

## Current result

Quadratic ridge achieved pooled leave-one-pressure-level-out R² = 0.901, MAE = 135.5 sensor counts, and RMSE = 168.6 sensor counts. Unregularized quadratic regression was essentially tied (RMSE = 168.6 counts; the difference was 0.03 counts). Fold-level performance varied: R² was negative at 0 kPa and near zero at 50 kPa. This evaluates interpolation within one actuator and test setup; it does not establish transfer to another material formulation or actuator design.

## Source

Joshi, S. and Paik, J. (2019). “Multi-DoF Force Characterization of Soft Actuators.” *IEEE Robotics and Automation Letters*, 4(4), 3679–3686. DOI: 10.1109/LRA.2019.2927936.

Dataset: https://zenodo.org/records/3627901, DOI: 10.5281/zenodo.3627901.

## Interactive interface

Install the packages in `requirements.txt`, then start the app with:

```bash
streamlit run app.py
```

The page provides pressure and displacement sliders, displays a single predicted response, and plots the model's predicted response across the displacement range at the selected pressure. Predictions remain raw sensor counts, not newtons.

## Interactive interface

Install the packages in `requirements.txt`, then start the app with:

```bash
streamlit run app.py
```

The page provides pressure and displacement sliders, shows a predicted response, and plots the predicted response curve at the selected pressure. Values are raw sensor counts, not newtons.
