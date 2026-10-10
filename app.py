"""Interactive Streamlit interface for the saved soft actuator model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "soft_actuator_response_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Soft Actuator Response Explorer", page_icon="🤖", layout="centered")
st.title("Soft Actuator Response Explorer")
st.write(
    "Explore how the trained regression model estimates the pulling-response "
    "sensor signal from pressure and normalized displacement."
)

with st.sidebar:
    st.header("Choose actuator state")
    pressure = st.slider("Pressure (kPa)", min_value=0, max_value=200, value=100, step=1)
    displacement = st.slider(
        "Normalized displacement", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )

model = load_model()
prediction = float(model.predict([[pressure, displacement]])[0])
left, middle, right = st.columns(3)
left.metric("Pressure", f"{pressure} kPa")
middle.metric("Displacement", f"{displacement:.2f}")
right.metric("Predicted sensor response", f"{prediction:,.1f} counts")

st.subheader("Predicted response across displacement")
displacements = np.linspace(0, 1, 101)
features = np.column_stack((np.full_like(displacements, pressure), displacements))
curve = pd.DataFrame(
    {"Predicted response (raw counts)": model.predict(features)},
    index=pd.Index(displacements, name="Normalized displacement"),
)
st.line_chart(curve)
st.caption(f"The selected state is displacement {displacement:.2f} at {pressure} kPa.")

st.warning(
    "Values are raw sensor counts, not force in newtons. Load-cell calibration "
    "factors were not supplied with the dataset."
)
st.info(
    "The model was evaluated on one actuator and test setup. Validation scores "
    "varied by pressure level, with weaker results at 0 and 50 kPa. Use the app "
    "to explore the fitted model, not as a calibrated control or safety tool."
)
