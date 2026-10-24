import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load TensorFlow only when the neural-network model is selected.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
DEFAULT_MODEL_NAME = "Random Forest"

st.set_page_config(page_title="UK Crop Yield Predictor", page_icon="🌾", layout="wide")


@st.cache_data
def load_feature_info():
    path = MODEL_DIR / "feature_info.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model(model_name):
    if model_name == "Neural Network":
        from tensorflow import keras
        path = MODEL_DIR / "neural_network_model.keras"
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path.name}")
        return keras.models.load_model(path)

    filename = {
        "Random Forest": "random_forest_model.joblib",
        "Linear Regression": "linear_regression_model.joblib",
    }[model_name]
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.name}")
    return joblib.load(path)


@st.cache_resource
def load_nn_preprocessor():
    path = MODEL_DIR / "preprocessor.joblib"
    if not path.exists():
        raise FileNotFoundError("preprocessor.joblib is required for the Neural Network.")
    return joblib.load(path)


st.title("🌾 UK Crop Yield Prediction")
st.markdown(
    "Estimate crop yield from region, crop type, year and climate inputs. "
    "Predictions are model estimates, not official agricultural forecasts."
)

try:
    feature_info = load_feature_info()
    categorical_features = feature_info["categorical_features"]
    numerical_features = feature_info["numerical_features"]
except Exception as exc:
    st.error(f"Could not load feature information: {exc}")
    st.stop()

available_models = ["Random Forest", "Linear Regression", "Neural Network"]
selected_model = st.sidebar.selectbox(
    "Prediction model",
    available_models,
    index=available_models.index(DEFAULT_MODEL_NAME),
)
st.sidebar.header("Input features")

# These options follow the lists in the original notebook's Streamlit interface.
# Unknown categories are ignored by the fitted OneHotEncoder, but use training labels
# where possible to keep inputs consistent with the training data.
regions = [
    "East Midlands",
    "Eastern",
    "North East",
    "North West and Merseyside",
    "South East and London",
    "South West",
    "West Midlands",
    "Yorkshire & The Humber",
]
crop_types = ["wheat", "barley", "oats"]

input_values = {}
if "region" in categorical_features:
    input_values["region"] = st.sidebar.selectbox("Region", regions)
if "crop_type" in categorical_features:
    input_values["crop_type"] = st.sidebar.selectbox("Crop type", crop_types)

if "year" in numerical_features:
    input_values["year"] = st.sidebar.number_input(
        "Year", min_value=2000, max_value=2050, value=2023, step=1
    )
if "tas" in numerical_features:
    input_values["tas"] = st.sidebar.slider(
        "Average temperature (°C)", -5.0, 25.0, 10.0, 0.1
    )
if "tasmin" in numerical_features:
    input_values["tasmin"] = st.sidebar.slider(
        "Minimum temperature (°C)", -10.0, 20.0, 5.0, 0.1
    )
if "tasmax" in numerical_features:
    input_values["tasmax"] = st.sidebar.slider(
        "Maximum temperature (°C)", 0.0, 35.0, 15.0, 0.1
    )
if "sun" in numerical_features:
    input_values["sun"] = st.sidebar.slider(
        "Sunshine", 0.0, 2000.0, 150.0, 1.0
    )
if "rainfall" in numerical_features:
    input_values["rainfall"] = st.sidebar.slider(
        "Rainfall (mm)", 0.0, 2000.0, 70.0, 1.0
    )

if "tas_rainfall_interaction" in numerical_features:
    input_values["tas_rainfall_interaction"] = (
        input_values.get("tas", 0.0) * input_values.get("rainfall", 0.0)
    )
if "prev_year_yield" in numerical_features:
    input_values["prev_year_yield"] = st.sidebar.number_input(
        "Previous year's yield (tonnes/ha)",
        min_value=0.0,
        value=7.5,
        step=0.1,
    )

if st.sidebar.button("Predict crop yield", type="primary"):
    expected_features = categorical_features + numerical_features
    missing = [name for name in expected_features if name not in input_values]
    if missing:
        st.error("Missing input features: " + ", ".join(missing))
        st.stop()

    input_df = pd.DataFrame(
        [{name: input_values[name] for name in expected_features}],
        columns=expected_features,
    )

    try:
        model = load_model(selected_model)
        if selected_model == "Neural Network":
            preprocessor = load_nn_preprocessor()
            model_input = preprocessor.transform(input_df)
            prediction = model.predict(model_input, verbose=0)
            predicted_yield = float(np.asarray(prediction).reshape(-1)[0])
        else:
            # The Random Forest and Linear Regression artefacts are full sklearn pipelines.
            prediction = model.predict(input_df)
            predicted_yield = float(np.asarray(prediction).reshape(-1)[0])

        st.subheader("Prediction result")
        st.metric(
            label=f"Estimated yield — {input_values.get('crop_type', 'selected crop')}",
            value=f"{predicted_yield:.2f} tonnes/ha",
        )
        st.caption(
            f"Model: {selected_model} · Region: {input_values.get('region', 'not specified')} · "
            f"Year: {input_values.get('year', 'not specified')}"
        )
        with st.expander("View submitted inputs"):
            st.dataframe(input_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "If this is a model-loading error, check that the dependency versions "
            "match those used when the models were trained."
        )
else:
    st.info("Choose the inputs in the sidebar, then select **Predict crop yield**.")
