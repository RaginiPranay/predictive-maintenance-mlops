
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Hugging Face model repository
MODEL_REPO_ID = "RaginiPranay/predictive-maintenance-model"

# Load the best model from Hugging Face Model Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="best_model.pkl"
    )
    model = joblib.load(model_path)
    return model

model = load_model()

st.set_page_config(
    page_title="Predictive Maintenance App",
    page_icon="🔧",
    layout="centered"
)

st.title("🔧 Predictive Maintenance for Engine Health")
st.write(
    "This application predicts whether an engine is likely to be operating normally "
    "or may require maintenance based on sensor readings."
)

st.subheader("Enter Engine Sensor Readings")

# Input fields based on the actual observed ranges in the dataset
engine_rpm = st.number_input(
    "Engine RPM",
    min_value=0.0,
    max_value=2500.0,
    value=746.0,
    step=10.0
)

lub_oil_pressure = st.number_input(
    "Lubrication Oil Pressure",
    min_value=0.0,
    max_value=8.0,
    value=3.16,
    step=0.1
)

fuel_pressure = st.number_input(
    "Fuel Pressure",
    min_value=0.0,
    max_value=25.0,
    value=6.20,
    step=0.1
)

coolant_pressure = st.number_input(
    "Coolant Pressure",
    min_value=0.0,
    max_value=8.0,
    value=2.17,
    step=0.1
)

lub_oil_temp = st.number_input(
    "Lubrication Oil Temperature",
    min_value=60.0,
    max_value=100.0,
    value=76.82,
    step=0.1
)

coolant_temp = st.number_input(
    "Coolant Temperature",
    min_value=50.0,
    max_value=210.0,
    value=78.35,
    step=0.1
)

# Create dataframe using the same column names used during model training
input_data = pd.DataFrame({
    "Engine rpm": [engine_rpm],
    "Lub oil pressure": [lub_oil_pressure],
    "Fuel pressure": [fuel_pressure],
    "Coolant pressure": [coolant_pressure],
    "lub oil temp": [lub_oil_temp],
    "Coolant temp": [coolant_temp]
})

st.subheader("Input Summary")
st.dataframe(input_data)

if st.button("Predict Engine Condition"):
    prediction = int(model.predict(input_data)[0])

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0]
        class_1_probability = round(probability[1] * 100, 2)
    else:
        class_1_probability = None

    if prediction == 1:
        st.error("Prediction: Maintenance / Fault Risk Detected")
        st.write(
            "The model predicts that the engine may require maintenance or may be showing signs of faulty operation."
        )
    else:
        st.success("Prediction: Normal Engine Condition")
        st.write(
            "The model predicts that the engine is likely operating under normal conditions."
        )

    if class_1_probability is not None:
        st.write(f"Probability of maintenance/faulty condition: {class_1_probability}%")

st.markdown("---")
st.write(
    "Note: This model is intended to support proactive maintenance decisions. "
    "Final maintenance actions should also consider technician review and operational context."
)
