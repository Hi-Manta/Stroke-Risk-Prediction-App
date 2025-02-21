#import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBRegressor

# Page Configuration
st.set_page_config(page_title="Stroke Risk Prediction", page_icon="🧠", layout="wide")

# Load trained model
model_path = "GB_model.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as gb_file:
        GB_model = pickle.load(gb_file)
else:
    st.error("❌ Model file not found! Please train and save the model before running this app.")
    st.stop()

# 🌟 UI Improvements
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#4A90E2;"> 🧠 Stroke Risk Prediction</h1>
        <p style="font-size:18px;">Enter your health details below, and our AI will predict your stroke risk.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# 💡 User-friendly Input Form
st.subheader("📝 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("📅 Age", min_value=18, max_value=100, value=30, step=1)

    chest_pain = st.toggle("❤️ Chest Pain")
    shortness_of_breath = st.toggle("💨 Shortness of Breath")
    irregular_heartbeat = st.toggle("💓 Irregular Heartbeat")
    fatigue = st.toggle("😴 Fatigue & Weakness")
    dizziness = st.toggle("🌀 Dizziness")
    swelling = st.toggle("💧 Swelling (Edema)")
    pain_neck = st.toggle("🤕 Pain in Neck/Jaw/Shoulder/Back")

with col2:
    sweating = st.toggle("💦 Excessive Sweating")
    cough = st.toggle("🤧 Persistent Cough")
    nausea = st.toggle("🤢 Nausea/Vomiting")
    high_bp = st.toggle("🩸 High Blood Pressure")
    chest_discomfort = st.toggle("💔 Chest Discomfort (Activity)")
    cold_hands = st.toggle("❄️ Cold Hands/Feet")
    sleep_apnea = st.toggle("😴 Snoring/Sleep Apnea")
    anxiety = st.toggle("😟 Anxiety/Feeling of Doom")

# Convert toggles to binary (True -> 1, False -> 0)
feature_order = [
    "Chest Pain", "Shortness of Breath", "Irregular Heartbeat", "Fatigue & Weakness",
    "Dizziness", "Swelling (Edema)", "Pain in Neck/Jaw/Shoulder/Back", "Excessive Sweating",
    "Persistent Cough", "Nausea/Vomiting", "High Blood Pressure", "Chest Discomfort (Activity)",
    "Cold Hands/Feet", "Snoring/Sleep Apnea", "Anxiety/Feeling of Doom", "Age"
]

input_data = np.array([
    int(chest_pain), int(shortness_of_breath), int(irregular_heartbeat),
    int(fatigue), int(dizziness), int(swelling), int(pain_neck), int(sweating),
    int(cough), int(nausea), int(high_bp), int(chest_discomfort), int(cold_hands),
    int(sleep_apnea), int(anxiety), age
]).reshape(1, -1)

st.divider()

# 🚀 Predict when button is clicked
if st.button("🔍 Predict Stroke Risk", use_container_width=True):
    risk_percentage = GB_model.predict(input_data)[0]

    # Ensure risk_percentage is within 0-100 range
    risk_percentage = np.clip(risk_percentage, 0, 100)

    # Display risk level based on percentage
    if risk_percentage < 20:
        risk_status = "🟢 Low Risk"
        risk_color = "green"
    elif risk_percentage < 50:
        risk_status = "🟡 Moderate Risk"
        risk_color = "orange"
    else:
        risk_status = "🔴 High Risk"
        risk_color = "red"

    # Display results
    st.subheader("📊 Prediction Results")
    st.markdown(f"<h3 style='color:{risk_color};'> {risk_status} ({risk_percentage:.2f}%) </h3>", unsafe_allow_html=True)
    st.progress(int(risk_percentage))
