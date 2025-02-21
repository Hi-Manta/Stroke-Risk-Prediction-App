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
    age = st.slider("📅 Age (years)", min_value=18, max_value=100, value=30, step=1, help="Select your age.")

    st.markdown("**Health Symptoms** (Toggle if present)")
    chest_pain = st.checkbox("❤️ Chest Pain", value=False, help="Do you experience any chest pain?")
    shortness_of_breath = st.checkbox("💨 Shortness of Breath", value=False, help="Do you feel short of breath?")
    irregular_heartbeat = st.checkbox("💓 Irregular Heartbeat", value=False, help="Do you have irregular heartbeats?")
    fatigue = st.checkbox("😴 Fatigue & Weakness", value=False, help="Do you often feel fatigued or weak?")
    dizziness = st.checkbox("🌀 Dizziness", value=False, help="Do you experience dizziness?")
    swelling = st.checkbox("💧 Swelling (Edema)", value=False, help="Do you have swelling in your legs or feet?")
    pain_neck = st.checkbox("🤕 Pain in Neck/Jaw/Shoulder/Back", value=False, help="Do you feel pain in your neck, jaw, shoulder, or back?")

with col2:
    st.markdown("**Other Symptoms** (Toggle if present)")
    sweating = st.checkbox("💦 Excessive Sweating", value=False, help="Do you sweat excessively?")
    cough = st.checkbox("🤧 Persistent Cough", value=False, help="Do you have a persistent cough?")
    nausea = st.checkbox("🤢 Nausea/Vomiting", value=False, help="Do you feel nauseous or have vomiting?")
    high_bp = st.checkbox("🩸 High Blood Pressure", value=False, help="Have you been diagnosed with high blood pressure?")
    chest_discomfort = st.checkbox("💔 Chest Discomfort (Activity)", value=False, help="Do you feel discomfort in your chest during physical activity?")
    cold_hands = st.checkbox("❄️ Cold Hands/Feet", value=False, help="Do you have cold hands or feet?")
    sleep_apnea = st.checkbox("😴 Snoring/Sleep Apnea", value=False, help="Do you snore or have sleep apnea?")
    anxiety = st.checkbox("😟 Anxiety/Feeling of Doom", value=False, help="Do you feel anxious or have a feeling of doom?")

# Convert toggles to binary (True -> 1, False -> 0)
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
