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
    st.error("❌ Oops! Model file not found! Please train and save the model before running this app.")
    st.stop()

# 🌟 UI Improvements
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#4A90E2;"> 🧠 Stroke Risk Prediction</h1>
        <p style="font-size:18px;">Welcome! Please enter your health details below, and we'll help you assess your stroke risk.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# 💡 User-friendly Input Form
st.subheader("📝 Let's Get Your Information")

st.write("Please provide the following details. If you're unsure about any symptoms, feel free to leave them unchecked.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("📅 Your Age (years)", min_value=18, max_value=100, value=30, step=1, help="Select your age.")

    st.markdown("**Do you currently experience any of the following symptoms?**")
    chest_pain = st.checkbox("❤️ Chest Pain", value=False, help="Check this if you feel any chest pain.")
    shortness_of_breath = st.checkbox("💨 Shortness of Breath", value=False, help="Check this if you feel short of breath.")
    irregular_heartbeat = st.checkbox("💓 Irregular Heartbeat", value=False, help="Check this if your heartbeat feels irregular.")
    fatigue = st.checkbox("😴 Fatigue or Weakness", value=False, help="Check this if you often feel tired or weak.")
    dizziness = st.checkbox("🌀 Dizziness", value=False, help="Check this if you experience dizziness.")
    swelling = st.checkbox("💧 Swelling in Legs or Feet", value=False, help="Check this if you have swelling in your legs or feet.")
    pain_neck = st.checkbox("🤕 Pain in Neck/Jaw/Shoulder/Back", value=False, help="Check this if you feel pain in these areas.")

with col2:
    st.markdown("**And any of these symptoms?**")
    sweating = st.checkbox("💦 Excessive Sweating", value=False, help="Check this if you sweat excessively.")
    cough = st.checkbox("🤧 Persistent Cough", value=False, help="Check this if you have a cough that won't go away.")
    nausea = st.checkbox("🤢 Nausea or Vomiting", value=False, help="Check this if you feel nauseous or have vomited.")
    high_bp = st.checkbox("🩸 High Blood Pressure", value=False, help="Check this if you have been diagnosed with high blood pressure.")
    chest_discomfort = st.checkbox("💔 Chest Discomfort during Activity", value=False, help="Check this if you feel discomfort during physical activity.")
    cold_hands = st.checkbox("❄️ Cold Hands or Feet", value=False, help="Check this if your hands or feet feel cold.")
    sleep_apnea = st.checkbox("😴 Snoring or Sleep Apnea", value=False, help="Check this if you snore or have sleep apnea.")
    anxiety = st.checkbox("😟 Anxiety or Feeling of Doom", value=False, help="Check this if you often feel anxious or uneasy.")

# Convert checkbox inputs to binary (True -> 1, False -> 0)
input_data = np.array([
    int(chest_pain), int(shortness_of_breath), int(irregular_heartbeat),
    int(fatigue), int(dizziness), int(swelling), int(pain_neck), int(sweating),
    int(cough), int(nausea), int(high_bp), int(chest_discomfort), int(cold_hands),
    int(sleep_apnea), int(anxiety), age
]).reshape(1, -1)

st.divider()

# 🚀 Predict when button is clicked
if st.button("🔍 Calculate My Stroke Risk", use_container_width=True):
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
    st.subheader("📊 Your Prediction Results")
    st.markdown(f"<h3 style='color:{risk_color};'> {risk_status} (Risk Level: {risk_percentage:.2f}%) </h3>", unsafe_allow_html=True)
    st.progress(int(risk_percentage))

    # Additional information
    st.write("Based on your input, this is your estimated stroke risk. If you have concerns, please consult a healthcare professional.")
