#import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

#create raw dataset
raw_df = pd.read_csv("stroke_risk_dataset.csv")

#create target datasets
binary_target = raw_df["At Risk (Binary)"]
risk_target = raw_df["Stroke Risk (%)"]

#remove target columns from training dataset
train = raw_df.drop(["At Risk (Binary)", "Stroke Risk (%)"], axis=1)

# #train RandomForestClassifier for classification
# RF_model = RandomForestClassifier(n_estimators=200, max_features=0.1, min_samples_split=5, min_samples_leaf=2, max_samples= 50000).fit(train, binary_target)

#train XGBRegressor for regression 
# GB_model = XGBRegressor(random_state=42, learning_rate=0.1, n_estimators=250).fit(train, risk_target)

# Page Configuration
st.set_page_config(page_title="Stroke Risk Prediction", page_icon="🧠", layout="wide")

# Load trained models
if not os.path.exists("GB_model.pkl"):
    st.error("❌ Model files not found! Please train and save the models before running this app.")
    st.stop()

with open("GB_model.pkl", "rb") as gb_file:
    GB_model = pickle.load(gb_file)

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
    st.subheader("📊 Prediction Results")

    # Ensure risk_percentage is within 0-100 range
    risk_percentage = np.clip(risk_percentage, 0, 100)
    # Display progress bar
    st.progress(int(risk_percentage))
    # Display risk percentage
    st.write(f"🩺 **Estimated Stroke Risk:** **{risk_percentage:.2f}%**")
