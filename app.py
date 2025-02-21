#import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

#create raw dataset
raw_df = pd.read_csv("stroke_risk_dataset.csv")

#create target datasets
Binary_target = raw_df["At Risk (Binary)"]
risk_target = raw_df["Stroke Risk (%)"]

#remove target columns from training dataset
train = raw_df.drop(["At Risk (Binary)", "Stroke Risk (%)"], axis=1)

#split train and test dataset for both case
X_train, X_test, y_binary_train, y_binary_test = train_test_split(train, Binary_target, test_size=0.2, random_state=42)
X_train, X_test, y_risk_train, y_risk_test = train_test_split(train, risk_target, test_size=0.2, random_state=42)

#train RandomForestClassifier for classification
RF_model = RandomForestClassifier(n_estimators=200, max_features=0.1, min_samples_split=5, min_samples_leaf=2, max_samples= 50000).fit(X_train, y_binary_train)

#train XGBRegressor for regression 
GB_model = XGBRegressor(random_state=42, learning_rate=0.1, n_estimators=250).fit(X_train, y_risk_train)

# Load your pre-trained models
with open("RF_model.pkl", "rb") as rf_file:
    RF_model = pickle.load(rf_file)

with open("GB_model.pkl", "rb") as gb_file:
    GB_model = pickle.load(gb_file)

# Page title
st.title("Stroke Risk Prediction")

# Collect user input
st.header("Enter Patient Details")

age = st.number_input("Age", min_value=18, max_value=100, step=1)

# Binary inputs (0 = No, 1 = Yes)
chest_pain = st.radio("Chest Pain", [0, 1])
shortness_of_breath = st.radio("Shortness of Breath", [0, 1])
irregular_heartbeat = st.radio("Irregular Heartbeat", [0, 1])
fatigue = st.radio("Fatigue & Weakness", [0, 1])
dizziness = st.radio("Dizziness", [0, 1])
swelling = st.radio("Swelling (Edema)", [0, 1])
pain_neck = st.radio("Pain in Neck/Jaw/Shoulder/Back", [0, 1])
sweating = st.radio("Excessive Sweating", [0, 1])
cough = st.radio("Persistent Cough", [0, 1])
nausea = st.radio("Nausea/Vomiting", [0, 1])
high_bp = st.radio("High Blood Pressure", [0, 1])
chest_discomfort = st.radio("Chest Discomfort (Activity)", [0, 1])
cold_hands = st.radio("Cold Hands/Feet", [0, 1])
sleep_apnea = st.radio("Snoring/Sleep Apnea", [0, 1])
anxiety = st.radio("Anxiety/Feeling of Doom", [0, 1])

# Combine inputs into an array
input_data = np.array([
    chest_pain, shortness_of_breath, irregular_heartbeat,
    fatigue, dizziness, swelling, pain_neck, sweating, 
    cough, nausea, high_bp, chest_discomfort, cold_hands, 
    sleep_apnea, anxiety, age
]).reshape(1, -1)

# Predict when the button is clicked
if st.button("Predict Stroke Risk"):
    risk_binary = RF_model.predict(input_data)[0]
    risk_percentage = GB_model.predict(input_data)[0] * 100  # Convert to percentage

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**At Risk (Binary)**: {'Yes' if risk_binary == 1 else 'No'}")
    st.write(f"**Stroke Risk (%)**: {risk_percentage:.2f}%")
