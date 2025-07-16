import streamlit as st
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier

# Load model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.load_model("model.json")

# Load the training feature names
with open("features.json", "r") as f:
    feature_names = json.load(f)

# Streamlit UI
st.title("🧠 Customer Churn Prediction")

# Simple form (you can expand this later)
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 10.0, 150.0, 70.0)
total_charges = tenure * monthly_charges

contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])

# Create input dictionary
user_input = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract_Month-to-month': 0,
    'Contract_One year': 0,
    'Contract_Two year': 0
}

# Set the correct contract one-hot column
user_input[f'Contract_{contract}'] = 1

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Make sure all features exist and are in the correct order
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# Predict
if st.button("Predict Churn"):
    result = model.predict(input_df)
    st.write("🔔 Churn Risk:" if result[0] == 1 else "✅ Customer Likely to Stay")
