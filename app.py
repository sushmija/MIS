import streamlit as st
import pandas as pd
import joblib
import json

# Load trained model, scaler, and feature names
model = joblib.load('hospital_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

st.title("🏥 Hospital Management & Patient Loyalty Prediction")
st.write("Enter the following details:")

user_input = {}
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=1.0)
    user_input[feature] = value

if st.button("Predict Loyalty Score"):
    input_df = pd.DataFrame([user_input], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Loyalty Score: {prediction[0]}")

st.sidebar.header("About")
st.sidebar.info("This app predicts customer loyalty scores for healthcare customers based on provided data.")
