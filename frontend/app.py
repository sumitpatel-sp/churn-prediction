# app.py

import streamlit as st
import requests

st.title("Customer Churn Prediction")

BACKEND_URL = "http://localhost:8000/predict"


def user_input():

    data = {}

    data["SeniorCitizen"] = st.selectbox("Senior Citizen", ["0", "1"])
    data["MonthlyCharges"] = st.number_input("Monthly Charges")
    data["TotalCharges"] = st.number_input("Total Charges")
    data["gender"] = st.selectbox("Gender", ["Male", "Female"])
    data["Partner"] = st.selectbox("Partner", ["Yes", "No"])
    data["Dependents"] = st.selectbox("Dependents", ["Yes", "No"])
    data["PhoneService"] = st.selectbox("Phone Service", ["Yes", "No"])
    data["MultipleLines"] = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    data["InternetService"] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    data["OnlineSecurity"] = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    data["OnlineBackup"] = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    data["DeviceProtection"] = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    data["TechSupport"] = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    data["StreamingTV"] = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    data["StreamingMovies"] = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    data["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    data["PaperlessBilling"] = st.selectbox("Paperless Billing", ["Yes", "No"])
    data["PaymentMethod"] = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    data["tenure"] = st.number_input("Tenure")

    return data


input_data = user_input()

if st.button("Predict"):

    response = requests.post(BACKEND_URL, json=input_data)

    if response.status_code == 200:
        result = response.json()
        st.success(result["prediction"])
        st.write(f"Confidence: {result['confidence']:.2f}%")
    else:
        st.error("Backend error")


response = requests.post(BACKEND_URL, json=input_data)

response = requests.post(BACKEND_URL, json=input_data)

st.write("Status:", response.status_code)
st.write("Response:", response.text)