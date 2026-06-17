import streamlit as st
import requests
import os
API_URL = os.environ.get("API_URL") or "http://localhost:8000/predict"
try:
    API_URL = st.secrets.get("API_URL", API_URL)
except Exception:
    pass 

st.set_page_config(page_title="Heart Disease Detection", page_icon="❤️")
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient details to assess heart disease risk")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=1, max_value=120, value=50)

        Sex = st.selectbox("Sex", options=["M", "F"])

        ChestPainType = st.selectbox(
            "Chest Pain Type",
            options=["TA", "ATA", "NAP", "ASY"],
            format_func=lambda x: {
                "TA": "Typical Angina",
                "ATA": "Atypical Angina",
                "NAP": "Non-Anginal Pain",
                "ASY": "Asymptomatic"
            }[x]
        )

        RestingBP = st.number_input("Resting Blood Pressure", min_value=0, max_value=250, value=120)
        Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=700, value=200)
        FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])

        RestingECG = st.selectbox(
            "Resting ECG Results",
            options=["Normal", "ST", "LVH"]
        )

    with col2:
        MaxHR = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

        ExerciseAngina = st.selectbox(
            "Exercise Induced Angina",
            options=["Y", "N"],
            format_func=lambda x: "Yes" if x == "Y" else "No"
        )

        Oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)

        ST_Slope = st.selectbox(
            "Slope of ST Segment",
            options=["Up", "Flat", "Down"]
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()

            st.divider()
            if result["prediction"] == 1:
                st.error("⚠️ Heart Disease Risk Detected")
            else:
                st.success("✅ No Heart Disease Detected")

            col_a, col_b = st.columns(2)
            col_a.metric("Probability", f"{result['probability']*100:.1f}%")
            col_b.metric("Risk Level", result["risk_level"])

        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API — make sure FastAPI server is running on port 8000")