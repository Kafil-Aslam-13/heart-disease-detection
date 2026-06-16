import requests
import streamlit as st
API_URL="http://localhost:8000/predict"


st.set_page_config(page_title="Heart Disease Detection", page_icon="❤️")
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient details to assess heart disease risk")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age      = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex      = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        cp       = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
        chol     = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
        restecg  = st.selectbox("Resting ECG Results", options=[0, 1, 2])


    with col2:
        thalach  = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang    = st.selectbox("Exercise Induced Angina", options=[0, 1])
        oldpeak  = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
        slope    = st.selectbox("Slope of ST Segment", options=[0, 1, 2])
        ca       = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3])
        thal     = st.selectbox("Thal", options=[0, 1, 2])

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
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

    
