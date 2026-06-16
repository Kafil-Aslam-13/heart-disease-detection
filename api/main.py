# api/main.py
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from api.schemas import PatientInput, PredictionOutput


app = FastAPI(
    title="Heart Disease Detection API",
    description="Predicts heart disease risk from clinical features",
    version="1.0.0"
)

# load model and preprocessor ONCE when the app starts
# not inside the endpoint — that would reload it every request
MODEL_PATH        = "pipeline/stage_04_model_training/artifacts/best_model.joblib"
PREPROCESSOR_PATH = "pipeline/stage_03_preprocessing/artifacts/preprocessor.joblib"

try:
    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except FileNotFoundError as e:
    model        = None
    preprocessor = None
    print(f"WARNING — model files not found: {e}")


@app.get("/")
def health_check():
    """Simple endpoint to check if API is running."""
    return {"status": "API is running", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # DataFrame creation MUST be inside try
        input_df = pd.DataFrame([{
            "age":      patient.age,
            "sex":      patient.sex,
            "cp":       patient.cp,
            "trestbps": patient.trestbps,
            "chol":     patient.chol,
            "fbs":      patient.fbs,
            "restecg":  patient.restecg,
            "thalach":  patient.thalach,
            "exang":    patient.exang,
            "oldpeak":  patient.oldpeak,
            "slope":    patient.slope,
            "ca":       patient.ca,
            "thal":     patient.thal
        }])

        processed = preprocessor.transform(input_df)

        prediction  = int(model.predict(processed)[0])
        probability = float(model.predict_proba(processed)[0][1])

        if probability < 0.3:
            risk_level = "LOW"
        elif probability < 0.7:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"

        return PredictionOutput(
            prediction  = prediction,
            probability = round(probability, 4),
            risk_level  = risk_level
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")