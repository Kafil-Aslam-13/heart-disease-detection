from pydantic import BaseModel , Field

class PatientInput(BaseModel):
    """
    One row of patient clinical data.
    Matches the 13 features used to train the model.
    """
    age:      int   = Field(..., ge=1,  le=120,  description="Age in years")
    sex:      int   = Field(..., ge=0,  le=1,    description="0=female, 1=male")
    cp:       int   = Field(..., ge=0,  le=3,    description="Chest pain type")
    trestbps: int   = Field(..., ge=80, le=200,  description="Resting blood pressure")
    chol:     int   = Field(..., ge=100,le=600,  description="Serum cholesterol mg/dl")
    fbs:      int   = Field(..., ge=0,  le=1,    description="Fasting blood sugar > 120 mg/dl")
    restecg:  int   = Field(..., ge=0,  le=2,    description="Resting ECG results")
    thalach:  int   = Field(..., ge=60, le=220,  description="Max heart rate achieved")
    exang:    int   = Field(..., ge=0,  le=1,    description="Exercise induced angina")
    oldpeak:  float = Field(..., ge=0.0,le=6.2,  description="ST depression induced by exercise")
    slope:    int   = Field(..., ge=0,  le=2,    description="Slope of peak exercise ST segment")
    ca:       int   = Field(..., ge=0,  le=3,    description="Number of major vessels colored")
    thal:     int   = Field(..., ge=0,  le=2,    description="0=normal, 1=fixed defect, 2=reversable")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
                "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
                "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 2
            }
        }


class PredictionOutput(BaseModel):
    """
    What the API sends back after prediction.
    """
    prediction:   int     # 0 = no disease, 1 = disease
    probability:  float   # confidence score between 0 and 1
    risk_level:   str     # LOW, MODERATE, HIGH