from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class PatientInput(BaseModel):
    """
    One row of patient clinical data (Fedesoriano heart failure dataset schema).
    Categorical fields use the dataset's actual string categories.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Age": 52,
                "Sex": "M",
                "ChestPainType": "ATA",
                "RestingBP": 125,
                "Cholesterol": 212,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 168,
                "ExerciseAngina": "N",
                "Oldpeak": 1.0,
                "ST_Slope": "Up"
            }
        }
    )

    Age:            int = Field(..., ge=1, le=120, description="Age in years")
    Sex:            Literal["M", "F"] = Field(..., description="M=male, F=female")
    ChestPainType:  Literal["TA", "ATA", "NAP", "ASY"] = Field(..., description="Chest pain type")
    RestingBP:      int = Field(..., ge=0, le=250, description="Resting blood pressure")
    Cholesterol:    int = Field(..., ge=0, le=700, description="Serum cholesterol mg/dl")
    FastingBS:      int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    RestingECG:     Literal["Normal", "ST", "LVH"] = Field(..., description="Resting ECG result")
    MaxHR:          int = Field(..., ge=60, le=220, description="Max heart rate achieved")
    ExerciseAngina: Literal["Y", "N"] = Field(..., description="Exercise induced angina")
    Oldpeak:        float = Field(..., ge=-3.0, le=7.0, description="ST depression")
    ST_Slope:       Literal["Up", "Flat", "Down"] = Field(..., description="Slope of peak exercise ST segment")


class PredictionOutput(BaseModel):
    """
    API response after prediction.
    """
    prediction:  int      # 0 = no disease, 1 = disease
    probability: float    # model confidence (0–1)
    risk_level:  str      # LOW / MODERATE / HIGH