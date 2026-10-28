from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/liver_rf_model.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = bundle["threshold"]
feature_order = bundle["features"]

app = FastAPI(title="Liver Disease Random Forest API")


class LiverFeatures(BaseModel):
    age: int = Field(gt=0, le=120)
    gender: Literal["Male", "Female"]
    tot_bilirubin: float = Field(ge=0)
    direct_bilirubin: float = Field(ge=0)
    alkphos: float = Field(ge=0)        # alkaline phosphatase
    sgpt: float = Field(ge=0)           # ALT
    sgot: float = Field(ge=0)           # AST
    tot_proteins: float = Field(gt=0)
    albumin: float = Field(gt=0)
    ag_ratio: float = Field(ge=0)       # albumin/globulin ratio


@app.get("/")
def ready_to_serve():
    return {"status": "ready to serve..."}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# endpoint for making predictions with the liver random forest
@app.post("/predict")
def predict(data: LiverFeatures):
    row = pd.DataFrame([data.model_dump()])[feature_order]
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "class_name": ["no liver disease", "liver disease"][prediction],
        "probability": round(probability, 4),
        "threshold": round(threshold, 2),
    }