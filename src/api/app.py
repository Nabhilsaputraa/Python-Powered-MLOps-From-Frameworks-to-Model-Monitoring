from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
from typing import List
import joblib
import pandas as pd
import numpy as np

# FastAPI app
app = FastAPI(title="House Price Prediction API")

# Load model & preprocessor
try:
    model = joblib.load("artifacts/model.pkl")
    preprocessor = joblib.load("artifacts/preprocessor.pkl")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

# Request schema (satu row)
class HouseFeatures(BaseModel):
    OverallQual: conint(ge=1, le=10)
    GrLivArea: confloat(gt=0)
    GarageCars: conint(ge=0)
    GarageArea: confloat(ge=0)
    TotalBsmtSF: confloat(ge=0)
    FirstFlrSF: confloat(ge=0)  # nanti di-mapping ke '1stFlrSF'
    FullBath: conint(ge=0)
    TotRmsAbvGrd: conint(ge=0)
    YearBuilt: conint(ge=1800)

# Response schema (untuk batch)
class PricePrediction(BaseModel):
    predicted_prices: List[float]

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/predict", response_model=PricePrediction)
def predict_batch(features_list: List[HouseFeatures]):
    try:
        # Convert semua object ke dict
        input_dicts = [f.dict() for f in features_list]

        # Mapping nama feature sesuai model
        feature_mapping = {
            'OverallQual': 'OverallQual',
            'GrLivArea': 'GrLivArea',
            'GarageCars': 'GarageCars',
            'GarageArea': 'GarageArea',
            'TotalBsmtSF': 'TotalBsmtSF',
            'FirstFlrSF': '1stFlrSF',  # mapping disini
            'FullBath': 'FullBath',
            'TotRmsAbvGrd': 'TotRmsAbvGrd',
            'YearBuilt': 'YearBuilt'
        }

        # Buat DataFrame dengan kolom benar
        X_input = pd.DataFrame([
            {feature_mapping[k]: v for k, v in d.items()} 
            for d in input_dicts
        ])

        # Transform data
        X_input_processed = preprocessor.transform(X_input)

        # Handle sparse matrix
        if hasattr(X_input_processed, 'toarray'):
            X_input_processed = X_input_processed.toarray()

        # Pastikan 2D
        if len(X_input_processed.shape) == 1:
            X_input_processed = X_input_processed.reshape(1, -1)

        # Predict
        preds = model.predict(X_input_processed)

        return {"predicted_prices": preds.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
