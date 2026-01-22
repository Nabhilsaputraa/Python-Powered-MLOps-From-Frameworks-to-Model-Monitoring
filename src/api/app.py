from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
import joblib
import pandas as pd
import numpy as np

# Logger setup
from src.utils.logger import get_logger
logger = get_logger(__name__)

# FastAPI app instance
app = FastAPI(title="House Price Prediction API")

# Load model and preprocessor
try:
    model = joblib.load("artifacts/model.pkl")
    preprocessor = joblib.load("artifacts/preprocessor.pkl")
    logger.info("Model and preprocessor loaded successfully.")
    logger.info(f"Model type: {type(model)}")
except Exception as e:
    logger.error(f"Error loading model or preprocessor: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed.")

# Request schema
class HouseFeatures(BaseModel):
    OverallQual: conint(ge=1, le=10)
    GrLivArea: confloat(gt=0)
    GarageCars: conint(ge=0)
    GarageArea: confloat(ge=0)
    TotalBsmtSF: confloat(ge=0)
    FirstFlrSF: confloat(ge=0)  # Note: ini akan diubah menjadi '1stFlrSF'
    FullBath: conint(ge=0)
    TotRmsAbvGrd: conint(ge=0)
    YearBuilt: conint(ge=1800)

# Response schema
class PricePrediction(BaseModel):
    predicted_price: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PricePrediction)
def predict_price(features: HouseFeatures):
    try:
        logger.info(f"Received prediction request: {features.dict()}")
        
        # Convert to dict and rename FirstFlrSF to 1stFlrSF
        input_dict = features.dict()
        
        # Mapping nama feature dari input API ke nama yang digunakan model
        feature_mapping = {
            'OverallQual': 'OverallQual',
            'GrLivArea': 'GrLivArea', 
            'GarageCars': 'GarageCars',
            'GarageArea': 'GarageArea',
            'TotalBsmtSF': 'TotalBsmtSF',
            'FirstFlrSF': '1stFlrSF',  # Mapping disini
            'FullBath': 'FullBath',
            'TotRmsAbvGrd': 'TotRmsAbvGrd',
            'YearBuilt': 'YearBuilt'
        }
        
        # Buat DataFrame dengan nama kolom yang benar
        X_input = pd.DataFrame([{
            feature_mapping[key]: value 
            for key, value in input_dict.items()
        }])
        
        logger.info(f"X_input columns: {X_input.columns.tolist()}")
        logger.info(f"X_input shape before transform: {X_input.shape}")
        logger.info(f"X_input dtypes: {X_input.dtypes.to_dict()}")

        # Debug: Cek preprocessor
        logger.info(f"Preprocessor type: {type(preprocessor)}")
        if hasattr(preprocessor, 'transformers_'):
            logger.info(f"Preprocessor has {len(preprocessor.transformers_)} transformers")
        
        # Transform data
        X_input_processed = preprocessor.transform(X_input)
        logger.info(f"X_input_processed shape: {X_input_processed.shape}")
        logger.info(f"X_input_processed type: {type(X_input_processed)}")
        
        # Handle sparse matrix
        if hasattr(X_input_processed, 'toarray'):
            X_input_processed = X_input_processed.toarray()
            logger.info(f"Converted sparse matrix to dense array")
        
        # Ensure 2D array
        if len(X_input_processed.shape) == 1:
            X_input_processed = X_input_processed.reshape(1, -1)
        elif len(X_input_processed.shape) == 3:
            X_input_processed = X_input_processed.reshape(X_input_processed.shape[0], -1)
        
        logger.info(f"Final shape for prediction: {X_input_processed.shape}")
        
        # Predict
        prediction = model.predict(X_input_processed)
        logger.info(f"Raw prediction output: {prediction}")
        
        if isinstance(prediction, (np.ndarray, list)):
            prediction = float(prediction[0])
        
        logger.info(f"Prediction made successfully: {prediction:.2f}")
        
        return {"predicted_price": float(prediction)}

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")