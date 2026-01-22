import mlflow
import numpy as np
import joblib
import os
import json
import datetime
from datetime import datetime
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#! Logger setup 
from src.utils.logger import get_logger
logger = get_logger(__name__)

#! Create artifacts directory
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

#! Training function 
def train_model(X, y, preprocessor):
    logger.info("Starting model training...")

    #? Split data
    logger.info("Splitting data into training and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    mlflow.set_experiment("House_Price_Prediction")
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        #? Model initialization
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

        #? Training
        logger.info("Training RandomForestRegressor...")
        model.fit(X_train, y_train)
        
        #? Predictions & matrics
        logger.info("Evaluating model on validation set...")
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        matrics = {"RMSE": rmse, "MAE": mae, "R2_Score": r2}
        mlflow.log_metrics(matrics)
        
        #? Save Artifacts
        model_path = f"{ARTIFACTS_DIR}/model.pkl"
        preprocessor_path = f"{ARTIFACTS_DIR}/preprocessor.pkl"
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        #? Log model to MLflow Model Registry
        mlflow.sklearn.log_model(model, "model", registered_model_name="HousePriceModel")

        #? log Metrics to json
        metrics_file = "logs/metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({**matrics, "timestamp": datetime.now().isoformat(), "run_id": mlflow.active_run().info.run_id}, f, indent=4)
        logger.info(f"Model training completed. Metrics: {matrics}")

        #? Print model performance
        print(f"Validation RMSE: {rmse}")
        print(f"Validation MSE: {mse}")
        print(f"Validation MAE: {mae}")
        print(f"Validation R2 Score: {r2}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
    
    return model


