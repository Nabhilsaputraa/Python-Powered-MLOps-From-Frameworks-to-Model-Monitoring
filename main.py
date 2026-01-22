import os
from src.data.data_loader import load_data
from src.data.data_preprocessor import preprocess_data
from src.models.trainer import train_model

ARTIFACTS_DIR = "artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

df = load_data()  
X, y, preprocessor = preprocess_data(df, target='SalePrice')
model = train_model(X, y, preprocessor)

print("✅ Training complete. Artifacts saved in ./artifacts")
