import pandas as pd
from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)

def load_data():
    config = load_config()
    path = config["data"]["raw_path"]
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Data loaded with shape {df.shape}")
    return df
