from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from src.utils.logger import get_logger
logger = get_logger(__name__)

#! Missing Value Handling
Non_Feature_Columns = [
    'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'LotFrontage', 'GarageQual',
    'GarageFinish', 'GarageType'
]

def fill_missing_values(X):
    x = X.copy()
    for col in Non_Feature_Columns:
        if col in X.columns:
            X[col] = X[col].fillna('None')
    return X

#! Preprocessing Pipeline
def preprocessing_pipeline(X):
    logger.info("Creating preprocessing pipeline")

    #? Numerical features
    num_features = X.select_dtypes(include=['int64', 'float64']).columns
    num_pipline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    #? Categorical features
    cat_features = X.select_dtypes(include=['object']).columns
    cat_pipeline = Pipeline([
        ('fill_none', FunctionTransformer(fill_missing_values)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    #? Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    return preprocessor

#! Preprocessing function
def preprocess_data(df, target='SalePrice'):
    logger.info("Starting data preprocessing")
    X = df.drop(columns=[target])
    y = df[target]

    preprocessor = preprocessing_pipeline(X)
    X_processed = preprocessor.fit_transform(X)

    logger.info("Data preprocessing completed")
    return X_processed, y, preprocessor
