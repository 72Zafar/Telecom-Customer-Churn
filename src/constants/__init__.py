import os
from datetime import datetime

# for mongo connection
DATABASE_NAME = "telecom_customer_churn"
COLLECTION_NAME = "telecom_customer_churn_data"
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = ""
ARTIFACTS_DIR: str = "artifacts"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "Customer Status"
CURRENT_YEAR = datetime.now().year
PREPOCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "telecom_customer_churn.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

""" 
Data ingestion related constant start with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "telecom_customer_churn_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.25

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"


"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
Model Trainer related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE : float = 0.7
MODEL_TRAINER_ITERATIONS : int = 500
MODEL_TRAINER_LEARNING_RATE : float = 0.03
MODEL_TRAINER_DEPTH : int = 4
MODEL_TRAINER_L2_LEAF_REG : int = 5
MODEL_TRAINER_BAGGING_TEMPERATURE : float = 0.5
MODEL_TRAINER_BORDER_COUNT : int = 128
MODEL_TRAINER_RANDOM_SEED : int = 42
MODEL_TRAINER_LOSS_FUNCTION : str = 'Logloss'
MODEL_TRAINER_VERBOSE : bool = False

"""
MODEL Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "my-model-mlopspro-end"
MODEL_PUSHER_S3_KEY = "model-registry"


APP_HOST = "0.0.0.0"
APP_PORT = 5000