from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from  src.exception import MyException
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.logger import logging
from src.utils.main_utils import load_object, read_yaml_file
import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.entity.s3_estimator import proj1Estimatoe
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            # load schema config for drop columns and other schema-driven operations
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e, sys)
        
    

    def get_best_model(self)-> Optional[proj1Estimatoe]:
        """
        Method Name : get_best_model
        Description : This function is used to get model from productioon stage.

        Output: Returns model object if available in  s3 bucket 
        on Failure : Write an exception log then raise an exceptiono
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj1estimator = proj1Estimatoe(bucket_name=bucket_name, model_path=model_path)

            if proj1estimator.is_model_present(model_path=model_path):
                return proj1estimator
            return None
        
        except Exception as e:
            raise MyException(e, sys)
        
    def _drop_columns(self, dataframe: pd.DataFrame)-> pd.DataFrame:
        try:
            raw = self._schema_config.get("drop_columns")
            # Accept either a YAML list or a comma-separated string
            if raw is None:
                return dataframe
            if isinstance(raw, str):
                # split by comma and strip spaces
                cols = [c.strip() for c in raw.split(",") if c.strip()]
            elif isinstance(raw, (list, tuple)):
                cols = list(raw)
            else:
                # unexpected type
                cols = list(raw)

            # Determine which columns actually exist to avoid KeyError
            existing = [c for c in cols if c in dataframe.columns]
            missing = [c for c in cols if c not in dataframe.columns]
            if missing:
                logging.warning(f"Drop columns not found in dataframe and will be ignored: {missing}")

            if existing:
                dataframe = dataframe.drop(columns=existing, axis=1)

            return dataframe
        except Exception as e:
            raise MyException(e, sys)
        
        
    def _remove_joined_customers(self, dataframe: pd.DataFrame)-> pd.DataFrame:
        try:
            # If the column does not exist, skip this filtering step
            if "Customer Status" not in dataframe.columns:
                logging.warning("Column 'Customer Status' not found; skipping joined-customer filtering.")
                return dataframe

            dataframe = dataframe[dataframe["Customer Status"] != "Joined"]
            return dataframe
        except Exception as e:
            raise MyException(e, sys)
        
    def fill_missing_value(self, dataframe:pd.DataFrame)-> pd.DataFrame:
        """
        Fill missing values with the mode & madian of the column.
        """
        try:
            df = dataframe.copy()
            df["Offer"] = df["Offer"].fillna("missing")

            df["Avg Monthly Long Distance Charges"] = df["Avg Monthly Long Distance Charges"].fillna(df["Avg Monthly Long Distance Charges"].median())

            df["Multiple Lines"] = df["Multiple Lines"].fillna(df["Multiple Lines"].mode()[0])

            df["Avg Monthly GB Download"] = df["Avg Monthly GB Download"].fillna(df["Avg Monthly GB Download"].median())

            df["Online Security"] = df["Online Security"].fillna(df["Online Security"].mode()[0])

            df["Online Backup"] = df["Online Backup"].fillna(df["Online Backup"].mode()[0])

            df["Device Protection Plan"] = df["Device Protection Plan"].fillna(df["Device Protection Plan"].mode()[0])

            df["Premium Tech Support"] = df["Premium Tech Support"].fillna(df["Premium Tech Support"].mode()[0])

            df["Streaming TV"] = df["Streaming TV"].fillna(df["Streaming TV"].mode()[0])

            df["Streaming Movies"] = df["Streaming Movies"].fillna(df["Streaming Movies"].mode()[0])

            df["Streaming Music"] = df["Streaming Music"].fillna(df["Streaming Music"].mode()[0])

            df["Unlimited Data"] = df["Unlimited Data"].fillna(df["Unlimited Data"].mode()[0])

            return df
            
        except Exception as e:
            raise MyException(e, sys)
        
    def encode_categorical(self, dataframe: pd.DataFrame, encoders: dict = None)-> tuple:
        """
        Encode categorical columns using LabelEncoder.
        If `encoders` is None, fit LabelEncoders on `dataframe` and return (dataframe, encoders).
        If `encoders` is provided, use them to transform the dataframe and return (dataframe, encoders).
        """
        try:
            categorical_cols = list(dataframe.select_dtypes(include=["object", "bool"]).columns)
            if encoders is None:
                encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    col_values = dataframe[col].astype(str)
                    le.fit(col_values)
                    known_classes = set(le.classes_)
                    # ensure an explicit 'unknown' class exists so unseen labels can be mapped
                    if 'unknown' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'unknown')
                    dataframe[col] = le.transform(col_values)
                    encoders[col] = {"le": le, "classes": known_classes}
                return dataframe, encoders
            else:
                # transform using provided encoders (do not fit)
                for col in categorical_cols:
                    if col in encoders:
                        le = encoders[col]["le"]
                        known = encoders[col]["classes"]
                        col_values = dataframe[col].astype(str)
                        # replace unseen labels with 'unknown' before transforming
                        col_values_clean = col_values.where(col_values.isin(known), other='unknown')
                        dataframe[col] = le.transform(col_values_clean)
                    else:
                        # if encoder missing for a column, fit a new one and store it
                        le = LabelEncoder()
                        col_values = dataframe[col].astype(str)
                        le.fit(col_values)
                        known_classes = set(le.classes_)
                        if 'unknown' not in le.classes_:
                            le.classes_ = np.append(le.classes_, 'unknown')
                        dataframe[col] = le.transform(col_values)
                        encoders[col] = {"le": le, "classes": known_classes}
                return dataframe, encoders
        except Exception as e:
            raise MyException(e, sys)
        
    def standardize_numerical(self, dataframe: pd.DataFrame)-> pd.DataFrame:
        """
        Standardize numerical columns using StandardScaler.
        """
        try:
            numerical_cols = list(dataframe.select_dtypes(include=["float64", "int64"]).columns)
            scaler = StandardScaler()
            dataframe[numerical_cols] = scaler.fit_transform(dataframe[numerical_cols])
            return dataframe
        except Exception as e:
            raise MyException(e, sys)
        
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try: 
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x,y = test_df.drop(columns=[TARGET_COLUMN],axis=1),test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            X = self._remove_joined_customers(x)
            X = self._drop_columns(X)
            X = self.fill_missing_value(X)
            # encode_categorical returns (dataframe, encoders)
            X, _ = self.encode_categorical(X)
            X = self.standardize_numerical(X)
            logging.info("Test data transformed for prediction...")

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists")

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score of this model: {trained_model_f1_score}")

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for  production model..")
                y_hat_best_model = best_model.predict(X)
                # use weighted average to be robust to label types (strings)
                best_model_f1_score = f1_score(y, y_hat_best_model, average="weighted")
                logging.info(f"F1_Score-Preduction model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")

            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference = trained_model_f1_score - tmp_best_model_score)
            
            logging.info(f"Result: {result}")
            return result
        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("_____________________________________________________________________")
            logging.info("Initialized model evaluation component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifactv = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path = s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifactv}")

            return model_evaluation_artifactv
        except Exception as e:
            raise MyException(e, sys)
    