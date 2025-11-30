import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.artifact_entity import DataTransformationArtifact
from src.exception import MyException
from src.logger import logging
# from config import drop_columns
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


# class for data transformation

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise MyException(e, sys)
        
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
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
    
    # Drop rows where "Customer Status" is "Joined"
    def _remove_joined_customers(self, dataframe: pd.DataFrame)-> pd.DataFrame:
        try:
            dataframe = dataframe[dataframe["Customer Status"] != "Joined"]
            return dataframe
        except Exception as e:
            raise MyException(e, sys)
    
    def fill_missing_value(self, dataframe: pd.DataFrame)-> pd.DataFrame:
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
        
    
    def standardize_numerical(self, X_train, X_test):
        """
        Standardize numerical arrays using StandardScaler.

        Fits scaler on X_train only and transforms X_test with the same scaler.
        Returns (X_train_scaled, X_test_scaled, scaler)
        """
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        except Exception as e:
            raise MyException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiate data transformation.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train & Test Data Loaded")

            # Remove joined customers from full dataframes before splitting
            train_df = self._remove_joined_customers(dataframe=train_df)
            test_df = self._remove_joined_customers(dataframe=test_df)

            # Split into input features and target after filtering
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Train & Test Data Splitted")

            # Drop unwanted columns defined in schema
            input_feature_train_df = self._drop_columns(dataframe=input_feature_train_df)
            input_feature_test_df = self._drop_columns(dataframe=input_feature_test_df)
            logging.info("Columns Dropped")

            input_feature_train_df = self.fill_missing_value(dataframe=input_feature_train_df)
            input_feature_test_df = self.fill_missing_value(dataframe=input_feature_test_df)
            logging.info("Missing Values Filled")

            # Encode categorical: fit on train, apply to test
            input_feature_train_df, encoders = self.encode_categorical(dataframe=input_feature_train_df)
            input_feature_test_df, _ = self.encode_categorical(dataframe=input_feature_test_df, encoders=encoders)
            logging.info("Categorical Columns Encoded")

            # Standardize numerical (fit scaler on train only)
            input_feature_train_arr, input_feature_test_arr, scaler = self.standardize_numerical(X_train=input_feature_train_df, X_test=input_feature_test_df)

            logging.info("Data Transformation Completed !!!")

            logging.info("Applying RandomUnderSampler for handling imblanced data ")

            # Apply RandomUnderSampler only to training data
            random_sampler = RandomUnderSampler(random_state=42)
            input_feature_train_resampled, target_feature_train_resampled = random_sampler.fit_resample(input_feature_train_arr, target_feature_train_df)

            logging.info("Applying RandomUnderSampler Completed !!!")

            train_arr = np.c_[input_feature_train_resampled, np.array(target_feature_train_resampled)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Train and Test Array Created")

            # Persist transformer objects (scaler + encoders + feature columns)
            transformer = {
                "scaler": scaler,
                "encoders": encoders,
                "feature_columns": list(input_feature_train_df.columns)
            }

            save_object(self.data_transformation_config.transformed_object_file_path, transformer)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving tranformation object and transformed files..")

            logging.info("Data Transformation Artifact Created successfully")

            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )
        
        except Exception as e:
            raise MyException(e, sys) from e