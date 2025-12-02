import sys
from typing import Tuple

import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay,accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from  src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel


class ModelTrainer:
    def __init__(self, data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model trainer
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train:np.array, test:np.array)->Tuple[object,object]:
        """
        Method Name : get_model_object_and_report
        Description : This function trains a CatBoost model with specified hyperparameters
        """
        try:
            logging.info("Training CatBoost model with specified hyperparameters")
            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            # Initialize CatBoost classifier
            model = CatBoostClassifier(iterations=self.model_trainer_config._iterations,
                                       learning_rate=self.model_trainer_config._learning_rate,
                                       depth=self.model_trainer_config._depth,
                                       l2_leaf_reg=self.model_trainer_config._l2_leaf_reg,
                                       bagging_temperature=self.model_trainer_config._bagging_temperature,
                                       border_count=self.model_trainer_config._border_count,
                                       random_seed=self.model_trainer_config._random_seed,
                                       loss_function=self.model_trainer_config._loss_function,
                                       verbose=self.model_trainer_config._verbose)
            
            # fit the model
            logging.info("Fitting the model")
            model.fit(x_train, y_train)
            logging.info("Model fitted")

            # prediction and evaluation metrics
            y_pred = model.predict(x_test)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, x_train, y_train, cv=skf, scoring="accuracy")
            logging.info(f"Cross-validation scores: {scores}")

            accuracy = accuracy_score(y_test, y_pred)

            # Compute metrics robustly for string labels
            unique_labels = np.unique(y_test)
            logging.info(f"Unique labels in y_test: {unique_labels}")
            if len(unique_labels) == 2:
                # Prefer explicit positive label if present (common case: 'Churned')
                if "Churned" in unique_labels:
                    pos_label = "Churned"
                else:
                    # fallback to the second label
                    pos_label = unique_labels[1]

                f1 = f1_score(y_test, y_pred, pos_label=pos_label)
                precision = precision_score(y_test, y_pred, pos_label=pos_label)
                recall = recall_score(y_test, y_pred, pos_label=pos_label)
            else:
                # Multiclass or unusual labels: use weighted averages
                f1 = f1_score(y_test, y_pred, average="weighted")
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")

            # creating metric artifact
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            return model, metric_artifact
        
        except Exception as e:
            raise MyException(e,sys) from e
        
    def initiate_model_trainer(self)->  ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer methid of ModelTrainer class")

        """
        Method Name : initiate_model_trainer
        Description : Tis function intiates the model training steps

        Output : Returns model trainer artifact
        on Failure : Write an exception log and then raise an exception 
        """    
        try:
            print("-"* 50)
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Train-Test Data Loaded")

            #  Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")

            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e