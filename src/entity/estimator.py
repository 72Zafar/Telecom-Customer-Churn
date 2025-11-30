import sys

import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.yes:int = 0
        self.no:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))

class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Function accepts preprocessed inputs (with all custom transformations already applied),
        applies scaling using preprocessing_object, and performs prediction on transformed features.
        Handles both sklearn Pipeline and dict of transformers.
        """
        try:
            logging.info("Starting prediction process.")

            if isinstance(self.preprocessing_object, dict):
                # Handle dict of transformers: scaler, encoders, feature_columns
                df = dataframe.copy()

                # Normalizer for robust matching: lower + keep alnum only
                def _norm(s: str) -> str:
                    return "".join(ch for ch in str(s).lower() if ch.isalnum())

                # build normalized -> actual column map for incoming df
                incoming_map = {_norm(c): c for c in df.columns}

                encoders = self.preprocessing_object.get("encoders", {})
                scaler = self.preprocessing_object.get("scaler")
                feature_columns = self.preprocessing_object.get("feature_columns", list(df.columns))

                # map training feature names -> actual incoming column names
                mapped_cols = []
                for feat in feature_columns:
                    nfeat = _norm(feat)
                    if nfeat in incoming_map:
                        mapped_cols.append(incoming_map[nfeat])
                    else:
                        # try common alternate forms (spaces <-> underscores)
                        alt = feat.replace(" ", "_")
                        if alt in df.columns:
                            mapped_cols.append(alt)
                        else:
                            # try to find any incoming column whose normalized form equals nfeat
                            found = None
                            for inc_col in df.columns:
                                if _norm(inc_col) == nfeat:
                                    found = inc_col
                                    break
                            if found:
                                mapped_cols.append(found)
                            else:
                                # column is missing: create a default-filled column (zeros)
                                logging.warning(f"Feature column '{feat}' not found in incoming data — filling with 0s")
                                df[feat] = 0
                                mapped_cols.append(feat)

                # Apply encoders using same matching logic (encoders keys are training names)
                for enc_col, enc_data in encoders.items():
                    # find actual incoming column name
                    actual_col = None
                    if _norm(enc_col) in incoming_map:
                        actual_col = incoming_map[_norm(enc_col)]
                    elif enc_col in df.columns:
                        actual_col = enc_col
                    else:
                        # try alternate forms
                        alt = enc_col.replace(" ", "_")
                        if alt in df.columns:
                            actual_col = alt
                        else:
                            # try matching by normalized names among df columns
                            for inc_col in df.columns:
                                if _norm(inc_col) == _norm(enc_col):
                                    actual_col = inc_col
                                    break
                    if actual_col is None:
                        logging.debug(f"No matching incoming column for encoder '{enc_col}', skipping encoder")
                        continue

                    le = enc_data["le"]
                    known = enc_data.get("classes", set())
                    col_values = df[actual_col].astype(str)
                    col_values_clean = col_values.where(col_values.isin(known), other="unknown")
                    df[actual_col] = le.transform(col_values_clean)

                # Finally, select in the trained feature order (using mapped_cols) and scale
                df_selected = df[mapped_cols].values if mapped_cols else df.values
                if scaler is not None:
                    transformed_feature = scaler.transform(df_selected)
                else:
                    transformed_feature = df_selected

            # Step 2: Perform prediction using the trained model
            logging.info("Using the trained model to get predictions")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e


    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"