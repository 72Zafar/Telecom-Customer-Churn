import sys
import pandas as pd
from src.entity.config_entity import ChurnPredictionCongig
from src.entity.s3_estimator import proj1Estimatoe
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class ChurnData():
    def __init__(
        self,
        Gender,
        Married,
        Offer,
        Phone_Service,
        Multiple_Lines,
        Internet_Service,
        Internet_Type,
        Online_Security,
        Online_Backup,
        Device_Protection_Plan,
        Premium_Tech_Support,
        Streaming_TV,
        Streaming_Movies,
        Streaming_Music,
        Unlimited_Data,
        Contract,
        Paperless_Billing,
        Payment_Method,
        Age,
        Number_of_Dependents,
        Number_of_Referrals,
        Tenure_in_Months,
        Avg_Monthly_Long_Distance_Charges,
        Avg_Monthly_GB_Download,
        Monthly_Charge,
        Total_Charges,
    ):
        """
        ChurnData constructor
        input: all features of the trained model for prediction
        """
        try:
            self.Gender = Gender
            self.Married = Married
            self.Offer = Offer
            self.Phone_Service = Phone_Service
            self.Multiple_Lines = Multiple_Lines
            self.Internet_Service = Internet_Service
            self.Internet_Type = Internet_Type
            self.Online_Security = Online_Security
            self.Online_Backup = Online_Backup
            self.Device_Protection_Plan = Device_Protection_Plan
            self.Premium_Tech_Support = Premium_Tech_Support
            self.Streaming_TV = Streaming_TV
            self.Streaming_Movies = Streaming_Movies
            self.Streaming_Music = Streaming_Music
            self.Unlimited_Data = Unlimited_Data
            self.Contract = Contract
            self.Paperless_Billing = Paperless_Billing
            self.Payment_Method = Payment_Method
            self.Age = Age
            self.Number_of_Dependents = Number_of_Dependents
            self.Number_of_Referrals = Number_of_Referrals
            self.Tenure_in_Months = Tenure_in_Months
            self.Avg_Monthly_Long_Distance_Charges = Avg_Monthly_Long_Distance_Charges
            self.Avg_Monthly_GB_Download = Avg_Monthly_GB_Download
            self.Monthly_Charge = Monthly_Charge
            self.Total_Charges = Total_Charges

        except Exception as e:
            raise MyException(e, sys) from e
        
    def get_churn_input_data_frame(self)-> DataFrame:
        """
        This function return a dictionary from churn data class input
        """
        logging.info("Entered get_churn_input_data_frame method as part of ChurnData class")

        try:
            input_data = {
                "Gender":[self.Gender],
                "Married":[self.Married],
                "Offer":[self.Offer],
                "Phone_Service":[self.Phone_Service],
                "Multiple_Lines":[self.Multiple_Lines],
                "Internet_Service":[self.Internet_Service],
                "Internet_Type":[self.Internet_Type],
                "Online_Security":[self.Online_Security],
                "Online_Backup":[self.Online_Backup],
                "Device_Protection_Plan":[self.Device_Protection_Plan],
                "Premium_Tech_Support":[self.Premium_Tech_Support],
                "Streaming_TV":[self.Streaming_TV],
                "Streaming_Movies":[self.Streaming_Movies],
                "Streaming_Music":[self.Streaming_Music],
                "Unlimited_Data":[self.Unlimited_Data],
                "Contract":[self.Contract],
                "Paperless_Billing":[self.Paperless_Billing],
                "Payment_Method":[self.Payment_Method],
                "Age":[self.Age],
                "Number_of_Dependents":[self.Number_of_Dependents],
                "Number_of_Referrals":[self.Number_of_Referrals],
                "Tenure_in_Months":[self.Tenure_in_Months],
                "Avg_Monthly_Long_Distance_Charges":[self.Avg_Monthly_Long_Distance_Charges],
                "Avg_Monthly_GB_Download":[self.Avg_Monthly_GB_Download],
                "Monthly_Charge":[self.Monthly_Charge],
                "Total_Charges":[self.Total_Charges]
            }

            logging.info("Created churn data dict")
            logging.info("Exited get_churn_input_data_frame method as part of ChurnData class")

            return pd.DataFrame(input_data)
        except Exception as e:
            raise MyException(e, sys) from e
        
class ChurnDataClassifer:
    def __init__(self, prediction_pipeline_config: ChurnPredictionCongig = ChurnPredictionCongig()) -> None:
        """
        :param prediction_pipeline_config: configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)
        
    def predict(self, dataframe)-> str:
        """
        This is the method of ChurnDataClassifer
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of ChurnDataClassifer class")
            model = proj1Estimatoe(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)
            return result
        except Exception as e:
            raise MyException(e, sys)