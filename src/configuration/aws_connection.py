import boto3
import os
from src.constants import AWS_ACCESS_KEY_ID_ENV_KEY, AWS_SECRET_ACCESS_KEY_ENV_KEY,REGION_NAME

class S3client:
    
    s3_client = None
    s3_resource = None
    def __init__(self, region_name = REGION_NAME):
        """
        This class gets aws credentials from env_varible and creates an connection s2 bucket and raise exception when environment variable is not set 
        """
        if S3client.s3_resource== None or S3client.s3_client == None:
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)
            if __access_key_id is None:
                raise Exception (f"Environment variable '{AWS_ACCESS_KEY_ID_ENV_KEY}' not set.")
            if __secret_access_key is None:
                raise Exception (f"Environment variable '{AWS_SECRET_ACCESS_KEY_ENV_KEY}' not set.")
            
            S3client.s3_resource = boto3.resource(
                "s3",
                aws_access_key_id = __access_key_id,
                aws_secret_access_key = __secret_access_key,
                region_name = region_name
            )

            S3client.s3_client = boto3.client(
                "s3",
                aws_access_key_id = __access_key_id,
                aws_secret_access_key = __secret_access_key,
                region_name = region_name
            )

            self.s3_resource = S3client.s3_resource
            self.s3_client = S3client.s3_client