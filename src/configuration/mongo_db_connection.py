import os
import sys
import pymongo
import certifi

from src.exception import MyException
from src.constants import MONGODB_URL_KEY, DATABASE_NAME
from src.logger import logging

# Load the certificate authority file to avoid timeout errors when connecting to MongoDB Atlas
ca = certifi.where()

class MongoDBClient:
    """
    MongoDB Client  is responsible for ectablishing a connection to the MongoDB database.

    Attributes:
    -----------
    client: MongoClient
        A shared MongoClient instance for the class.
    database : Database
        The specific database instance that MongoDBClient connects to.

    Methods:
    --------
    __init__(database_name: str)-> None:
        Initializes the MongoDB connection using the given database name.
    """
    client = None # Shared MongoClient instance across all MongoDBClient instances

    def __init__(self, database_name: str = DATABASE_NAME)-> None:
        """
        Initializes a connection to the MongoDB database. If no existing connection is found, it establishes a new one.

        Parameters:
        ----------
        database_name : str, optional
            Name of the MongoDB database to connect to. Default is set by DATABASE_NAME constant.

        Raises:
        ------
        MyException
            If there is an issue connecting to MongoDB or if the environment variable for the MongoDB URL is not set.
        """
        try:
            # Check if a mongoDB client connection has already been established, if not , create a newone
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY) # Retrieve MingoDB URL from environment variable    
                if mongo_db_url is None:
                    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' not set.")
                
                # Establish a new MongoDB client connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

            #  Use the shared MongoClient for this instance
            self.client = MongoDBClient.client
            self.database = self.client[database_name] # connect to the specified database
            self.database_name = database_name
            logging.info(f"MongoDB connection successfully.")

        except Exception as e:
            # Raise a custom exception if any error occurs during the connection process
            raise MyException(e, sys) 
                         