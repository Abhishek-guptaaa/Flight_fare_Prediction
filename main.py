import os
import sys
import pandas as pd
from Config.config import Config
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation import DataTransformation

def main():
    try:
        logging.info("Starting model training and evaluation")

        # Data Ingestion
        data_ingestion = DataIngestion()
        raw_data_path = data_ingestion.initiate_data_ingestion()

        # Update Config with the ingested raw data path
        Config.RAW_DATA_PATH = raw_data_path

        # Data Cleaning
        data_cleaning = DataCleaning()
        cleaned_data_path = data_cleaning.initiate_data_cleaning()

        # Update Config with the cleaned data path
        Config.CLEANED_DATA_PATH = cleaned_data_path

        # Data Transformation
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(Config.CLEANED_DATA_PATH)

        logging.info("Model training and evaluation completed successfully")

    except Exception as e:
        logging.error(f"Error in main script: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
