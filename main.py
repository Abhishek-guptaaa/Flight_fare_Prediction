import os
import sys
import pandas as pd
from Config.config import Config
from src.exception.exception import CustomException
from src.logger.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation import DataTransformation


def main():
    try:
        logging.info("Starting model training and evaluation")

        # Data Ingestion
        data_ingestion = DataIngestion()
        Config.RAW_DATA_PATH = data_ingestion.initiate_data_ingestion()

    

        # Data Cleaning
        data_cleaning = DataCleaning()
        Config.CLEANED_DATA_PATH = data_cleaning.initiate_data_cleaning()


        # Data Transformation
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(Config.CLEANED_DATA_PATH)

        # logging.info("Model training and evaluation completed successfully")

    except Exception as e:
        logging.error(f"Error in main script: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
