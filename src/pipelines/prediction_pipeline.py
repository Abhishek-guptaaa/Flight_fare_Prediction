import os
import sys
import pandas as pd
import joblib
from Config.config import Config
from src.exception.exception import CustomException
from src.logger.logger import logging

class PredictionPipeline:
    def __init__(self):
        self.preprocessor_path = Config.PREPROCESSOR_PATH
        self.model_path = Config.MODEL_PATH  # Ensure this path is correctly set in Config
        self.preprocessor = None
        self.model = None

    def load_preprocessor(self):
        try:
            self.preprocessor = joblib.load(self.preprocessor_path)
            logging.info("Preprocessor loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading preprocessor: {e}")
            raise CustomException(e, sys)

    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise CustomException(e, sys)

    def preprocess_data(self, data):
        try:
            # Apply preprocessor transformations
            categorical_features = data.select_dtypes(include=['object'])
            categorical_features_encoded = categorical_features.apply(self.preprocessor.transform)

            numerical_features = data.select_dtypes(exclude=['object'])
            data_transformed = pd.concat([categorical_features_encoded, numerical_features], axis=1)
            return data_transformed
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise CustomException(e, sys)

    def make_predictions(self, input_data):
        try:
            # Preprocess the input data
            processed_data = self.preprocess_data(input_data)

            # Make predictions
            predictions = self.model.predict(processed_data)
            return predictions
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            raise CustomException(e, sys)

# Example usage
if __name__ == "__main__":
    try:
        # Create a prediction pipeline instance
        prediction_pipeline = PredictionPipeline()
        
        # Load preprocessor and model
        prediction_pipeline.load_preprocessor()
        prediction_pipeline.load_model()

        # Load new data for prediction
        new_data = pd.read_csv('path_to_new_data.csv')  # Replace with your data path
        
        # Make predictions
        predictions = prediction_pipeline.make_predictions(new_data)
        print(predictions)
    except Exception as e:
        logging.error(f"Error in prediction pipeline: {e}")
        raise CustomException(e, sys)
