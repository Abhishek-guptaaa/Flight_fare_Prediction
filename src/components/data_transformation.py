import os
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from Config.config import Config
from src.exception.exception import CustomException
from src.logger.logger import logging

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = Config.PREPROCESSOR_PATH

    def initiate_data_transformation(self, cleaned_data_path):
        try:
            # Load cleaned data
            df = pd.read_csv(cleaned_data_path)

            # Separate features and target
            X = df.drop(columns=['Price'])
            y = df['Price']

            # Define categorical and numerical features
            categorical_features = ['Source', 'Destination', 'Airline']
            numerical_features = ['Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min', 'Total_Stops']

            # Define the preprocessing for numerical and categorical features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(), categorical_features)
                ])

            # Fit and transform data
            X_transformed = preprocessor.fit_transform(X)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

            # Save the preprocessor
            os.makedirs('models', exist_ok=True)
            joblib.dump(preprocessor, self.preprocessor_obj_file_path)

            logging.info("Data transformation complete and preprocessor saved.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)
