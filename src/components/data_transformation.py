import os
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Config.config import Config
from src.exception import CustomException
from src.logger import logger

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

            # Encode categorical features
            categorical_data = X.select_dtypes(exclude=['int64', 'float', 'int32'])
            numerical_data = X.select_dtypes(include=['int64', 'float', 'int32'])

            le = LabelEncoder()
            categorical_data = categorical_data.apply(le.fit_transform)

            X_transformed = pd.concat([categorical_data, numerical_data], axis=1)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

            # Save the preprocessor
            os.makedirs('models', exist_ok=True)
            joblib.dump(le, self.preprocessor_obj_file_path)

            logger.info("Data transformation complete and preprocessor saved.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)
