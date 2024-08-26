# import os
# import sys
# import joblib
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from Config.config import Config
# from src.exception.exception import CustomException
# from src.logger.logger import logging

# class DataTransformation:
#     def __init__(self):
#         self.preprocessor_obj_file_path = Config.PREPROCESSOR_PATH

#     def initiate_data_transformation(self, cleaned_data_path):
#         try:
#             # Load cleaned data
#             df = pd.read_csv(cleaned_data_path)

#             # Separate features and target
#             X = df.drop(columns=['Price'])
#             y = df['Price']

#             # Encode categorical features
#             categorical_data = X.select_dtypes(exclude=['int64', 'float', 'int32'])
#             numerical_data = X.select_dtypes(include=['int64', 'float', 'int32'])

#             le = LabelEncoder()
#             categorical_data = categorical_data.apply(le.fit_transform)

#             X_transformed = pd.concat([categorical_data, numerical_data], axis=1)

#             # Split the data
#             X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

#             # Save the preprocessor
#             os.makedirs('models', exist_ok=True)
#             joblib.dump(le, self.preprocessor_obj_file_path)

#             logging.info("Data transformation complete and preprocessor saved.")
#             return X_train, X_test, y_train, y_test
#         except Exception as e:
#             raise CustomException(e, sys)



import os
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Config.config import Config
from src.exception.exception import CustomException
from src.logger.logger import logging

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = Config.PREPROCESSOR_PATH
        self.X_TEST_TRANSFORMED_PATH = Config.X_TEST_TRANSFORMED_PATH
        self.Y_TEST_PATH = Config.Y_TEST_PATH

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

            # Save the test data and target values
            os.makedirs(os.path.dirname(self.X_TEST_TRANSFORMED_PATH), exist_ok=True)
            pd.DataFrame(X_test).to_csv(self.X_TEST_TRANSFORMED_PATH, index=False)
            pd.DataFrame(y_test).to_csv(self.Y_TEST_PATH, index=False)

            logging.info("Data transformation complete and preprocessor saved.")
            logging.info(f"Transformed test data saved to {self.X_TEST_TRANSFORMED_PATH}")
            logging.info(f"Test target values saved to {self.Y_TEST_PATH}")

            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)
