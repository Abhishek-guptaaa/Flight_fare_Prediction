import os

class Config:
    RAW_DATA_PATH = os.path.join('artifacts', 'raw.csv')
    CLEANED_DATA_PATH = os.path.join('artifacts', 'cleaned_data.csv')
    PREPROCESSOR_PATH = os.path.join('models', 'preprocessor.pkl')

    PREPROCESSOR_PATH='models/preprocessor.pkl'
    MODEL_PATH = 'models/model.pkl'
    X_TEST_TRANSFORMED_PATH = 'artifacts/X_test_transformed.csv'
    Y_TEST_PATH = 'artifacts/y_test.csv'
    
    DROP_COLUMNS = ['id']

    APP_HOST = '0.0.0.0'
    APP_PORT = 8080


