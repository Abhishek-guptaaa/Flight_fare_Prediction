import os

class Config:
    RAW_DATA_PATH = os.path.join('artifacts', 'raw.csv')
    CLEANED_DATA_PATH = os.path.join('artifacts', 'cleaned_data.csv')
    PREPROCESSOR_PATH = os.path.join('models', 'preprocessor.pkl')

    CLEANED_DATA_PATH = 'notebook/cleaned_data.csv'
    PREPROCESSOR_PATH='models/preprocessor.pkl'
    MODEL_PATH = 'models/model.pkl'
    X_TEST_TRANSFORMED_PATH = 'notebook/X_test_transformed.csv'
    Y_TEST_PATH = 'notebook/y_test.csv'
    
    DROP_COLUMNS = ['id']

    APP_HOST = '0.0.0.0'
    APP_PORT = 8080


