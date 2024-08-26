import os

class Config:
    RAW_DATA_PATH = os.path.join('artifacts', 'raw.csv')
    CLEANED_DATA_PATH = os.path.join('artifacts', 'cleaned_data.csv')
    PREPROCESSOR_PATH = os.path.join('models', 'preprocessor.pkl')


