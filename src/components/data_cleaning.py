import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logger
from src.config import Config

class DataCleaning:
    def __init__(self):
        pass

    def clean_data(self, df):
        try:
            # Remove null values
            df.dropna(inplace=True)

            # Remove duplicates
            df.drop_duplicates(keep='first', inplace=True)

            # Standardize 'Additional_Info'
            df["Additional_Info"] = df["Additional_Info"].replace({'No Info': 'No info'})

            # Convert Duration to minutes
            df['Duration'] = df['Duration'].str.replace("h", '*60').str.replace(' ', '+').str.replace('m', '*1').apply(eval)

            # Extracting Journey day and month from Date_of_Journey
            df["Journey_day"] = df['Date_of_Journey'].str.split('/').str[0].astype(int)
            df["Journey_month"] = df['Date_of_Journey'].str.split('/').str[1].astype(int)
            df.drop(["Date_of_Journey"], axis=1, inplace=True)

            # Extracting Dep_hour and Dep_min from Dep_Time
            df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
            df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
            df.drop(["Dep_Time"], axis=1, inplace=True)

            # Extracting Arrival_hour and Arrival_min from Arrival_Time
            df["Arrival_hour"] = pd.to_datetime(df.Arrival_Time).dt.hour
            df["Arrival_min"] = pd.to_datetime(df.Arrival_Time).dt.minute
            df.drop(["Arrival_Time"], axis=1, inplace=True)

            # Handling Total_Stops
            df['Total_Stops'].replace(['1 stop', 'non-stop', '2 stops', '3 stops', '4 stops'], [1, 0, 2, 3, 4], inplace=True)

            # Handling Airline
            df["Airline"].replace({'Multiple carriers Premium economy': 'Other',
                                   'Jet Airways Business': 'Other',
                                   'Vistara Premium economy': 'Other',
                                   'Trujet': 'Other'}, inplace=True)

            # Handling Additional_Info
            df["Additional_Info"].replace({'Change airports': 'Other',
                                           'Business class': 'Other',
                                           '1 Short layover': 'Other',
                                           'Red-eye flight': 'Other',
                                           '2 Long layover': 'Other'}, inplace=True)

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_cleaning(self):
        try:
            df = pd.read_csv(Config.RAW_DATA_PATH)
            df = self.clean_data(df)
            cleaned_data_path = Config.CLEANED_DATA_PATH
            df.to_csv(cleaned_data_path, index=False)
            logger.info("Data cleaning completed.")
            return cleaned_data_path
        except Exception as e:
            raise CustomException(e, sys)
