import os
import sys
import pandas as pd
from src.exception.exception import CustomException
from src.logger.logger import logging
from Config.config import Config

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
            df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.hour
            df["Dep_min"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.minute
            df.drop(["Dep_Time"], axis=1, inplace=True)

            # Extract only the time part from Arrival_Time
            df["Arrival_Time"] = df["Arrival_Time"].str.extract(r'(\d{2}:\d{2})')  # Extract time part

            # Convert to datetime, assuming the extracted time is in HH:MM format
            df["Arrival_Time"] = pd.to_datetime(df["Arrival_Time"], format="%H:%M", errors='coerce')

            # Extract hour and minute
            df["Arrival_hour"] = df["Arrival_Time"].dt.hour
            df["Arrival_min"] = df["Arrival_Time"].dt.minute
            df.drop(["Arrival_Time"], axis=1, inplace=True)

            # Map total stops
            df['Total_Stops'] = df['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})

            # Handling Airline
            df["Airline"] = df["Airline"].replace({
                'Multiple carriers Premium economy': 'Other',
                'Jet Airways Business': 'Other',
                'Vistara Premium economy': 'Other',
                'Trujet': 'Other'
            })

            # Handling Additional_Info
            df["Additional_Info"] = df["Additional_Info"].replace({
                'Change airports': 'Other',
                'Business class': 'Other',
                '1 Short layover': 'Other',
                'Red-eye flight': 'Other',
                '2 Long layover': 'Other'
            })

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_cleaning(self):
        try:
            df = pd.read_csv(Config.RAW_DATA_PATH)
            df = self.clean_data(df)
            cleaned_data_path = Config.CLEANED_DATA_PATH
            df.to_csv(cleaned_data_path, index=False)
            logging.info("Data cleaning completed.")
            return cleaned_data_path
        except Exception as e:
            raise CustomException(e, sys)
