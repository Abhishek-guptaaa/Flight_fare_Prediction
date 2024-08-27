# Flight Fare Prediction

Project Overview

The Flight Fare Prediction project aims to build a machine learning model to predict flight prices based on various features such as airline, source and destination cities, total stops, and journey timings. This project involves several stages, including data ingestion, cleaning, transformation, model training, and evaluation.

conda activate venv/

git clone https://github.com/Abhishek-guptaaa/Flight_fare_Prediction.git


Features
Data Cleaning: Handles missing values, outlier detection, and data normalization.

Data Transformation: Encodes categorical features, scales numerical features, and prepares data for modeling.

Model Training: Trains multiple regression models including Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBoost, and AdaBoost.

Model Evaluation: Evaluates models based on metrics such as R² score, RMSE, and MAE.

Model Saving: Saves the best-performing model for future use.

Prerequisites
Python 3.x
Required Python libraries: numpy, pandas, scikit-learn, xgboost, joblib, mlflow, flask, matplotlib, seaborn


### MLflow Integration
This project uses MLflow for tracking experiments, logging parameters, and evaluating model performance. Ensure MLflow is properly configured to log metrics and parameters.