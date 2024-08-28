# Flight Fare Prediction

Project Overview

The Flight Fare Prediction project aims to build a machine learning model to predict flight prices based on various features such as airline, source and destination cities, total stops, and journey timings. This project involves several stages, including data ingestion, cleaning, transformation, model training, and evaluation.

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

# Project Demo How to Work
[RecordedVideo (2).webm](https://github.com/user-attachments/assets/41fcf329-f31f-4f54-9446-df18afe74efa)


# How to Run This Project
1. conda activate venv/
2. git clone https://github.com/Abhishek-guptaaa/Flight_fare_Prediction.git
3. pip install -r requirements.txt
4. python mongo.py
5. python main.py
6. python app.py


# AWS-CICD-Deployment-with-Github-Actions
## 1. Login to AWS console.

## 2. Create IAM user for deployment

#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker image
- Save the URI: 730335610052.dkr.ecr.us-east-1.amazonaws.com/flight_fare

# 4. Create EC2 machine (Ubuntu)

# 5. Open EC2 and Install docker in EC2 Machine:

#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

# 6. Configure EC2 as self-hosted runner:
setting>actions>runner>new self hosted runner> choose os> then run command one by one

# 7. Setup github secrets:
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI =   730335610052.dkr.ecr.us-east-1.amazonaws.com

ECR_REPOSITORY_NAME = flight_fare


