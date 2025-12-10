# CreditCard-Fraud-Detection
This project is an end-to-end Machine Learning + Data Visualization Dashboard for detecting fraudulent credit card transactions. It includes data preprocessing, EDA (Exploratory Data Analysis), ML model training, evaluation, and simple visualization.

#prject structure
CreditCard-Fraud-Detection/
│
├── Data/
│   └── creditcard.csv              # Dataset (not uploaded to GitHub because it is >100 MB)
│
├── Notebooks/
│   ├── fraud_detection_eda_model.ipynb
│   ├── ML_Models.ipynb
│   └── prediction_Visualization.ipynb
│
├── Visuals/
│   └── Graphs.ipynb
│
└── README.md

Project Description

The goal of this project is to identify fraudulent credit card transactions using machine learning models.

You perform:

✅ Data Cleaning and Preprocessing

Handling class imbalance

Scaling numerical features

Splitting dataset into train/test

✅ Exploratory Data Analysis

Fraud vs Non-Fraud distribution

Amount pattern analysis

Correlation heatmaps

Visual graphs

✅ Machine Learning Models

You trained and evaluated:

Logistic Regression

Random Forest Classifier

XGBoost (optional)

✅ Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

📊 Results

Fraud detection is an imbalanced problem, so metrics like Recall and F1-Score matter the most.
The Random Forest model usually performs best on such datasets.

(You can replace this section later with exact numbers from your notebook.)

🚀 How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Open the Notebooks

Run using VS Code or Jupyter:

Notebooks/ML_Models.ipynb
Notebooks/fraud_detection_eda_model.ipynb
Notebooks/prediction_Visualization.ipynb

⚠️ Important Note

The dataset (creditcard.csv) is NOT uploaded to GitHub because it is 143 MB and GitHub has a 100 MB limit.

You can download the dataset from:
🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Data/creditcard.csv

📌 Future Improvements

Add Streamlit dashboard

Model retraining pipeline

API endpoint using FastAPI

Deploy on AWS/GCP

👤 Author

Rashmi Kulkarni
MCA Student | Data Analyst & ML Learner

