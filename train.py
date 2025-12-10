import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load Data
data = pd.read_csv("data/creditcard.csv")

# Split features/target
X = data.drop("Class", axis=1)
y = data["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Create model folder if not exists
os.makedirs("../model", exist_ok=True)

# Save model + scaler
joblib.dump(model, "../model/fraud_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")

print("Model and scaler saved successfully!")
