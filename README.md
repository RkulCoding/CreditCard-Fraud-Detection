"variant":"document",
"id":"48319",
"title":"README Template for Credit Card Fraud Detection"
}

# Credit Card Fraud Detection (Machine Learning Project)

### 🎯 Objective

This project is an end-to-end Machine Learning + Data Visualization Dashboard for detecting fraudulent credit card transactions. It includes data preprocessing, EDA (Exploratory Data Analysis), ML model training, evaluation, and simple visualizations.

---

## 📌 Dataset

* Source: Kaggle
* Total transactions: 284,807
* Fraud transactions: Highly imbalanced dataset
* Features: Time, Amount, V1–V28 (PCA components)

---

## 🛠️ Tools & Technologies

* **Python**
* **Pandas**, **NumPy**
* **Matplotlib**, **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

---

## 📂 Project Structure

```
CreditCard-Fraud-Detection/
│── data/
│── notebooks/
│── scripts/
│── visuals/
│── README.md
```

---

## 🧹 Step 1: Data Preprocessing

* Handle imbalance using undersampling
* Scale "Amount" and "Time"
* Remove missing values
* Prepare train/test split

---

## 📊 Step 2: Exploratory Data Analysis

* Fraud vs Legit distribution
* Correlation Heatmap
* Amount distribution
* Time-based transaction patterns

### 🔥 Sample Visual

![Fraud vs Legit](visuals/fraud_vs_legit.png)

---

## 🤖 Step 3: Model Training

Tried multiple models:

* Logistic Regression
* Decision Tree
* Random Forest (Best)

### Best Model: **Random Forest**

* Accuracy: ~96%
* Precision: High for fraud class
* Handles imbalance well

---

## 📈 Step 4: Evaluation

* Confusion Matrix
* Classification Report
* ROC Curve

---

## 🧠 Key Insights

* Fraud transactions have lower amount values
* Certain PCA components correlate strongly with fraud
* Imbalanced dataset → need careful resampling

---

## 🚀 How to Run the Project

```
pip install -r requirements.txt
cd scripts
python train_model.py
```

---

## 📬 Contact

Feel free to connect for collaboration or improvements.
