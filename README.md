# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

This project detects fraudulent credit card transactions using Machine Learning. A Random Forest Classifier was trained on an imbalanced credit card transaction dataset to classify transactions as **Fraud** or **Normal**.

The project also includes a **Streamlit web application** where users can upload a CSV file, predict fraudulent transactions, view prediction results, and download the results.

---

## 🚀 Features

- Upload credit card transaction CSV file
- Predict fraudulent transactions
- Display prediction results
- Show total Normal and Fraud transactions
- Download prediction results as CSV
- User-friendly Streamlit interface

---

## 🛠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

---

## 📂 Dataset

The project uses the **Credit Card Fraud Detection Dataset**.

Dataset Features:

- Time
- V1 to V28 (PCA transformed features)
- Amount
- Class (Target)

Target:

- 0 → Normal Transaction
- 1 → Fraud Transaction

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Exploration & Exploratory Data Analysis (EDA)
3. Data Preprocessing
   - Data Cleaning
   - Feature Scaling
4. Train-Test Split
5. Handling Class Imbalance using SMOTE
6. Model Training
   - Logistic Regression
   - Naive Bayes
   - Decision Tree
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
7. Ensemble Learning using Random Forest
8. Model Evaluation
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC Score
   - Confusion Matrix
   - Classification Report
9. Model Comparison
10. Best Model Selection
11. Model Serialization using Joblib
12. Streamlit Web Application Development
13. Batch Prediction using CSV Upload
14. Deployment

---

## 🤖 Model Used

**Random Forest Classifier (Ensemble Learning - Bagging)**

Random Forest is an ensemble learning algorithm that builds multiple decision trees using bootstrap sampling and random feature selection. The final prediction is made by majority voting, improving accuracy and reducing overfitting.
---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

## 📁 Project Structure

```
CreditCardFraudDetection/
│
├── app.py
├── Creditcard1.ipynb
├── random_forest_creditcard.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
├── README.md
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/TamaliM174/creditcardfrauddetection.git
```

Go to project folder

```bash
cd Credit-Card-Fraud-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## 📷 Application

### Upload CSV

- Upload transaction dataset
- Click **Predict**

### Results

- Displays predictions
- Shows Fraud and Normal transaction counts
- Download prediction results as CSV

---

## 📌 Future Improvements

- Add Probability Score
- Interactive Charts
- Feature Importance Visualization
- Real-time API Prediction

---

## 👨‍💻 Author

Tamali Mahapatra