# telco-churn-prediction
Churn Prediction using ML models with EDA
# Telco Customer Churn Prediction
# Telco Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-blue?style=flat-square&logo=linkedin)](www.linkedin.com/in/suyash-kulkarni-yes777)

A machine learning project to predict whether a customer will churn (leave the service) based on their demographic, account, and usage data. Built with Python, Pandas, Seaborn, Scikit-learn, and XGBoost.

## Problem Statement

Customer churn is a major concern for telecom companies. Retaining existing customers is far more cost-effective than acquiring new ones. Using historical data, we aim to predict whether a customer is likely to churn, so proactive retention strategies can be applied.

---

## Dataset

-  `data/Telco-Customer-Churn.csv`
- Source: [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Features include:
  - `gender`, `SeniorCitizen`, `Partner`, `tenure`
  - `Contract`, `MonthlyCharges`, `TotalCharges`, etc.
  - `Churn` (Target)

---

## Exploratory Data Analysis (EDA)

We performed the following:
- Churn distribution with percentage annotations
- Monthly charges distribution by churn (Violin Plot)
- KDE plots for tenure vs churn
- Correlation heatmaps for numeric features
- Stacked bar plots for churn by contract type
- Pairplots for selected features

---

## Model Training

- Data preprocessing:
  - Handled missing values
  - Label encoding for categorical variables
  - StandardScaler for feature scaling

- ML Models used:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier (Best performing)

- Model evaluation:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix & ROC AUC Curve

---

##  Final Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 79%      | 74%       | 68%    | 71%      |
| Random Forest      | 81%      | 77%       | 72%    | 74%      |
| XGBoost            | **82%**  | **78%**   | **74%**| **76%**  |

---

##  Tech Stack

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

---
##To clone this Repo
git clone https://github.com/Gitforsuyash/telco-churn-prediction.git
cd telco-churn-prediction
---
---
## To Install requirements

pip install -r requirements.txt
---
---
## To open notebook 
jupyter notebook notebook/telco_churn_prediction.ipynb

---
## Business Insight
Key drivers of churn include:

Month-to-month contracts

Lack of online security/tech support

Higher monthly charges

## Churn can be reduced by:
Offering long-term contracts
Bundling services
Offering loyalty discounts
---

## Contribution
Feel free to fork, raise issues, or submit PRs for improvements!!
---
## 📧 Contact
Created by Suyash — aspiring AI/ML engineer. Connect on @linkedin or raise issues for discussion.
