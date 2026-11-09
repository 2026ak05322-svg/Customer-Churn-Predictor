# Subscription Service Customer Churn Predictor

## a. Problem Statement
Retaining active subscribers is a major operational challenge for modern digital service platforms. Identifying at-risk customers early allows companies to deploy targeted retention strategies. The goal of this project is to build a binary classification system using 5 machine learning models to predict whether a subscriber will churn (`1`) or remain active (`0`) based on customer demographic data and usage patterns.

## b. Dataset Description
- **Target Column**: `Churn` (Binary: 0 for Active, 1 for Churned)
- **Dataset Size**: 1,200 instances (Exceeds required minimum of 500 instances)
- **Feature Size**: 14 input features (Exceeds required minimum of 12 features)

### Input Features:
1. **Age**: Numerical age of the subscriber (18 to 70 years).
2. **Tenure_Months**: Number of months the customer has been with the service.
3. **Monthly_Bill**: Monthly service cost (USD).
4. **Total_Usage_GB**: Internet bandwidth data usage in gigabytes.
5. **Support_Calls**: Frequency of customer care interactions.
6. **Payment_Delay_Days**: Number of days payment has been overdue.
7. **Multiple_Lines**: Binary flag representing subscription to multiple telephone lines.
8. **Device_Protection**: Binary flag indicating active equipment coverage plan.
9. **Tech_Support_Plan**: Binary flag representing prioritized tech support tier.
10. **Streaming_TV**: Binary flag representing subscription to video service streaming.
11. **Paperless_Billing**: Binary flag indicating electronic delivery of invoices.
12. **Has_Promo_Code**: Binary flag indicating registration using promotional codes.
13. **Contract_Length_Years**: Categorical contract tier (0: Month-to-month, 1: One-year, 2: Two-year).
14. **Family_Plan_Members**: Number of household members linked to the contract.

## c. GitHub Repository Link
- **GitHub Repository**: [https://github.com/<your-username>/customer-churn-classifier](https://github.com/<your-username>/customer-churn-classifier)
- **Live Streamlit App**: [https://customer-churn-classifier.streamlit.app](https://customer-churn-classifier.streamlit.app)

## d. Models Used & Performance Evaluation Table

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Decision Tree** | 0.8833 | 0.9101 | 0.9030 | 0.9255 | 0.9141 | 0.7329 |
| **K-Nearest Neighbors (KNN)** | 0.8292 | 0.9023 | 0.8529 | 0.9006 | 0.8761 | 0.6039 |
| **Naive Bayes** | 0.9292 | 0.9771 | 0.9286 | 0.9689 | 0.9483 | 0.8378 |
| **Support Vector Machine (SVM)** | 0.9375 | 0.9928 | 0.9398 | 0.9689 | 0.9541 | 0.8571 |

### Overall Winner for your dataset?
**Decision Tree Classifier / SVM** are excellent real-world choices. SVM achieves a highly balanced evaluation profile with an accuracy of **93.75%** and an outstanding AUC Score of **0.9928**.

### Observations on Model Performance
- **Logistic Regression**: Linear separation achieves optimal metrics on the synthetic test boundary parameters.
- **Decision Tree**: Captures multi-level decision pathways well with an Accuracy of 88.33%.
- **KNN**: A strong baseline performer reaching an AUC of 90.23%.
- **Naive Bayes**: Performs exceptionally with a 92.92% Accuracy and stable classification indicators.
- **Support Vector Machine (SVM)**: A highly robust non-linear performer achieving 93.75% Accuracy and an outstanding AUC Score of 99.28%.
