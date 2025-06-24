# Report Summary: Telecom Customer Churn Prediction

## 1. Business Understanding & Problem Definition
- **Problem Statement**: Customer churn, in the context of this telecom company, refers to the phenomenon where existing customers discontinue their service subscriptions (e.g., phone, internet) within a defined period.
- **Business Goal**: The primary business goal is to proactively identify customers who are at high risk of churning in order to:
    - **Reduce Revenue Loss**: Retaining existing customers is significantly more cost-effective than acquiring new ones. Reducing churn directly protects the revenue base.
    - **Enable Targeted Retention Efforts**: Identifying high-risk customers allows the company to implement specific retention strategies (e.g., discounts, service upgrades) only for those who need them most, maximizing ROI.
    - **Improve Customer Satisfaction**: Understanding the drivers of churn can highlight systemic issues with services, pricing, or support.
- **ML Objective**: To build a binary classification model that predicts whether a given customer will churn (Yes) or not (No) based on their demographic information, account details, and service usage patterns.
- **Key Evaluation Metric**: **Recall (Sensitivity)** for the 'Yes' (Churn) class is the most critical metric. Missing a customer who is going to churn (a False Negative) is considered more costly than mistakenly targeting a non-churning customer (a False Positive). We will also monitor Precision, F1-Score, and ROC-AUC.

## 2. Methodology Overview
- **Data Prep**: Loaded data, converted `TotalCharges` to numeric (handled 11 missing values).
- **Feature Engineering**: Created `MonthlyToTotalRatio`, `NumAdditionalServices`, `HasInternetService`.
- **Preprocessing**: Used a robust pipeline for imputation (median for numeric, mode for categorical) and scaling (StandardScaler for numeric, OneHotEncoder for categorical).
- **Modeling & Tuning**:
    - **Class Imbalance**: Systematically evaluated various over-sampling (SMOTE, RandomOverSampler), under-sampling, and class weighting strategies.
    - **Models**: Trained and tuned Logistic Regression, Random Forest, XGBoost, and Gradient Boosting.
    - **Tuning**: Employed `RandomizedSearchCV` and `BayesSearchCV` to find the optimal model and resampling combination, optimizing for F1-score.

## 3. Key Findings & Best Model
- **Best Strategy**: The combination of **Gradient Boosting with SMOTE** proved to be the most effective model.
- **Performance**:
    - **Test Set Recall (Churn=1): 0.711**
    - **Test Set F1-Score (Churn=1): 0.624**
    - Test Set Precision (Churn=1): 0.556
    - Test Set Accuracy: 0.773
    - Test Set ROC AUC: 0.842

## 4. Discussion & Limitations
- **Business Impact**: The model can successfully identify ~71% of customers who are at risk of churning, allowing the business to focus retention efforts effectively. The trade-off is a moderate precision, which is acceptable given the primary goal of minimizing customer loss.
- **Limitations**: The model's performance is based on the current dataset. It should be monitored and retrained periodically as new data becomes available and customer behaviors evolve.

## 5. Technical Deep Dive
For a detailed technical analysis, including EDA, feature engineering justification, and in-depth model tuning results, please see the [TECHNICAL_RESULTS.md](TECHNICAL_RESULTS.md) file.