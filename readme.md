# Report Summary: Telecom Customer Churn Prediction

## 1. Objective & Context
- **Goal:** Predict customer churn for a telecom company using the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset (7043 customers, ~26.5% churn rate).
- **Purpose:** Enable proactive, targeted retention efforts to reduce revenue loss by identifying customers likely to leave.
- **Scope:** Data exploration, preprocessing, feature engineering, handling class imbalance (SMOTE), training/tuning classifiers (Logistic Regression, Random Forest, Gradient Boosting), and evaluating performance, focusing on Recall for the churn class.

## 2. Methodology
- **Data Prep:** Dropped `customerID`, converted `TotalCharges` to numeric (handled 11 missing values).
- **Feature Engineering:** Created `MonthlyToTotalRatio`, `NumAdditionalServices`, `HasInternetService`.
- **Preprocessing:**
    - 80/20 stratified train-test split.
    - Pipeline: Median imputation + scaling for numeric; Mode imputation + One-Hot Encoding for categorical.
- **Modeling & Tuning:**
    - **Baseline:** Logistic Regression.
    - **Ensembles:** Random Forest, Gradient Boosting.
    - **Imbalance Handling:** Primarily used SMOTE (found effective via tuning).
    - **Tuning:** `RandomizedSearchCV` (RF, GB) & `BayesSearchCV` (RF) optimizing for F1-score.
- **Evaluation:** Prioritized **Recall** for churn class (minimize missed churners). Also tracked Precision, F1-Score, Accuracy, ROC-AUC.

## 3. Key Findings
- **Baseline (Logistic Regression):** Recall 0.553, F1 0.612, AUC 0.849.
- **Tuned Random Forest (SMOTE):** Recall ~0.69, F1 ~0.61, AUC ~0.84. (Randomized vs Bayesian search yielded similar results).
- **Tuned Gradient Boosting (SMOTE - Best Model):**
    - **Test Set Recall (Churn=1): 0.711**
    - **Test Set F1-Score (Churn=1): 0.624**
    - Test Set Precision (Churn=1): 0.556
    - Test Set Accuracy: 0.773
    - Test Set ROC AUC: 0.842
- **Comparison:** Ensemble models with SMOTE significantly improved Recall over the baseline. Gradient Boosting performed slightly better than Random Forest. Higher Recall came with lower Precision (~0.55).

## 4. Discussion & Limitations
- **Interpretation:** Tuned ensemble models (especially GB with SMOTE) effectively predict churn (AUC ~0.84).
- **Business Impact:** The best model identifies ~71% of actual churners, enabling efficient targeting for retention campaigns.
- **Limitations:**
    - Model based on static data; requires periodic retraining.
    - Precision/Recall trade-off: ~44% of predicted churners might be false positives (cost consideration needed).
    - Detailed feature importance analysis not included.

## 5. Conclusion
- Machine learning models, particularly tuned Gradient Boosting with SMOTE, successfully predict telecom churn.
- The best model achieved **Recall=0.711** and **F1=0.624** for the churn class, identifying over 70% of at-risk customers.
- This provides a valuable tool for data-driven customer retention, despite the trade-off with precision.
