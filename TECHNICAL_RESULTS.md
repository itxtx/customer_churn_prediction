# Technical Results: Customer Churn Prediction

This document provides a detailed overview of the technical analysis, feature engineering, and model tuning for the customer churn prediction project.

## 1. Exploratory Data Analysis (EDA)
- **Churn Rate**: The dataset has a churn rate of 26.5%, indicating a significant class imbalance.
- **Key Predictors**: EDA revealed that `tenure`, `MonthlyCharges`, and `Contract` type are strong initial indicators of churn. Month-to-month contracts have a churn rate of over 42%.

## 2. Data Processing and Feature Engineering

### 2.1. Evaluation of the Data Preprocessing Pipeline
The standard data preprocessing pipeline is robust and follows industry best practices. It employs a `ColumnTransformer` to apply distinct transformations to numeric and categorical features.
- **Numeric Features**: The pipeline uses `SimpleImputer` with a **median** strategy (robust to outliers) followed by `StandardScaler` (essential for models like Logistic Regression).
- **Categorical Features**: The pipeline uses `SimpleImputer` with a **most_frequent** strategy to handle missing values, followed by `OneHotEncoder`. The encoder is configured with `handle_unknown='ignore'`, a defensive, production-ready choice that prevents errors if new, unseen categories are encountered during prediction.

### 2.2. Impact Analysis of Engineered Features
The `FeatureEngineer` class introduces three high-quality features to capture more complex data relationships:
- **`MonthlyToTotalRatio`**: Calculated as `MonthlyCharges / TotalCharges`, this feature serves as a proxy for customer commitment. A high ratio (common for new customers) can indicate a lower switching cost and thus a higher churn risk. The implementation correctly handles edge cases like zero tenure and division-by-zero errors.
- **`NumAdditionalServices`**: This feature counts the number of "Yes" values across six add-on service columns (e.g., `OnlineSecurity`, `StreamingTV`). It aggregates information about a customer's integration into the service ecosystem. A higher number of services likely indicates a "stickier" customer.
- **`HasInternetService`**: This binary flag simplifies the original three-state `InternetService` column ('DSL', 'Fiber optic', 'No') into a simple `True/False` variable, helping simpler models more easily capture the primary effect of having an internet service.

## 3. Model Selection and Optimization

### 3.1. Comprehensive Review of Class Imbalance Strategies
The project's approach to addressing class imbalance was exceptionally thorough. The hyperparameter tuning process systematically evaluated a wide array of strategies:
- **No Resampling**: Served as a baseline.
- **Over-sampling**: Techniques like **SMOTE**, **RandomOverSampler**, and **ADASYN** were tested to increase the minority (churn) class representation.
- **Under-sampling**: Techniques like **RandomUnderSampler** were tested to reduce the majority class representation.
- **Internal Model Weighting**: The `class_weight` parameter of models was also explored.
This empirical, data-driven approach allowed for the determination of the most effective strategy for each model. For example, results showed that `RandomOverSampler` was the winning approach for Random Forest, while **SMOTE** was optimal for the final Gradient Boosting model.

### 3.2. Final Hyperparameters
The final, optimized hyperparameters for each model are detailed below.

#### Logistic Regression
- **Performance**: F1-Score: 0.612, Recall: 0.553, ROC AUC: 0.849
- **Final Hyperparameters**:
    - `classifier__C`: 0.001
    - `classifier__penalty`: 'l2'
    - `classifier__solver`: 'liblinear'

#### Random Forest
- **Performance**: F1-Score: ~0.61, Recall: ~0.69, ROC AUC: ~0.84
- **Final Hyperparameters**:
    - `classifier__bootstrap`: True
    - `classifier__class_weight`: None
    - `classifier__max_depth`: 12
    - `classifier__max_features`: 'log2'
    - `classifier__min_samples_leaf`: 6
    - `classifier__min_samples_split`: 8
    - `classifier__n_estimators`: 203
    - `resampler__sampling_strategy`: 0.999

#### XGBoost
- **Final Hyperparameters**:
    - `classifier__colsample_bytree`: 0.6
    - `classifier__gamma`: 0.06
    - `classifier__learning_rate`: 0.012
    - `classifier__max_depth`: 3
    - `classifier__n_estimators`: 550
    - `classifier__reg_alpha`: 0.02
    - `classifier__reg_lambda`: 0.168
    - `classifier__scale_pos_weight`: 0.8
    - `classifier__subsample`: 0.612
    - `resampler__sampling_strategy`: 1.0

#### Gradient Boosting (Best Model)
- **Performance**: F1-Score: 0.624, Recall: 0.711, ROC AUC: 0.842
- **Final Hyperparameters**:
    - `classifier__learning_rate`: 0.0117
    - `classifier__max_depth`: 4
    - `classifier__min_samples_leaf`: 4
    - `classifier__min_samples_split`: 3
    - `classifier__n_estimators`: 350
    - `classifier__subsample`: 0.736
    - `resampler__k_neighbors`: 4