# Telecom Customer Churn Prediction

This project builds a binary classification model to proactively identify customers at high risk of churning from a telecom company. By leveraging customer data, the model enables targeted retention efforts, helping to reduce revenue loss and improve customer satisfaction.

## Table of Contents
- [Telecom Customer Churn Prediction](#telecom-customer-churn-prediction)
  - [Table of Contents](#table-of-contents)
  - [1. Business Understanding \& Problem Definition](#1-business-understanding--problem-definition)
  - [2. Key Findings \& Best Model](#2-key-findings--best-model)
  - [3. Project Structure](#3-project-structure)
  - [4. Installation](#4-installation)
  - [5. Usage](#5-usage)
    - [Model Training](#model-training)
  - [6. Technical Deep Dive](#6-technical-deep-dive)

## 1. Business Understanding & Problem Definition
- **Problem Statement**: Customer churn, in the context of this telecom company, refers to the phenomenon where existing customers discontinue their service subscriptions.
- **Business Goal**: The primary business goal is to proactively identify customers who are at high risk of churning in order to:
    - **Reduce Revenue Loss**: Retaining existing customers is significantly more cost-effective than acquiring new ones.
    - **Enable Targeted Retention Efforts**: Identifying high-risk customers allows the company to implement specific retention strategies (e.g., discounts, service upgrades) only for those who need them most, maximizing ROI.
- **ML Objective**: To build a binary classification model that predicts whether a given customer will churn (Yes) or not (No) based on their account details and service usage patterns.
- **Key Evaluation Metric**: **Recall** for the "Yes" (Churn) class is the most critical metric. Missing a customer who is going to churn (a False Negative) is considered more costly than mistakenly targeting a non-churning customer (a False Positive).

## 2. Key Findings & Best Model
- **Best Strategy**: The combination of a **Gradient Boosting** model with a **SMOTE** resampling strategy proved to be the most effective combination.
- **Performance**:
    - **Test Set Recall (Churn=1): 0.711**
    - **Test Set F1-Score (Churn=1): 0.624**
    - Test Set ROC AUC: 0.842

## 3. Project Structure
The repository is organized as follows:
```
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── best_hyperparameters.json
├── notebooks/
│   ├── churn_notebook.ipynb
│   └── parameter_search.ipynb
├── src/
│   ├── init.py
│   ├── api.py
│   ├── config.py
│   ├── data_processing.py
│   ├── database.py
│   ├── main.py
│   ├── predict.py
│   └── train.py
├── tests/
│   └── test_data_processing.py
├── config.yaml
├── readme.md
└── requirements.txt
```
- **`data/`**: Contains the raw dataset.
- **`models/`**: Stores trained model artifacts and hyperparameters.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis (`churn_notebook.ipynb`) and hyperparameter tuning (`parameter_search.ipynb`).
- **`src/`**: Contains the main source code, including the FastAPI application (`api.py`), data processing logic (`data_processing.py`), and model training/prediction scripts (`train.py`, `predict.py`).
- **`tests/`**: Contains unit tests for the project.
- **`config.yaml`**: Configuration file for defining model parameters, feature lists, and file paths.

## 4. Installation
To set up the project locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/customer_churn_prediction.git](https://github.com/customer_churn_prediction.git)
    cd customer_churn_prediction
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 5. Usage

### Model Training
The entire model training and tuning process can be executed by running the `train.py` script. This script uses the settings in `config.yaml` to find the best hyperparameters and saves the resulting model pipeline to the `models/` directory.

```bash
python src/train.py
```

## 6. Technical Deep Dive
For a detailed technical analysis, including EDA, feature engineering, and in-depth model tuning results, please see the TECHNICAL_RESULTS.md file.