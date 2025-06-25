# Telecom Customer Churn Prediction

A comprehensive machine learning solution for predicting customer churn in the telecom industry. This project combines advanced ML models with business simulation capabilities to provide actionable insights for customer retention strategies.

## Key Features

- **High-Performance ML Models**: XGBoost, Random Forest, and Gradient Boosting models with automated hyperparameter tuning
- **Production-Ready API**: FastAPI web service for real-time predictions
- **Business Simulation Tools**: Monte Carlo simulations for scenario analysis and revenue impact assessment
- **Comprehensive Data Pipeline**: Automated data processing, feature engineering, and model validation
- **Interactive Notebooks**: Jupyter notebooks for analysis, experimentation, and visualization

## Table of Contents
- [Key Features](#key-features)
- [Business Understanding & Problem Definition](#business-understanding--problem-definition)
- [Model Performance & Results](#model-performance--results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Model Training](#model-training)
  - [Making Predictions](#making-predictions)
  - [Running Simulations](#running-simulations)
  - [API Usage](#api-usage)
- [Advanced Features](#advanced-features)
- [Business Impact Analysis](#business-impact-analysis)
- [Testing](#testing)
- [Technical Deep Dive](#technical-deep-dive)

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

## Usage

### Quick Start

1. **Train the model:**
   ```bash
   python src/main.py train
   ```

2. **Make a single prediction:**
   ```bash
   python src/main.py predict --customer-data '{"tenure": 12, "MonthlyCharges": 70.5, "Contract": "Month-to-month"}'
   ```

3. **Start the API server:**
   ```bash
   python src/api.py
   ```

### Model Training

The training process includes automated hyperparameter tuning and model comparison:

```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Train multiple models (XGBoost, Random Forest, Gradient Boosting)
- Perform hyperparameter optimization using Bayesian search
- Save the best model and hyperparameters

### Making Predictions

#### Single Customer Prediction
```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor()
predictor.load_model()

customer_data = {
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic"
}

result = predictor.predict_single(customer_data)
print(f"Churn Probability: {result['churn_probability']:.3f}")
print(f"Risk Level: {result['risk_level']}")
```

#### Batch Predictions
```python
customers_data = [customer_data1, customer_data2, customer_data3]
results = predictor.predict_batch(customers_data)
```

### Running Simulations

#### Basic Monte Carlo Simulation
```python
from src.simulation_utils import run_simulation

# Define customer distributions
baseline_distributions = {
    'tenure': ('uniform', [1, 72]),
    'MonthlyCharges': ('normal', [64.76, 30.09]),
    'Contract': ('choice', {
        'options': ['Month-to-month', 'One year', 'Two year'],
        'probabilities': [0.55, 0.24, 0.21]
    })
}

# Run simulation
results = run_simulation(predictor, baseline_distributions, num_simulations=10000)
print(f"Average churn rate: {results['churn_probability'].mean():.3f}")
```

#### Price Sensitivity Analysis
```python
from src.simulation_utils import run_price_sensitivity_analysis

price_factors = [0.9, 1.0, 1.1, 1.2, 1.3]  # -10% to +30% price changes
results = run_price_sensitivity_analysis(
    predictor, 
    baseline_distributions, 
    price_factors, 
    num_simulations=5000
)
print(results)
```

#### Scenario Comparison
```python
from src.simulation_utils import run_scenario_comparison

scenarios = {
    'High Price': {'MonthlyCharges': ('normal', [80, 25])},
    'Premium Service': {
        'MonthlyCharges': ('normal', [90, 20]),
        'Contract': ('choice', {
            'options': ['One year', 'Two year'],
            'probabilities': [0.4, 0.6]
        })
    }
}

comparison = run_scenario_comparison(predictor, baseline_distributions, scenarios)
print(comparison)
```

### API Usage

Start the FastAPI server:
```bash
uvicorn src.api:app --reload
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tenure": 12,
       "MonthlyCharges": 70.5,
       "Contract": "Month-to-month",
       "PaymentMethod": "Electronic check"
     }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "customers": [
         {"tenure": 12, "MonthlyCharges": 70.5},
         {"tenure": 24, "MonthlyCharges": 55.0}
       ]
     }'
```

## Advanced Features

### Data Processing Pipeline

The project includes a sophisticated data processing pipeline:

- **Missing Value Imputation**: Smart handling of missing TotalCharges based on tenure
- **Feature Engineering**: Automated creation of derived features like MonthlyToTotalRatio
- **Data Validation**: Comprehensive validation with detailed error reporting
- **Preprocessing**: Standardized numeric features and one-hot encoded categorical features

### Model Comparison and Selection

The training pipeline automatically compares multiple algorithms:
- XGBoost Classifier
- Random Forest Classifier  
- Gradient Boosting Classifier
- Logistic Regression (baseline)

Each with:
- Bayesian hyperparameter optimization
- SMOTE oversampling for class imbalance
- Cross-validation for robust evaluation

### Simulation Engine

Advanced Monte Carlo simulation capabilities:
- **Distribution Support**: Normal, uniform, choice, lognormal, exponential, and more
- **Parallel Processing**: Multi-core support for large simulations
- **Revenue Impact**: Calculate financial implications of churn scenarios
- **Scenario Testing**: Compare multiple business scenarios simultaneously

## Business Impact Analysis

### Revenue Impact Calculation
```python
from src.simulation_utils import calculate_revenue_impact

revenue_metrics = calculate_revenue_impact(simulation_results)
print(f"Expected churn rate: {revenue_metrics['churn_rate']:.2%}")
print(f"Revenue at risk: ${revenue_metrics['expected_lost_revenue']:,.2f}")
print(f"Revenue retention rate: {revenue_metrics['revenue_retention_rate']:.2%}")
```

### Key Business Metrics
- **Customer Lifetime Value Impact**
- **Revenue Retention Rates**
- **Cost-Benefit Analysis** of retention campaigns
- **Scenario Planning** for pricing strategies

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run specific test modules:
```bash
python -m pytest tests/test_data_processing.py -v
```

## Project Structure (Updated)

```
customer_churn_prediction/
├── data/
│   └── raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/                           # Trained models and artifacts
│   ├── best_model_xgboost.pkl
│   ├── best_model_random_forest.pkl
│   ├── best_model_gradient_boosting.pkl
│   └── best_hyperparameters.json
├── notebooks/                        # Analysis and experimentation
│   ├── churn_notebook.ipynb
│   ├── parameter_search.ipynb
│   ├── advanced_simulation.ipynb
│   └── price_impact_simulation.ipynb
├── src/                             # Main source code
│   ├── __init__.py
│   ├── api.py                       # FastAPI web service
│   ├── data_processing.py           # Data pipeline
│   ├── database.py                  # Database operations
│   ├── main.py                      # CLI interface
│   ├── predict.py                   # Prediction engine
│   ├── simulation_utils.py          # Monte Carlo simulations
│   └── train.py                     # Model training
├── tests/                           # Unit tests
│   ├── __init__.py
│   └── test_data_processing.py
├── graphs/                          # Generated visualizations
├── config.yaml                      # Configuration
├── requirements.txt                 # Dependencies
├── readme.md                        # This file
└── TECHNICAL_RESULTS.md            # Detailed technical analysis
```

## Configuration

The `config.yaml` file controls all aspects of the pipeline:

```yaml
data:
  raw_data_path: "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
  
features:
  numeric_features: ["tenure", "MonthlyCharges", "TotalCharges"]
  categorical_features: ["gender", "Partner", "Dependents", "PhoneService"]
  target_column: "Churn"
  
models:
  output_dir: "models/"
  algorithms: ["xgboost", "random_forest", "gradient_boosting"]
  
simulation:
  default_num_simulations: 10000
  parallel_jobs: -1
```

## Technical Deep Dive

For detailed technical analysis, including:
- Exploratory Data Analysis (EDA)
- Feature Engineering methodology
- Model performance comparisons
- Hyperparameter tuning results
- Business impact calculations

Please see the [TECHNICAL_RESULTS.md](TECHNICAL_RESULTS.md) file.
