# Telecom Customer Churn Prediction



## Project Structure

```text
customer_churn_prediction/
├── scripts/
│   ├── download_data.py
│   └── download_model.py
├── src/
│   ├── api.py
│   ├── data_processing.py
│   ├── main.py
│   ├── predict.py
│   ├── simulation_utils.py
│   └── train.py
├── tests/
│   └── fixtures/
├── config.yaml
├── requirements.txt
└── TECHNICAL_RESULTS.md
```

Generated assets are intentionally kept out of git:

- `data/raw/` for downloaded datasets
- `models/` for trained or downloaded model artifacts
- local databases, virtual environments, and notebook-generated binaries

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the dataset:

```bash
python scripts/download_data.py
```

The script writes to the path configured in `config.yaml`:

```yaml
data:
  raw_data_path: "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

If the public mirror changes, override it:

```bash
CHURN_DATA_URL="https://example.com/telco.csv" python scripts/download_data.py
```

## Train And Evaluate

Train the configured models and save the selected final pipeline:

```bash
python -m src.main train --config config.yaml
```

Training writes:

- `models/final_model.joblib`
- `models/training_report.json`

The default model registry target is configured as:

```yaml
models:
  registry_repo: "itxtx/customer-churn-prediction"
```

Model binaries are not committed to GitHub. To restore a previously published model from Hugging Face Hub:

```bash
python scripts/download_model.py
```

You can override the repo without editing config:

```bash
CHURN_MODEL_REPO="your-user/your-model-repo" python scripts/download_model.py
```

## Predict

Run batch prediction from CSV:

```bash
python -m src.main predict --batch-file tests/fixtures/customers.csv
```

Run one customer from JSON:

```bash
python -m src.main predict --customer-json '{"customerID":"demo-001","gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'
```

## Run The API

Start the API:

```bash
uvicorn src.api:app --reload
```

Available endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `POST /predict/explain`

The API returns `503` from prediction endpoints when `models/final_model.joblib` is not present. Run training or `scripts/download_model.py` first.

## Simulation Example

```python
from src.predict import ChurnPredictor
from src.simulation_utils import run_simulation, calculate_revenue_impact

predictor = ChurnPredictor()
predictor.load_model()

distributions = {
    "tenure": ("uniform", [1, 72]),
    "MonthlyCharges": ("normal", [64.76, 30.09]),
    "Contract": (
        "choice",
        {
            "options": ["Month-to-month", "One year", "Two year"],
            "probabilities": [0.55, 0.24, 0.21],
        },
    ),
}

results = run_simulation(predictor, distributions, num_simulations=1000, n_jobs=1)
impact = calculate_revenue_impact(results)
print(impact)
```

## Tests

Run the full local test suite:

```bash
python -m pytest tests/ -q
```

Useful smoke checks:

```bash
python -m src.main --help
python -m src.main train --config config.yaml
python -m src.main predict --batch-file tests/fixtures/customers.csv
```

## Configuration

See [TECHNICAL_RESULTS.md](TECHNICAL_RESULTS.md) for the current modeling summary and interpretation notes. Regenerate detailed metrics locally with:

```bash
python -m src.main train --config config.yaml
```
