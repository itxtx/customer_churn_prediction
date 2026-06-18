import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.predict import ChurnPredictor
from src.train import ModelTrainer


def write_test_config(tmp_path: Path) -> Path:
    config = {
        "data": {
            "raw_data_path": "tests/fixtures/telco_sample.csv",
            "test_size": 0.25,
            "random_state": 42,
        },
        "features": {
            "target_column": "Churn",
            "customer_id_column": "customerID",
            "numeric_features": ["tenure", "MonthlyCharges", "TotalCharges"],
            "categorical_features": [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
            ],
            "engineered_features": [
                "MonthlyToTotalRatio",
                "NumAdditionalServices",
                "HasInternetService",
            ],
        },
        "models": {
            "output_dir": str(tmp_path / "models"),
            "final_model_path": str(tmp_path / "models" / "final_model.joblib"),
            "registry_repo": "itxtx/customer-churn-prediction",
            "training_report_path": str(tmp_path / "models" / "training_report.json"),
        },
        "training": {
            "models_to_train": ["logistic"],
            "cv_folds": 2,
            "scoring_metric": "f1",
            "n_iter_search": 0,
            "tuning_strategy": "random",
        },
        "api": {"host": "0.0.0.0", "port": 8000, "reload": True},
        "logging": {"level": "INFO"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_trainer_saves_final_model_and_report(tmp_path):
    config_path = write_test_config(tmp_path)
    trainer = ModelTrainer(str(config_path))
    X, y = trainer.prepare_training_data()
    report = trainer.train_and_select_best_model(X, y)

    assert report["best_model"] == "logistic"
    assert Path(report["final_model_path"]).exists()
    assert (tmp_path / "models" / "training_report.json").exists()


def test_predictor_loads_final_model_and_predicts_batch(tmp_path):
    config_path = write_test_config(tmp_path)
    trainer = ModelTrainer(str(config_path))
    X, y = trainer.prepare_training_data()
    trainer.train_and_select_best_model(X, y)

    predictor = ChurnPredictor(str(config_path))
    assert predictor.load_model()

    customers = pd.read_csv("tests/fixtures/customers.csv").to_dict("records")
    results = predictor.predict_batch(customers)

    assert len(results) == 2
    assert set(results[0]) == {
        "customer_id",
        "churn_prediction",
        "churn_probability",
        "risk_level",
        "confidence",
    }
    assert results[0]["churn_prediction"] in {"Yes", "No"}
    assert 0 <= results[0]["churn_probability"] <= 1


def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Customer Churn Prediction System" in result.stdout


def test_cli_train_and_predict_smoke(tmp_path):
    config_path = write_test_config(tmp_path)

    train_result = subprocess.run(
        [sys.executable, "-m", "src.main", "train", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert train_result.returncode == 0, train_result.stderr
    assert json.loads(train_result.stdout)["best_model"] == "logistic"

    predict_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main",
            "predict",
            "--config",
            str(config_path),
            "--batch-file",
            "tests/fixtures/customers.csv",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert predict_result.returncode == 0, predict_result.stderr
    predictions = json.loads(predict_result.stdout)
    assert len(predictions) == 2
