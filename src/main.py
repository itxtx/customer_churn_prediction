import argparse
import json
import logging
from typing import Any, Dict, List, Union

import pandas as pd

from src.data_processing import DataProcessor
from src.predict import ChurnPredictor
from src.train import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_models(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Run the reproducible train/evaluate/save pipeline."""
    logger.info("Starting model training pipeline.")
    trainer = ModelTrainer(config_path)
    X, y = trainer.prepare_training_data()
    report = trainer.train_and_select_best_model(X, y)
    logger.info(
        "Training complete. Best model: %s (%s=%.4f)",
        report["best_model"],
        report["selection_metric"],
        report["best_score"],
    )
    return report


def make_predictions(
    customer_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    config_path: str = "config.yaml",
) -> List[Dict[str, Any]]:
    """Load the configured final model and make single or batch predictions."""
    predictor = ChurnPredictor(config_path)
    if not predictor.load_model():
        raise FileNotFoundError(
            "No final model found. Run training first or use scripts/download_model.py."
        )

    if isinstance(customer_data, dict):
        return [predictor.predict_single(customer_data)]
    return predictor.predict_batch(customer_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer Churn Prediction System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train and save the final model")
    train_parser.add_argument("--config", default="config.yaml", help="Path to config file")

    predict_parser = subparsers.add_parser("predict", help="Make churn predictions")
    predict_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    predict_input = predict_parser.add_mutually_exclusive_group(required=True)
    predict_input.add_argument("--batch-file", help="CSV file containing customers")
    predict_input.add_argument("--customer-json", help="JSON object for one customer")
    predict_parser.add_argument(
        "--output",
        help="Optional CSV output path. Results are printed as JSON when omitted.",
    )

    validate_parser = subparsers.add_parser("validate-data", help="Validate configured raw data")
    validate_parser.add_argument("--config", default="config.yaml", help="Path to config file")

    args = parser.parse_args()

    if args.command == "train":
        report = train_models(args.config)
        print(json.dumps(report, indent=2))
    elif args.command == "predict":
        if args.batch_file:
            customer_data = pd.read_csv(args.batch_file).to_dict("records")
        else:
            customer_data = json.loads(args.customer_json)

        predictions = make_predictions(customer_data, args.config)
        if args.output:
            pd.DataFrame(predictions).to_csv(args.output, index=False)
            logger.info("Predictions saved to %s", args.output)
        else:
            print(json.dumps(predictions, indent=2))
    elif args.command == "validate-data":
        processor = DataProcessor(args.config)
        raw_path = processor.config["data"]["raw_data_path"]
        df = processor.load_data(raw_path)
        df = processor.clean_data(df)
        df = processor.calculate_derived_features(df)
        issues = processor.validate_data(df)
        print(json.dumps(issues, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
