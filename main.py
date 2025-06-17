import argparse
import logging
import yaml
from typing import Dict, Any

from database import Database
from data_processing import DataProcessor
from train import ModelTrainer
from predict import ChurnPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_models(config_path: str = "config.yaml"):
    """
    Train all models using the configuration file.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("Starting model training pipeline...")
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize components
    data_processor = DataProcessor(config_path)
    trainer = ModelTrainer(config_path)
    
    # Load and process data
    logger.info("Loading data...")
    df = data_processor.load_data(config['data']['raw_data_path'])
    
    # Clean data
    logger.info("Cleaning data...")
    df = data_processor.clean_data(df)
    
    # Add engineered features
    logger.info("Engineering features...")
    df = data_processor.calculate_derived_features(df)
    
    # Cap outliers
    numeric_features = config['features']['numeric_features'] + ['MonthlyToTotalRatio']
    df = data_processor.cap_outliers(df, numeric_features)
    
    # Prepare features and target
    X, y = data_processor.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Train all models
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Save training report
    trainer.save_training_report(results)
    
    logger.info("Training complete! Check the models/ directory for saved models and reports.")


def make_predictions(customer_data: Dict[str, Any], config_path: str = "config.yaml"):
    """
    Make predictions for new customer data.
    
    Args:
        customer_data: Dictionary or list of dictionaries with customer features
        config_path: Path to configuration file
    """
    # Initialize predictor
    predictor = ChurnPredictor(config_path)
    
    # Load the best model
    if not predictor.load_model():
        logger.error("Failed to load model. Make sure you've trained models first.")
        return
    
    # Make prediction
    if isinstance(customer_data, dict):
        # Single customer
        result = predictor.predict_single(customer_data)
        explanation = predictor.explain_prediction(customer_data)
        
        logger.info("\nPrediction Result:")
        logger.info(f"Customer ID: {result['customer_id']}")
        logger.info(f"Churn Prediction: {result['churn_prediction']}")
        logger.info(f"Churn Probability: {result['churn_probability']:.3f}")
        logger.info(f"Risk Level: {result['risk_level']}")
        
        logger.info("\nImportant Factors:")
        for factor in explanation['important_factors']:
            logger.info(f"  - {factor}")
        
        logger.info("\nRecommendations:")
        for rec in explanation['recommendations']:
            logger.info(f"  - {rec}")
    else:
        # Batch prediction
        results = predictor.predict_batch(customer_data)
        predictor.save_predictions(results)
        logger.info(f"Batch predictions saved for {len(results)} customers")


def import_data(csv_path: str, config_path: str = "config.yaml"):
    """
    Import customer data from CSV to database.
    
    Args:
        csv_path: Path to CSV file
        config_path: Path to configuration file
    """
    db = Database(config_path)
    db.create_customers_table()
    
    success, failed = db.import_from_csv(csv_path)
    logger.info(f"Import complete: {success} succeeded, {failed} failed")
    
    # Show statistics
    stats = db.get_churn_statistics()
    logger.info(f"\nDatabase Statistics:")
    logger.info(f"Total Customers: {stats['total_customers']}")
    logger.info(f"Churned Customers: {stats['churned_customers']}")
    logger.info(f"Churn Rate: {stats['churn_rate']:.2f}%")
    logger.info(f"Average Tenure: {stats['avg_tenure']:.1f} months")
    logger.info(f"Average Monthly Charges: ${stats['avg_monthly_charges']:.2f}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Customer Churn Prediction System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--config', default='config.yaml', help='Path to config file')
    predict_parser.add_argument('--customer-id', help='Customer ID for single prediction')
    predict_parser.add_argument('--batch-file', help='CSV file for batch predictions')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import data to database')
    import_parser.add_argument('csv_file', help='Path to CSV file')
    import_parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_models(args.config)
    
    elif args.command == 'predict':
        if args.customer_id:
            # Example customer data - in real use, this would come from user input or database
            customer_data = {
                "customerID": args.customer_id,
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
            make_predictions(customer_data, args.config)
        
        elif args.batch_file:
            # Load batch data from CSV
            import pandas as pd
            batch_data = pd.read_csv(args.batch_file).to_dict('records')
            make_predictions(batch_data, args.config)
        
        else:
            logger.error("Please specify --customer-id or --batch-file")
    
    elif args.command == 'import':
        import_data(args.csv_file, args.config)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()