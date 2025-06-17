import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
import logging
import yaml
import joblib
import os

from data_processing import DataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Handles loading models and making predictions on new customer data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ChurnPredictor with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models_dir = self.config['models']['output_dir']
        self.data_processor = DataProcessor(config_path)
        self.model = None
        self.model_name = None
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file. If None, loads the best model.
            
        Returns:
            Boolean indicating success
        """
        try:
            if model_path is None:
                # Find the best model in the models directory
                model_files = [f for f in os.listdir(self.models_dir) 
                             if f.startswith('best_model_') and f.endswith('.pkl')]
                
                if not model_files:
                    logger.error("No best model found in models directory")
                    return False
                
                model_path = os.path.join(self.models_dir, model_files[0])
                self.model_name = model_files[0].replace('best_model_', '').replace('.pkl', '')
            
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def prepare_single_customer(self, customer_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare a single customer's data for prediction.
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            Prepared DataFrame
        """
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Clean data
        df = self.data_processor.clean_data(df)
        
        # Add engineered features
        df = self.data_processor.calculate_derived_features(df)
        
        # Remove customerID if present
        if self.data_processor.customer_id_column in df.columns:
            customer_id = df[self.data_processor.customer_id_column].iloc[0]
            df = df.drop(columns=[self.data_processor.customer_id_column])
        else:
            customer_id = None
            
        return df, customer_id
    
    def prepare_batch_customers(self, customers_data: Union[List[Dict], pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare batch customer data for prediction.
        
        Args:
            customers_data: List of dictionaries or DataFrame with customer features
            
        Returns:
            Prepared DataFrame
        """
        # Convert to DataFrame if needed
        if isinstance(customers_data, list):
            df = pd.DataFrame(customers_data)
        else:
            df = customers_data.copy()
        
        # Clean data
        df = self.data_processor.clean_data(df)
        
        # Add engineered features
        df = self.data_processor.calculate_derived_features(df)
        
        # Store customer IDs if present
        if self.data_processor.customer_id_column in df.columns:
            customer_ids = df[self.data_processor.customer_id_column].tolist()
            df = df.drop(columns=[self.data_processor.customer_id_column])
        else:
            customer_ids = None
            
        return df, customer_ids
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single customer.
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare data
        df, customer_id = self.prepare_single_customer(customer_data)
        
        # Validate data
        validation_issues = self.data_processor.validate_data(df)
        if validation_issues['missing_columns']:
            logger.warning(f"Missing columns: {validation_issues['missing_columns']}")
        
        # Make prediction
        try:
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0, 1]  # Probability of churn
            
            result = {
                'customer_id': customer_id,
                'churn_prediction': 'Yes' if prediction == 1 else 'No',
                'churn_probability': float(probability),
                'risk_level': self._get_risk_level(probability),
                'confidence': float(max(self.model.predict_proba(df)[0]))
            }
            
            logger.info(f"Prediction for customer {customer_id}: {result['churn_prediction']} "
                       f"(probability: {result['churn_probability']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'customer_id': customer_id,
                'error': str(e)
            }
    
    def predict_batch(self, customers_data: Union[List[Dict], pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple customers.
        
        Args:
            customers_data: List of dictionaries or DataFrame with customer features
            
        Returns:
            List of prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare data
        df, customer_ids = self.prepare_batch_customers(customers_data)
        
        # Validate data
        validation_issues = self.data_processor.validate_data(df)
        if validation_issues['missing_columns']:
            logger.warning(f"Missing columns: {validation_issues['missing_columns']}")
            # Only proceed if missing columns are not critical for prediction
            critical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']
            missing_critical = [col for col in critical_columns if col in validation_issues['missing_columns']]
            if missing_critical:
                raise ValueError(f"Missing critical columns for prediction: {missing_critical}")
        
        # Make predictions
        try:
            predictions = self.model.predict(df)
            probabilities = self.model.predict_proba(df)[:, 1]
            
            results = []
            for i in range(len(predictions)):
                result = {
                    'customer_id': customer_ids[i] if customer_ids else f"customer_{i}",
                    'churn_prediction': 'Yes' if predictions[i] == 1 else 'No',
                    'churn_probability': float(probabilities[i]),
                    'risk_level': self._get_risk_level(probabilities[i]),
                    'confidence': float(max(self.model.predict_proba(df)[i]))
                }
                results.append(result)
            
            # Summary statistics
            churn_count = sum(1 for p in predictions if p == 1)
            logger.info(f"Batch prediction complete: {len(predictions)} customers, "
                       f"{churn_count} predicted to churn ({churn_count/len(predictions)*100:.1f}%)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            return [{'error': str(e)}]
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Categorize churn risk based on probability.
        
        Args:
            probability: Churn probability
            
        Returns:
            Risk level category
        """
        if probability >= 0.8:
            return 'Very High'
        elif probability >= 0.6:
            return 'High'
        elif probability >= 0.4:
            return 'Medium'
        elif probability >= 0.2:
            return 'Low'
        else:
            return 'Very Low'
    
    def explain_prediction(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide explanation for a prediction (feature contributions).
        
        Args:
            customer_data: Dictionary with customer features
            
        Returns:
            Dictionary with prediction and explanations
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get prediction first
        prediction_result = self.predict_single(customer_data)
        
        # Prepare data
        df, _ = self.prepare_single_customer(customer_data)
        
        # Get feature names after preprocessing
        preprocessor = self.model.named_steps.get('preprocessor', None)
        if preprocessor is None:
            # For imblearn pipeline
            preprocessor = self.model.steps[0][1]
        
        feature_names = preprocessor.get_feature_names_out()
        
        # Calculate feature contributions (simplified approach)
        # For tree-based models, we could use SHAP values for better explanations
        explanations = {
            'prediction': prediction_result,
            'important_factors': self._get_important_factors(customer_data),
            'recommendations': self._get_retention_recommendations(
                customer_data, 
                prediction_result['churn_probability']
            )
        }
        
        return explanations
    
    def _get_important_factors(self, customer_data: Dict[str, Any]) -> List[str]:
        """
        Identify important factors for the customer's churn risk.
        
        Args:
            customer_data: Customer features
            
        Returns:
            List of important factors
        """
        factors = []
        
        # Check contract type
        if customer_data.get('Contract') == 'Month-to-month':
            factors.append("Month-to-month contract (higher churn risk)")
        
        # Check tenure
        tenure = customer_data.get('tenure', 0)
        if tenure < 6:
            factors.append(f"New customer (tenure: {tenure} months)")
        
        # Check payment method
        if customer_data.get('PaymentMethod') == 'Electronic check':
            factors.append("Electronic check payment (associated with higher churn)")
        
        # Check services
        if customer_data.get('InternetService') != 'No':
            if customer_data.get('OnlineSecurity') == 'No':
                factors.append("No online security service")
            if customer_data.get('TechSupport') == 'No':
                factors.append("No tech support service")
        
        # Check charges
        monthly = customer_data.get('MonthlyCharges', 0)
        if monthly > 70:
            factors.append(f"High monthly charges (${monthly:.2f})")
        
        return factors
    
    def _get_retention_recommendations(self, customer_data: Dict[str, Any], 
                                     churn_probability: float) -> List[str]:
        """
        Generate retention recommendations based on customer profile.
        
        Args:
            customer_data: Customer features
            churn_probability: Predicted churn probability
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if churn_probability < 0.3:
            recommendations.append("Low risk - maintain regular engagement")
            return recommendations
        
        # High risk recommendations
        if customer_data.get('Contract') == 'Month-to-month':
            recommendations.append("Offer incentive to switch to annual contract")
        
        if customer_data.get('tenure', 0) < 12:
            recommendations.append("Implement new customer retention program")
        
        if customer_data.get('OnlineSecurity') == 'No' or customer_data.get('TechSupport') == 'No':
            recommendations.append("Bundle additional services with discount")
        
        if customer_data.get('MonthlyCharges', 0) > 70:
            recommendations.append("Review pricing and offer competitive rate")
        
        if churn_probability > 0.7:
            recommendations.append("Priority intervention - personal outreach recommended")
        
        return recommendations
    
    def save_predictions(self, predictions: List[Dict[str, Any]], 
                        filename: str = "predictions.csv") -> str:
        """
        Save predictions to a CSV file.
        
        Args:
            predictions: List of prediction results
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        df = pd.DataFrame(predictions)
        filepath = os.path.join(self.models_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Predictions saved to {filepath}")
        return filepath