import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
import logging
import yaml
import joblib
import os

from src.data_processing import DataProcessor

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
            model_path: Path to the model file. If None, loads the xgboost model.
            
        Returns:
            Boolean indicating success
        """
        try:
            if model_path is None:
                # Use the xgboost model by default
                model_path = os.path.join(self.models_dir, 'xgboost_model.pkl')
                self.model_name = 'xgboost'
                
                if not os.path.exists(model_path):
                    logger.error(f"XGBoost model not found at {model_path}")
                    return False
            
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model type: {type(self.model)}")
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
    
    def prepare_batch_customers(self, customers_data: Union[List[Dict], pd.DataFrame]) -> tuple[pd.DataFrame, list]:
        """
        Prepare batch customer data for prediction. This is the single source of truth for data prep.
        """
        if isinstance(customers_data, list):
            df = pd.DataFrame(customers_data)
        else:
            df = customers_data.copy()
        
        # Clean data and add engineered features
        df = self.data_processor.clean_data(df)
        df = self.data_processor.calculate_derived_features(df)
        
        # Store and drop customer IDs
        customer_ids = None
        if self.data_processor.customer_id_column in df.columns:
            customer_ids = df[self.data_processor.customer_id_column].tolist()
            df = df.drop(columns=[self.data_processor.customer_id_column])
            
        return df, customer_ids
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single customer by wrapping the batch method.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # This guarantees a single prediction uses the exact same logic as a batch prediction.
        prediction_result = self.predict_batch([customer_data])
        return prediction_result[0]

    def predict_batch(self, customers_data: Union[List[Dict], pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple customers using a fully vectorized approach.
        This method is significantly faster and more robust than a looped approach.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # 1. Prepare data using the single, reliable prep method
        df, customer_ids = self.prepare_batch_customers(customers_data)
        
        # Create a DataFrame to hold the results
        results_df = pd.DataFrame(index=df.index)

        # 2. Make predictions ONCE for the entire batch
        predictions = self.model.predict(df)
        proba_matrix = self.model.predict_proba(df)
        
        # 3. Assign all results using efficient, vectorized operations
        results_df['churn_probability'] = proba_matrix[:, 1]
        results_df['confidence'] = np.max(proba_matrix, axis=1)
        
        if customer_ids is not None:
            results_df['customer_id'] = customer_ids
        else:
            results_df['customer_id'] = [f"customer_{i}" for i in range(len(df))]

        conditions = [
            (results_df['churn_probability'] >= 0.8),
            (results_df['churn_probability'] >= 0.6),
            (results_df['churn_probability'] >= 0.4),
            (results_df['churn_probability'] >= 0.2)
        ]
        choices = ['Very High', 'High', 'Medium', 'Low']
        results_df['risk_level'] = np.select(conditions, choices, default='Very Low')
        
        results_df['churn_prediction'] = pd.Series(predictions, index=df.index).map({0: 'No', 1: 'Yes'})
        
        # Log a summary
        churn_count = (results_df['churn_prediction'] == 'Yes').sum()
        logger.info(f"Batch prediction complete: {len(df)} customers, "
                    f"{churn_count} predicted to churn ({churn_count/len(df)*100:.1f}%)")
        
        # 4. Return as a list of dictionaries to match the original output format
        final_columns = ['customer_id', 'churn_prediction', 'churn_probability', 'risk_level', 'confidence']
        return results_df[final_columns].to_dict(orient='records')

    def _get_risk_level(self, probability: float) -> str:
        """
        Categorize churn risk based on probability. (Kept from original)
        """
        if probability >= 0.8: return 'Very High'
        elif probability >= 0.6: return 'High'
        elif probability >= 0.4: return 'Medium'
        elif probability >= 0.2: return 'Low'
        else: return 'Very Low'

    
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