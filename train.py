import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import yaml
import joblib
from datetime import datetime
import os

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from data_processing import DataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, tuning, and evaluation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ModelTrainer with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models_dir = self.config['models']['output_dir']
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.data_processor = DataProcessor(config_path)
        self.best_model = None
        self.best_score = 0
        
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                                  pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def create_model_pipeline(self, model_name: str, preprocessor, use_smote: bool = True):
        """
        Create a complete pipeline with preprocessing and model.
        
        Args:
            model_name: Name of the model ('logistic', 'random_forest', 'gradient_boosting')
            preprocessor: Preprocessing pipeline
            use_smote: Whether to use SMOTE for handling imbalanced data
            
        Returns:
            Complete pipeline
        """
        # Select model based on name
        if model_name == 'logistic':
            model = LogisticRegression(**self.config['models']['logistic_regression'])
        elif model_name == 'random_forest':
            model = RandomForestClassifier(**self.config['models']['random_forest'])
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(**self.config['models']['gradient_boosting'])
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Create pipeline with or without SMOTE
        if use_smote:
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=self.config['data']['random_state'])),
                ('classifier', model)
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
        
        logger.info(f"Created pipeline for {model_name} (SMOTE: {use_smote})")
        return pipeline
    
    def get_param_distributions(self, model_name: str) -> Dict[str, List]:
        """
        Get hyperparameter distributions for RandomizedSearchCV.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of parameter distributions
        """
        if model_name == 'logistic':
            return {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        elif model_name == 'random_forest':
            return {
                'classifier__n_estimators': [50, 100, 200, 300],
                'classifier__max_depth': [None, 10, 20, 30, 40],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['sqrt', 'log2', None]
            }
        elif model_name == 'gradient_boosting':
            return {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7, 10],
                'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
                'classifier__min_samples_split': [2, 5, 10]
            }
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def tune_hyperparameters(self, pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                           model_name: str) -> Pipeline:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            pipeline: Model pipeline
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            
        Returns:
            Best tuned pipeline
        """
        param_distributions = self.get_param_distributions(model_name)
        
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=self.config['training']['n_iter_search'],
            cv=self.config['training']['cv_folds'],
            scoring=self.config['training']['scoring_metric'],
            n_jobs=-1,
            random_state=self.config['data']['random_state'],
            verbose=1
        )
        
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        random_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def evaluate_model(self, pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str) -> Dict[str, Any]:
        """
        Evaluate model performance on test set.
        
        Args:
            pipeline: Trained model pipeline
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Focus on recall for churn class (class 1)
        recall_churn = recall_score(y_test, y_pred, pos_label=1)
        f1_churn = f1_score(y_test, y_pred, pos_label=1)
        
        # Calculate ROC and PR curves
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        results = {
            'model_name': model_name,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'recall_churn': recall_churn,
            'f1_churn': f1_churn,
            'roc_curve': {'fpr': fpr, 'tpr': tpr},
            'pr_curve': {'precision': precision, 'recall': recall},
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba
        }
        
        logger.info(f"\n{model_name} Performance:")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"Recall (Churn): {recall_churn:.4f}")
        logger.info(f"F1 Score (Churn): {f1_churn:.4f}")
        logger.info(f"Accuracy: {classification_rep['accuracy']:.4f}")
        
        return results
    
    def cross_validate_model(self, pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                           model_name: str) -> Dict[str, float]:
        """
        Perform cross-validation to get robust performance estimates.
        
        Args:
            pipeline: Model pipeline
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            
        Returns:
            Dictionary of cross-validation scores
        """
        cv_folds = self.config['training']['cv_folds']
        
        # Multiple scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_scores = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds,
                                   scoring=metric, n_jobs=-1)
            cv_scores[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            
        logger.info(f"\n{model_name} Cross-Validation Results:")
        for metric, values in cv_scores.items():
            logger.info(f"{metric}: {values['mean']:.4f} (+/- {values['std']:.4f})")
        
        return cv_scores
    
    def train_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Train and evaluate all models.
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            
        Returns:
            Dictionary of results for all models
        """
        preprocessor = self.data_processor.create_preprocessing_pipeline()
        use_smote = self.config['training']['use_smote']
        
        models = ['logistic', 'random_forest', 'gradient_boosting']
        all_results = {}
        
        for model_name in models:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*50}")
            
            # Create pipeline
            pipeline = self.create_model_pipeline(model_name, preprocessor, use_smote)
            
            # Tune hyperparameters
            tuned_pipeline = self.tune_hyperparameters(pipeline, X_train, y_train, model_name)
            
            # Cross-validation
            cv_scores = self.cross_validate_model(tuned_pipeline, X_train, y_train, model_name)
            
            # Evaluate on test set
            test_results = self.evaluate_model(tuned_pipeline, X_test, y_test, model_name)
            
            # Combine results
            all_results[model_name] = {
                'pipeline': tuned_pipeline,
                'cv_scores': cv_scores,
                'test_results': test_results
            }
            
            # Track best model based on recall for churn class
            if test_results['recall_churn'] > self.best_score:
                self.best_score = test_results['recall_churn']
                self.best_model = model_name
                
                # Save best model
                self.save_model(tuned_pipeline, f"best_model_{model_name}.pkl")
        
        logger.info(f"\nBest model: {self.best_model} with recall={self.best_score:.4f}")
        return all_results
    
    def save_model(self, pipeline, filename: str) -> str:
        """
        Save trained model to disk.
        
        Args:
            pipeline: Trained model pipeline
            filename: Name of the file
            
        Returns:
            Full path to saved model
        """
        filepath = os.path.join(self.models_dir, filename)
        joblib.dump(pipeline, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def save_training_report(self, results: Dict[str, Dict], filename: str = "training_report.txt"):
        """
        Save detailed training report.
        
        Args:
            results: Dictionary of all model results
            filename: Name of the report file
        """
        filepath = os.path.join(self.models_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("Customer Churn Model Training Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for model_name, model_results in results.items():
                f.write(f"\n{model_name.upper()}\n")
                f.write("-"*40 + "\n")
                
                # Cross-validation scores
                f.write("\nCross-Validation Scores:\n")
                for metric, scores in model_results['cv_scores'].items():
                    f.write(f"  {metric}: {scores['mean']:.4f} (+/- {scores['std']:.4f})\n")
                
                # Test set performance
                test_res = model_results['test_results']
                f.write("\nTest Set Performance:\n")
                f.write(f"  ROC AUC: {test_res['roc_auc']:.4f}\n")
                f.write(f"  Recall (Churn): {test_res['recall_churn']:.4f}\n")
                f.write(f"  F1 Score (Churn): {test_res['f1_churn']:.4f}\n")
                f.write(f"  Accuracy: {test_res['classification_report']['accuracy']:.4f}\n")
                
                # Confusion Matrix
                f.write("\nConfusion Matrix:\n")
                f.write(f"{test_res['confusion_matrix']}\n")
                
                # Classification Report
                f.write("\nDetailed Classification Report:\n")
                for class_label, metrics in test_res['classification_report'].items():
                    if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                        f.write(f"  Class {class_label}:\n")
                        if isinstance(metrics, dict):
                            for metric, value in metrics.items():
                                f.write(f"    {metric}: {value:.4f}\n")
                
            f.write(f"\n\nBest Model: {self.best_model} (Recall: {self.best_score:.4f})\n")
        
        logger.info(f"Training report saved to {filepath}")
        
    def get_feature_importance(self, pipeline, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.
        
        Args:
            pipeline: Trained pipeline
            feature_names: List of feature names after preprocessing
            
        Returns:
            DataFrame of feature importances
        """
        # Get the classifier from the pipeline
        if hasattr(pipeline, 'named_steps'):
            classifier = pipeline.named_steps['classifier']
        else:
            # For imblearn pipeline
            classifier = pipeline.steps[-1][1]
        
        # Check if the model has feature_importances_
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            
            # Create DataFrame
            importance_df = self.data_processor.get_feature_importance_df(
                feature_names, importances
            )
            
            return importance_df
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()