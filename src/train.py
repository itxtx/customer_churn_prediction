import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import yaml
import joblib
from datetime import datetime
import os

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, recall_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline # Use ImbPipeline for pipelines with resampling

# Import XGBoost and skopt for Bayesian Optimization
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from src.data_processing import DataProcessor
from src.data_processing import FeatureEngineer

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
        
    def _calculate_scale_pos_weight(self, y_train: pd.Series) -> float:
        """
        Calculates scale_pos_weight for XGBoost to handle class imbalance.
        """
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = 1.0 if pos_count == 0 else neg_count / pos_count
        logger.info(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")
        return scale_pos_weight

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
    
    def create_model_pipeline(self, model_name: str, preprocessor: ColumnTransformer) -> ImbPipeline:
        """
        Create a complete pipeline with preprocessing and model classifier,
        and optionally a resampler. This structure is flexible for tuning.
        Args:
            model_name: Name of the model ('logistic', 'random_forest', 'gradient_boosting', 'xgboost')
            preprocessor: Preprocessing pipeline
        Returns:
            An ImbPipeline object with preprocessor, resampler placeholder, and classifier.
        """
        if model_name == 'logistic':
            classifier = LogisticRegression(random_state=self.config['data']['random_state'], max_iter=1000)
        elif model_name == 'random_forest':
            classifier = RandomForestClassifier(random_state=self.config['data']['random_state'], n_jobs=-1)
        elif model_name == 'gradient_boosting':
            classifier = GradientBoostingClassifier(random_state=self.config['data']['random_state'])
        elif model_name == 'xgboost':
            classifier = XGBClassifier(
                random_state=self.config['data']['random_state'],
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False # Suppress warning in newer XGBoost versions
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        pipeline = ImbPipeline([
            ('engineer', FeatureEngineer()), # <-- Feature engineering is now the first step
            ('preprocessor', preprocessor),
            ('resampler', None),  # Placeholder for resampling
            ('classifier', classifier)
        ])
        
        logger.info(f"Created base pipeline for {model_name}.")
        return pipeline

    def get_param_distributions(self, model_name: str, y_train: Optional[pd.Series] = None) -> List[Dict[str, Any]]:
        """
        Get hyperparameter distributions and resampling options for tuning.
        """
        resampling_options = [
            (None, {}),  # No resampling
            (SMOTE(random_state=self.config['data']['random_state']), {'resampler__k_neighbors': [3, 5, 7]}),
            # FIX: Ensure sampling_strategy is a float < 1.0 for RandomUnderSampler
            (RandomUnderSampler(random_state=self.config['data']['random_state'], sampling_strategy=0.5), {}),
            (RandomOverSampler(random_state=self.config['data']['random_state']), {'resampler__sampling_strategy': [0.7, 1.0]}),
        ]
        
        classifier_param_ranges = {}
        if model_name == 'logistic':
            classifier_param_ranges = {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        elif model_name == 'random_forest':
            classifier_param_ranges = {
                'classifier__n_estimators': [100, 200, 300, 400],
                'classifier__max_depth': [5, 10, 15, 20, None],
                'classifier__max_features': ['sqrt', 'log2', 0.3],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 3, 5],
                'classifier__bootstrap': [True, False]
            }
        elif model_name == 'gradient_boosting':
            classifier_param_ranges = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 4, 5, 6],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 3, 5],
                'classifier__subsample': [0.7, 0.8, 0.9, 1.0]
            }
        elif model_name == 'xgboost':
            scale_pos_weight_val = self._calculate_scale_pos_weight(y_train) if y_train is not None else 1
            classifier_param_ranges = {
                'classifier__n_estimators': [100, 200, 300, 400, 500],
                'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7, 9],
                'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
                'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'classifier__gamma': [0, 0.1, 0.5, 1, 2],
                'classifier__reg_alpha': [0, 0.01, 0.1, 1],
                'classifier__reg_lambda': [0.1, 1, 5, 10]
            }
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        param_grid_list = []
        for resampler_instance, resampler_params in resampling_options:
            config = {
                'resampler': [resampler_instance],
                **classifier_param_ranges
            }
            config.update(resampler_params)

            # Special handling for class_weight and scale_pos_weight
            if model_name in ['logistic', 'random_forest']:
                if resampler_instance is None:
                    config['classifier__class_weight'] = [None, 'balanced']
                    if model_name == 'random_forest':
                        config['classifier__class_weight'].append('balanced_subsample')
                else:
                    config['classifier__class_weight'] = [None] # Don't use internal weighting with external resampling
            elif model_name == 'xgboost':
                if resampler_instance is None:
                    config['classifier__scale_pos_weight'] = [1, scale_pos_weight_val]
                else:
                    config['classifier__scale_pos_weight'] = [1] # Set to 1 when external resampling is used

            param_grid_list.append(config)
            
        return param_grid_list

    def tune_hyperparameters(self, pipeline: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series,
                        model_name: str, tuning_strategy: str = 'random') -> Pipeline:
        """
        Perform hyperparameter tuning using RandomizedSearchCV, BayesSearchCV, or a hybrid approach.
        
        Args:
            pipeline: Base model pipeline (with preprocessor and resampler placeholder)
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            tuning_strategy: 'random' for RandomizedSearchCV, 'bayes' for BayesSearchCV, 
                            'hybrid' for Random->Bayes sequential search
            
        Returns:
            Best tuned pipeline
        """
        param_distributions = self.get_param_distributions(model_name, y_train)
        random_state = self.config['data']['random_state']
        cv_strategy = StratifiedKFold(n_splits=self.config['training']['cv_folds'],
                                    shuffle=True, random_state=random_state)
        scoring_metric = self.config['training']['scoring_metric']
        n_iter_search = self.config['training']['n_iter_search']
        
        if tuning_strategy == 'hybrid':
            logger.info(f"Starting Hybrid tuning for {model_name}...")
            
            # Step 1: Broad exploration with RandomizedSearchCV
            logger.info("Step 1: Broad exploration with RandomizedSearchCV...")
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_distributions,
                n_iter=n_iter_search,
                cv=cv_strategy,
                scoring=scoring_metric,
                refit=True,
                n_jobs=-1,
                random_state=random_state,
                verbose=1
            )
            random_search.fit(X_train, y_train)
            
            best_random_params = random_search.best_params_
            best_random_score = random_search.best_score_
            logger.info(f"Random search best params: {best_random_params}")
            logger.info(f"Random search best score: {best_random_score:.4f}")
            
            # Step 2: Create refined search space around best parameters
            logger.info("Step 2: Creating refined search space for BayesSearchCV...")
            refined_search_space = self._create_refined_search_space(best_random_params, model_name)
            
            # Step 3: Focused exploitation with BayesSearchCV
            logger.info("Step 3: Focused exploitation with BayesSearchCV...")
            bayes_search = BayesSearchCV(
                estimator=pipeline,
                search_spaces=refined_search_space,
                n_iter=max(n_iter_search // 2, 10),  # Use half iterations for focused search
                cv=cv_strategy,
                scoring=scoring_metric,
                n_jobs=-1,
                refit=True,
                random_state=random_state,
                verbose=1
            )
            bayes_search.fit(X_train, y_train)
            
            logger.info(f"Bayes search best params: {bayes_search.best_params_}")
            logger.info(f"Bayes search best score: {bayes_search.best_score_:.4f}")
            logger.info(f"Improvement over random search: {bayes_search.best_score_ - best_random_score:.4f}")
            
            return bayes_search.best_estimator_
        
        elif tuning_strategy == 'random':
            tuner = RandomizedSearchCV(
                pipeline,
                param_distributions=param_distributions,
                n_iter=n_iter_search,
                cv=cv_strategy,
                scoring=scoring_metric,
                refit=scoring_metric,
                n_jobs=-1,
                random_state=random_state,
                verbose=1
            )
            logger.info(f"Starting RandomizedSearchCV for {model_name} (n_iter={n_iter_search})...")
        
        elif tuning_strategy == 'bayes':
            # Convert param distributions to Bayes search space
            bayes_search_space = self._convert_to_bayes_space(param_distributions[0], model_name)
            
            tuner = BayesSearchCV(
                estimator=pipeline,
                search_spaces=bayes_search_space,
                n_iter=n_iter_search,
                cv=cv_strategy,
                scoring=scoring_metric,
                n_jobs=-1,
                refit=True,
                random_state=random_state,
                verbose=1
            )
            logger.info(f"Starting BayesSearchCV for {model_name} (n_iter={n_iter_search})...")
        
        else:
            raise ValueError(f"Unknown tuning strategy: {tuning_strategy}")

        if tuning_strategy != 'hybrid':
            tuner.fit(X_train, y_train)
            logger.info(f"Best parameters for {model_name}: {tuner.best_params_}")
            logger.info(f"Best CV score: {tuner.best_score_:.4f}")
            return tuner.best_estimator_

    def _create_refined_search_space(self, best_params: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Create a refined search space for BayesSearchCV based on RandomizedSearchCV results.
        
        Args:
            best_params: Best parameters from RandomizedSearchCV
            model_name: Name of the model
            
        Returns:
            Refined search space for BayesSearchCV
        """
        refined_space = {}
        
        # Handle resampler - keep the best one found
        if 'resampler' in best_params:
            refined_space['resampler'] = Categorical([best_params['resampler']])
        
        # Define refinement ranges for different parameter types
        for param, value in best_params.items():
            if param == 'resampler':
                continue
                
            # Classifier parameters
            if param.startswith('classifier__'):
                param_name = param.replace('classifier__', '')
                
                # Integer parameters
                if param_name in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                    if value is None:  # Handle None for max_depth
                        refined_space[param] = Categorical([None, 20, 30, 40])
                    else:
                        # Create range around best value (±20%)
                        lower = max(1, int(value * 0.8))
                        upper = max(lower + 1, int(value * 1.2))  # Ensure upper is at least lower + 1
                        refined_space[param] = Integer(lower, upper)
                
                # Float parameters
                elif param_name in ['learning_rate', 'subsample', 'colsample_bytree']:
                    # Create range around best value (±30%)
                    lower = max(0.001, value * 0.7)
                    upper = min(1.0, value * 1.3)
                    if param_name == 'learning_rate':
                        refined_space[param] = Real(lower, upper, prior='log-uniform')
                    else:
                        refined_space[param] = Real(lower, upper, prior='uniform')
                
                # Regularization parameters
                elif param_name in ['C', 'gamma', 'reg_alpha', 'reg_lambda']:
                    # Wider range for regularization parameters
                    if value == 0:
                        refined_space[param] = Real(0, 0.1, prior='uniform')
                    else:
                        lower = value * 0.1
                        upper = value * 10
                        refined_space[param] = Real(lower, upper, prior='log-uniform')
                
                # Categorical parameters
                elif param_name in ['penalty', 'solver', 'max_features', 'bootstrap', 'class_weight']:
                    refined_space[param] = Categorical([value])
                
                # Scale pos weight for XGBoost
                elif param_name == 'scale_pos_weight':
                    if value == 1:
                        refined_space[param] = Categorical([1])
                    else:
                        lower = max(1, value * 0.8)
                        upper = value * 1.2
                        refined_space[param] = Real(lower, upper, prior='uniform')
            
            # Resampler parameters
            elif param.startswith('resampler__'):
                param_name = param.replace('resampler__', '')
                if param_name == 'k_neighbors':
                    # Narrow range around best k
                    lower = max(1, value - 2)
                    upper = value + 2
                    refined_space[param] = Integer(lower, upper)
                elif param_name == 'sampling_strategy':
                    # Small range around best sampling strategy
                    lower = max(0.5, value - 0.1)
                    upper = min(1.0, value + 0.1)
                    refined_space[param] = Real(lower, upper, prior='uniform')
        
        logger.info(f"Refined search space created with {len(refined_space)} parameters")
        return refined_space

    def _convert_to_bayes_space(self, param_dict: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Convert parameter distributions to BayesSearchCV format.
        
        Args:
            param_dict: Parameter distribution dictionary
            model_name: Name of the model
            
        Returns:
            Search space for BayesSearchCV
        """
        bayes_space = {}
        
        for key, values in param_dict.items():
            if not isinstance(values, list):
                values = [values]
            
            if len(values) == 1:
                bayes_space[key] = Categorical(values)
            elif key == 'resampler':
                # Special handling for resampler objects
                bayes_space[key] = Categorical(values)
            elif any(substring in key for substring in ['n_estimators', 'max_depth', 'min_samples', 'k_neighbors']):
                if None in values:
                    bayes_space[key] = Categorical(values)
                else:
                    min_val = min(v for v in values if v is not None)
                    max_val = max(v for v in values if v is not None)
                    if min_val == max_val:
                        max_val = min_val + 1  # Ensure we have a valid range
                    bayes_space[key] = Integer(min_val, max_val)
            elif any(substring in key for substring in ['learning_rate', 'subsample', 'colsample_bytree', 
                                                    'gamma', 'reg_alpha', 'reg_lambda', 'C']):
                bayes_space[key] = Real(min(values), max(values), 
                                    prior='log-uniform' if 'learning_rate' in key or 'C' in key else 'uniform')
            else:
                bayes_space[key] = Categorical(values)
        
        return bayes_space
        
    
    def evaluate_model(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
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
        accuracy = accuracy_score(y_test, y_pred) # Explicitly get accuracy
        
        # Calculate ROC and PR curves
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        results = {
            'model_name': model_name,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(), # Convert to list for JSON compatibility
            'roc_auc': float(roc_auc),
            'recall_churn': float(recall_churn),
            'f1_churn': float(f1_churn),
            'accuracy': float(accuracy),
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}, # Convert arrays to lists
            'pr_curve': {'precision': precision.tolist(), 'recall': recall.tolist()}, # Convert arrays to lists
            'test_predictions': y_pred.tolist(), # Convert to list
            'test_probabilities': y_pred_proba.tolist() # Convert to list
        }
        
        logger.info(f"\n{model_name} Performance:")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"Recall (Churn): {recall_churn:.4f}")
        logger.info(f"F1 Score (Churn): {f1_churn:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        return results
    
    def cross_validate_model(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                           model_name: str) -> Dict[str, Dict]:
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
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'scores': scores.tolist()
            }
            
        logger.info(f"\n{model_name} Cross-Validation Results:")
        for metric, values in cv_scores.items():
            logger.info(f"{metric}: {values['mean']:.4f} (+/- {values['std']:.4f})")
        
        return cv_scores
    

        
        
    def train_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Train and evaluate all models specified in config or hardcoded.
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            
        Returns:
            Dictionary of results for all models
        """
        preprocessor = self.data_processor.create_preprocessing_pipeline()
        
        # Define models to train based on config or default
        models_to_train = self.config['models'].get('models_to_train', 
                                                     ['logistic', 'random_forest', 'gradient_boosting', 'xgboost'])
        tuning_strategy = self.config['training'].get('tuning_strategy', 'random') # 'random' or 'bayes'

        all_results = {}
        
        for model_name in models_to_train:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name.upper()} Model")
            logger.info(f"{'='*50}")
            
            # Create base pipeline (with preprocessor and resampler placeholder)
            base_pipeline = self.create_model_pipeline(model_name, preprocessor)
            
            # Tune hyperparameters using specified strategy
            tuned_pipeline = self.tune_hyperparameters(base_pipeline, X_train, y_train, model_name, tuning_strategy)
            
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
            
            self.save_model(tuned_pipeline, f"best_model_{model_name}.pkl") 
            # Track best model based on recall for churn class
            if test_results['recall_churn'] > self.best_score:
                self.best_score = test_results['recall_churn']
                self.best_model = model_name
                

                
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Complete. Best model: {self.best_model} with recall={self.best_score:.4f}")
        logger.info(f"{'='*50}")
        return all_results
    
    def save_model(self, pipeline: Pipeline, filename: str) -> str:
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
                f.write(f"  Accuracy: {test_res['accuracy']:.4f}\n") # Use directly from test_res
                
                # Confusion Matrix
                f.write("\nConfusion Matrix:\n")
                f.write(f"{np.array(test_res['confusion_matrix'])}\n") # Convert back to numpy array for printing format
                
                # Classification Report
                f.write("\nDetailed Classification Report:\n")
                # classification_report output_dict format has keys like '0', '1', 'accuracy', etc.
                for class_label, metrics in test_res['classification_report'].items():
                    if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                        f.write(f"  Class {class_label}:\n")
                        if isinstance(metrics, dict): # Ensure it's a dict for metrics
                            for metric_name, value in metrics.items():
                                f.write(f"    {metric_name}: {value:.4f}\n")
                    elif class_label in ['macro avg', 'weighted avg']:
                        f.write(f"  {class_label}:\n")
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                f.write(f"    {metric_name}: {value:.4f}\n")
                
            f.write(f"\n\nBest Model: {self.best_model} (Recall: {self.best_score:.4f})\n")
        
        logger.info(f"Training report saved to {filepath}")
        
    def get_feature_importance(self, pipeline: Pipeline, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.
        
        Args:
            pipeline: Trained pipeline
            feature_names: List of feature names after preprocessing
            
        Returns:
            DataFrame of feature importances
        """
        # Get the classifier from the pipeline
        # For ImbPipeline, classifier is at the last step.
        classifier = pipeline.steps[-1][1]
        
        # Check if the model has feature_importances_
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            
            # Create DataFrame directly
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
            
            return importance_df
        elif hasattr(classifier, 'coef_'): # For linear models like LogisticRegression
            # For binary classification, coef_ is usually 2D for multi-class, but 1D for binary
            # If 2D, take the coefficients for the positive class (index 1)
            coefs = classifier.coef_
            if coefs.ndim > 1:
                coefs = coefs[0] # Assuming positive class is the first row for binary coefs
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coefs) # Use absolute value for importance
            })
            
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
            
            return importance_df
        else:
            logger.warning(f"Model {classifier.__class__.__name__} does not have feature_importances_ or coef_ attribute.")
            return pd.DataFrame()