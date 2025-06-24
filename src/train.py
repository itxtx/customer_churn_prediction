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

# Import the actual data processing classes
from src.data_processing import DataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, tuning, evaluation, and final model training."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ModelTrainer with configuration.

        Args:
            config_path: Path to the configuration file
        """
        # Create a dummy config if not present for standalone execution
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.models_dir = self.config['models']['output_dir']
        os.makedirs(self.models_dir, exist_ok=True)

        self.data_processor = DataProcessor(config_path)
        self.best_model_name = None
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

    def create_model_pipeline(self, model_name: str, preprocessor: ColumnTransformer) -> ImbPipeline:
        """
        Create a complete pipeline with preprocessing, feature engineering, a resampler placeholder, and a classifier.

        Args:
            model_name: Name of the model ('logistic', 'random_forest', etc.)
            preprocessor: Preprocessing pipeline for transformations.

        Returns:
            An ImbPipeline object ready for tuning.
        """
        if model_name == 'logistic':
            classifier = LogisticRegression(random_state=self.config['data']['random_state'], max_iter=1000, n_jobs=-1)
        elif model_name == 'random_forest':
            classifier = RandomForestClassifier(random_state=self.config['data']['random_state'], n_jobs=-1)
        elif model_name == 'gradient_boosting':
            classifier = GradientBoostingClassifier(random_state=self.config['data']['random_state'], n_jobs=-1)
        elif model_name == 'xgboost':
            classifier = XGBClassifier(
                random_state=self.config['data']['random_state'],
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1  
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        pipeline = ImbPipeline([

            ('preprocessor', preprocessor),
            ('resampler', 'passthrough'),  # Placeholder for resampling
            ('classifier', classifier)
        ])

        logger.info(f"Created base pipeline for {model_name}.")
        return pipeline

    def get_param_distributions(self, model_name: str, y_train: Optional[pd.Series] = None) -> List[Dict[str, Any]]:
        """
        Get hyperparameter distributions for a given model, including resampling options.

        Args:
            model_name: The name of the model.
            y_train: The training target series, used for calculating scale_pos_weight.

        Returns:
            A list of parameter grids for hyperparameter search.
        """
        resampling_options = [
            ('passthrough', {}),
            (SMOTE(random_state=self.config['data']['random_state']), {'resampler__k_neighbors': [3, 5]}),
            (RandomUnderSampler(random_state=self.config['data']['random_state'], sampling_strategy=0.5), {}),
            (RandomOverSampler(random_state=self.config['data']['random_state']), {'resampler__sampling_strategy': [0.7, 1.0]})
        ]

        # Define parameter ranges for each classifier
        param_ranges = {
            'logistic': {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear']
            },
            'random_forest': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 3, 5]
            },
            'gradient_boosting': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 4, 5]
            },
            'xgboost': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.8, 0.9, 1.0],
                'classifier__colsample_bytree': [0.8, 0.9, 1.0]
            }
        }
        classifier_param_ranges = param_ranges.get(model_name)
        if not classifier_param_ranges:
             raise ValueError(f"Unknown model name: {model_name}")

        param_grid_list = []
        for resampler_instance, resampler_params in resampling_options:
            config = {
                'resampler': [resampler_instance],
                **classifier_param_ranges
            }
            config.update(resampler_params)
            param_grid_list.append(config)

        return param_grid_list

    def tune_hyperparameters(self, pipeline: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series,
                             model_name: str) -> Pipeline:
        """
        Perform hyperparameter tuning using the strategy specified in the config.

        Args:
            pipeline: Base model pipeline.
            X_train: Training features.
            y_train: Training target.
            model_name: Name of the model.

        Returns:
            Best tuned pipeline from the search.
        """
        tuning_strategy = self.config['training']['tuning_strategy']
        param_distributions = self.get_param_distributions(model_name, y_train)
        cv_strategy = StratifiedKFold(n_splits=self.config['training']['cv_folds'],
                                      shuffle=True, random_state=self.config['data']['random_state'])
        scoring_metric = self.config['training']['scoring_metric']
        n_iter_search = self.config['training']['n_iter_search']

        search_strategies = {
            'random': RandomizedSearchCV,
            'bayes': BayesSearchCV
        }
        search_class = search_strategies.get(tuning_strategy)

        if not search_class:
            raise ValueError(f"Unknown tuning strategy: {tuning_strategy}")

        tuner = search_class(
            pipeline,
            param_distributions if tuning_strategy == 'random' else self._convert_to_bayes_space(param_distributions, model_name),
            n_iter=n_iter_search,
            cv=cv_strategy,
            scoring=scoring_metric,
            refit=True,
            n_jobs=-1,
            random_state=self.config['data']['random_state'],
            verbose=1
        )
        logger.info(f"Starting {tuning_strategy.capitalize()}SearchCV for {model_name} (n_iter={n_iter_search})...")
        tuner.fit(X_train, y_train)

        logger.info(f"Best parameters for {model_name}: {tuner.best_params_}")
        logger.info(f"Best CV score ({scoring_metric}): {tuner.best_score_:.4f}")
        return tuner.best_estimator_

    def _convert_to_bayes_space(self, param_list: List[Dict[str, Any]], model_name: str) -> List[Tuple[Dict[str, Any]]]:
        """Converts a list of param grids to a format suitable for BayesSearchCV."""
        bayes_spaces = []
        for params in param_list:
            space = {}
            for key, values in params.items():
                if isinstance(values[0], str) or not np.isscalar(values[0]):
                    space[key] = Categorical(values)
                elif all(isinstance(v, int) for v in values):
                    space[key] = Integer(min(values), max(values))
                else:
                    space[key] = Real(min(values), max(values))
            bayes_spaces.append(space)
        return bayes_spaces


    def evaluate_model(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                       model_name: str) -> Dict[str, Any]:
        """
        Evaluate model performance on the test set.

        Args:
            pipeline: Trained model pipeline.
            X_test: Test features.
            y_test: Test target.
            model_name: Name of the model.

        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Determine the positive label based on target data type
        if y_test.dtype == 'object' or y_test.dtype == 'string':
            # String labels like 'Yes'/'No'
            pos_label = 'Yes'
        else:
            # Integer labels like 1/0 (encoded)
            pos_label = 1

        f1 = f1_score(y_test, y_pred, pos_label=pos_label)
        recall = recall_score(y_test, y_pred, pos_label=pos_label)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=pos_label)

        results = {
            'model_name': model_name,
            'f1': f1,
            'recall': recall,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()},
        }

        logger.info(f"\n{model_name} Test Performance:")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")

        return results

    def find_and_save_best_pipeline(self, X: pd.DataFrame, y: pd.Series) -> str:
        """
        Trains, tunes, and evaluates all models to find the best one, then saves it.

        Args:
            X: Full features DataFrame.
            y: Full target Series.

        Returns:
            The file path of the saved best pipeline.
        """
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        preprocessor = self.data_processor.create_preprocessing_pipeline()
        models_to_train = self.config['models'].get('models_to_train', ['random_forest', 'xgboost'])
        scoring_metric = self.config['training']['scoring_metric']

        best_pipeline = None
        best_score = -1
        best_model_name = ""

        for model_name in models_to_train:
            logger.info(f"\n{'='*50}\nTraining and Tuning {model_name.upper()} Model\n{'='*50}")

            base_pipeline = self.create_model_pipeline(model_name, preprocessor)
            tuned_pipeline = self.tune_hyperparameters(base_pipeline, X_train, y_train, model_name)
            test_results = self.evaluate_model(tuned_pipeline, X_test, y_test, model_name)

            current_score = test_results.get(scoring_metric)
            if current_score is None:
                raise ValueError(f"Scoring metric '{scoring_metric}' not found in evaluation results.")

            if current_score > best_score:
                best_score = current_score
                best_pipeline = tuned_pipeline
                best_model_name = model_name
                logger.info(f"New best model found: {best_model_name} with {scoring_metric}: {best_score:.4f}")

        if best_pipeline:
            logger.info(f"\nOverall Best Model: {best_model_name.upper()} (Best {scoring_metric}: {best_score:.4f})")
            filepath = self.save_model(best_pipeline, self.config['models']['best_pipeline_name'])
            return filepath
        else:
            logger.error("No models were trained successfully.")
            return ""
        
    def train_final_model(self, X: pd.DataFrame, y: pd.Series, best_pipeline_path: str) -> str:
        """
        Loads the best pipeline, trains it on the full dataset, and saves the final model.

        Args:
            X: The full features DataFrame.
            y: The full target Series.
            best_pipeline_path: Path to the pickled best pipeline file.

        Returns:
            The file path of the saved final model.
        """
        logger.info(f"\n{'='*50}\nStarting Final Model Training\n{'='*50}")

        if not os.path.exists(best_pipeline_path):
            logger.error(f"Best pipeline file not found at '{best_pipeline_path}'.")
            logger.error("Please run `find_and_save_best_pipeline` first to create it.")
            return ""

        logger.info(f"Loading best pipeline from: {best_pipeline_path}")
        final_pipeline = joblib.load(best_pipeline_path)

        logger.info("Training the final model on the entire dataset...")
        final_pipeline.fit(X, y)
        logger.info("Final model training complete.")

        final_model_name = self.config['models'].get('final_model_name', 'final_model.pkl')
        filepath = self.save_model(final_pipeline, final_model_name)
        return filepath


    def save_model(self, pipeline: Pipeline, filename: str) -> str:
        """
        Save trained model to the output directory specified in the config.

        Args:
            pipeline: Trained model pipeline to save.
            filename: Name of the file.

        Returns:
            The full path to the saved model file.
        """
        filepath = os.path.join(self.models_dir, filename)
        joblib.dump(pipeline, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def train_fresh_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, best_params_dict: Dict[str, Dict]) -> Tuple[Dict[str, Dict], str, float]:
        """
        Train fresh models using best hyperparameters found previously.
        
        Args:
            X_train: Training features DataFrame.
            X_test: Test features DataFrame.
            y_train: Training target Series.
            y_test: Test target Series.
            best_params_dict: Dictionary mapping model names to their best hyperparameters.
            
        Returns:
            Tuple of (results_dict, best_model_name, best_score)
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from sklearn.preprocessing import LabelEncoder
        
        all_results = {}
        best_score = -1
        best_model_name = ""
        scoring_metric = self.config['training']['scoring_metric']
        
        logger.info(f"\n{'='*50}\nTraining Fresh Models with Best Hyperparameters\n{'='*50}")
        
        # Create label encoder for target variable
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        for model_name, params in best_params_dict.items():
            logger.info(f"\nTraining {model_name}...")
            
            # Extract classifier parameters (remove pipeline prefixes)
            classifier_params = {}
            resampler = RandomOverSampler(random_state=42)  # Default resampler
            
            for key, value in params.items():
                if key.startswith('classifier__'):
                    # Remove 'classifier__' prefix
                    clean_key = key.replace('classifier__', '')
                    classifier_params[clean_key] = value
                elif key.startswith('resampler__'):
                    # Handle resampler parameters
                    if 'sampling_strategy' in key:
                        resampler = RandomOverSampler(sampling_strategy=value, random_state=42)
                    elif 'k_neighbors' in key:
                        resampler = SMOTE(k_neighbors=value, random_state=42)
            
            # Create fresh pipeline
            if model_name == 'random_forest':
                classifier = RandomForestClassifier(random_state=self.config['data']['random_state'], n_jobs=-1, **classifier_params)
            elif model_name == 'xgboost':
                classifier = XGBClassifier(random_state=self.config['data']['random_state'], **classifier_params)
            elif model_name == 'gradient_boosting':
                classifier = GradientBoostingClassifier(random_state=self.config['data']['random_state'], **classifier_params)
            elif model_name == 'logistic':
                from sklearn.linear_model import LogisticRegression
                classifier = LogisticRegression(random_state=self.config['data']['random_state'], **classifier_params)
            else:
                logger.warning(f"Unknown model name: {model_name}. Skipping.")
                continue
            
            pipeline = ImbPipeline([
                ('preprocessor', self.data_processor.create_preprocessing_pipeline(X_train)),
                ('resampler', resampler),
                ('classifier', classifier)
            ])
            
            # Fit on training data only (using encoded labels)
            pipeline.fit(X_train, y_train_encoded)
            
            # Evaluate on test data (using encoded labels)
            test_results = self.evaluate_model(pipeline, X_test, y_test_encoded, model_name)
            
            # Store both evaluation results and the fitted pipeline
            all_results[model_name] = {
                **test_results,
                'pipeline': pipeline,
                'label_encoder': label_encoder
            }
            
            # Track best model
            current_score = test_results.get(scoring_metric, 0)
            if current_score > best_score:
                best_score = current_score
                best_model_name = model_name
            
            logger.info(f"Finished training {model_name}.")
        
        logger.info(f"\nBest model: {best_model_name} with {scoring_metric}: {best_score:.4f}")
        logger.info("\nAll fresh models trained and evaluated!")
        return all_results, best_model_name, best_score
    
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
    def save_training_report(self, results: Dict[str, Dict], best_model_info: Tuple[str, float]):
        """Saves a detailed training report comparing all models."""
        filename = self.config['models']['report_name']
        filepath = os.path.join(self.models_dir, filename)
        best_model_name, best_score = best_model_info
        scoring_metric = self.config['training']['scoring_metric']

        with open(filepath, 'w') as f:
            f.write("Customer Churn Model Training Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
            f.write(f"Best model determined by '{scoring_metric}' score on the test set.\n")
            f.write("="*80 + "\n\n")

            for model_name, model_results in results.items():
                f.write(f"\n{model_name.upper()}\n" + "-"*40 + "\n")
                test_res = model_results['test_results']
                f.write("Test Set Performance:\n")
                f.write(f"  ROC AUC: {test_res['roc_auc']:.4f}\n")
                f.write(f"  Recall (Churn): {test_res['recall_churn']:.4f}\n")
                f.write(f"  F1 Score (Churn): {test_res['f1_churn']:.4f}\n")
                f.write(f"  Accuracy: {test_res['accuracy']:.4f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(f"{np.array(test_res['confusion_matrix'])}\n")
            
            f.write("\n\n" + "="*80 + "\n")
            f.write(f"OVERALL BEST MODEL: {best_model_name.upper()} (Test {scoring_metric}: {best_score:.4f})\n")
            f.write("="*80 + "\n")

        logger.info(f"Training report saved to {filepath}")