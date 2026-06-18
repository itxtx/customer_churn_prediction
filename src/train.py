import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real
except ImportError:  # pragma: no cover - only used when optional bayes mode is selected.
    BayesSearchCV = None
    Categorical = Integer = Real = None

from src.data_processing import DataProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train, compare, and persist churn prediction pipelines."""

    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.config_path = config_path
        self.models_dir = self.config["models"]["output_dir"]
        os.makedirs(self.models_dir, exist_ok=True)
        self.data_processor = DataProcessor(config_path)

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load, clean, engineer, and split features from configured raw data."""
        raw_data_path = self.config["data"]["raw_data_path"]
        df = self.data_processor.load_data(raw_data_path)
        df = self.data_processor.clean_data(df)
        df = self.data_processor.calculate_derived_features(df)
        return self.data_processor.prepare_features(df)

    def encode_target(self, y: pd.Series) -> pd.Series:
        """Encode churn labels as 0/1 while keeping numeric labels unchanged."""
        if pd.api.types.is_numeric_dtype(y):
            return y.astype(int)

        mapping = {"No": 0, "Yes": 1}
        encoded = y.map(mapping)
        if encoded.isna().any():
            bad_values = sorted(y[encoded.isna()].dropna().astype(str).unique())
            raise ValueError(f"Target values must be Yes/No or 0/1. Invalid values: {bad_values}")
        return encoded.astype(int)

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        test_size = self.config["data"]["test_size"]
        random_state = self.config["data"]["random_state"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        logger.info("Data split - Train: %s, Test: %s", X_train.shape, X_test.shape)
        logger.info("Class distribution - Train: %s", y_train.value_counts().to_dict())
        logger.info("Class distribution - Test: %s", y_test.value_counts().to_dict())
        return X_train, X_test, y_train, y_test

    def create_model_pipeline(
        self, model_name: str, preprocessor: ColumnTransformer
    ) -> ImbPipeline:
        random_state = self.config["data"]["random_state"]

        if model_name == "logistic":
            classifier = LogisticRegression(random_state=random_state, max_iter=1000)
        elif model_name == "random_forest":
            classifier = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        elif model_name == "gradient_boosting":
            classifier = GradientBoostingClassifier(random_state=random_state)
        elif model_name == "xgboost":
            classifier = XGBClassifier(
                random_state=random_state,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return ImbPipeline(
            [
                ("preprocessor", preprocessor),
                ("resampler", "passthrough"),
                ("classifier", classifier),
            ]
        )

    def get_param_distributions(
        self, model_name: str, y_train: Optional[pd.Series] = None
    ) -> List[Dict[str, Any]]:
        random_state = self.config["data"]["random_state"]
        min_class_count = int(y_train.value_counts().min()) if y_train is not None else 6

        resampling_options: List[Tuple[Any, Dict[str, Any]]] = [
            ("passthrough", {}),
            (RandomUnderSampler(random_state=random_state, sampling_strategy=0.7), {}),
            (
                RandomOverSampler(random_state=random_state),
                {"resampler__sampling_strategy": [0.7, 1.0]},
            ),
        ]
        if min_class_count > 3:
            resampling_options.append(
                (
                    SMOTE(random_state=random_state),
                    {"resampler__k_neighbors": [min(3, min_class_count - 1)]},
                )
            )

        param_ranges = {
            "logistic": {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs"],
            },
            "random_forest": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 3],
            },
            "gradient_boosting": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 4],
            },
            "xgboost": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__max_depth": [3, 5],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0],
            },
        }

        if model_name not in param_ranges:
            raise ValueError(f"Unknown model name: {model_name}")

        grids = []
        for resampler, resampler_params in resampling_options:
            grid = {"resampler": [resampler], **param_ranges[model_name]}
            grid.update(resampler_params)
            grids.append(grid)
        return grids

    def tune_hyperparameters(
        self,
        pipeline: ImbPipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
    ) -> ImbPipeline:
        n_iter_search = int(self.config["training"].get("n_iter_search", 20))
        if n_iter_search <= 0:
            logger.info("Skipping hyperparameter search for %s; fitting base pipeline.", model_name)
            pipeline.fit(X_train, y_train)
            return pipeline

        tuning_strategy = self.config["training"]["tuning_strategy"]
        cv_strategy = StratifiedKFold(
            n_splits=self.config["training"]["cv_folds"],
            shuffle=True,
            random_state=self.config["data"]["random_state"],
        )
        scoring_metric = self.config["training"]["scoring_metric"]
        param_distributions = self.get_param_distributions(model_name, y_train)

        if tuning_strategy == "random":
            tuner = RandomizedSearchCV(
                pipeline,
                param_distributions,
                n_iter=n_iter_search,
                cv=cv_strategy,
                scoring=scoring_metric,
                refit=True,
                n_jobs=-1,
                random_state=self.config["data"]["random_state"],
                verbose=1,
            )
        elif tuning_strategy == "bayes":
            if BayesSearchCV is None:
                raise ImportError("scikit-optimize is required for tuning_strategy='bayes'.")
            tuner = BayesSearchCV(
                pipeline,
                self._convert_to_bayes_space(param_distributions),
                n_iter=n_iter_search,
                cv=cv_strategy,
                scoring=scoring_metric,
                refit=True,
                n_jobs=-1,
                random_state=self.config["data"]["random_state"],
                verbose=1,
            )
        else:
            raise ValueError("training.tuning_strategy must be 'random' or 'bayes'.")

        logger.info("Starting %s search for %s.", tuning_strategy, model_name)
        tuner.fit(X_train, y_train)
        logger.info("Best parameters for %s: %s", model_name, tuner.best_params_)
        logger.info("Best CV %s: %.4f", scoring_metric, tuner.best_score_)
        return tuner.best_estimator_

    def _convert_to_bayes_space(self, param_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        bayes_spaces = []
        for params in param_list:
            space = {}
            for key, values in params.items():
                if isinstance(values[0], str) or not np.isscalar(values[0]):
                    space[key] = Categorical(values)
                elif all(isinstance(v, int) and not isinstance(v, bool) for v in values):
                    space[key] = Integer(min(values), max(values))
                else:
                    space[key] = Real(min(values), max(values))
            bayes_spaces.append(space)
        return bayes_spaces

    def evaluate_model(
        self, pipeline: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, Any]:
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        results = {
            "model_name": model_name,
            "f1": float(f1_score(y_test, y_pred, pos_label=1)),
            "recall": float(recall_score(y_test, y_pred, pos_label=1)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "classification_report": classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["No", "Yes"],
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        }

        logger.info(
            "%s Test Performance - F1: %.4f, Recall: %.4f, ROC AUC: %.4f",
            model_name,
            results["f1"],
            results["recall"],
            results["roc_auc"],
        )
        return results

    def train_and_select_best_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        y_encoded = self.encode_target(y)
        X_train, X_test, y_train, y_test = self.split_data(X, y_encoded)
        preprocessor = self.data_processor.create_preprocessing_pipeline(X_train=X_train)
        models_to_train = self.config["training"]["models_to_train"]
        scoring_metric = self.config["training"]["scoring_metric"]

        best_pipeline = None
        best_model_name = ""
        best_score = -1.0
        model_results: Dict[str, Dict[str, Any]] = {}

        for model_name in models_to_train:
            logger.info("Training and tuning %s.", model_name)
            pipeline = self.create_model_pipeline(model_name, preprocessor)
            tuned_pipeline = self.tune_hyperparameters(pipeline, X_train, y_train, model_name)
            test_results = self.evaluate_model(tuned_pipeline, X_test, y_test, model_name)
            model_results[model_name] = test_results

            current_score = test_results[scoring_metric]
            if current_score > best_score:
                best_score = current_score
                best_pipeline = tuned_pipeline
                best_model_name = model_name

        if best_pipeline is None:
            raise RuntimeError("No models were trained successfully.")

        logger.info("Refitting best model %s on the full dataset.", best_model_name)
        best_pipeline.fit(X, y_encoded)
        final_model_path = self.config["models"]["final_model_path"]
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        joblib.dump(best_pipeline, final_model_path)
        logger.info("Final model saved to %s", final_model_path)

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "best_model": best_model_name,
            "selection_metric": scoring_metric,
            "best_score": float(best_score),
            "final_model_path": final_model_path,
            "models": model_results,
        }
        self.save_training_report(report)
        return report

    def save_training_report(self, report: Dict[str, Any]) -> str:
        report_path = self.config["models"]["training_report_path"]
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as file:
            json.dump(report, file, indent=2)
        logger.info("Training report saved to %s", report_path)
        return report_path
