import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.metrics import roc_curve, average_precision_score, f1_score, log_loss
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.numeric_features = []
        self.categorical_features = []
        self.target = None
        self.models = {}
        self.pipeline = None
        self.feature_importances = None
        
    def prepare_data(self, 
                    data: pd.DataFrame,
                    target_column: str,
                    numeric_features: List[str],
                    categorical_features: List[str]) -> None:
        """Previous implementation remains the same"""
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target = target_column
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.drop(target_column, axis=1),
            data[target_column],
            test_size=0.2,
            random_state=self.random_state
        )

    def perform_eda(self) -> Dict[str, Any]:
        """Enhanced EDA with additional analyses"""
        eda_results = {
            'missing_values': self.X_train.isnull().sum(),
            'numeric_stats': self.X_train[self.numeric_features].describe(),
            'categorical_stats': {
                feature: self.X_train[feature].value_counts() 
                for feature in self.categorical_features
            },
            'correlations': self.X_train[self.numeric_features].corr()
        }
        
        # Target distribution analysis
        eda_results['target_distribution'] = self.y_train.value_counts()
        eda_results['imbalance_ratio'] = (
            self.y_train.value_counts()[0] / self.y_train.value_counts()[1]
        )
        
        # Additional numeric feature analysis
        eda_results['numeric_skewness'] = self.X_train[self.numeric_features].skew()
        eda_results['numeric_kurtosis'] = self.X_train[self.numeric_features].kurtosis()
        
        return eda_results

    def plot_eda_results(self) -> None:
        """Generate EDA visualizations"""
        # Set up the matplotlib figure
        plt.style.use('seaborn')
        
        # Numeric features distribution
        fig, axes = plt.subplots(len(self.numeric_features), 2, figsize=(15, 5*len(self.numeric_features)))
        for idx, feature in enumerate(self.numeric_features):
            # Histogram
            sns.histplot(data=self.X_train, x=feature, ax=axes[idx, 0])
            axes[idx, 0].set_title(f'{feature} Distribution')
            
            # Box plot by target
            sns.boxplot(data=pd.concat([self.X_train, self.y_train], axis=1), 
                       x=self.y_train.name, y=feature, ax=axes[idx, 1])
            axes[idx, 1].set_title(f'{feature} by {self.y_train.name}')
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.X_train[self.numeric_features].corr(), annot=True, cmap='coolwarm')
        plt.title('Feature Correlations')
        plt.show()
        
        # Target distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x=self.y_train)
        plt.title('Target Distribution')
        plt.show()

    def build_ensemble_model(self) -> None:
        """Enhanced model building with additional algorithms"""
        preprocessor = self.create_preprocessing_pipeline()
        
        # Define base models
        rf = RandomForestClassifier(random_state=self.random_state)
        gb = GradientBoostingClassifier(random_state=self.random_state)
        ada = AdaBoostClassifier(random_state=self.random_state)
        lr = LogisticRegression(random_state=self.random_state)
        svc = SVC(probability=True, random_state=self.random_state)
        
        # Create pipelines for each model
        models_config = {
            'random_forest': (rf, {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, 30, None],
                'classifier__min_samples_split': [2, 5, 10]
            }),
            'gradient_boosting': (gb, {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.01, 0.1, 0.3],
                'classifier__max_depth': [3, 4, 5]
            }),
            'adaboost': (ada, {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 1.0]
            }),
            'logistic_regression': (lr, {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }),
            'svc': (svc, {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['rbf', 'linear'],
                'classifier__gamma': ['scale', 'auto']
            })
        }
        
        # Create and store model pipelines
        for name, (model, param_grid) in models_config.items():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            self.models[name] = RandomizedSearchCV(
                pipeline, param_grid, n_iter=10,
                cv=5, random_state=self.random_state, n_jobs=-1,
                scoring='roc_auc'
            )

    def evaluate_models(self) -> Dict[str, Any]:
        """Enhanced evaluation with additional metrics"""
        evaluation_results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate various metrics
            evaluation_results[name] = {
                'classification_report': classification_report(
                    self.y_test, y_pred, output_dict=True
                ),
                'roc_auc': roc_auc_score(self.y_test, y_prob),
                'average_precision': average_precision_score(self.y_test, y_prob),
                'log_loss': log_loss(self.y_test, y_prob),
                'f1_score': f1_score(self.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
                'cross_val_scores': cross_val_score(
                    model.best_estimator_, self.X_train, self.y_train, 
                    cv=5, scoring='roc_auc'
                ).tolist()
            }
            
        return evaluation_results

    def plot_model_performance(self, evaluation_results: Dict[str, Any]) -> None:
        """Plot model performance comparisons"""
        # Set up the style
        plt.style.use('seaborn')
        
        # ROC curves
        plt.figure(figsize=(10, 6))
        for name, model in self.models.items():
            y_prob = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {evaluation_results[name]["roc_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.show()
        
        # Precision-Recall curves
        plt.figure(figsize=(10, 6))
        for name, model in self.models.items():
            y_prob = model.predict_proba(self.X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
            plt.plot(recall, precision, 
                    label=f'{name} (AP = {evaluation_results[name]["average_precision"]:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.show()
        
        # Model comparison bar plot
        metrics = ['roc_auc', 'f1_score', 'average_precision']
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(self.models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[model][metric] for model in self.models]
            plt.bar(x + i*width, values, width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width, self.models.keys(), rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self) -> None:
        """Plot feature importance comparison"""
        plt.figure(figsize=(12, 6))
        feature_importance = self.get_feature_importance()
        
        # Plot feature importance for each model
        for name in self.models.keys():
            plt.figure(figsize=(10, 6))
            importance_scores = feature_importance[name].sort_values(ascending=True)
            importance_scores.plot(kind='barh')
            plt.title(f'Feature Importance - {name}')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()

def main():
    
    dataset_path = Path("/Users/tom/.cache/kagglehub/datasets/vermakeshav/churn-datacsv/versions/1/churn.csv")

    # Example usage
    data = pd.read_csv(dataset_path)
    
    predictor = ChurnPredictor(random_state=42)
    
    numeric_features = ['tenure', 'monthly_charges', 'total_charges']
    categorical_features = ['gender', 'internet_service', 'contract_type']
    
    # Prepare data and perform EDA
    predictor.prepare_data(
        data,
        target_column='churn',
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )
    
    # Perform and visualize EDA
    eda_results = predictor.perform_eda()
    predictor.plot_eda_results()
    
    # Build and train models
    predictor.build_ensemble_model()
    predictor.train_models()
    
    # Evaluate and visualize results
    evaluation_results = predictor.evaluate_models()
    predictor.plot_model_performance(evaluation_results)
    predictor.plot_feature_importance()
    
    # Print detailed evaluation results
    print("\nModel Evaluation Results:")
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name}:")
        print(f"ROC AUC: {results['roc_auc']:.3f}")
        print(f"Average Precision: {results['average_precision']:.3f}")
        print(f"F1 Score: {results['f1_score']:.3f}")
        print(f"Log Loss: {results['log_loss']:.3f}")
        print(f"Cross-validation ROC AUC scores: {np.mean(results['cross_val_scores']):.3f} "
              f"(±{np.std(results['cross_val_scores']):.3f})")

if __name__ == "__main__":
    main()