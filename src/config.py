# src/config.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

RANDOM_SEED = 42
CV_SPLITS = 5
N_ITERATIONS = 50
SCORING_METRIC = 'f1'

# --- Define the models and their parameter grids ---

MODELS = {
    'RandomForest': {
        'estimator': RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        'params': {
            'classifier__n_estimators': [100, 200, 300, 400],
            'classifier__max_depth': [5, 10, 15, 20, None],
            'classifier__max_features': ['sqrt', 'log2'],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 3, 5],
            'classifier__bootstrap': [True, False]
        }
    },
    'GradientBoosting': {
        'estimator': GradientBoostingClassifier(random_state=RANDOM_SEED),
        'params': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 4, 5, 6],
            'classifier__subsample': [0.7, 0.8, 0.9, 1.0]
        }
    },
    'XGBoost': {
        'estimator': XGBClassifier(random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'classifier__n_estimators': [100, 200, 300, 400],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 5, 7, 9],
            'classifier__gamma': [0, 0.1, 0.5, 1],
            'classifier__subsample': [0.7, 0.8, 0.9, 1.0]
        }
    }
}

# --- Define resampling techniques ---

RESAMPLING_OPTIONS = {
    'SMOTE': (SMOTE(random_state=RANDOM_SEED), {'resampler__k_neighbors': [3, 5, 7]}),
    'RandomUnderSampler': (RandomUnderSampler(random_state=RANDOM_SEED), {'resampler__sampling_strategy': [0.3, 0.5, 0.7]}),
    'RandomOverSampler': (RandomOverSampler(random_state=RANDOM_SEED), {'resampler__sampling_strategy': [0.5, 0.7, 1.0]})
}