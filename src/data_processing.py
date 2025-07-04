import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


    
    
class DataProcessor:
    """Handles all data preprocessing and feature engineering operations."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize DataProcessor with configuration.

        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.numeric_features = self.config['features']['numeric_features']
        self.categorical_features = self.config['features']['categorical_features']
        self.target_column = self.config['features']['target_column']
        self.customer_id_column = self.config['features']['customer_id_column']

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath: Path to the CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and data type conversions.
        This function now also performs a safer imputation for TotalCharges.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Convert TotalCharges to numeric (handle empty strings and spaces)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = df['TotalCharges'].replace(['', ' '], np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

            # Safer imputation for TotalCharges
            # For new customers (tenure=0), TotalCharges should be 0.
            # For other missing TotalCharges, impute based on MonthlyCharges * tenure.
            # This avoids leaking info if NaNs are churn-related, as MonthlyCharges & tenure are known.
            missing_total_charges = df['TotalCharges'].isna()

            # Fill TotalCharges for tenure = 0 (these should be 0)
            df.loc[(df['tenure'] == 0) & missing_total_charges, 'TotalCharges'] = 0

            # Fill remaining NaNs in TotalCharges based on MonthlyCharges * tenure
            # This is a more robust imputation that doesn't assume MonthlyCharges is the TotalCharges for a churner
            df.loc[missing_total_charges, 'TotalCharges'] = \
                df.loc[missing_total_charges, 'MonthlyCharges'] * df.loc[missing_total_charges, 'tenure']

            logger.info(f"Converted and imputed TotalCharges. Found {df['TotalCharges'].isna().sum()} remaining missing values (should be 0)")


        # Convert SeniorCitizen to string for consistency
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

        return df

    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Create engineered features and drop the original source columns.
            """
            df = df.copy()

            # Monthly to Total Charges Ratio (logic is fine, no changes needed here)
            if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
                epsilon = 1e-6 
                df['MonthlyToTotalRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + epsilon)
                df.loc[df['tenure'] == 0, 'MonthlyToTotalRatio'] = 1 
                df['MonthlyToTotalRatio'] = df['MonthlyToTotalRatio'].fillna(0)
                df['MonthlyToTotalRatio'] = df['MonthlyToTotalRatio'].clip(upper=1)
                logger.info("Created MonthlyToTotalRatio feature")

            # Number of Additional Services
            service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies']
            existing_services = [col for col in service_columns if col in df.columns]
            if existing_services:
                df['NumAdditionalServices'] = (df[existing_services] == 'Yes').sum(axis=1)
                logger.info("Created NumAdditionalServices feature")
                # ----> ADD THIS LINE TO DROP THE SOURCE COLUMNS <----
                df = df.drop(columns=existing_services)
                logger.info(f"Dropped original service columns: {existing_services}")

            # Has Internet Service (binary)
            if 'InternetService' in df.columns:
                df['HasInternetService'] = (df['InternetService'] != 'No').astype(int)
                logger.info("Created HasInternetService feature")
                # ----> ADD THIS LINE TO DROP THE SOURCE COLUMN <----
                df = df.drop(columns=['InternetService'])
                logger.info("Dropped original InternetService column")
                
            numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if 'SeniorCitizen' in numerical_features:
                numerical_features.remove('SeniorCitizen') # Remove if present

            categorical_features = df.select_dtypes(include=['object']).columns.tolist()
            # Add 'SeniorCitizen' to categorical features
            categorical_features.append('SeniorCitizen')

            # Include newly engineered features if they are numeric
            engineered_numeric = ['MonthlyToTotalRatio', 'NumAdditionalServices', 'HasInternetService']
            for feat in engineered_numeric:
                if feat in df.columns and feat not in numerical_features:
                    # Check if it's actually numeric in the DataFrame
                    if pd.api.types.is_numeric_dtype(df[feat]):
                        numerical_features.append(feat)

            return df

    def cap_outliers(self, df: pd.DataFrame, columns: List[str],
                     lower_percentile: float = 0.01,
                     upper_percentile: float = 0.99) -> pd.DataFrame:
        """
        Cap outliers in specified columns using percentile method.

        Args:
            df: Input DataFrame
            columns: List of columns to cap outliers
            lower_percentile: Lower percentile threshold
            upper_percentile: Upper percentile threshold

        Returns:
            DataFrame with capped outliers
        """
        df = df.copy()

        for col in columns:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                lower_bound = df[col].quantile(lower_percentile)
                upper_bound = df[col].quantile(upper_percentile)

                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Capped outliers in {col}: [{lower_bound:.2f}, {upper_bound:.2f}]")

        return df

    def get_median_values(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
        """
        Calculate median values for specified numeric columns.

        Args:
            df: Input DataFrame
            columns: List of columns to calculate medians

        Returns:
            Dictionary of column names and their median values
        """
        medians = {}
        for col in columns:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                medians[col] = df[col].median()
                logger.info(f"Median for {col}: {medians[col]:.2f}")
        return medians
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Drop customer ID if present
        if self.customer_id_column in df.columns:
            df = df.drop(columns=[self.customer_id_column])

        # Separate features and target
        if self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

        else:
            raise ValueError(f"Target column {self.target_column} not found in dataframe")

        return X, y

    def create_preprocessing_pipeline(self, X_train: pd.DataFrame = None) -> ColumnTransformer:
        """
        Create a preprocessing pipeline for numeric and categorical features.

        Args:
            X_train: Training data to dynamically detect feature types (optional)

        Returns:
            ColumnTransformer object with preprocessing steps
        """
        # Initialize feature lists
        numeric_features = []
        categorical_features = []
        
        if X_train is not None:
            # Dynamic feature detection like in parameter search notebook
            numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if 'SeniorCitizen' in numeric_features:
                numeric_features.remove('SeniorCitizen') # Remove if present

            categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
            # Add 'SeniorCitizen' to categorical features
            categorical_features.append('SeniorCitizen')

            # Include newly engineered features if they are numeric
            engineered_numeric = ['MonthlyToTotalRatio', 'NumAdditionalServices', 'HasInternetService']
            for feat in engineered_numeric:
                if feat in X_train.columns and feat not in numeric_features:
                     # Check if it's actually numeric in the DataFrame
                     if pd.api.types.is_numeric_dtype(X_train[feat]):
                          numeric_features.append(feat)
        else:
            # Fallback to config-based feature lists
            engineered_features = self.config['features'].get('engineered_features', [])
            numeric_features = self.numeric_features.copy()
            categorical_features = self.categorical_features.copy()
            
            if 'SeniorCitizen' in numeric_features:
                numeric_features.remove('SeniorCitizen')
            if 'SeniorCitizen' not in categorical_features:
                categorical_features.append('SeniorCitizen')
            
            engineered_numeric = ['MonthlyToTotalRatio', 'NumAdditionalServices', 'HasInternetService']
            for feat in engineered_numeric:
                if feat in engineered_features and feat not in numeric_features:
                    numeric_features.append(feat)

        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), # Median imputation for remaining NaNs
            ('scaler', StandardScaler())
        ])

        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Most frequent for categorical NaNs
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough' # Keep other columns (if any), although none expected after explicit definition
        )

        logger.info("Created preprocessing pipeline")
        return preprocessor
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate data quality and return issues found.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary of validation issues by category
        """
        issues = {
            'missing_columns': [],
            'missing_values': [],
            'data_type_issues': [],
            'value_issues': []
        }
        
        # Define a comprehensive list of expected columns for validation.
        # Ensure that newly engineered features that are expected to be present
        # after calculate_derived_features are also included here for robust validation.
        expected_columns = (
            [col for col in self.numeric_features if col != 'TotalCharges'] + # Exclude TotalCharges to check separately after cleaning
            self.categorical_features +
            ['TotalCharges', 'MonthlyToTotalRatio', 'NumAdditionalServices', 'HasInternetService'] +
            [self.target_column, self.customer_id_column]
        )
        
        # Check for required columns
        for col in expected_columns:
            if col not in df.columns:
                issues['missing_columns'].append(col)
        
        # Check for missing values after cleaning
        # This check should be run *after* clean_data and calculate_derived_features
        # to catch any remaining NaNs.
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues['missing_values'].append(f"{col}: {missing_count} missing values")
        
        # Check data types for expected numeric features
        # Assuming these are the final numeric features after all processing
        final_numeric_features = (self.numeric_features + 
                                  ['MonthlyToTotalRatio', 'NumAdditionalServices'])
        for col in final_numeric_features:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues['data_type_issues'].append(f"{col} should be numeric, but is {df[col].dtype}")
        
        # Check data types for expected categorical features (after potential str conversion)
        final_categorical_features = self.categorical_features + ['HasInternetService']
        for col in final_categorical_features:
            if col in df.columns and not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                issues['data_type_issues'].append(f"{col} should be categorical/object, but is {df[col].dtype}")

        # Check for invalid values in specific columns
        if 'SeniorCitizen' in df.columns:
            # After conversion to str, SeniorCitizen should only contain '0' or '1'
            valid_senior_citizen_values = ['0', '1']
            invalid_senior_citizen = ~df['SeniorCitizen'].astype(str).isin(valid_senior_citizen_values)
            if invalid_senior_citizen.any():
                issues['value_issues'].append(f"SeniorCitizen has {invalid_senior_citizen.sum()} invalid values (not '0' or '1')")
        
        # Check for 'No phone service' or 'No internet service' in columns where 'No' is expected
        # This is a general check for categorical consistency, adjust as needed
        service_cols_to_check = [
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        for col in service_cols_to_check:
            if col in df.columns:
                # Check for values that aren't 'Yes', 'No', or 'No internet/phone service'
                # after general cleaning. This might be too strict depending on data source.
                unique_values = df[col].unique()
                for val in unique_values:
                    if val not in ['Yes', 'No', 'No phone service', 'No internet service', np.nan]:
                        issues['value_issues'].append(f"{col} has unexpected value: '{val}'")

        return issues