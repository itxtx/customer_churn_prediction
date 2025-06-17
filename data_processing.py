import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml

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
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Convert TotalCharges to numeric (handle empty strings and spaces)
        if 'TotalCharges' in df.columns:
            # First replace any empty strings or spaces with NaN
            df['TotalCharges'] = df['TotalCharges'].replace(['', ' '], np.nan)
            # Then convert to numeric
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

            logger.info(f"Converted TotalCharges to numeric. Found {df['TotalCharges'].isna().sum()} missing values")
        
        # Convert SeniorCitizen to string for consistency
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
        
        return df
    
    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features based on existing columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()
        
        # Monthly to Total Charges Ratio
        if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
            # Ensure both columns are numeric
            df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Calculate ratio with proper handling of zeros and NaN
            df['MonthlyToTotalRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
            df['MonthlyToTotalRatio'] = df['MonthlyToTotalRatio'].fillna(0)
            logger.info("Created MonthlyToTotalRatio feature")
        
        # Number of Additional Services
        service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
        existing_services = [col for col in service_columns if col in df.columns]
        
        if existing_services:
            df['NumAdditionalServices'] = (df[existing_services] == 'Yes').sum(axis=1)
            logger.info("Created NumAdditionalServices feature")
        
        # Has Internet Service (binary)
        if 'InternetService' in df.columns:
            df['HasInternetService'] = (df['InternetService'] != 'No').astype(int)
            logger.info("Created HasInternetService feature")
        
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
            
            # Convert target to binary
            if y.dtype == 'object':
                y = (y == 'Yes').astype(int)
        else:
            X = df
            y = None
        
        return X, y
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create a preprocessing pipeline for numeric and categorical features.
        
        Returns:
            ColumnTransformer object with preprocessing steps
        """
        # Update feature lists to include engineered features
        numeric_features = self.numeric_features + ['MonthlyToTotalRatio', 'NumAdditionalServices']
        categorical_features = self.categorical_features + ['HasInternetService']
        
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
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
        
        # Check for required columns
        all_features = (self.numeric_features + self.categorical_features + 
                       [self.target_column, self.customer_id_column])
        
        for col in all_features:
            if col not in df.columns:
                issues['missing_columns'].append(col)
        
        # Check for missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues['missing_values'].append(f"{col}: {missing_count} missing values")
        
        # Check data types
        for col in self.numeric_features:
            if col in df.columns and df[col].dtype not in ['float64', 'int64']:
                issues['data_type_issues'].append(f"{col} should be numeric")
        
        # Check for invalid values
        if 'SeniorCitizen' in df.columns:
            valid_values = ['0', '1', 0, 1]
            invalid = ~df['SeniorCitizen'].isin(valid_values)
            if invalid.any():
                issues['value_issues'].append(f"SeniorCitizen has {invalid.sum()} invalid values")
        
        return issues
    
    def get_feature_importance_df(self, feature_names: List[str], 
                                 importances: np.ndarray) -> pd.DataFrame:
        """
        Create a DataFrame of feature importances sorted by importance.
        
        Args:
            feature_names: List of feature names
            importances: Array of feature importance values
            
        Returns:
            DataFrame with features and their importance scores
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        return importance_df