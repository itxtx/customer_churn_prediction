import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from data_processing import DataProcessor


@pytest.fixture
def sample_data():
    """Create sample customer data for testing."""
    return pd.DataFrame({
        'customerID': ['1234', '5678', '9012'],
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'tenure': [12, 24, 6],
        'MonthlyCharges': [50.0, 75.0, 100.0],
        'TotalCharges': ['600.0', '1800.0', ' '],  # Include empty string
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'Contract': ['Month-to-month', 'Two year', 'One year'],
        'Churn': ['No', 'Yes', 'No']
    })


@pytest.fixture
def data_processor(tmp_path):
    """Create DataProcessor instance with test config."""
    # Create temporary config file
    config_content = """
database:
  path: "test.db"
  table_name: "customers"

features:
  target_column: "Churn"
  customer_id_column: "customerID"
  
  numeric_features:
    - "tenure"
    - "MonthlyCharges"
    - "TotalCharges"
  
  categorical_features:
    - "gender"
    - "SeniorCitizen"
    - "Partner"
    - "InternetService"
    - "OnlineSecurity"
    - "Contract"
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    
    return DataProcessor(str(config_path))


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_load_data(self, data_processor, tmp_path, sample_data):
        """Test data loading from CSV."""
        # Save sample data to CSV
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Load data
        loaded_df = data_processor.load_data(str(csv_path))
        
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == list(sample_data.columns)
        
    def test_load_data_file_not_found(self, data_processor):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            data_processor.load_data("nonexistent_file.csv")
    
    def test_clean_data(self, data_processor, sample_data):
        """Test data cleaning functionality."""
        cleaned_df = data_processor.clean_data(sample_data)
        
        # Check TotalCharges conversion
        assert cleaned_df['TotalCharges'].dtype == 'float64'
        # Empty string should be converted to a value based on MonthlyCharges * tenure
        assert cleaned_df['TotalCharges'].isna().sum() == 0  # No NaNs after imputation
        
        # Check SeniorCitizen conversion
        assert cleaned_df['SeniorCitizen'].dtype == 'object'
        assert all(cleaned_df['SeniorCitizen'].isin(['0', '1']))
    
    def test_calculate_derived_features(self, data_processor, sample_data):
        """Test feature engineering."""
        # Clean data first
        df = data_processor.clean_data(sample_data)
        
        # Calculate features
        df_with_features = data_processor.calculate_derived_features(df)
        
        # Check new features exist
        assert 'MonthlyToTotalRatio' in df_with_features.columns
        assert 'HasInternetService' in df_with_features.columns
        
        # Check calculations
        assert df_with_features['HasInternetService'].iloc[0] == 1  # DSL
        assert df_with_features['HasInternetService'].iloc[2] == 0  # No
        
        # Check ratio calculation
        expected_ratio = 50.0 / (600.0 + 1)
        assert abs(df_with_features['MonthlyToTotalRatio'].iloc[0] - expected_ratio) < 0.001
    
    def test_cap_outliers(self, data_processor):
        """Test outlier capping."""
        # Create data with outliers
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 100],  # 100 is outlier
            'feature2': [-50, 10, 20, 30, 40, 50]  # -50 is outlier
        })
        
        capped_df = data_processor.cap_outliers(df, ['feature1', 'feature2'])
        
        # Check that extreme values are capped
        assert capped_df['feature1'].max() < 100
        assert capped_df['feature2'].min() > -50
    
    def test_get_median_values(self, data_processor, sample_data):
        """Test median calculation."""
        medians = data_processor.get_median_values(
            sample_data, 
            ['tenure', 'MonthlyCharges']
        )
        
        assert medians['tenure'] == 12  # Median of [6, 12, 24]
        assert medians['MonthlyCharges'] == 75.0  # Median of [50, 75, 100]
    
    def test_prepare_features(self, data_processor, sample_data):
        """Test feature preparation for modeling."""
        X, y = data_processor.prepare_features(sample_data)
        
        # Check that customerID is removed
        assert 'customerID' not in X.columns
        
        # Check that target is separated
        assert 'Churn' not in X.columns
        assert len(y) == 3
        
        # Check target conversion to binary
        assert all(y.isin([0, 1]))
        assert y.iloc[1] == 1  # 'Yes' -> 1
    
    def test_create_preprocessing_pipeline(self, data_processor):
        """Test preprocessing pipeline creation."""
        pipeline = data_processor.create_preprocessing_pipeline()
        
        # Check pipeline structure
        assert hasattr(pipeline, 'transformers')
        assert len(pipeline.transformers) >= 2  # Numeric and categorical
        
        # Check transformer names
        transformer_names = [t[0] for t in pipeline.transformers]
        assert 'num' in transformer_names
        assert 'cat' in transformer_names
    
    def test_validate_data(self, data_processor):
        """Test data validation."""
        # Create data with various issues
        df = pd.DataFrame({
            'customerID': ['1234', '5678'],
            'gender': ['Male', None],  # Missing value
            'tenure': ['invalid', 12],  # Wrong data type
            'SeniorCitizen': [0, 5],  # Invalid value
            # Missing required column 'MonthlyCharges'
        })
        
        issues = data_processor.validate_data(df)
        
        assert 'MonthlyCharges' in issues['missing_columns']
        assert any('gender' in issue for issue in issues['missing_values'])
        assert any('tenure' in issue for issue in issues['data_type_issues'])
        assert any('SeniorCitizen' in issue for issue in issues['value_issues'])


class TestIntegration:
    """Integration tests for complete data processing pipeline."""
    
    def test_full_pipeline(self, data_processor, sample_data):
        """Test complete data processing pipeline."""
        # Clean data
        df = data_processor.clean_data(sample_data)
        
        # Add features
        df = data_processor.calculate_derived_features(df)
        
        # Prepare for modeling
        X, y = data_processor.prepare_features(df)
        
        # Create and fit preprocessing pipeline
        preprocessor = data_processor.create_preprocessing_pipeline()
        
        # Fit transform (should work without errors)
        X_transformed = preprocessor.fit_transform(X)
        
        # Check output shape
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > len(X.columns)  # Due to one-hot encoding


if __name__ == "__main__":
    pytest.main([__file__])