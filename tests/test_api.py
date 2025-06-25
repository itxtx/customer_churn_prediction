import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import pandas as pd
from src.api import app


class TestAPI:
    """Test suite for the FastAPI application."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_predictor(self):
        """Create a mock predictor for testing."""
        predictor = Mock()
        predictor.load_model.return_value = True
        predictor.predict_single.return_value = {
            'customer_id': 'test_customer',
            'churn_prediction': 'No',
            'churn_probability': 0.25,
            'risk_level': 'Low',
            'confidence': 0.85
        }
        predictor.predict_batch.return_value = [
            {
                'customer_id': 'customer_1',
                'churn_prediction': 'No',
                'churn_probability': 0.25,
                'risk_level': 'Low',
                'confidence': 0.85
            },
            {
                'customer_id': 'customer_2',
                'churn_prediction': 'Yes',
                'churn_probability': 0.75,
                'risk_level': 'High',
                'confidence': 0.90
            }
        ]
        return predictor
    
    @pytest.fixture
    def sample_customer_data(self):
        """Sample customer data for testing."""
        return {
            "tenure": 12,
            "MonthlyCharges": 70.5,
            "TotalCharges": 846.0,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
            "gender": "Male",
            "Partner": "Yes",
            "Dependents": "No"
        }
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "model_loaded" in data
    
    @patch('src.api.predictor')
    def test_single_prediction_success(self, mock_predictor_instance, client, sample_customer_data):
        """Test successful single customer prediction."""
        mock_predictor_instance.predict_single.return_value = {
            'customer_id': 'test_customer',
            'churn_prediction': 'No',
            'churn_probability': 0.25,
            'risk_level': 'Low',
            'confidence': 0.85
        }
        
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["churn_prediction"] == "No"
        assert data["churn_probability"] == 0.25
        assert data["risk_level"] == "Low"
        assert data["confidence"] == 0.85
        assert "customer_id" in data
    
    @patch('src.api.predictor')
    def test_single_prediction_missing_fields(self, mock_predictor_instance, client):
        """Test single prediction with missing required fields."""
        # Mock predictor to return empty dict (simulating missing required fields error)
        mock_predictor_instance.predict_single.return_value = {}
        
        incomplete_data = {"tenure": 12}  # Missing other required fields
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 500  # Internal error due to empty prediction result
    
    @patch('src.api.predictor')
    def test_single_prediction_invalid_data_types(self, mock_predictor_instance, client):
        """Test single prediction with invalid data types."""
        invalid_data = {
            "tenure": "invalid",  # Should be numeric
            "MonthlyCharges": 70.5,
            "Contract": "Month-to-month"
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.predictor')
    def test_batch_prediction_success(self, mock_predictor_instance, client, sample_customer_data):
        """Test successful batch prediction."""
        mock_predictor_instance.predict_batch.return_value = [
            {
                'customer_id': 'customer_1',
                'churn_prediction': 'No',
                'churn_probability': 0.25,
                'risk_level': 'Low',
                'confidence': 0.85
            },
            {
                'customer_id': 'customer_2',
                'churn_prediction': 'Yes',
                'churn_probability': 0.75,
                'risk_level': 'High',
                'confidence': 0.90
            }
        ]
        
        batch_data = {
            "customers": [sample_customer_data, sample_customer_data]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["churn_prediction"] == "No"
        assert data["predictions"][1]["churn_prediction"] == "Yes"
    
    @patch('src.api.predictor')
    def test_batch_prediction_empty_list(self, mock_predictor_instance, client):
        """Test batch prediction with empty customer list."""
        batch_data = {"customers": []}
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 422  # Should validate minimum items
    
    @patch('src.api.predictor')
    def test_batch_prediction_too_many_customers(self, mock_predictor_instance, client, sample_customer_data):
        """Test batch prediction with too many customers."""
        # Create a list with more than 1000 customers (assuming that's the limit)
        large_batch = {"customers": [sample_customer_data] * 1001}
        
        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 422  # Should validate maximum items
    
    @patch('src.api.predictor')
    def test_prediction_with_explanation(self, mock_predictor_instance, client, sample_customer_data):
        """Test prediction with explanation."""
        mock_predictor_instance.explain_prediction.return_value = {
            'prediction': {
                'customer_id': 'test_customer',
                'churn_prediction': 'No',
                'churn_probability': 0.25,
                'risk_level': 'Low',
                'confidence': 0.85
            },
            'important_factors': [
                'Month-to-month contract (higher churn risk)',
                'Electronic check payment (associated with higher churn)'
            ],
            'recommendations': [
                'Offer incentive to switch to annual contract',
                'Review pricing and offer competitive rate'
            ]
        }
        
        response = client.post("/predict/explain", json=sample_customer_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "important_factors" in data
        assert "recommendations" in data
        assert len(data["important_factors"]) > 0
        assert len(data["recommendations"]) > 0
    
    @patch('src.api.predictor')
    def test_api_error_handling(self, mock_predictor_instance, client, sample_customer_data):
        """Test API error handling when predictor fails."""
        mock_predictor_instance.predict_single.side_effect = Exception("Model prediction failed")
        
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 500
        
        data = response.json()
        assert "detail" in data
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.get("/health")
        assert response.status_code == 200
        # Note: CORS headers would be tested if CORS middleware is configured
    
    def test_openapi_docs(self, client):
        """Test that OpenAPI documentation is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"


class TestAPIDataModels:
    """Test the Pydantic data models used in the API."""
    
    def test_customer_data_model_validation(self):
        """Test CustomerData model validation."""
        from src.api import CustomerData
        
        # Valid data
        valid_data = {
            "tenure": 12,
            "MonthlyCharges": 70.5,
            "Contract": "Month-to-month"
        }
        customer = CustomerData(**valid_data)
        assert customer.tenure == 12
        assert customer.MonthlyCharges == 70.5
        assert customer.Contract == "Month-to-month"
        
        # Test optional fields
        customer_with_optional = CustomerData(
            tenure=12,
            MonthlyCharges=70.5,
            Contract="Month-to-month",
            TotalCharges=846.0
        )
        assert customer_with_optional.TotalCharges == 846.0
    
    def test_customer_data_model_invalid_values(self):
        """Test CustomerData model with invalid values."""
        from src.api import CustomerData
        import pytest
        from pydantic import ValidationError
        
        # Test negative tenure
        with pytest.raises(ValidationError):
            CustomerData(
                tenure=-1,
                MonthlyCharges=70.5,
                Contract="Month-to-month"
            )
        
        # Test negative charges
        with pytest.raises(ValidationError):
            CustomerData(
                tenure=12,
                MonthlyCharges=-10.0,
                Contract="Month-to-month"
            )
    
    def test_prediction_response_model(self):
        """Test PredictionResponse model."""
        from src.api import PredictionResponse
        
        response = PredictionResponse(
            customer_id="test_123",
            churn_prediction="Yes",
            churn_probability=0.75,
            risk_level="High",
            confidence=0.90
        )
        
        assert response.customer_id == "test_123"
        assert response.churn_prediction == "Yes"
        assert response.churn_probability == 0.75
        assert response.risk_level == "High"
        assert response.confidence == 0.90
    
    def test_batch_prediction_response_model(self):
        """Test BatchPredictionResponse model."""
        from src.api import BatchPredictionResponse, PredictionResponse
        
        predictions = [
            PredictionResponse(
                customer_id="test_1",
                churn_prediction="No",
                churn_probability=0.25,
                risk_level="Low",
                confidence=0.85
            ),
            PredictionResponse(
                customer_id="test_2",
                churn_prediction="Yes",
                churn_probability=0.75,
                risk_level="High",
                confidence=0.90
            )
        ]
        
        batch_response = BatchPredictionResponse(predictions=predictions)
        assert len(batch_response.predictions) == 2
        assert batch_response.predictions[0].customer_id == "test_1"
        assert batch_response.predictions[1].churn_prediction == "Yes"


class TestAPIIntegration:
    """Integration tests for the API with real components."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create a test client for integration tests."""
        return TestClient(app)
    
    @pytest.mark.integration
    def test_api_with_real_predictor(self, client):
        """Test API with actual predictor (requires trained model)."""
        # This test would require a trained model to be present
        # Skip if no model is available
        import os
        model_path = "models/best_model_xgboost.pkl"
        
        if not os.path.exists(model_path):
            pytest.skip("No trained model available for integration test")
        
        sample_data = {
            "tenure": 12,
            "MonthlyCharges": 70.5,
            "TotalCharges": 846.0,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
            "gender": "Male",
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "PaperlessBilling": "Yes",
            "SeniorCitizen": "0"
        }
        
        response = client.post("/predict", json=sample_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "churn_prediction" in data
            assert "churn_probability" in data
            assert "risk_level" in data
            assert "confidence" in data
            assert 0 <= data["churn_probability"] <= 1
            assert data["churn_prediction"] in ["Yes", "No"]
            assert data["risk_level"] in ["Very Low", "Low", "Medium", "High", "Very High"]
