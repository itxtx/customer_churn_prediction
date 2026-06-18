import anyio
import httpx
import pytest

from src import api


class ASGITestClient:
    def request(self, method, path, **kwargs):
        return anyio.run(self._request, method, path, kwargs)

    async def _request(self, method, path, kwargs):
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


@pytest.fixture
def client():
    return ASGITestClient()


@pytest.fixture
def sample_customer_data():
    return {
        "customerID": "test_customer",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
    }


@pytest.fixture
def loaded_predictor(monkeypatch):
    class MockPredictor:
        model = object()
        final_model_path = "models/final_model.joblib"

        def predict_single(self, customer):
            return {
                "customer_id": customer.get("customerID", "customer_0"),
                "churn_prediction": "No",
                "churn_probability": 0.25,
                "risk_level": "Low",
                "confidence": 0.75,
            }

        def predict_batch(self, customers):
            return [
                {
                    "customer_id": customer.get("customerID", f"customer_{index}"),
                    "churn_prediction": "Yes" if index else "No",
                    "churn_probability": 0.75 if index else 0.25,
                    "risk_level": "High" if index else "Low",
                    "confidence": 0.75,
                }
                for index, customer in enumerate(customers)
            ]

        def explain_prediction(self, customer):
            return {
                "prediction": self.predict_single(customer),
                "important_factors": ["Month-to-month contract"],
                "recommendations": ["Offer incentive to switch to annual contract"],
            }

    monkeypatch.setattr(api, "predictor", MockPredictor())


def test_health_check_reports_model_state(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"healthy", "degraded"}
    assert "model_loaded" in data
    assert "model_path" in data


def test_single_prediction_success(client, sample_customer_data, loaded_predictor):
    response = client.post("/predict", json=sample_customer_data)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "test_customer"
    assert data["churn_prediction"] == "No"
    assert data["churn_probability"] == 0.25
    assert data["risk_level"] == "Low"


def test_single_prediction_requires_full_schema(client, loaded_predictor):
    response = client.post("/predict", json={"tenure": 12})
    assert response.status_code == 422


def test_single_prediction_invalid_types(client, loaded_predictor, sample_customer_data):
    sample_customer_data["tenure"] = "invalid"
    response = client.post("/predict", json=sample_customer_data)
    assert response.status_code == 422


def test_prediction_requires_loaded_model(client, monkeypatch, sample_customer_data):
    class MissingPredictor:
        model = None
        final_model_path = "models/final_model.joblib"

    monkeypatch.setattr(api, "predictor", MissingPredictor())
    response = client.post("/predict", json=sample_customer_data)
    assert response.status_code == 503


def test_batch_prediction_success(client, sample_customer_data, loaded_predictor):
    response = client.post(
        "/predict/batch",
        json={"customers": [sample_customer_data, sample_customer_data]},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["total_customers"] == 2
    assert data["high_risk_count"] == 1


def test_batch_prediction_bounds(client, sample_customer_data, loaded_predictor):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    response = client.post(
        "/predict/batch",
        json={"customers": [sample_customer_data] * 1001},
    )
    assert response.status_code == 422


def test_prediction_with_explanation(client, sample_customer_data, loaded_predictor):
    response = client.post("/predict/explain", json=sample_customer_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "important_factors" in data
    assert "recommendations" in data


def test_removed_customer_management_routes_are_not_public(client):
    assert client.get("/customers/test_customer").status_code == 404
    assert client.get("/statistics").status_code == 404


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
