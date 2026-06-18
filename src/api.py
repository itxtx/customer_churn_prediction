import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from src.predict import ChurnPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

predictor = ChurnPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Customer Churn Prediction API.")
    if predictor.load_model():
        logger.info("Model loaded successfully.")
    else:
        logger.warning("Model is not loaded. Prediction endpoints will return 503.")
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for serving the configured customer churn model.",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerData(BaseModel):
    """Validated model input matching the Telco churn feature schema."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    )

    customerID: Optional[str] = Field(None, description="Customer ID")
    gender: str = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen (0/1)")
    Partner: str = Field(..., description="Whether customer has partner")
    Dependents: str = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, description="Number of months with company")
    PhoneService: str = Field(..., description="Whether customer has phone service")
    MultipleLines: str = Field(..., description="Whether customer has multiple lines")
    InternetService: str = Field(..., description="Type of internet service")
    OnlineSecurity: str = Field(..., description="Whether customer has online security")
    OnlineBackup: str = Field(..., description="Whether customer has online backup")
    DeviceProtection: str = Field(..., description="Whether customer has device protection")
    TechSupport: str = Field(..., description="Whether customer has tech support")
    StreamingTV: str = Field(..., description="Whether customer has streaming TV")
    StreamingMovies: str = Field(..., description="Whether customer has streaming movies")
    Contract: str = Field(..., description="Contract type")
    PaperlessBilling: str = Field(..., description="Whether customer has paperless billing")
    PaymentMethod: str = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")


class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_prediction: str
    churn_probability: float
    risk_level: str
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    processing_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    timestamp: datetime


def _customer_to_dict(customer: CustomerData) -> Dict[str, Any]:
    return customer.model_dump()


def _ensure_model_loaded() -> None:
    if predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run training or scripts/download_model.py first.",
        )


@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if predictor.model is not None else "degraded",
        model_loaded=predictor.model is not None,
        model_path=predictor.final_model_path,
        timestamp=datetime.now(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerData):
    _ensure_model_loaded()
    try:
        result = predictor.predict_single(_customer_to_dict(customer))
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    _ensure_model_loaded()
    try:
        import time

        start_time = time.time()
        customers_data = [_customer_to_dict(customer) for customer in request.customers]
        results = predictor.predict_batch(customers_data)
        high_risk_count = sum(
            1 for result in results if result.get("risk_level") in ["High", "Very High"]
        )
        predictions = [PredictionResponse(**result) for result in results]
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=high_risk_count,
            processing_time_seconds=time.time() - start_time,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail="Batch prediction failed") from exc


@app.post("/predict/explain")
async def explain_prediction(customer: CustomerData):
    _ensure_model_loaded()
    try:
        return predictor.explain_prediction(_customer_to_dict(customer))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Explanation error")
        raise HTTPException(status_code=500, detail="Explanation failed") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["api"]["reload"],
    )
