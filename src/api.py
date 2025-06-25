from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import yaml
from datetime import datetime
import uvicorn
from src.predict import ChurnPredictor
from src.database import Database

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0"
)

# Load configuration
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Initialize predictor and database
predictor = ChurnPredictor()
db = Database()


# Pydantic models for request/response validation
class CustomerData(BaseModel):
    """Model for customer data input."""
    customerID: Optional[str] = Field(None, description="Customer ID")
    gender: Optional[str] = Field(None, description="Customer gender")
    SeniorCitizen: Optional[int] = Field(None, description="Whether customer is senior citizen (0/1)")
    Partner: Optional[str] = Field(None, description="Whether customer has partner")
    Dependents: Optional[str] = Field(None, description="Whether customer has dependents")
    tenure: Optional[int] = Field(None, ge=0, description="Number of months with company")
    PhoneService: Optional[str] = Field(None, description="Whether customer has phone service")
    MultipleLines: Optional[str] = Field(None, description="Whether customer has multiple lines")
    InternetService: Optional[str] = Field(None, description="Type of internet service")
    OnlineSecurity: Optional[str] = Field(None, description="Whether customer has online security")
    OnlineBackup: Optional[str] = Field(None, description="Whether customer has online backup")
    DeviceProtection: Optional[str] = Field(None, description="Whether customer has device protection")
    TechSupport: Optional[str] = Field(None, description="Whether customer has tech support")
    StreamingTV: Optional[str] = Field(None, description="Whether customer has streaming TV")
    StreamingMovies: Optional[str] = Field(None, description="Whether customer has streaming movies")
    Contract: Optional[str] = Field(None, description="Contract type")
    PaperlessBilling: Optional[str] = Field(None, description="Whether customer has paperless billing")
    PaymentMethod: Optional[str] = Field(None, description="Payment method")
    MonthlyCharges: Optional[float] = Field(None, ge=0, description="Monthly charges")
    TotalCharges: Optional[float] = Field(None, ge=0, description="Total charges")

    class Config:
        schema_extra = {
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
                "TotalCharges": 29.85
            }
        }


class PredictionResponse(BaseModel):
    """Model for prediction response."""
    customer_id: Optional[str]
    churn_prediction: str
    churn_probability: float
    risk_level: str
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchPredictionResponse(BaseModel):
    """Model for batch prediction response."""
    predictions: List[PredictionResponse]
    total_customers: int = 0
    high_risk_count: int = 0
    processing_time_seconds: float = 0.0


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: datetime


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting API server...")
    
    # Load model
    if not predictor.load_model():
        logger.error("Failed to load model!")
    else:
        logger.info("Model loaded successfully")
    
    # Test database connection
    try:
        db.create_customers_table()
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model is not None,
        database_connected=True,  # Simplified - could add actual check
        timestamp=datetime.now()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerData):
    """
    Predict churn for a single customer.
    
    Args:
        customer: Customer data
        
    Returns:
        Prediction response with churn probability and risk level
    """
    try:
        # Convert to dict
        customer_dict = customer.dict()
        
        # Make prediction
        result = predictor.predict_single(customer_dict)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class BatchPredictionRequest(BaseModel):
    """Model for batch prediction request."""
    customers: List[CustomerData] = Field(..., min_items=1, max_items=1000)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers.
    
    Args:
        customers: List of customer data
        
    Returns:
        Batch prediction response with all predictions
    """
    try:
        import time
        start_time = time.time()
        
        # Convert to list of dicts
        customers_data = [customer.dict() for customer in request.customers]
        
        # Make predictions
        results = predictor.predict_batch(customers_data)
        
        # Count high risk customers
        high_risk_count = sum(1 for r in results 
                            if r.get('risk_level') in ['High', 'Very High'])
        
        # Convert to PredictionResponse objects
        predictions = [PredictionResponse(**r) for r in results if 'error' not in r]
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=high_risk_count,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/explain")
async def explain_prediction(customer: CustomerData):
    """
    Get prediction with explanation for a single customer.
    
    Args:
        customer: Customer data
        
    Returns:
        Prediction with explanations and recommendations
    """
    try:
        # Convert to dict
        customer_dict = customer.dict()
        
        # Get explanation
        explanation = predictor.explain_prediction(customer_dict)
        
        return JSONResponse(content=explanation)
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customers/{customer_id}")
async def get_customer(customer_id: str):
    """
    Get customer data by ID.
    
    Args:
        customer_id: Customer ID
        
    Returns:
        Customer data if found
    """
    customer = db.get_customer(customer_id)
    
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    return customer


@app.post("/customers/{customer_id}/predict")
async def predict_for_existing_customer(customer_id: str):
    """
    Predict churn for an existing customer in the database.
    
    Args:
        customer_id: Customer ID
        
    Returns:
        Prediction response
    """
    # Get customer from database
    customer = db.get_customer(customer_id)
    
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Remove non-feature columns
    for col in ['created_at', 'updated_at', 'Churn']:
        customer.pop(col, None)
    
    # Make prediction
    result = predictor.predict_single(customer)
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return PredictionResponse(**result)


@app.get("/statistics")
async def get_statistics():
    """
    Get churn statistics from the database.
    
    Returns:
        Dictionary with churn statistics
    """
    try:
        stats = db.get_churn_statistics()
        return stats
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/customers/{customer_id}/save-prediction")
async def save_prediction(customer_id: str, background_tasks: BackgroundTasks):
    """
    Make prediction and save it to the database.
    
    Args:
        customer_id: Customer ID
        background_tasks: FastAPI background tasks
        
    Returns:
        Success message
    """
    # Get customer
    customer = db.get_customer(customer_id)
    
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Make prediction
    for col in ['created_at', 'updated_at', 'Churn']:
        customer.pop(col, None)
    
    result = predictor.predict_single(customer)
    
    # Save prediction in background
    background_tasks.add_task(
        save_prediction_to_db, 
        customer_id, 
        result
    )
    
    return {"message": "Prediction queued for saving", "prediction": result}


def save_prediction_to_db(customer_id: str, prediction: Dict[str, Any]):
    """
    Background task to save prediction to database.
    
    Args:
        customer_id: Customer ID
        prediction: Prediction result
    """
    try:
        # Update customer record with prediction
        update_data = {
            'Churn': prediction['churn_prediction'],
            'churn_probability': prediction['churn_probability'],
            'last_prediction_date': datetime.now()
        }
        
        db.update_customer(customer_id, update_data)
        logger.info(f"Saved prediction for customer {customer_id}")
        
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")


if __name__ == "__main__":

    
    # Run the API
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['reload']
    )