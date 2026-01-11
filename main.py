from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from contextlib import asynccontextmanager
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Global model variable
model = None
MODEL_PATH = "model.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        raise
    
    yield
    
    # Shutdown (if needed)
    # Cleanup code can go here

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ChurnPredictionRequest(BaseModel):
    """Request model for churn prediction"""
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: Literal["Yes", "No"] = Field(..., description="Is the customer a senior citizen")
    Partner: Literal["Yes", "No"] = Field(..., description="Does the customer have a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Does the customer have dependents")
    tenure: int = Field(..., ge=0, le=72, description="Number of months the customer has been with the company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Does the customer have phone service")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., description="Does the customer have multiple lines")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Type of internet service")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., description="Does the customer have online security")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., description="Does the customer have online backup")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., description="Does the customer have device protection")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., description="Does the customer have tech support")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., description="Does the customer have streaming TV")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., description="Does the customer have streaming movies")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract type")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Does the customer use paperless billing")
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": "No",
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
                "MonthlyCharges": 79.85,
                "TotalCharges": 1000.0
            }
        }
    )

# Response Models
class ChurnPredictionResponse(BaseModel):
    """Response model for churn prediction"""
    prediction: int = Field(..., description="Predicted class (0=No Churn, 1=Churn)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    retention_probability: float = Field(..., ge=0, le=1, description="Probability of retention")
    risk_level: str = Field(..., description="Risk level category")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    message: str

# Helper Functions
def prepare_input_data(input_dict: dict) -> pd.DataFrame:
    """
    Prepare input data to match the model's expected feature format.
    Based on the training notebook: prepare_telco_df function
    """
    # Start with a dictionary for the encoded features
    encoded = {}
    
    # Binary encoding: Gender (Male=0, Female=1)
    encoded['gender'] = 1 if input_dict['gender'] == "Female" else 0
    
    # Binary encoding: SeniorCitizen (No=0, Yes=1)
    encoded['SeniorCitizen'] = 1 if input_dict['SeniorCitizen'] == "Yes" else 0
    
    # Binary encoding: Partner, Dependents, PhoneService, PaperlessBilling (Yes=1, No=0)
    encoded['Partner'] = 1 if input_dict['Partner'] == "Yes" else 0
    encoded['Dependents'] = 1 if input_dict['Dependents'] == "Yes" else 0
    encoded['PhoneService'] = 1 if input_dict['PhoneService'] == "Yes" else 0
    encoded['PaperlessBilling'] = 1 if input_dict['PaperlessBilling'] == "Yes" else 0
    
    # Numeric features
    encoded['tenure'] = int(input_dict['tenure'])
    encoded['MonthlyCharges'] = float(input_dict['MonthlyCharges'])
    # Convert TotalCharges to numeric (handle potential string values)
    try:
        encoded['TotalCharges'] = float(input_dict['TotalCharges'])
    except (ValueError, TypeError):
        encoded['TotalCharges'] = 0.0
    
    # MultipleLines encoding (No phone service=0, No=0, Yes=1)
    if input_dict['MultipleLines'] == "Yes":
        encoded['MultipleLines'] = 1
    else:
        encoded['MultipleLines'] = 0
    
    # Service columns: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
    # Yes=1, No=0, "No internet service"=0
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        encoded[col] = 1 if input_dict[col] == "Yes" else 0
    
    # One-hot encoding with drop_first=True
    # InternetService: drops "DSL" (first category alphabetically)
    encoded['InternetService_Fiber optic'] = 1 if input_dict['InternetService'] == "Fiber optic" else 0
    encoded['InternetService_No'] = 1 if input_dict['InternetService'] == "No" else 0
    
    # Contract: drops "Month-to-month" (first category alphabetically)
    encoded['Contract_One year'] = 1 if input_dict['Contract'] == "One year" else 0
    encoded['Contract_Two year'] = 1 if input_dict['Contract'] == "Two year" else 0
    
    # PaymentMethod: drops "Bank transfer (automatic)" (first category alphabetically)
    encoded['PaymentMethod_Credit card (automatic)'] = 1 if input_dict['PaymentMethod'] == "Credit card (automatic)" else 0
    encoded['PaymentMethod_Electronic check'] = 1 if input_dict['PaymentMethod'] == "Electronic check" else 0
    encoded['PaymentMethod_Mailed check'] = 1 if input_dict['PaymentMethod'] == "Mailed check" else 0
    
    # Convert to DataFrame with correct column order (matching model's expected order)
    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'InternetService_Fiber optic', 'InternetService_No',
        'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    # Create DataFrame with all features in order
    input_df = pd.DataFrame([encoded])
    
    # Ensure all columns are present (fill missing with 0)
    for feature in feature_order:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Reorder columns to match expected order
    input_df = input_df[feature_order]
    
    return input_df

def get_risk_level(churn_probability: float) -> str:
    """Determine risk level based on churn probability"""
    if churn_probability > 0.7:
        return "CRITICAL"
    elif churn_probability > 0.5:
        return "HIGH"
    elif churn_probability > 0.3:
        return "MEDIUM"
    else:
        return "LOW"

# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        message="Model loaded and ready" if model is not None else "Model not loaded"
    )

@app.post("/predict", response_model=ChurnPredictionResponse, tags=["Prediction"])
async def predict_churn(request: ChurnPredictionRequest):
    """
    Predict customer churn probability
    
    This endpoint takes customer information and returns:
    - Prediction (0=No Churn, 1=Churn)
    - Churn probability (0-1)
    - Retention probability (0-1)
    - Risk level (LOW, MEDIUM, HIGH, CRITICAL)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # Convert request to dictionary
        input_dict = request.model_dump()
        
        # Prepare input data
        input_df = prepare_input_data(input_dict)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Get probabilities
        churn_prob = float(prediction_proba[1])
        retention_prob = float(prediction_proba[0])
        
        # Determine risk level
        risk_level = get_risk_level(churn_prob)
        
        # Create response
        response = ChurnPredictionResponse(
            prediction=int(prediction),
            churn_probability=churn_prob,
            retention_probability=retention_prob,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", tags=["Prediction"])
async def predict_churn_batch(requests: list[ChurnPredictionRequest]):
    """
    Predict churn for multiple customers at once
    
    Accepts a list of customer data and returns predictions for all
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        results = []
        for request in requests:
            input_dict = request.dict()
            input_df = prepare_input_data(input_dict)
            
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            churn_prob = float(prediction_proba[1])
            retention_prob = float(prediction_proba[0])
            risk_level = get_risk_level(churn_prob)
            
            results.append({
                "prediction": int(prediction),
                "churn_probability": churn_prob,
                "retention_probability": retention_prob,
                "risk_level": risk_level,
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "predictions": results,
            "total": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

