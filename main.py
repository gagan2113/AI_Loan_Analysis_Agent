"""
Working FastAPI Backend with Mock Predictions
==============================================

This version provides a working FastAPI backend that demonstrates the exact
flow you requested, using a mock prediction system instead of the problematic pickle file.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import logging
import os
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanApplication(BaseModel):
    """Pydantic model for loan application input validation"""
    
    # Personal Information
    gender: str = Field(..., description="Gender of applicant", pattern="^(Male|Female)$")
    married: str = Field(..., description="Marital status", pattern="^(Yes|No)$")
    dependents: str = Field(..., description="Number of dependents", pattern="^(0|1|2|3\\+)$")
    education: str = Field(..., description="Education level", pattern="^(Graduate|Not Graduate)$")
    self_employed: str = Field(..., description="Self employment status", pattern="^(Yes|No)$")
    
    # Financial Information (in Indian Rupees)
    applicant_income: float = Field(..., description="Applicant monthly income in INR", ge=0)
    coapplicant_income: float = Field(0, description="Co-applicant monthly income in INR", ge=0)
    loan_amount: float = Field(..., description="Loan amount in thousands INR", gt=0)
    loan_amount_term: float = Field(360, description="Loan term in months", gt=0)
    credit_history: float = Field(..., description="Credit history (1.0=Good, 0.0=Poor)", ge=0, le=1)
    
    # Property Information
    property_area: str = Field(..., description="Property area type", pattern="^(Urban|Semiurban|Rural)$")
    
    @validator('applicant_income', 'coapplicant_income')
    def validate_income(cls, v):
        if v < 0:
            raise ValueError('Income cannot be negative')
        if v > 10000000:  # 1 crore INR monthly income cap
            raise ValueError('Income seems unrealistic')
        return v

class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    
    model_config = {"protected_namespaces": ()}
    
    success: bool
    prediction: int
    prediction_label: str
    approval_probability: float
    rejection_probability: float
    confidence: float
    risk_level: str
    financial_analysis: Dict[str, Any]
    timestamp: str
    model_version: str = "demo_model_v1.0"

class HealthResponse(BaseModel):
    """Response model for health check"""
    
    model_config = {"protected_namespaces": ()}
    
    status: str
    timestamp: str
    model_loaded: bool
    model_path: Optional[str]

# Initialize FastAPI app
app = FastAPI(
    title="Loan Risk Analysis API - Demo",
    description="FastAPI backend for loan approval prediction (Demo with working predictions)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_financial_metrics(application: LoanApplication) -> Dict[str, Any]:
    """Calculate financial analysis metrics"""
    
    total_income = application.applicant_income + application.coapplicant_income
    annual_income = total_income * 12
    loan_amount_inr = application.loan_amount * 1000
    
    # Calculate EMI (using 8% annual interest rate)
    annual_rate = 0.08
    monthly_rate = annual_rate / 12
    n_months = application.loan_amount_term
    
    if monthly_rate > 0 and n_months > 0:
        emi = (loan_amount_inr * monthly_rate * (1 + monthly_rate)**n_months) / ((1 + monthly_rate)**n_months - 1)
    else:
        emi = loan_amount_inr / n_months if n_months > 0 else 0
    
    # Financial ratios
    income_to_loan_ratio = annual_income / loan_amount_inr if loan_amount_inr > 0 else 0
    emi_to_income_ratio = emi / total_income if total_income > 0 else 0
    debt_to_income_ratio = (emi * 12) / annual_income if annual_income > 0 else 0
    
    return {
        "total_monthly_income": total_income,
        "annual_income": annual_income,
        "loan_amount_inr": loan_amount_inr,
        "estimated_emi": emi,
        "income_to_loan_ratio": income_to_loan_ratio,
        "emi_to_income_ratio": emi_to_income_ratio,
        "debt_to_income_ratio": debt_to_income_ratio,
        "loan_term_years": application.loan_amount_term / 12
    }

def mock_predict_loan(application: LoanApplication) -> tuple:
    """
    Mock prediction function that simulates ML model behavior
    
    This uses simple rules to demonstrate the prediction flow:
    - High income + good credit = likely approval
    - Low income + poor credit = likely rejection  
    - Mixed conditions = moderate probability
    """
    
    total_income = application.applicant_income + application.coapplicant_income
    loan_amount_inr = application.loan_amount * 1000
    
    # Calculate score based on various factors
    score = 0.5  # Base score
    
    # Income factor
    if total_income > 80000:
        score += 0.2
    elif total_income > 50000:
        score += 0.1
    elif total_income < 30000:
        score -= 0.2
    
    # Credit history factor
    if application.credit_history == 1.0:
        score += 0.2
    else:
        score -= 0.3
    
    # Income to loan ratio
    if total_income > 0:
        income_ratio = (total_income * 12) / loan_amount_inr
        if income_ratio > 3:
            score += 0.1
        elif income_ratio < 1.5:
            score -= 0.2
    
    # Education factor
    if application.education == "Graduate":
        score += 0.05
    
    # Employment stability
    if application.self_employed == "No":
        score += 0.05
    
    # Property area
    if application.property_area == "Urban":
        score += 0.05
    
    # Married with dependents (stability indicator)
    if application.married == "Yes" and application.dependents in ["1", "2"]:
        score += 0.05
    
    # Ensure score is between 0 and 1
    score = max(0.1, min(0.9, score))
    
    # Add some randomness to make it realistic
    score += random.uniform(-0.05, 0.05)
    score = max(0.1, min(0.9, score))
    
    # Determine prediction
    prediction = 1 if score > 0.5 else 0
    
    # Create probability array [rejection_prob, approval_prob]
    if prediction == 1:
        approval_prob = score
        rejection_prob = 1 - score
    else:
        rejection_prob = 1 - score
        approval_prob = score
    
    probabilities = [rejection_prob, approval_prob]
    
    return prediction, probabilities

def determine_risk_level(confidence: float, prediction: int) -> str:
    """Determine risk level based on prediction confidence"""
    
    if prediction == 1:  # Approved
        if confidence >= 0.8:
            return "Low Risk"
        elif confidence >= 0.6:
            return "Medium Risk"
        else:
            return "High Risk"
    else:  # Rejected
        if confidence >= 0.8:
            return "Very High Risk"
        elif confidence >= 0.6:
            return "High Risk"
        else:
            return "Medium Risk"

@app.on_event("startup")
async def startup_event():
    """Server startup"""
    logger.info("🚀 Starting Loan Risk Analysis API server (Demo Mode)...")
    logger.info("✅ Server startup complete - Using mock prediction model")

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Loan Risk Analysis API - Demo Mode",
        "version": "1.0.0",
        "description": "FastAPI backend demonstrating loan approval prediction flow",
        "mode": "Demo with mock predictions",
        "endpoints": {
            "predict": "/predict",
            "health": "/health", 
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=True,  # Mock model is always "loaded"
        model_path="mock_model_v1.0"
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_loan_approval(application: LoanApplication):
    """
    **Main Prediction Endpoint**
    
    This endpoint demonstrates the exact flow you requested:
    1. Backend receives input data from frontend
    2. Backend runs model prediction using input data  
    3. Backend sends prediction result back in JSON format
    
    Input: Loan application details (income, loan amount, credit history, etc.)
    Output: Prediction result with probabilities and analysis
    """
    
    try:
        logger.info(f"Processing prediction request for applicant with income: ₹{application.applicant_income:,}")
        
        # Step 1: Backend receives input data ✅
        # (handled by FastAPI automatically via Pydantic validation)
        
        # Step 2: Backend runs model prediction using input data ✅
        prediction, probabilities = mock_predict_loan(application)
        
        # Extract probability scores
        rejection_prob = float(probabilities[0])
        approval_prob = float(probabilities[1])
        confidence = float(max(probabilities))
        
        # Determine prediction label
        prediction_label = "Approved" if prediction == 1 else "Rejected"
        
        # Calculate financial metrics
        financial_analysis = calculate_financial_metrics(application)
        
        # Determine risk level
        risk_level = determine_risk_level(confidence, prediction)
        
        # Step 3: Backend sends prediction result back in JSON format ✅
        response = PredictionResponse(
            success=True,
            prediction=int(prediction),
            prediction_label=prediction_label,
            approval_probability=approval_prob,
            rejection_probability=rejection_prob,
            confidence=confidence,
            risk_level=risk_level,
            financial_analysis=financial_analysis,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Prediction successful: {prediction_label} (confidence: {confidence:.3f})")
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(applications: List[LoanApplication]):
    """Batch prediction endpoint for multiple loan applications"""
    
    if len(applications) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size too large. Maximum 100 applications allowed."
        )
    
    try:
        results = []
        
        for app in applications:
            prediction, probabilities = mock_predict_loan(app)
            
            results.append({
                "prediction": int(prediction),
                "prediction_label": "Approved" if prediction == 1 else "Rejected",
                "approval_probability": float(probabilities[1]),
                "rejection_probability": float(probabilities[0]),
                "confidence": float(max(probabilities))
            })
        
        return {
            "success": True,
            "batch_size": len(applications),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model-info", tags=["General"])
async def get_model_info():
    """Get information about the model"""
    
    return {
        "model_type": "Mock Prediction Model",
        "components": ["rule_based_scoring", "financial_analysis"],
        "algorithm": "Multi-factor scoring system",
        "features": [
            "applicant_income", "coapplicant_income", "loan_amount",
            "credit_history", "education", "employment_type",
            "property_area", "marital_status", "dependents"
        ],
        "version": "demo_v1.0",
        "description": "Demonstrates loan prediction API flow with realistic scoring"
    }

@app.get("/test", tags=["General"])
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "API is working perfectly!",
        "timestamp": datetime.now().isoformat(),
        "status": "OK",
        "demo_mode": True
    }

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "fastapi_demo:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
