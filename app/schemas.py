"""
Pydantic data schemas for request validation and response formatting.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class CustomerInput(BaseModel):
    """
    Schema representing customer subscription and billing attributes.
    """
    gender: str = Field(default="Female", description="Gender of customer ('Male', 'Female')")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="Whether customer is a senior citizen (1, 0)")
    Partner: str = Field(default="No", description="Whether customer has a partner ('Yes', 'No')")
    Dependents: str = Field(default="No", description="Whether customer has dependents ('Yes', 'No')")
    tenure: int = Field(default=1, ge=0, description="Number of months customer has stayed with company")
    PhoneService: str = Field(default="Yes", description="Whether customer has phone service ('Yes', 'No')")
    MultipleLines: str = Field(default="No", description="Whether customer has multiple lines ('Yes', 'No', 'No phone service')")
    InternetService: str = Field(default="DSL", description="Customer's internet service provider ('DSL', 'Fiber optic', 'No')")
    OnlineSecurity: str = Field(default="No", description="Whether customer has online security ('Yes', 'No', 'No internet service')")
    OnlineBackup: str = Field(default="No", description="Whether customer has online backup ('Yes', 'No', 'No internet service')")
    DeviceProtection: str = Field(default="No", description="Whether customer has device protection ('Yes', 'No', 'No internet service')")
    TechSupport: str = Field(default="No", description="Whether customer has tech support ('Yes', 'No', 'No internet service')")
    StreamingTV: str = Field(default="No", description="Whether customer has streaming TV ('Yes', 'No', 'No internet service')")
    StreamingMovies: str = Field(default="No", description="Whether customer has streaming movies ('Yes', 'No', 'No internet service')")
    Contract: str = Field(default="Month-to-month", description="Contract term ('Month-to-month', 'One year', 'Two year')")
    PaperlessBilling: str = Field(default="Yes", description="Whether customer has paperless billing ('Yes', 'No')")
    PaymentMethod: str = Field(default="Electronic check", description="Payment method ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')")
    MonthlyCharges: float = Field(default=29.85, ge=0.0, description="The amount charged to the customer monthly")
    TotalCharges: Optional[float] = Field(default=None, description="The total amount charged to the customer (computed if None)")

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def set_total_charges(cls, v, info):
        if v is None or v == "" or v == " ":
            # Default to MonthlyCharges or 0.0
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    model_config = {
        "json_schema_extra": {
            "example": {
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
    }


class PredictionResponse(BaseModel):
    """
    Schema for single customer prediction response.
    """
    churn_probability: float = Field(..., description="Estimated probability of churning (0.0 to 1.0)")
    churn_prediction: int = Field(..., description="Binary churn prediction at default 0.5 threshold (1 = Churn, 0 = Retain)")
    churn_prediction_optimal: int = Field(..., description="Binary churn prediction at business-optimal decision threshold")
    optimal_threshold: float = Field(..., description="Business-optimal decision threshold that maximizes retention ROI")
    risk_level: str = Field(..., description="Categorical risk tier: 'Low Risk', 'Medium Risk', or 'High Risk'")
    retention_recommendation: str = Field(..., description="Actionable retention recommendation for customer success team")
    model_name: str = Field(..., description="Name of the model deployed")


class BatchPredictionRequest(BaseModel):
    """
    Schema for batch prediction request.
    """
    customers: List[CustomerInput] = Field(..., description="List of customer records to evaluate")


class BatchPredictionResponse(BaseModel):
    """
    Schema for batch prediction response.
    """
    total_records: int
    predicted_churners_count: int
    predicted_churn_rate: float
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    artifacts_path: str


class ModelMetadataResponse(BaseModel):
    model_name: str
    optimal_threshold: float
    default_threshold: float
    test_roc_auc: float
    test_pr_auc: float
    test_recall_at_optimal: float
    test_precision_at_optimal: float
    best_hyperparameters: dict
    feature_count: int
