"""
FastAPI REST API for Customer Churn Prediction.
Exposes endpoints for health checking, model metadata, single and batch predictions.
"""

import os
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse,
)

# Global variables for model and metadata
MODEL_PIPELINE = None
MODEL_METADATA: Dict[str, Any] = {}
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "pipeline_metadata.json")


def load_artifacts():
    """Load model pipeline and metadata from disk."""
    global MODEL_PIPELINE, MODEL_METADATA

    if os.path.exists(MODEL_PATH):
        try:
            MODEL_PIPELINE = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            MODEL_PIPELINE = None
    else:
        MODEL_PIPELINE = None

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                MODEL_METADATA = json.load(f)
        except Exception as e:
            print(f"Error loading metadata from {METADATA_PATH}: {e}")
            MODEL_METADATA = {}
    else:
        MODEL_METADATA = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to manage loading model on startup."""
    load_artifacts()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-ready machine learning service for predicting customer churn in subscription businesses. "
        "Engineered with Scikit-learn, FastAPI, and data-driven business impact thresholds."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


def get_retention_recommendation(probability: float, customer: dict, optimal_thresh: float) -> str:
    """
    Generate tailored, actionable retention recommendations based on risk score and customer drivers.
    """
    contract = customer.get("Contract", "")
    has_tech = customer.get("TechSupport", "") == "Yes"
    tenure = customer.get("tenure", 0)

    if probability < 0.25:
        return "Customer is highly satisfied with low churn risk. Maintain regular service notifications and standard engagement."
    elif probability < optimal_thresh:
        recs = ["Moderate churn risk."]
        if contract == "Month-to-month":
            recs.append("Offer 1-year contract with 10% loyalty discount.")
        if not has_tech:
            recs.append("Promote complimentary 2-month tech support add-on.")
        return " ".join(recs)
    else:
        recs = ["CRITICAL RETENTION ALERT (High Risk)."]
        if contract == "Month-to-month":
            recs.append("Immediate outreach required: Offer annual contract upgrade with 15% discount + free device protection.")
        if tenure < 12:
            recs.append("Customer in vulnerable early-tenure cohort (<12m): Schedule check-in call from Customer Success.")
        if not has_tech:
            recs.append("Bundle priority technical assistance.")
        return " ".join(recs)


@app.get("/", tags=["General"])
def read_root():
    """Root endpoint providing service overview and documentation links."""
    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "model_loaded": MODEL_PIPELINE is not None,
        "model_name": MODEL_METADATA.get("model_name", "Not loaded"),
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check endpoint to verify API and model readiness."""
    is_loaded = MODEL_PIPELINE is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=MODEL_METADATA.get("model_name"),
        artifacts_path=ARTIFACTS_DIR,
    )


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Monitoring"])
def get_model_metadata():
    """Retrieves metadata, validation metrics, and optimal decision threshold for the active model."""
    if not MODEL_METADATA:
        # Try reloading in case it was created recently
        load_artifacts()
    if not MODEL_METADATA:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is not available. Please run the training pipeline first.",
        )

    return ModelMetadataResponse(
        model_name=MODEL_METADATA.get("model_name", "Unknown"),
        optimal_threshold=MODEL_METADATA.get("optimal_threshold", 0.5),
        default_threshold=MODEL_METADATA.get("default_threshold", 0.5),
        test_roc_auc=MODEL_METADATA.get("test_roc_auc", 0.0),
        test_pr_auc=MODEL_METADATA.get("test_pr_auc", 0.0),
        test_recall_at_optimal=MODEL_METADATA.get("test_recall_at_optimal", 0.0),
        test_precision_at_optimal=MODEL_METADATA.get("test_precision_at_optimal", 0.0),
        best_hyperparameters=MODEL_METADATA.get("best_hyperparameters", {}),
        feature_count=len(MODEL_METADATA.get("feature_names", [])),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(customer: CustomerInput):
    """
    Predict churn for a single customer.
    Returns:
    - churn_probability: continuous probability [0.0, 1.0]
    - churn_prediction: binary prediction at 0.5 threshold
    - churn_prediction_optimal: binary prediction at business-optimal threshold
    - risk_level: categorical tier (Low, Medium, High)
    - retention_recommendation: personalized proactive retention strategy
    """
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        load_artifacts()
    if MODEL_PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train the pipeline first.",
        )

    customer_dict = customer.model_dump()
    df_input = pd.DataFrame([customer_dict])

    try:
        # Run inference through full pipeline
        proba = float(MODEL_PIPELINE.predict_proba(df_input)[0, 1])
        optimal_thresh = float(MODEL_METADATA.get("optimal_threshold", 0.5))

        pred_default = int(proba >= 0.5)
        pred_optimal = int(proba >= optimal_thresh)

        if proba < 0.25:
            risk = "Low Risk"
        elif proba < optimal_thresh:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        recommendation = get_retention_recommendation(proba, customer_dict, optimal_thresh)

        return PredictionResponse(
            churn_probability=round(proba, 4),
            churn_prediction=pred_default,
            churn_prediction_optimal=pred_optimal,
            optimal_threshold=round(optimal_thresh, 2),
            risk_level=risk,
            retention_recommendation=recommendation,
            model_name=MODEL_METADATA.get("model_name", "Classifier"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for a batch of customers.
    High-throughput batch endpoint returning predictions and cohort-level summary statistics.
    """
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        load_artifacts()
    if MODEL_PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train the pipeline first.",
        )

    if not request.customers:
        return BatchPredictionResponse(
            total_records=0,
            predicted_churners_count=0,
            predicted_churn_rate=0.0,
            predictions=[],
        )

    records = [c.model_dump() for c in request.customers]
    df_batch = pd.DataFrame(records)

    try:
        probas = MODEL_PIPELINE.predict_proba(df_batch)[:, 1]
        optimal_thresh = float(MODEL_METADATA.get("optimal_threshold", 0.5))

        predictions: List[PredictionResponse] = []
        churner_count = 0

        for i, proba in enumerate(probas):
            p = float(proba)
            pred_default = int(p >= 0.5)
            pred_optimal = int(p >= optimal_thresh)
            if pred_optimal == 1:
                churner_count += 1

            if p < 0.25:
                risk = "Low Risk"
            elif p < optimal_thresh:
                risk = "Medium Risk"
            else:
                risk = "High Risk"

            rec = get_retention_recommendation(p, records[i], optimal_thresh)

            predictions.append(
                PredictionResponse(
                    churn_probability=round(p, 4),
                    churn_prediction=pred_default,
                    churn_prediction_optimal=pred_optimal,
                    optimal_threshold=round(optimal_thresh, 2),
                    risk_level=risk,
                    retention_recommendation=rec,
                    model_name=MODEL_METADATA.get("model_name", "Classifier"),
                )
            )

        total = len(predictions)
        return BatchPredictionResponse(
            total_records=total,
            predicted_churners_count=churner_count,
            predicted_churn_rate=round(churner_count / total, 4) if total > 0 else 0.0,
            predictions=predictions,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch inference error: {str(e)}",
        )
