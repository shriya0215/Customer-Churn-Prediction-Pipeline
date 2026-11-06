"""
Integration tests for FastAPI REST endpoints.
"""

import pytest

from app.api import (
    app,
    load_artifacts,
    read_root,
    health_check,
    get_model_metadata,
    predict_single,
    predict_batch,
)
from app.schemas import CustomerInput, BatchPredictionRequest


@pytest.fixture(autouse=True)
def setup_artifacts():
    load_artifacts()


def test_root_endpoint_direct():
    """Test GET / returns valid service details."""
    res = read_root()
    assert res["status"] == "online"
    assert res["service"] == "Customer Churn Prediction API"
    assert res["model_loaded"] is True


def test_health_endpoint_direct():
    """Test GET /health returns healthy status."""
    res = health_check()
    assert res.status == "healthy"
    assert res.model_loaded is True
    assert res.model_name is not None


def test_model_metadata_endpoint_direct():
    """Test GET /model/metadata returns valid model evaluation information."""
    res = get_model_metadata()
    assert res.model_name is not None
    assert res.optimal_threshold > 0.0
    assert res.test_roc_auc > 0.70
    assert res.feature_count > 0


def test_predict_single_high_risk_customer():
    """Test /predict with a customer with high churn characteristics."""
    high_risk_input = CustomerInput(
        gender="Female",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        tenure=1,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=95.50,
        TotalCharges=95.50,
    )

    res = predict_single(high_risk_input)
    assert 0.0 <= res.churn_probability <= 1.0
    assert res.churn_probability > 0.40
    assert res.risk_level in ["Medium Risk", "High Risk"]
    assert len(res.retention_recommendation) > 10
    assert res.optimal_threshold > 0.0


def test_predict_single_low_risk_customer():
    """Test /predict with a customer with low churn characteristics."""
    low_risk_input = CustomerInput(
        gender="Male",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="Yes",
        tenure=65,
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="Yes",
        DeviceProtection="Yes",
        TechSupport="Yes",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Two year",
        PaperlessBilling="No",
        PaymentMethod="Credit card (automatic)",
        MonthlyCharges=60.0,
        TotalCharges=3900.0,
    )

    res = predict_single(low_risk_input)
    assert 0.0 <= res.churn_probability <= 1.0
    assert res.churn_probability < 0.20
    assert res.churn_prediction == 0
    assert res.risk_level == "Low Risk"


def test_predict_batch_customers():
    """Test /predict/batch returns bulk inference and cohort metrics."""
    batch_req = BatchPredictionRequest(
        customers=[
            CustomerInput(
                gender="Female",
                SeniorCitizen=0,
                Partner="No",
                Dependents="No",
                tenure=1,
                PhoneService="Yes",
                MultipleLines="No",
                InternetService="Fiber optic",
                OnlineSecurity="No",
                OnlineBackup="No",
                DeviceProtection="No",
                TechSupport="No",
                StreamingTV="Yes",
                StreamingMovies="Yes",
                Contract="Month-to-month",
                PaperlessBilling="Yes",
                PaymentMethod="Electronic check",
                MonthlyCharges=95.50,
                TotalCharges=95.50,
            ),
            CustomerInput(
                gender="Male",
                SeniorCitizen=0,
                Partner="Yes",
                Dependents="Yes",
                tenure=65,
                PhoneService="Yes",
                MultipleLines="Yes",
                InternetService="DSL",
                OnlineSecurity="Yes",
                OnlineBackup="Yes",
                DeviceProtection="Yes",
                TechSupport="Yes",
                StreamingTV="No",
                StreamingMovies="No",
                Contract="Two year",
                PaperlessBilling="No",
                PaymentMethod="Credit card (automatic)",
                MonthlyCharges=60.0,
                TotalCharges=3900.0,
            ),
        ]
    )

    res = predict_batch(batch_req)
    assert res.total_records == 2
    assert len(res.predictions) == 2
    assert 0.0 <= res.predicted_churn_rate <= 1.0
