"""
Unit tests for data cleaning, feature engineering, and pipeline integrity.
"""

import os
import pytest
import pandas as pd
import numpy as np
import joblib

from src.data_loader import clean_churn_data
from src.feature_engineering import ChurnFeatureEngineer, build_preprocessor


@pytest.fixture
def sample_raw_dataframe():
    """Create a realistic sample DataFrame for testing."""
    return pd.DataFrame([
        {
            "customerID": "0001-TEST1",
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
            "TotalCharges": "29.85",
            "Churn": "No",
        },
        {
            "customerID": "0002-TEST2",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 0,  # Zero tenure edge case
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 115.0,
            "TotalCharges": " ",  # Whitespace edge case
            "Churn": "Yes",
        },
    ])


def test_clean_churn_data(sample_raw_dataframe):
    """Test data cleaning removes customerID, handles spaces in TotalCharges, and encodes target."""
    X, y = clean_churn_data(sample_raw_dataframe)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert len(X) == 2
    assert len(y) == 2
    assert y.tolist() == [0, 1]
    # Verify TotalCharges converted to numeric and whitespace imputed to 0.0
    assert X["TotalCharges"].iloc[1] == 0.0
    assert np.issubdtype(X["TotalCharges"].dtype, np.number)


def test_feature_engineering_transformer(sample_raw_dataframe):
    """Test custom feature engineering transformer generates all required features."""
    X, _ = clean_churn_data(sample_raw_dataframe)
    fe = ChurnFeatureEngineer()
    X_fe = fe.transform(X)

    # 1. Check tenure_group
    assert "tenure_group" in X_fe.columns
    assert X_fe["tenure_group"].iloc[0] == "0-12m"

    # 2. Check avg_monthly_spend
    assert "avg_monthly_spend" in X_fe.columns
    assert X_fe["avg_monthly_spend"].iloc[0] == pytest.approx(29.85 / 2.0)

    # 3. Check num_services count
    assert "num_services" in X_fe.columns
    # Row 0 has only OnlineBackup == 'Yes' -> num_services = 1
    assert X_fe["num_services"].iloc[0] == 1
    # Row 1 has PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies -> 8
    assert X_fe["num_services"].iloc[1] == 8

    # 4. Check charge_ratio
    assert "charge_ratio" in X_fe.columns
    assert X_fe["charge_ratio"].iloc[0] > 0

    # 5. Check has_support_services
    assert "has_support_services" in X_fe.columns
    assert X_fe["has_support_services"].iloc[0] == 0
    assert X_fe["has_support_services"].iloc[1] == 1

    # 6. Check is_month_to_month
    assert "is_month_to_month" in X_fe.columns
    assert X_fe["is_month_to_month"].iloc[0] == 1
    assert X_fe["is_month_to_month"].iloc[1] == 0


def test_preprocessor_fit_transform(sample_raw_dataframe):
    """Test full preprocessor builds expected numerical matrix."""
    X, _ = clean_churn_data(sample_raw_dataframe)
    preprocessor, num_cols, cat_cols = build_preprocessor()

    matrix = preprocessor.fit_transform(X)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == len(X)
    assert matrix.shape[1] > len(num_cols) + len(cat_cols)  # One-hot expansion


def test_saved_model_artifact():
    """Verify saved best model joblib can be loaded and make valid predictions."""
    model_path = "artifacts/best_model.joblib"
    assert os.path.exists(model_path), "Model artifact does not exist"

    pipeline = joblib.load(model_path)
    assert hasattr(pipeline, "predict_proba")

    # Predict on test record
    test_sample = pd.DataFrame([{
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
    }])

    probas = pipeline.predict_proba(test_sample)
    assert probas.shape == (1, 2)
    assert 0.0 <= probas[0, 1] <= 1.0
