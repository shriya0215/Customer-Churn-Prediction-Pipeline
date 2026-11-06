"""
Feature Engineering and Preprocessing Pipeline for Customer Churn Prediction.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for churn domain-specific feature engineering.
    Engineers:
    1. tenure_group: Binned tenure cohorts ('0-12m', '13-24m', '25-48m', '49-60m', '61+m')
    2. avg_monthly_spend: Historical average spend TotalCharges / (tenure + 1)
    3. num_services: Count of active subscriptions/services
    4. charge_ratio: MonthlyCharges / (TotalCharges + 1.0)
    5. has_support_services: Binary flag for OnlineSecurity or TechSupport
    6. is_month_to_month: Binary flag for Month-to-month contract
    """

    def __init__(self):
        self.service_columns = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Ensure numeric columns are properly typed
        if "tenure" in X.columns:
            X["tenure"] = pd.to_numeric(X["tenure"], errors="coerce").fillna(0)
        if "MonthlyCharges" in X.columns:
            X["MonthlyCharges"] = pd.to_numeric(X["MonthlyCharges"], errors="coerce").fillna(0.0)
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)

        # 1. Feature: tenure_group
        if "tenure" in X.columns:
            bins = [-np.inf, 12, 24, 48, 60, np.inf]
            labels = ["0-12m", "13-24m", "25-48m", "49-60m", "61+m"]
            X["tenure_group"] = pd.cut(X["tenure"], bins=bins, labels=labels).astype(str)

        # 2. Feature: avg_monthly_spend
        if "TotalCharges" in X.columns and "tenure" in X.columns:
            X["avg_monthly_spend"] = X["TotalCharges"] / (X["tenure"] + 1.0)

        # 3. Feature: num_services (Count of active services)
        active_services_count = pd.Series(0, index=X.index)
        for col in self.service_columns:
            if col in X.columns:
                active_services_count += (X[col].astype(str) == "Yes").astype(int)
        X["num_services"] = active_services_count

        # 4. Feature: charge_ratio (Monthly burden vs total investment)
        if "MonthlyCharges" in X.columns and "TotalCharges" in X.columns:
            X["charge_ratio"] = X["MonthlyCharges"] / (X["TotalCharges"] + 1.0)

        # 5. Feature: has_support_services (Tech support or Online security)
        has_security = X["OnlineSecurity"].astype(str) == "Yes" if "OnlineSecurity" in X.columns else False
        has_tech = X["TechSupport"].astype(str) == "Yes" if "TechSupport" in X.columns else False
        X["has_support_services"] = (has_security | has_tech).astype(int)

        # 6. Feature: is_month_to_month
        if "Contract" in X.columns:
            X["is_month_to_month"] = (X["Contract"].astype(str) == "Month-to-month").astype(int)

        return X


def build_preprocessor() -> Tuple[Pipeline, List[str], List[str]]:
    """
    Creates a full preprocessing pipeline that:
    1. Applies ChurnFeatureEngineer
    2. Applies ColumnTransformer:
       - Standardizes numerical features
       - One-hot encodes categorical features
    """
    numerical_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "avg_monthly_spend",
        "num_services",
        "charge_ratio",
    ]

    categorical_features = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "tenure_group",
        "has_support_services",
        "is_month_to_month",
    ]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    full_preprocessor = Pipeline([
        ("feature_engineer", ChurnFeatureEngineer()),
        ("col_transformer", col_transformer),
    ])

    return full_preprocessor, numerical_features, categorical_features
