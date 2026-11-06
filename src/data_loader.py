"""
Data loading and initial cleaning module for Customer Churn Prediction.
"""

from typing import Tuple
import pandas as pd
import numpy as np


def load_raw_data(filepath: str = "data/customer_churn.csv") -> pd.DataFrame:
    """Load the raw CSV data."""
    df = pd.read_csv(filepath)
    return df


def clean_churn_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean the dataset:
    - Drop customerID if present
    - Convert TotalCharges from string to numeric (handles whitespace strings)
    - Handle missing values in TotalCharges
    - Map target variable Churn ('Yes'/'No') to binary (1/0)
    
    Returns:
        X: DataFrame of feature columns
        y: Series of binary target values
    """
    df = df.copy()

    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Clean TotalCharges: convert whitespace or invalid strings to NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Customers with tenure == 0 have NaN TotalCharges; set TotalCharges to 0.0 or MonthlyCharges
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Clean SeniorCitizen to string category or int
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Encode target if present
    y = None
    if "Churn" in df.columns:
        y = df["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).astype(int)
        X = df.drop(columns=["Churn"])
    else:
        X = df

    return X, y
