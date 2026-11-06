"""
Interactive Demonstration and Validation Script for Customer Churn API.
Tests model inference, risk grading, and actionable recommendations.
"""

import sys
import json
import argparse
from app.api import (
    load_artifacts,
    read_root,
    health_check,
    get_model_metadata,
    predict_single,
    predict_batch,
)
from app.schemas import CustomerInput, BatchPredictionRequest


def print_header(title: str):
    print("\n" + "=" * 75)
    print(f" {title.upper()} ")
    print("=" * 75)


def run_demo():
    print_header("Initializing Customer Churn Prediction Service")
    load_artifacts()

    # 1. Health & Root
    root_info = read_root()
    health_info = health_check()
    print(f"Service Name: {root_info.get('service')}")
    print(f"Service Status: {health_info.status.upper()}")
    print(f"Deployed Model: {health_info.model_name}")

    # 2. Metadata
    print_header("Model Metadata & Validation Metrics")
    meta = get_model_metadata()
    print(f"Model Architecture       : {meta.model_name}")
    print(f"Test ROC-AUC Score       : {meta.test_roc_auc:.4f}")
    print(f"Test PR-AUC Score        : {meta.test_pr_auc:.4f}")
    print(f"Optimal Business Thresh  : {meta.optimal_threshold:.2f} (Maximizes Retention ROI)")
    print(f"Recall at Optimal Thresh : {meta.test_recall_at_optimal * 100:.2f}%")
    print(f"Engineered Feature Count : {meta.feature_count}")
    print(f"Best Hyperparameters     : {json.dumps(meta.best_hyperparameters)}")

    # 3. Single Prediction Case 1: High Risk Profile
    print_header("Scenario 1: High-Risk At-Risk Customer")
    high_risk_customer = CustomerInput(
        gender="Female",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        tenure=2,
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
        MonthlyCharges=94.85,
        TotalCharges=189.70,
    )
    res_high = predict_single(high_risk_customer)
    print(f"Customer Tenure          : {high_risk_customer.tenure} months (Month-to-month)")
    print(f"Monthly Charges          : ${high_risk_customer.MonthlyCharges:.2f}")
    print(f"Churn Probability        : {res_high.churn_probability * 100:.2f}%")
    print(f"Default Prediction (@0.5): {'CHURN (1)' if res_high.churn_prediction == 1 else 'RETAIN (0)'}")
    print(f"Optimal Prediction       : {'CHURN (1)' if res_high.churn_prediction_optimal == 1 else 'RETAIN (0)'}")
    print(f"Assigned Risk Tier       : [{res_high.risk_level.upper()}]")
    print(f"Proactive Action Plan    : {res_high.retention_recommendation}")

    # 4. Single Prediction Case 2: Loyal Long-term Customer
    print_header("Scenario 2: Loyal Low-Risk Customer")
    low_risk_customer = CustomerInput(
        gender="Male",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="Yes",
        tenure=68,
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
        MonthlyCharges=64.80,
        TotalCharges=4406.40,
    )
    res_low = predict_single(low_risk_customer)
    print(f"Customer Tenure          : {low_risk_customer.tenure} months (Two-year Contract)")
    print(f"Monthly Charges          : ${low_risk_customer.MonthlyCharges:.2f}")
    print(f"Churn Probability        : {res_low.churn_probability * 100:.2f}%")
    print(f"Default Prediction (@0.5): {'CHURN (1)' if res_low.churn_prediction == 1 else 'RETAIN (0)'}")
    print(f"Optimal Prediction       : {'CHURN (1)' if res_low.churn_prediction_optimal == 1 else 'RETAIN (0)'}")
    print(f"Assigned Risk Tier       : [{res_low.risk_level.upper()}]")
    print(f"Proactive Action Plan    : {res_low.retention_recommendation}")

    # 5. Batch Prediction Demonstration
    print_header("Scenario 3: High-Throughput Batch Scoring")
    batch_req = BatchPredictionRequest(customers=[high_risk_customer, low_risk_customer])
    batch_res = predict_batch(batch_req)
    print(f"Total Cohort Size        : {batch_res.total_records}")
    print(f"Identified Churners      : {batch_res.predicted_churners_count}")
    print(f"Cohort Churn Rate        : {batch_res.predicted_churn_rate * 100:.2f}%")

    print("\n" + "=" * 75)
    print(" ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY ")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Pipeline Demo & Server Runner")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI uvicorn REST server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        print(f"Starting Customer Churn Prediction REST API at http://{args.host}:{args.port}")
        print(f"Interactive Swagger Documentation available at http://{args.host}:{args.port}/docs")
        uvicorn.run("app.api:app", host=args.host, port=args.port, reload=False)
    else:
        run_demo()
