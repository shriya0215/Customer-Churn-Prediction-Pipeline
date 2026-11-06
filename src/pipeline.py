"""
End-to-end training and deployment pipeline orchestration.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_raw_data, clean_churn_data
from src.feature_engineering import build_preprocessor
from src.models import get_candidate_models, tune_and_train_model
from src.evaluate import (
    compute_metrics,
    compute_business_impact,
    plot_evaluation_curves,
    plot_feature_importance,
)


def get_feature_names(preprocessor, X_sample: pd.DataFrame) -> list:
    """Extracts output feature names from the fitted preprocessor."""
    try:
        # Fit on sample to learn categories if needed
        fe = preprocessor.named_steps["feature_engineer"]
        ct = preprocessor.named_steps["col_transformer"]
        X_trans = fe.transform(X_sample)
        ct.fit(X_trans)
        return list(ct.get_feature_names_out())
    except Exception as e:
        print(f"Warning extracting feature names: {e}")
        return []


def run_pipeline(
    data_path: str = "data/customer_churn.csv",
    artifacts_dir: str = "artifacts",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Executes the complete machine learning lifecycle:
    1. Ingestion & Data Cleaning
    2. Stratified Train/Test Split
    3. Feature Engineering & Preprocessor Definition
    4. Candidate Model Training & Cross-Validation Tuning
    5. Comprehensive Model Evaluation & Business Impact Analysis
    6. Selection of Best Performer
    7. Artifact Generation and Persistence
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    plots_dir = os.path.join(artifacts_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Ingesting and Cleaning Data")
    print("=" * 70)
    raw_df = load_raw_data(data_path)
    X, y = clean_churn_data(raw_df)
    print(f"Total dataset shape: {X.shape}, Target distribution: {np.bincount(y)}")

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    print("\n" + "=" * 70)
    print("STEP 2: Building Feature Engineering & Preprocessing Pipeline")
    print("=" * 70)
    preprocessor, num_cols, cat_cols = build_preprocessor()
    print(f"Numerical columns ({len(num_cols)}): {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    print("\n" + "=" * 70)
    print("STEP 3: Model Training & Hyperparameter Tuning (5-Fold CV)")
    print("=" * 70)
    candidate_models = get_candidate_models()
    trained_models = {}
    cv_results_summary = {}

    for name, (estimator, param_grid) in candidate_models.items():
        print(f"\n--> Tuning and Training: {name}...")
        # Fresh preprocessor clone for each model
        model_preprocessor, _, _ = build_preprocessor()
        best_pipeline, cv_res = tune_and_train_model(
            model_name=name,
            base_estimator=estimator,
            param_grid=param_grid,
            preprocessor=model_preprocessor,
            X_train=X_train,
            y_train=y_train,
            cv_folds=5,
            scoring="roc_auc",
        )
        trained_models[name] = best_pipeline
        cv_results_summary[name] = cv_res
        print(f"    Best CV ROC-AUC: {cv_res['best_cv_score']:.4f}")
        print(f"    Best Params: {cv_res['best_params']}")

    print("\n" + "=" * 70)
    print("STEP 4: Test Set Evaluation & Business Impact Analysis")
    print("=" * 70)
    models_evaluation = {}
    summary_table = []

    for name, pipeline in trained_models.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred_default = (y_proba >= 0.5).astype(int)

        metrics = compute_metrics(y_test, y_pred_default, y_proba)
        best_thresh, max_val, curve_data = compute_business_impact(y_test.values, y_proba)

        # Predictions at optimal business threshold
        y_pred_optimal = (y_proba >= best_thresh).astype(int)
        opt_metrics = compute_metrics(y_test, y_pred_optimal, y_proba)

        models_evaluation[name] = {
            "y_true": y_test.values,
            "y_proba": y_proba,
            "metrics": metrics,
            "optimal_threshold": round(best_thresh, 2),
            "optimal_business_value": max_val,
            "optimal_metrics": opt_metrics,
            "best_params": cv_results_summary[name]["best_params"],
            "cv_score": cv_results_summary[name]["best_cv_score"],
        }

        summary_table.append({
            "Model": name,
            "CV ROC-AUC": round(cv_results_summary[name]["best_cv_score"], 4),
            "Test ROC-AUC": round(metrics["roc_auc"], 4),
            "Test PR-AUC": round(metrics["pr_auc"], 4),
            "Recall @0.5": round(metrics["recall"], 4),
            "Precision @0.5": round(metrics["precision"], 4),
            "F1 @0.5": round(metrics["f1_score"], 4),
            "Optimal Thresh": round(best_thresh, 2),
            "Recall @Opt": round(opt_metrics["recall"], 4),
            "Max Net Value ($)": round(max_val, 2),
        })

    summary_df = pd.DataFrame(summary_table)
    print(summary_df.to_string(index=False))

    # Select Best Model based on highest Business Net Value & ROC-AUC
    # We rank by business value first, then test ROC-AUC
    summary_df_sorted = summary_df.sort_values(
        by=["Max Net Value ($)", "Test ROC-AUC"], ascending=[False, False]
    )
    best_model_name = summary_df_sorted.iloc[0]["Model"]
    best_pipeline = trained_models[best_model_name]
    best_eval = models_evaluation[best_model_name]

    print("\n" + "=" * 70)
    print(f"STEP 5: Selected Best Model: {best_model_name}")
    print("=" * 70)
    print(f"Reason: Maximizes customer retention ROI ($ {best_eval['optimal_business_value']:,.2f})")
    print(f"Optimal decision threshold: {best_eval['optimal_threshold']}")
    print(f"ROC-AUC: {best_eval['metrics']['roc_auc']:.4f}, PR-AUC: {best_eval['metrics']['pr_auc']:.4f}")

    print("\n" + "=" * 70)
    print("STEP 6: Generating Diagnostic Visualizations")
    print("=" * 70)
    plot_evaluation_curves(models_evaluation, output_dir=plots_dir)
    print(f"Saved ROC and PR curves to {plots_dir}")

    # Extract feature names from preprocessor and plot feature importances
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    ct = fitted_preprocessor.named_steps["col_transformer"]
    feature_names = list(ct.get_feature_names_out())
    plot_feature_importance(best_pipeline, feature_names, output_dir=plots_dir, top_n=15)
    print(f"Saved feature importance chart to {plots_dir}")

    print("\n" + "=" * 70)
    print("STEP 7: Persisting Production Artifacts")
    print("=" * 70)
    # Save best end-to-end model pipeline
    model_artifact_path = os.path.join(artifacts_dir, "best_model.joblib")
    joblib.dump(best_pipeline, model_artifact_path)
    print(f"Saved full best model pipeline to: {model_artifact_path}")

    # Save evaluation metrics json
    eval_metrics_serializable = {}
    for name, data in models_evaluation.items():
        eval_metrics_serializable[name] = {
            "cv_score": data["cv_score"],
            "metrics": data["metrics"],
            "optimal_threshold": data["optimal_threshold"],
            "optimal_business_value": data["optimal_business_value"],
            "optimal_metrics": data["optimal_metrics"],
            "best_params": data["best_params"],
        }

    metrics_file = os.path.join(artifacts_dir, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(eval_metrics_serializable, f, indent=4)
    print(f"Saved evaluation metrics to: {metrics_file}")

    # Save metadata for API
    metadata = {
        "model_name": best_model_name,
        "optimal_threshold": best_eval["optimal_threshold"],
        "default_threshold": 0.5,
        "test_roc_auc": best_eval["metrics"]["roc_auc"],
        "test_pr_auc": best_eval["metrics"]["pr_auc"],
        "test_recall_at_optimal": best_eval["optimal_metrics"]["recall"],
        "test_precision_at_optimal": best_eval["optimal_metrics"]["precision"],
        "best_hyperparameters": best_eval["best_params"],
        "feature_names": feature_names,
        "input_features": list(X.columns),
    }
    meta_file = os.path.join(artifacts_dir, "pipeline_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to: {meta_file}")

    return {
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "metadata": metadata,
        "eval_metrics": eval_metrics_serializable,
    }


if __name__ == "__main__":
    run_pipeline()
