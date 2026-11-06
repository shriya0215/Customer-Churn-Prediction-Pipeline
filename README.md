# Customer Churn Prediction Pipeline

An end-to-end, production-ready supervised machine learning pipeline that predicts customer churn for subscription-based businesses. The project encompasses robust data ingestion, missing value remediation, domain-specific feature engineering, multi-model hyperparameter optimization via cross-validation, financial impact analysis, and deployment as a REST API with FastAPI.

---

## 📌 Architecture & Project Structure

```
edquest2/
├── data/
│   └── customer_churn.csv          # Telco Churn benchmark dataset (7,043 records)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Ingestion, schema cleaning, edge-case remediation
│   ├── feature_engineering.py      # Custom Scikit-Learn transformers & preprocessor
│   ├── models.py                   # Model definitions, hyperparameter grids, Stratified 5-Fold CV
│   ├── evaluate.py                 # Multi-metric evaluation, business ROI threshold analysis, plots
│   └── pipeline.py                 # End-to-end training and artifact persistence workflow
├── app/
│   ├── __init__.py
│   ├── schemas.py                  # Pydantic v2 data models with validation & field schemas
│   └── api.py                      # FastAPI REST application (health, metadata, single & batch inference)
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py            # Unit tests for preprocessing & feature engineering
│   └── test_api.py                 # Integration tests for API endpoints
├── artifacts/
│   ├── best_model.joblib           # Serialized end-to-end Scikit-Learn pipeline
│   ├── evaluation_metrics.json     # Quantitative validation metrics across all models
│   ├── pipeline_metadata.json      # Metadata, hyperparameters, and optimal decision threshold
│   └── plots/
│       ├── roc_curves.png          # Comparative Receiver Operating Characteristic curves
│       ├── pr_curves.png           # Comparative Precision-Recall curves
│       └── feature_importance.png  # Top predictive feature weights
├── run_demo.py                     # Interactive verification CLI & API runner
├── requirements.txt                # Pinned production dependencies
└── README.md                       # Comprehensive system documentation
```

---

## 📊 1. Data Cleaning & Preprocessing

- **Dataset**: Telco Customer Churn (7,043 customers, 21 attributes).
- **Target Variable**: `Churn` (`'Yes'` = 26.54%, `'No'` = 73.46%).
- **Edge Cases Handled**:
  - `TotalCharges`: Cleaned 11 occurrences of blank whitespace strings (`" "`) resulting from zero-tenure new accounts; converted to numeric float and imputed to `0.0`.
  - `customerID`: Stripped identifier column to prevent spurious memorization.
  - `SeniorCitizen`: Standardized as binary indicator.
  - `Categorical Encoding`: One-hot encoded categorical variables using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` inside a `ColumnTransformer`.
  - `Numerical Scaling`: Standardized using `StandardScaler` to prevent feature dominance and ensure convergence.

---

## ⚙️ 2. Domain Feature Engineering

The pipeline implements a custom Scikit-learn transformer (`ChurnFeatureEngineer`) that generates domain-specific features:

| Feature Name | Type | Formula / Logic | Business Rationale |
| :--- | :--- | :--- | :--- |
| `tenure_group` | Categorical | Binned: `0-12m`, `13-24m`, `25-48m`, `49-60m`, `61+m` | Customers in the first 12 months exhibit significantly higher churn rates. |
| `avg_monthly_spend` | Numerical | $\frac{\text{TotalCharges}}{\text{tenure} + 1.0}$ | Measures true historical monthly spending compared to nominal contract rate. |
| `num_services` | Numerical | Count of active add-on services | Customers with more bundled services (Phone, Internet, Security, TV) are stickier. |
| `charge_ratio` | Numerical | $\frac{\text{MonthlyCharges}}{\text{TotalCharges} + 1.0}$ | Short-term financial burden relative to historical customer investment. |
| `has_support_services`| Binary | $\text{OnlineSecurity} = 1 \lor \text{TechSupport} = 1$ | High-touch support dramatically suppresses churn risk. |
| `is_month_to_month` | Binary | $\text{Contract} == \text{'Month-to-month'}$ | Month-to-month contracts are the #1 predictor of customer attrition. |

---

## 🏆 3. Model Training & Cross-Validation Tuning

Three distinct model families were trained using 5-fold `StratifiedKFold` cross-validation:

1. **Logistic Regression**: Linear baseline with $L_2$ regularization penalty and balanced class weights.
2. **Random Forest Classifier**: Non-linear bagging ensemble tuned over tree depths and splitting criteria.
3. **Gradient Boosting Classifier**: Sequential boosting ensemble optimizing residual errors over learning rates and tree depths.

### Hyperparameter Search Grids
- **Logistic Regression**: `C`: `[0.1, 1.0, 5.0]`, `class_weight`: `[None, 'balanced']`.
- **Random Forest**: `n_estimators`: `[100]`, `max_depth`: `[6, 10]`, `class_weight`: `[None, 'balanced']`.
- **Gradient Boosting**: `n_estimators`: `[100]`, `learning_rate`: `[0.05, 0.1]`, `max_depth`: `[3, 4]`.

---

## 📈 4. Comparative Model Evaluation

Evaluation on the unseen holdout test set (1,409 customers):

| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Recall @0.5 | Precision @0.5 | F1-Score | Optimal Threshold | Recall @Optimal | Net Business Value ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8483** | **0.8460** | **0.6568** | 0.5267 | 0.6633 | **0.5872** | **0.06** | **98.13%** | **+$57,000** |
| **Gradient Boosting** | 0.8492 | 0.8455 | 0.6618 | 0.5000 | 0.6775 | 0.5754 | 0.05 | 99.20% | +$55,300 |
| **Random Forest** | 0.8470 | 0.8424 | 0.6573 | 0.4733 | 0.6604 | 0.5514 | 0.06 | 98.66% | +$54,400 |

### 💼 Business Impact & Cost-Benefit Analysis
In subscription businesses, the financial penalty of a **False Negative** (losing a customer worth \$500 Customer Lifetime Value) far outweighs a **False Positive** (giving an unnecessary \$50 retention perk to a loyal customer):
- Net TP Value = $+\$250$ (retained customer CLV benefit minus retention incentive cost).
- Cost per FP = $-\$50$ (unnecessary discount).
- Cost per FN = $-\$500$ (unmitigated lost CLV).

By shifting from the arbitrary default threshold of `0.50` to the **data-driven optimal threshold of `0.06`**, the retention team captures **98.13% of all churning customers** (vs. only 52.67% at 0.50), generating **+\$57,000 in net retained revenue** on the test cohort.

---

## 🚀 5. REST API Deployment (FastAPI)

The API is served with FastAPI and Pydantic v2 validation.

### API Endpoints

#### 1. `GET /`
- Returns service status, model status, and Swagger documentation link.

#### 2. `GET /health`
- Liveness and readiness probe reporting whether the pipeline artifact is loaded.

#### 3. `GET /model/metadata`
- Returns active model name, hyperparameter configurations, test ROC-AUC, PR-AUC, and the optimal decision threshold.

#### 4. `POST /predict`
- Predicts churn for an individual customer.
- **Request Body**:
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 94.85,
  "TotalCharges": 189.70
}
```
- **Response**:
```json
{
  "churn_probability": 0.8181,
  "churn_prediction": 1,
  "churn_prediction_optimal": 1,
  "optimal_threshold": 0.06,
  "risk_level": "High Risk",
  "retention_recommendation": "CRITICAL RETENTION ALERT (High Risk). Immediate outreach required: Offer annual contract upgrade with 15% discount + free device protection. Customer in vulnerable early-tenure cohort (<12m): Schedule check-in call from Customer Success. Bundle priority technical assistance.",
  "model_name": "Logistic Regression"
}
```

#### 5. `POST /predict/batch`
- Vectorized batch endpoint scoring hundreds or thousands of customer records simultaneously. Returns total records, predicted churner count, aggregate cohort churn rate, and itemized customer predictions.

---

## 🧪 6. Execution & Verification Guide

### 1. Run the Training Pipeline
Trains all 3 models with 5-fold CV, runs business threshold optimization, generates visual plots, and saves artifacts:
```bash
python -m src.pipeline
```

### 2. Run the Unit & Integration Test Suite
Executes all 10 automated test cases (data cleaning, feature engineering, artifact loading, API schemas, single & batch inference):
```bash
python -m pytest -v
```

### 3. Run the End-to-End Verification Demo
Runs model inference against high-risk and low-risk customer profiles:
```bash
python run_demo.py
```

### 4. Start the Production API Server
To launch the FastAPI server and view interactive Swagger UI:
```bash
python run_demo.py --serve --port 8000
```
Open your browser to:
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`
