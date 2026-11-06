"""
Model definitions, training, and hyperparameter tuning with Cross-Validation.
"""

from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def get_candidate_models() -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """
    Returns candidate models and their hyperparameter grids for tuning.
    Includes:
    1. Logistic Regression (Linear baseline with regularization and class weighting)
    2. Random Forest (Bagging ensemble with tree depth & class weight tuning)
    3. Gradient Boosting (Sequential boosting ensemble with learning rate tuning)
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {
                "classifier__C": [0.1, 1.0, 5.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [6, 10],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "classifier__n_estimators": [100],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 4],
            },
        ),
    }
    return models


def tune_and_train_model(
    model_name: str,
    base_estimator: Any,
    param_grid: Dict[str, list],
    preprocessor: Pipeline,
    X_train,
    y_train,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Constructs a full Pipeline (preprocessor + classifier),
    tunes hyperparameters using StratifiedKFold cross-validation,
    and returns the best estimator pipeline and cv metrics.
    """
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", base_estimator),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        refit=True,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    results = {
        "model_name": model_name,
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
        "scoring_metric": scoring,
    }

    return best_pipeline, results
