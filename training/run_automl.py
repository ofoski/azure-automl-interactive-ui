"""AutoML helpers for the Streamlit app."""

import pandas as pd

from ml_pipeline import get_ml_client, run_automl_job
from ml_pipeline.data import register_train_test_data

PRIMARY_METRICS = {
    "Classification": "accuracy",
    "Regression": "r2_score",
}


def detect_problem_type(dataframe: pd.DataFrame, target_column: str) -> dict:
    """Classify the target column as Classification or Regression."""
    target = dataframe[target_column].dropna()
    if target.empty:
        raise ValueError("Target column has no non-null values.")

    if not pd.api.types.is_numeric_dtype(target):
        return {"problem_type": "Classification", "reason": "Target is non-numeric."}

    unique_count = int(target.nunique())
    if unique_count <= 2:
        return {"problem_type": "Classification", "reason": "Target is binary."}

    if pd.api.types.is_integer_dtype(target) and unique_count <= 20:
        return {
            "problem_type": "Classification",
            "reason": "Target is integer with low unique values.",
        }

    return {"problem_type": "Regression", "reason": "Target appears continuous."}


def get_primary_metric(problem_type: str) -> str:
    """Return the primary metric string for a given problem type."""
    if problem_type not in PRIMARY_METRICS:
        raise ValueError("Unsupported problem type. Use 'Classification' or 'Regression'.")
    return PRIMARY_METRICS[problem_type]


def submit_automl_job(
    csv_path: str,
    target_column: str,
    problem_type: str,
    data_name: str | None = None,
    subscription_id: str | None = None,
) -> str:
    """Register data and submit an AutoML training job, return the job name."""
    ml_client = get_ml_client(subscription_id=subscription_id)
    train_data_id = register_train_test_data(
        ml_client=ml_client,
        local_csv_path=csv_path,
        target_column=target_column,
        problem_type=problem_type,
        name=data_name,
    )

    job_name = run_automl_job(
        ml_client=ml_client,
        problem_type=problem_type,
        training_data=train_data_id,
        target_column=target_column,
        primary_metric=get_primary_metric(problem_type),
    )

    return job_name

