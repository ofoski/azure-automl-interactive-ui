"""AutoML helpers used by the Streamlit app."""

import pandas as pd

from ml_pipeline import get_ml_client, register_training_data, run_automl_job

DEFAULT_VM_SIZE = "Standard_DS11_v2"

PRIMARY_METRICS = {
    "Classification": "accuracy",
    "Regression": "r2_score",
}


def _ml_client(subscription_id: str | None = None):
    return get_ml_client(subscription_id=subscription_id)


def detect_problem_type(dataframe: pd.DataFrame, target_column: str) -> dict:
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
    if problem_type not in PRIMARY_METRICS:
        raise ValueError("Unsupported problem type. Use 'Classification' or 'Regression'.")
    return PRIMARY_METRICS[problem_type]


def submit_automl_job(
    csv_path: str,
    target_column: str,
    problem_type: str,
    data_name: str = "training-data",
    vm_size: str = DEFAULT_VM_SIZE,
    subscription_id: str | None = None,
) -> str:
    ml_client = _ml_client(subscription_id)
    data_asset = register_training_data(ml_client, csv_path, name=data_name)
    return run_automl_job(
        ml_client=ml_client,
        problem_type=problem_type,
        training_data=data_asset.id,
        target_column=target_column,
        primary_metric=get_primary_metric(problem_type),
        vm_size=vm_size,
    )


