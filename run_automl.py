"""AutoML helpers used by the Streamlit app."""

import pandas as pd
from azure.ai.ml.entities import Model

from ml_pipeline import get_ml_client, register_training_data, run_automl_job

DEFAULT_VM_SIZE = "Standard_DS11_v2"
DEFAULT_EXPERIMENT_NAME = "streamlit-automl-demo"

PRIMARY_METRICS = {
    "Classification": "accuracy",
    "Regression": "r2_score",
}


def detect_problem_type(dataframe: pd.DataFrame, target_column: str) -> dict:
    target = dataframe[target_column].dropna()
    if target.empty:
        raise ValueError("Target column has no non-null values.")

    # Minimal rule:
    # non-numeric target => Classification, numeric target => Regression
    if not pd.api.types.is_numeric_dtype(target):
        return {"problem_type": "Classification", "reason": "Target is non-numeric."}
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
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    subscription_id: str | None = None,
) -> str:
    safe_data_name = (data_name or "training-data").strip().replace(" ", "-")[:255]
    safe_experiment_name = (experiment_name or DEFAULT_EXPERIMENT_NAME).strip().replace(" ", "-")[:255]

    ml_client = get_ml_client(subscription_id=subscription_id)
    data_asset = register_training_data(ml_client, csv_path, name=safe_data_name)
    return run_automl_job(
        ml_client=ml_client,
        problem_type=problem_type,
        training_data=data_asset.id,
        target_column=target_column,
        primary_metric=get_primary_metric(problem_type),
        vm_size=vm_size,
        experiment_name=safe_experiment_name,
    )


def register_best_model(job_name: str, best_model: dict, subscription_id: str | None = None) -> dict:
    """Register the best model artifact from the best child run."""
    ml_client = get_ml_client(subscription_id=subscription_id)

    if best_model.get("model_id"):
        return best_model

    run_id = best_model.get("run_id")
    if not run_id:
        return best_model

    registered_name = f"best-model-{job_name}"[:255]

    try:
        model = Model(
            name=registered_name,
            path=f"runs:/{run_id}/outputs/model.pkl",
            type="custom_model",
            description=f"Best model from {job_name}",
        )
        created = ml_client.models.create_or_update(model)
        best_model["registered_model_name"] = getattr(created, "name", registered_name)
        best_model["registered_model_version"] = str(getattr(created, "version", "1"))
        best_model["model_id"] = f"azureml:{best_model['registered_model_name']}:{best_model['registered_model_version']}"
    except Exception:
        pass

    return best_model
