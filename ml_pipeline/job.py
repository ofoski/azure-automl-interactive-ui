"""Submit AutoML jobs to Azure ML."""

from uuid import uuid4

from azure.ai.ml import Input, automl
from azure.ai.ml.entities import JobResourceConfiguration

DEFAULT_MAX_TRIALS = 4
DEFAULT_TRIAL_TIMEOUT_MINUTES = 5
DEFAULT_TIMEOUT_MINUTES = 15
DEFAULT_EXPERIMENT_NAME = "streamlit-automl-demo"


def run_automl_job(
    *,
    ml_client,
    problem_type: str,
    training_data: str,
    target_column: str,
    primary_metric: str,
    vm_size: str | None,
    timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
    trial_timeout_minutes: int = DEFAULT_TRIAL_TIMEOUT_MINUTES,
    max_trials: int = DEFAULT_MAX_TRIALS,
) -> str:
    """Create and submit an AutoML job."""
    if problem_type not in ("Classification", "Regression"):
        raise ValueError(f"Invalid problem_type: {problem_type}")

    training_data_input = Input(type="mltable", path=training_data)

    if problem_type == "Classification":
        job = automl.classification(
            training_data=training_data_input,
            target_column_name=target_column,
            primary_metric=primary_metric,
            experiment_name=DEFAULT_EXPERIMENT_NAME,
        )
    else:
        job = automl.regression(
            training_data=training_data_input,
            target_column_name=target_column,
            primary_metric=primary_metric,
            experiment_name=DEFAULT_EXPERIMENT_NAME,
        )

    if vm_size:
        job.resources = JobResourceConfiguration(instance_type=vm_size)

    # Set a stable parent job name and use it across the app.
    parent_job_name = f"automl-{uuid4().hex[:16]}"
    job.name = parent_job_name

    job.set_limits(
        timeout_minutes=int(timeout_minutes),
        trial_timeout_minutes=int(trial_timeout_minutes),
        max_trials=int(max_trials),
    )

    ml_client.jobs.create_or_update(job)
    return parent_job_name
