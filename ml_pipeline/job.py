"""Submit AutoML jobs to Azure ML."""

from azure.ai.ml import Input, automl
from azure.ai.ml.entities import JobResourceConfiguration

DEFAULT_MAX_TRIALS = 3
DEFAULT_TRIAL_TIMEOUT_MINUTES = 5
DEFAULT_TIMEOUT_MINUTES = 15


def run_automl_job(
    *,
    ml_client,
    problem_type: str,
    training_data: str,
    target_column: str,
    primary_metric: str,
    vm_size: str | None,
    experiment_name: str,
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
            experiment_name=experiment_name,
        )
    else:
        job = automl.regression(
            training_data=training_data_input,
            target_column_name=target_column,
            primary_metric=primary_metric,
            experiment_name=experiment_name,
        )

    if vm_size:
        job.resources = JobResourceConfiguration(instance_type=vm_size)

    job.set_limits(
        timeout_minutes=int(timeout_minutes),
        trial_timeout_minutes=int(trial_timeout_minutes),
        max_trials=int(max_trials),
    )

    submitted_job = ml_client.jobs.create_or_update(job)
    return submitted_job.name
