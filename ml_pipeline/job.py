# ============================================================
# AUTOML JOB SUBMISSION
# ============================================================
from azure.ai.ml import Input, automl
from azure.ai.ml.entities import JobResourceConfiguration


def run_automl_job(
    *,
    ml_client,
    problem_type: str,
    training_data: str,
    target_column: str,
    primary_metric: str,
    compute_name: str | None,
    vm_size: str | None,
    experiment_name: str,
    timeout_minutes: int = 15,
    trial_timeout_minutes: int = 5,
    max_trials: int = 4,
    enable_early_termination: bool = True,
) -> str:
    """Create and submit an AutoML job and return the Azure job name."""

    if problem_type not in ("Classification", "Regression"):
        raise ValueError(
            f"Unsupported problem_type: {problem_type}. Supported: Classification, Regression"
        )

    training_data_input = Input(type="mltable", path=training_data)
    job_creator = (
        automl.classification
        if problem_type == "Classification"
        else automl.regression
    )

    job_args = {
        "training_data": training_data_input,
        "target_column_name": target_column,
        "primary_metric": primary_metric,
        "experiment_name": experiment_name,
    }
    if compute_name:
        job_args["compute"] = compute_name

    job = job_creator(**job_args)

    if vm_size:
        job.resources = JobResourceConfiguration(instance_type=vm_size)

    job.set_limits(
        timeout_minutes=timeout_minutes,
        trial_timeout_minutes=trial_timeout_minutes,
        max_trials=max_trials,
        enable_early_termination=enable_early_termination,
    )

    submitted_job = ml_client.jobs.create_or_update(job)
    return submitted_job.name