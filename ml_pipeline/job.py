# ============================================================
# AUTOML JOB SUBMISSION
# ============================================================
from azure.ai.ml import Input
from .config import job_type

def run_automl_job(
    *,
    ml_client,
    problem_type: str,
    training_data: str,
    target_column: str,
    primary_metric: str,
    compute_name: str,
    experiment_name: str,
    timeout_minutes: int = 15,
    trial_timeout_minutes: int = 5,
    max_trials: int = 4,
    enable_early_termination: bool = True,
) -> str:
    """Create and submit an AutoML job and return the Azure job name."""
    
    if problem_type not in job_type:
        supported = ", ".join(sorted(job_type.keys()))
        raise ValueError(
            f"Unsupported problem_type: {problem_type}. Supported: {supported}"
        )

    job_creator = job_type[problem_type]

    training_data_input = Input(type="mltable", path=training_data)

    job = job_creator(
        training_data=training_data_input,
        target_column_name=target_column,
        primary_metric=primary_metric,
        compute=compute_name,
        experiment_name=experiment_name,
    )

    job.set_limits(
        timeout_minutes=timeout_minutes,
        trial_timeout_minutes=trial_timeout_minutes,
        max_trials=max_trials,
        enable_early_termination=enable_early_termination,
    )

    submitted_job = ml_client.jobs.create_or_update(job)
    return submitted_job.name