# ============================================================
# AUTOML JOB SUBMISSION
# ============================================================
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
    limits=None,
) -> str:
    """
    Create and submit an AutoML job.
    
    Args:
        - ml_client: authenticated MLClient
        - problem_type: 'Classification' or 'Regression'
        - training_data: asset id (e.g., registered.id from register_training_data)
        - target_column: column to predict
        - primary_metric: metric to optimize (e.g., 'accuracy', 'r2_score')
        - compute_name: name of compute target (e.g., 'automl-cpu-cluster')
        - experiment_name: job name/experiment name
        - limits: dict with cost controls (e.g., {'max_trials': 2, 'timeout_minutes': 15})
    
    Returns:
        Job name (string) — use to track job in Azure ML Studio
    """
    
    if problem_type not in job_type:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    job_creator = job_type[problem_type]

    # Default cost limits (override with limits arg if needed)
    default_limits = {"max_trials": 3, "timeout_minutes": 15}
    merged_limits = {**default_limits, **(limits or {})}

    job = job_creator(
        training_data=training_data,
        target_column_name=target_column,
        primary_metric=primary_metric,
        compute=compute_name,
        experiment_name=experiment_name,
        limits=merged_limits,
    )

    submitted_job = ml_client.jobs.create_or_update(job)
    return submitted_job.name