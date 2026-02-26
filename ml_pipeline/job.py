# ============================================================
# AUTOML JOB SUBMISSION
# ============================================================
from azure.ai.ml import Input, automl
from azure.ai.ml.entities import JobResourceConfiguration

CLASSIFICATION_ALLOWED_ALGOS = [
    "LightGBM",
    "XGBoostClassifier",
    "RandomForest",
    "LogisticRegression",
]
REGRESSION_ALLOWED_ALGOS = [
    "LightGBM",
    "XGBoostRegressor",
    "RandomForest",
    "ElasticNet",
]
FAST_MAX_TRIALS = 2
FAST_TRIAL_TIMEOUT_MINUTES = 2
FAST_MAX_CONCURRENT_TRIALS = 1


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
    timeout_minutes: int | None = None,
    trial_timeout_minutes: int = FAST_TRIAL_TIMEOUT_MINUTES,
    max_trials: int = FAST_MAX_TRIALS,
    max_concurrent_trials: int = FAST_MAX_CONCURRENT_TRIALS,
    enable_early_termination: bool = True,
    enable_model_explainability: bool = True,
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

    limits_kwargs = {
        "trial_timeout_minutes": int(trial_timeout_minutes),
        "max_trials": int(max_trials),
        "enable_early_termination": enable_early_termination,
    }
    if timeout_minutes is not None and int(timeout_minutes) >= 15:
        limits_kwargs["timeout_minutes"] = int(timeout_minutes)
    safe_max_concurrent_trials = int(max_concurrent_trials)
    try:
        job.set_limits(max_concurrent_trials=safe_max_concurrent_trials, **limits_kwargs)
    except Exception:
        job.set_limits(**limits_kwargs)

    # Prefer explainable candidates: disable ensembles, keep explainability on.
    # Some SDK versions may not support all arguments, so apply best-effort.
    allowed_algorithms = (
        CLASSIFICATION_ALLOWED_ALGOS if problem_type == "Classification" else REGRESSION_ALLOWED_ALGOS
    )
    training_kwargs = {
        "enable_stack_ensemble": False,
        "enable_vote_ensemble": False,
    }
    if enable_model_explainability:
        training_kwargs["enable_model_explainability"] = True
    try:
        job.set_training(
            **training_kwargs,
            allowed_training_algorithms=allowed_algorithms,
        )
    except Exception:
        job.set_training(**training_kwargs)

    submitted_job = ml_client.jobs.create_or_update(job)
    return submitted_job.name
