"""AutoML job submission helper."""
from automl import get_ml_client, register_training_data, run_automl_job


def submit_automl_job(
    csv_path: str,
    target_column: str,
    problem_type: str,
    primary_metric: str,
    data_name: str = "training-data",
    compute_name: str = "automl-cpu-cluster",
    experiment_name: str = "automl-experiment",
    limits: dict = None,
) -> str:
    """
    Submit an AutoML job to Azure ML.
    
    Args:
        csv_path: Path to the CSV file
        target_column: Target column name
        problem_type: 'Classification' or 'Regression'
        primary_metric: Azure ML metric name (e.g., 'accuracy', 'r2_score')
        data_name: Name for registered data asset
        compute_name: Compute cluster name
        experiment_name: Experiment name
        limits: Optional dict with max_trials, timeout_minutes, etc.
    
    Returns:
        Job name (string)
    """
    # Get Azure ML client
    ml_client = get_ml_client()
    
    # Register training data
    registered_data = register_training_data(
        ml_client,
        csv_path,
        name=data_name
    )
    
    print(f"✓ Data registered: {registered_data.name}")
    
    # Submit AutoML job
    job_name = run_automl_job(
        ml_client=ml_client,
        problem_type=problem_type,
        training_data=registered_data.id,
        target_column=target_column,
        primary_metric=primary_metric,
        compute_name=compute_name,
        experiment_name=experiment_name,
        limits=limits
    )
    
    print(f"✓ Job submitted: {job_name}")
    
    return job_name
