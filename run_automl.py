"""Submit Azure AutoML jobs."""
from azure.core.exceptions import ResourceNotFoundError
from ml_pipeline import get_ml_client, register_training_data, run_automl_job


def submit_automl_job(
    csv_path: str,
    target_column: str,
    problem_type: str,
    primary_metric: str,
    data_name: str = "training-data",
    compute_name: str | None = None,
    experiment_name: str = "automl-experiment",
) -> str:
    """Register CSV data and submit an AutoML job. Returns the job name."""
    ml_client = get_ml_client()

    registered_data = register_training_data(
        ml_client,
        csv_path,
        name=data_name
    )

    print(f"✓ Data registered: {registered_data.name}")

    if compute_name:
        try:
            ml_client.compute.get(compute_name)
            resolved_compute_name = compute_name
        except ResourceNotFoundError:
            raise ValueError(
                f"Compute '{compute_name}' was not found in this workspace."
            )
    else:
        computes = list(ml_client.compute.list())
        if not computes:
            raise ValueError(
                "No compute targets found in this workspace. "
                "Create a CPU-based AML compute cluster, then rerun."
            )

        resolved_compute_name = None
        for compute in computes:
            if getattr(compute, "type", "") != "amlcompute":
                continue
            compute_size = (
                getattr(compute, "size", None)
                or getattr(compute, "vm_size", "")
                or ""
            ).upper()
            if "STANDARD_NC" in compute_size or "STANDARD_ND" in compute_size or "STANDARD_NV" in compute_size:
                continue
            resolved_compute_name = compute.name
            break

        if not resolved_compute_name:
            raise ValueError(
                "No CPU-based AML compute cluster found. "
                "Create a CPU cluster and rerun."
            )

    job_name = run_automl_job(
        ml_client=ml_client,
        problem_type=problem_type,
        training_data=registered_data.id,
        target_column=target_column,
        primary_metric=primary_metric,
        compute_name=resolved_compute_name,
        experiment_name=experiment_name,
    )

    print(f"✓ Job submitted: {job_name}")

    return job_name
