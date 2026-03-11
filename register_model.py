"""Register best model from a completed AutoML job."""

from azure.ai.ml.entities import Model

from ml_pipeline import get_ml_client


def register_best_model(
    job_name: str,
    subscription_id: str | None = None,
) -> dict:
    """Register best model from a completed AutoML parent job."""
    ml_client = get_ml_client(subscription_id=subscription_id)

    parent_job = ml_client.jobs.get(job_name)
    tags = getattr(parent_job, "tags", {}) or {}

    # AutoML parent exposes the selected best child run here.
    best_child_run_id = tags.get("automl_best_child_run_id")
    if not best_child_run_id:
        raise ValueError(
            f"AutoML best child run id not found for parent job '{job_name}'."
        )

    best_score = None
    children = ml_client.jobs.list(parent_job_name=job_name, max_results=200)
    for child in children:
        if getattr(child, "name", None) == best_child_run_id:
            child_props = getattr(child, "properties", {}) or {}
            best_score = child_props.get("score")
            break

    registered_name = f"best-model-{best_child_run_id}"
    source_path = f"azureml://jobs/{job_name}/outputs/best_model"

    # If already registered, return existing model info without re-registering.
    try:
        existing = next(ml_client.models.list(name=registered_name), None)
    except Exception:
        existing = None

    if existing is not None:
        existing_name = getattr(existing, "name", registered_name)
        existing_version = str(getattr(existing, "version", "1"))
        return {
            "run_id": best_child_run_id,
            "score": best_score,
            "registered_model_name": existing_name,
            "registered_model_version": existing_version,
            "model_id": f"azureml:{existing_name}:{existing_version}",
            "source_path": source_path,
            "already_registered": True,
        }

    model = Model(
        name=registered_name,
        path=source_path,
        type="custom_model",
        description=f"Best model from {job_name}",
    )
    created = ml_client.models.create_or_update(model)

    created_name = getattr(created, "name", registered_name)
    created_version = str(getattr(created, "version", "1"))
    return {
        "run_id": best_child_run_id,
        "score": best_score,
        "registered_model_name": created_name,
        "registered_model_version": created_version,
        "model_id": f"azureml:{created_name}:{created_version}",
        "source_path": source_path,
        "already_registered": False,
    }