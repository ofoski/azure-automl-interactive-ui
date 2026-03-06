"""Register best model from a completed AutoML job."""

from azure.ai.ml.entities import Model

from ml_pipeline import get_ml_client


def register_best_model(
    job_name: str,
    subscription_id: str | None = None,
) -> dict:
    """Register best-scoring child run model artifact."""
    ml_client = get_ml_client(subscription_id=subscription_id)

    children = list(ml_client.jobs.list(parent_job_name=job_name, max_results=100))
    best_score = None
    best_child_run_id = None

    for child in children:
        props = getattr(child, "properties", {}) or {}
        score_val = props.get("score")
        if score_val is None:
            continue
        try:
            score = float(score_val)
        except (TypeError, ValueError):
            continue

        if best_score is None or score > best_score:
            best_score = score
            best_child_run_id = getattr(child, "name", None)

    if not best_child_run_id:
        raise ValueError(f"No best child run score found for job {job_name}")

    registered_name = f"best-model-{job_name}"[:255]
    model = Model(
        name=registered_name,
        path=f"runs:/{best_child_run_id}/outputs/model.pkl",
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
    }