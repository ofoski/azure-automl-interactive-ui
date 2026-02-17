"""Submit Azure AutoML jobs and fetch concise job results."""

from ml_pipeline import get_ml_client, register_training_data, run_automl_job

CPU_GPU_MARKERS = ("STANDARD_NC", "STANDARD_ND", "STANDARD_NV")


def _is_cpu_vm(vm_size: str) -> bool:
    vm_upper = vm_size.upper()
    return not any(marker in vm_upper for marker in CPU_GPU_MARKERS)


def _normalize_metric_name(metric_value) -> str | None:
    if metric_value is None:
        return None
    metric_text = str(metric_value)
    if "." in metric_text:
        metric_text = metric_text.split(".")[-1]
    return metric_text.lower()


def _to_float(value) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _pick_first(mapping: dict | None, keys: list[str]):
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def submit_automl_job(
    csv_path: str,
    target_column: str,
    problem_type: str,
    primary_metric: str,
    data_name: str = "training-data",
    vm_size: str = "Standard_DS11_v2",
    experiment_name: str = "automl-experiment",
) -> str:
    """Register CSV data and submit a serverless CPU AutoML job."""
    if not _is_cpu_vm(vm_size):
        raise ValueError("Only CPU VM sizes are supported.")

    ml_client = get_ml_client()
    registered_data = register_training_data(ml_client, csv_path, name=data_name)

    job_name = run_automl_job(
        ml_client=ml_client,
        problem_type=problem_type,
        training_data=registered_data.id,
        target_column=target_column,
        primary_metric=primary_metric,
        compute_name=None,
        vm_size=vm_size,
        experiment_name=experiment_name,
    )

    return job_name


def _extract_scored_child_run(child_run) -> dict | None:
    properties = getattr(child_run, "properties", None)
    tags = getattr(child_run, "tags", None)

    score_value = _pick_first(
        properties,
        [
            "score",
            "best_score",
            "bestScore",
            "metric_value",
            "primary_metric_value",
            "primaryMetricValue",
        ],
    )
    if score_value is None:
        score_value = _pick_first(tags, ["score", "best_score", "primary_metric_value"])

    score = _to_float(score_value)
    if score is None:
        return None

    algorithm = _pick_first(
        properties,
        ["run_algorithm", "runAlgorithm", "algorithm", "training_algorithm"],
    ) or _pick_first(tags, ["run_algorithm", "algorithm"])

    model_name = _pick_first(
        properties,
        ["model_name", "modelName", "model", "model_id", "modelId"],
    ) or _pick_first(tags, ["model_name", "model"])

    return {
        "run_id": getattr(child_run, "name", None),
        "run_name": getattr(child_run, "display_name", None) or getattr(child_run, "name", None),
        "score": score,
        "algorithm": algorithm,
        "model_name": model_name,
    }


def get_automl_job_details(job_name: str) -> dict:
    """Fetch concise AutoML job details for UI display."""
    ml_client = get_ml_client()
    job = ml_client.jobs.get(job_name)

    details = {
        "job_name": getattr(job, "name", job_name),
        "status": getattr(job, "status", "Unknown"),
        "experiment_name": getattr(job, "experiment_name", "Unknown"),
        "primary_metric": _normalize_metric_name(getattr(job, "primary_metric", None)),
        "top_models": [],
    }

    try:
        child_runs = list(ml_client.jobs.list(parent_job_name=job_name))
    except Exception:
        child_runs = []

    scored_models = []
    for child_run in child_runs:
        model_row = _extract_scored_child_run(child_run)
        if model_row:
            scored_models.append(model_row)

    scored_models.sort(key=lambda item: item["score"], reverse=True)
    details["top_models"] = scored_models[:5]

    if details["top_models"]:
        best_model = details["top_models"][0]
        details["best_child_run_id"] = best_model["run_id"]
        details["best_run_name"] = best_model["run_name"]
        details["best_algorithm"] = best_model["algorithm"]
        details["best_model_name"] = best_model["model_name"]
        details["best_metric_value"] = best_model["score"]

        try:
            best_job = ml_client.jobs.get(best_model["run_id"])
            best_properties = getattr(best_job, "properties", None)
            details["feature_importance"] = _pick_first(
                best_properties,
                ["feature_importance", "featureImportance", "global_feature_importance"],
            )
        except Exception:
            details["feature_importance"] = None

    return details
