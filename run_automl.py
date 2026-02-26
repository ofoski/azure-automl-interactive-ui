"""Submit Azure AutoML jobs and fetch concise job results."""

import json
import re
import tempfile
from pathlib import Path

import pandas as pd

from ml_pipeline import get_ml_client, register_training_data, run_automl_job

DEFAULT_VM_SIZE = "Standard_DS11_v2"


def _sanitize_name(value: str, default_value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "-", str(value or ""))
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    if not cleaned:
        cleaned = default_value
    return cleaned[:255]


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


def _extract_numeric_metrics(mapping: dict | None) -> dict:
    if not isinstance(mapping, dict):
        return {}
    result = {}
    for key, value in mapping.items():
        score = _to_float(value)
        if score is not None:
            result[str(key)] = score
    return result


def _walk_nested(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from _walk_nested(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_nested(item)


def _to_feature_importance(value) -> dict | None:
    if value is None:
        return None

    parsed = value
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            return None

    if isinstance(parsed, dict):
        if all(isinstance(v, (int, float)) for v in parsed.values()):
            return parsed
        candidates = [
            "feature_importance",
            "featureImportance",
            "global_feature_importance",
            "globalFeatureImportance",
            "global_importance_values",
            "permutation_importance",
        ]
        for key in candidates:
            nested = parsed.get(key)
            nested_converted = _to_feature_importance(nested)
            if nested_converted:
                return nested_converted

    if isinstance(parsed, list):
        result = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            feature = item.get("feature") or item.get("feature_name") or item.get("name")
            importance = item.get("importance") or item.get("score") or item.get("value")
            feature_text = str(feature).strip() if feature is not None else ""
            score = _to_float(importance)
            if feature_text and score is not None:
                result[feature_text] = score
        if result:
            return result

    return None


def _to_confusion_matrix(value):
    if value is None:
        return None
    parsed = value
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            return None
    if isinstance(parsed, dict):
        matrix = parsed.get("matrix") or parsed.get("confusion_matrix") or parsed.get("data")
        if matrix is not None:
            labels = parsed.get("labels") or parsed.get("class_labels")
            return {"labels": labels, "matrix": matrix}
        if "0" in parsed or "1" in parsed:
            return {"labels": None, "matrix": parsed}
    if isinstance(parsed, list):
        return {"labels": None, "matrix": parsed}
    return None


def _extract_feature_importance_from_mapping(mapping: dict | None) -> dict | None:
    if not isinstance(mapping, dict):
        return None
    # Direct parse first.
    direct = _to_feature_importance(mapping)
    if direct:
        return direct
    # Then nested search.
    for key, value in _walk_nested(mapping):
        key_text = str(key).lower()
        if "importance" in key_text or "explain" in key_text:
            converted = _to_feature_importance(value)
            if converted:
                return converted
    return None


def _extract_confusion_matrix_from_mapping(mapping: dict | None):
    if not isinstance(mapping, dict):
        return None
    direct = _to_confusion_matrix(mapping)
    if direct:
        return direct
    for key, value in _walk_nested(mapping):
        key_text = str(key).lower()
        if "confusion" in key_text:
            converted = _to_confusion_matrix(value)
            if converted:
                return converted
    return None


def _extract_feature_importance_from_artifacts(ml_client, run_id: str) -> dict | None:
    output_candidates = ["model_explanation", "explanations", "explanation", "outputs"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        try:
            ml_client.jobs.download(
                name=run_id,
                download_path=str(base / "all"),
            )
        except Exception:
            pass
        for output_name in output_candidates:
            try:
                ml_client.jobs.download(
                    name=run_id,
                    download_path=str(base / output_name),
                    output_name=output_name,
                )
            except Exception:
                continue

        for json_file in base.rglob("*.json"):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            converted = _to_feature_importance(payload)
            if converted:
                return converted
        for csv_file in base.rglob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue
            cols_lower = {str(col).lower(): col for col in df.columns}
            feature_col = cols_lower.get("feature") or cols_lower.get("feature_name") or cols_lower.get("name")
            importance_col = cols_lower.get("importance") or cols_lower.get("score") or cols_lower.get("value")
            if feature_col and importance_col:
                rows = {}
                for _, row in df.iterrows():
                    feature = str(row.get(feature_col, "")).strip()
                    score = _to_float(row.get(importance_col))
                    if feature and score is not None:
                        rows[feature] = float(score)
                if rows:
                    return rows
    return None


def _extract_confusion_matrix_from_artifacts(ml_client, run_id: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        try:
            ml_client.jobs.download(
                name=run_id,
                download_path=str(base / "all"),
            )
        except Exception:
            return None

        for json_file in base.rglob("*.json"):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            converted = _extract_confusion_matrix_from_mapping(payload)
            if converted:
                return converted
    return None


def submit_automl_job(
    csv_path: str,
    target_column: str,
    problem_type: str,
    primary_metric: str,
    data_name: str = "training-data",
    vm_size: str = DEFAULT_VM_SIZE,
    experiment_name: str = "automl-experiment",
    subscription_id: str | None = None,
) -> str:
    """Register CSV data and submit a serverless AutoML job."""
    safe_data_name = _sanitize_name(data_name, "training-data")
    safe_experiment_name = _sanitize_name(experiment_name, "automl-experiment")

    ml_client = get_ml_client(subscription_id=subscription_id)
    registered_data = register_training_data(ml_client, csv_path, name=safe_data_name)

    job_name = run_automl_job(
        ml_client=ml_client,
        problem_type=problem_type,
        training_data=registered_data.id,
        target_column=target_column,
        primary_metric=primary_metric,
        compute_name=None,
        vm_size=vm_size,
        experiment_name=safe_experiment_name,
        enable_model_explainability=True,
    )

    return job_name


def _extract_scored_child_run(child_run) -> dict | None:
    properties = getattr(child_run, "properties", None)
    tags = getattr(child_run, "tags", None)
    numeric_metrics = {}
    numeric_metrics.update(_extract_numeric_metrics(tags))
    numeric_metrics.update(_extract_numeric_metrics(properties))

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
    feature_importance = _extract_feature_importance_from_mapping(properties) or _extract_feature_importance_from_mapping(tags)
    confusion_matrix = _extract_confusion_matrix_from_mapping(properties) or _extract_confusion_matrix_from_mapping(tags)

    return {
        "run_id": getattr(child_run, "name", None),
        "run_name": getattr(child_run, "display_name", None) or getattr(child_run, "name", None),
        "score": score,
        "algorithm": algorithm,
        "model_name": model_name,
        "feature_importance": feature_importance,
        "confusion_matrix": confusion_matrix,
        "metrics": numeric_metrics,
    }


def get_automl_job_details(
    job_name: str,
    subscription_id: str | None = None,
) -> dict:
    """Fetch concise AutoML job details for UI display."""
    ml_client = get_ml_client(subscription_id=subscription_id, ensure_resources=False)
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
    details["all_scored_models"] = scored_models
    details["top_models"] = scored_models[:5]

    if details["top_models"]:
        best_model = details["top_models"][0]
        details["best_child_run_id"] = best_model["run_id"]
        details["best_run_name"] = best_model["run_name"]
        details["best_algorithm"] = best_model["algorithm"]
        details["best_model_name"] = best_model["model_name"]
        details["best_metric_value"] = best_model["score"]

        details["feature_importance"] = best_model.get("feature_importance")
        details["feature_importance_source_run"] = best_model.get("run_id")
        details["confusion_matrix"] = best_model.get("confusion_matrix")
        details["confusion_matrix_source_run"] = best_model.get("run_id")
        if not details["feature_importance"]:
            for model in details["all_scored_models"]:
                try:
                    candidate_job = ml_client.jobs.get(model["run_id"])
                    candidate_properties = getattr(candidate_job, "properties", None)
                    candidate_tags = getattr(candidate_job, "tags", None)
                    candidate_importance = (
                        _extract_feature_importance_from_mapping(candidate_properties)
                        or _extract_feature_importance_from_mapping(candidate_tags)
                    )
                    candidate_confusion_matrix = (
                        _extract_confusion_matrix_from_mapping(candidate_properties)
                        or _extract_confusion_matrix_from_mapping(candidate_tags)
                    )
                    if candidate_importance:
                        details["feature_importance"] = candidate_importance
                        details["feature_importance_source_run"] = model.get("run_id")
                    if candidate_confusion_matrix and not details.get("confusion_matrix"):
                        details["confusion_matrix"] = candidate_confusion_matrix
                        details["confusion_matrix_source_run"] = model.get("run_id")
                    if details["feature_importance"] and details.get("confusion_matrix"):
                        break
                except Exception:
                    continue
        if not details["feature_importance"]:
            try:
                for model in details["all_scored_models"]:
                    details["feature_importance"] = _extract_feature_importance_from_artifacts(
                        ml_client=ml_client,
                        run_id=model["run_id"],
                    )
                    if details["feature_importance"]:
                        details["feature_importance_source_run"] = model.get("run_id")
                        break
            except Exception:
                pass
        if not details.get("confusion_matrix"):
            try:
                for model in details["all_scored_models"]:
                    details["confusion_matrix"] = _extract_confusion_matrix_from_artifacts(
                        ml_client=ml_client,
                        run_id=model["run_id"],
                    )
                    if details["confusion_matrix"]:
                        details["confusion_matrix_source_run"] = model.get("run_id")
                        break
            except Exception:
                pass

    return details
