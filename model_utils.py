"""Utilities to compare AutoML child model metrics."""

import pandas as pd


def extract_child_metrics(ml_client, parent_job_name: str) -> pd.DataFrame:
    """Parse model metrics from the parent job's semicolon-delimited tags."""
    parent = ml_client.jobs.get(parent_job_name)
    tags = getattr(parent, "tags", {}) or {}
    props = getattr(parent, "properties", {}) or {}

    best_run_id    = tags.get("automl_best_child_run_id", "")
    primary_metric = props.get("primary_metric", "score")

    def _split(key):
        val = tags.get(key, "")
        return [v.strip() for v in val.split(";")] if val else []

    algorithms    = _split("run_algorithm_000")
    scores        = _split("score_000")
    fit_times     = _split("fit_time_000")
    iterations    = _split("iteration_000")
    preprocessors = _split("run_preprocessor_000")

    rows = []
    for i, algo in enumerate(algorithms):
        if not algo:
            continue
        itr = int(iterations[i]) if i < len(iterations) and iterations[i].lstrip("-").isdigit() else i
        run_id = f"{parent_job_name}_{itr}"
        rows.append({
            "algorithm":    algo,
            "preprocessor": preprocessors[i] if i < len(preprocessors) else "",
            primary_metric: _safe_float(scores[i] if i < len(scores) else None),
            "fit_time_s":   _safe_float(fit_times[i] if i < len(fit_times) else None),
            "run_id":       run_id,
            "best":         run_id == best_run_id,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(axis=1, how="all")
    if primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=False).reset_index(drop=True)
    return df


def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
