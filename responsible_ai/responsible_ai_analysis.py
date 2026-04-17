"""Analysis functions called by the Responsible AI agent.

This module is the computation layer — responsible_ai_agent.py calls these
functions as tools and passes the results to the LLM for interpretation.

Design notes:
- The Azure ML client is created lazily on first use so importing this module
  never crashes before credentials are configured.
- fairness_analysis keeps a copy of the original (unencoded) X_test so that
  group labels show real values (e.g. "male"/"female") instead of numeric codes
  produced by the encoding step.
- recall_score is used instead of fairlearn's true_positive_rate because
  fairlearn crashes when a group contains only positive-class samples.
"""

from __future__ import annotations

import importlib
import pickle
import shutil
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
import dice_ml

from ml_pipeline import get_ml_client
from training.run_automl import detect_problem_type

# Lazily initialised on first use so that importing this module never crashes
# when Azure credentials are not yet configured (e.g. during unit tests or
# when the Streamlit app is loading before the user sets env vars).
_client = None


def _get_client():
    """Return the Azure ML client, creating it on the first call."""
    global _client
    if _client is None:
        _client = get_ml_client()
    return _client


class DiceModelAdapter:
    """Wrap an AutoML model to return numpy arrays for DiCE."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """Predict and convert output to a numpy array."""
        predictions = self.model.predict(X)
        return predictions.to_numpy() if hasattr(predictions, "to_numpy") else predictions

    def predict_proba(self, X):
        """Predict probabilities and convert output to a numpy array."""
        probabilities = self.model.predict_proba(X)
        return probabilities.to_numpy() if hasattr(probabilities, "to_numpy") else probabilities

def _version_kwargs(version: str | None) -> dict:
    """Build version or latest-label kwargs for Azure ML .get() calls."""
    return {"version": version} if version else {"label": "latest"}


def load_model(model_name: str, version: str | None = None):
    """Download a registered model from Azure ML and return the pipeline via MLflow."""
    if not version:
        raise ValueError("Model version is required. Pass the registered model version explicitly.")

    client = _get_client()
    info = client.models.get(name=model_name, **_version_kwargs(version))
    path = Path("model_artifacts") / info.name / str(info.version)
    path.mkdir(parents=True, exist_ok=True)
    client.models.download(name=info.name, version=info.version, download_path=str(path))

    # Azure ML SDK creates a redundant inner {name}/ folder inside the download path.
    # Flatten it so the layout becomes: path/mlflow-model/ (or path/best_model/).
    inner = path / info.name
    if inner.is_dir():
        for item in inner.iterdir():
            dest = path / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(str(inner))

    # Locate the MLflow model directory by the known Azure AutoML folder names.
    for candidate in ("mlflow-model", "best_model"):
        mlflow_dir = path / candidate
        if (mlflow_dir / "MLmodel").exists():
            with open(mlflow_dir / "model.pkl", "rb") as f:
                return pickle.load(f)

    raise FileNotFoundError(
        f"Could not find 'mlflow-model' or 'best_model' folder under {path}"
    )


def load_test_data(asset_name: str, version: str | None = None) -> pd.DataFrame:
    """Load an Azure ML MLTable data asset into a DataFrame."""
    asset = _get_client().data.get(name=asset_name, **_version_kwargs(version))
    return importlib.import_module("mltable").load(asset.path).to_pandas_dataframe()


def run_permutation_importance(
    model_name: str,
    test_asset_name: str,
    target_column: str,
    model_version: str | None = None,
    test_asset_version: str | None = None,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Compute permutation importance for a registered model and test asset."""
    model = load_model(model_name, model_version)
    df = load_test_data(test_asset_name, test_asset_version)

    X, y = df.drop(columns=[target_column]), df[target_column]
    scoring = "accuracy" if detect_problem_type(df, target_column)["problem_type"] == "Classification" else "r2"

    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, scoring=scoring)

    return (
        pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def error_analysis(model, X_test, y_test, task_type="regression"):
    """Compute mean error per group or bin for every feature."""
    df = X_test.copy()
    df["y_true"] = y_test.values
    df["y_pred"] = model.predict(X_test)
    df["error"] = (df["y_true"] - df["y_pred"]).abs() if task_type == "regression" else (df["y_true"] != df["y_pred"]).astype(int)

    results = {}
    for feature in X_test.columns:
        if df[feature].dtype == "object" or df[feature].nunique() <= 10:
            results[feature] = df.groupby(feature)["error"].mean().sort_values(ascending=False)
        else:
            bins = pd.qcut(df[feature], q=4, duplicates="drop")
            results[feature] = df.groupby(bins)["error"].mean().sort_values(ascending=False)

    return results


def fairness_analysis(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
    sensitive_features: list | None = None,
    n_bins: int = 4,
) -> dict:
    """Compute per-group fairness metrics for every feature."""
    from fairlearn.metrics import MetricFrame, selection_rate
    from sklearn.metrics import accuracy_score, mean_absolute_error, recall_score

    import numpy as np
    y_pred = model.predict(X_test)
    if hasattr(y_pred, "to_numpy"):
        y_pred = y_pred.to_numpy()

    # Flatten to 1-D and cast to plain numpy int/float so that sklearn's
    # confusion_matrix never receives a pandas nullable-boolean Series.
    # Nullable booleans produce a mix of int and None when numpy tries to
    # compute unique labels, causing: '<' not supported between int and NoneType.
    y_pred_arr = np.asarray(y_pred).ravel()
    y_true_arr = np.asarray(y_test).ravel()

    if task_type == "classification":
        y_pred_arr = y_pred_arr.astype(int)
        y_true_arr = y_true_arr.astype(int)

        # WHY WE USE recall_score INSTEAD OF fairlearn's true_positive_rate:
        #
        # fairlearn's true_positive_rate crashes when a group contains ONLY
        # positive-class samples (e.g. an age group where every customer churned).
        # It builds labels=[None, 1] for the missing class, then passes it to
        # sklearn's confusion_matrix which tries to sort it — crashing with:
        #   TypeError: '<' not supported between instances of 'int' and 'NoneType'
        #
        # recall_score computes the same metric (TP / (TP + FN)) but handles
        # this edge case safely via zero_division=0, returning 0 instead of crashing.
        def _true_positive_rate(y_true, y_pred):
            return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

        metrics = {
            "accuracy":           accuracy_score,
            "selection_rate":     selection_rate,
            "true_positive_rate": _true_positive_rate,
        }
    else:
        metrics = {"mean_abs_error": mean_absolute_error}

    features = sensitive_features if sensitive_features is not None else list(X_test.columns)
    results = {}
    for feature in features:
        # Use original (unencoded) column for group labels so that categorical
        # features display their real values instead of factorize() codes.
        col = X_test[feature]
        groups = col if col.dtype == "object" or col.nunique() <= 10 else pd.qcut(col, q=n_bins, duplicates="drop")

        # Convert group labels to str — makes them always sortable regardless of
        # whether the underlying column is int64, float, Categorical interval, or object.
        groups_str = groups.astype(str)

        results[feature] = MetricFrame(
            metrics=metrics,
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            sensitive_features=groups_str,
        ).by_group
    return results


def run_counterfactuals(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
    target_column: str,
    instance_index: int = 0,
    desired_class: str | int | None = "opposite",
    total_cfs: int = 5,
    features_to_vary: list[str] | None = None,
):
    """Generate counterfactual examples for a single test instance using DiCE.

    features_to_vary: columns DiCE is allowed to change. Pass a subset to lock
                      certain features constant (e.g. keep Age fixed at its current
                      value by omitting it from the list). If None, all features vary.
    """
    # DiCE needs NaN-free data to compute feature value ranges for candidate sampling.
    # The AutoML model still receives the original data (with NaN) for its predictions.
    X_train_cf = X_train.fillna(X_train.mode().iloc[0])
    X_test_cf = X_test.fillna(X_test.mode().iloc[0])

    # Keep track of which features are truly continuous in the original data.
    continuous_features = [
        col for col in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[col])
    ]

    # Convert non-numeric columns to stable train/test-aligned int codes for DiCE.
    # Also build a reverse map so we can decode int codes back to labels in the output.
    label_map: dict[str, dict] = {}
    for col in X_train_cf.columns:
        if not pd.api.types.is_numeric_dtype(X_train_cf[col]):
            train_cat = pd.Categorical(X_train_cf[col])
            X_train_cf[col] = train_cat.codes
            X_test_cf[col] = pd.Categorical(X_test_cf[col], categories=train_cat.categories).codes

            # Unknown categories in test become -1; map them to most frequent seen train code.
            fallback_code = pd.Series(X_train_cf[col]).mode().iloc[0]
            X_test_cf.loc[X_test_cf[col] == -1, col] = fallback_code

            # Reverse map: {0: 'France', 1: 'Germany', ...}
            label_map[col] = {i: cat for i, cat in enumerate(train_cat.categories)}

    # Normalise all dtypes to float64 — handles int8 (from Categorical.codes),
    # Int64 (nullable extension type from MLTable), and any other non-standard numeric.
    X_train_cf = X_train_cf.astype(float)
    X_test_cf = X_test_cf.astype(float)

    # Combine filled training features with the target so DiCE can learn the data distribution.
    train_cf = X_train_cf.copy()
    train_cf[target_column] = y_train.values

    try:
        data = dice_ml.Data(
            dataframe=train_cf,
            continuous_features=continuous_features,
            outcome_name=target_column,
        )
    except ValueError as exc:
        if "Unknown data type of feature" in str(exc):
            dtypes = ", ".join(f"{c}:{train_cf[c].dtype}" for c in train_cf.columns)
            raise ValueError(f"{exc} | DiCE input dtypes: {dtypes}") from exc
        raise

    # Wrap the AutoML pipeline in an adapter so DiCE can call predict() through it.
    dice_model = dice_ml.Model(
        model=DiceModelAdapter(model),
        backend="sklearn",
        model_type="classifier" if task_type == "classification" else "regressor",
    )
    exp = dice_ml.Dice(data, dice_model, method="random")

    # Pick the single row we want to generate counterfactuals for.
    query_instance = X_test_cf.iloc[[instance_index]]

    # Build kwargs — if the caller restricted which features to vary, pass that constraint.
    # Example: features_to_vary=["Pclass", "Fare"] keeps Age, Sex, etc. locked at their
    # current values, producing counterfactuals that only change actionable features.
    cf_kwargs = {"total_CFs": total_cfs}
    if features_to_vary:
        cf_kwargs["features_to_vary"] = features_to_vary

    def _decode(df):
        """Replace integer codes with original category labels in a DiCE output DataFrame."""
        if df is None:
            return df
        df = df.copy()
        for col, mapping in label_map.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda v: mapping.get(int(round(float(v))), v))
        return df

    if task_type == "classification":
        result = exp.generate_counterfactuals(
            query_instance,
            desired_class=desired_class,
            **cf_kwargs,
        )
        for ex in result.cf_examples_list:
            ex.test_instance_df = _decode(ex.test_instance_df)
            ex.final_cfs_df = _decode(ex.final_cfs_df)
        return result

    # Auto compute desired_range from actual value of that specific row
    current_pred = float(model.predict(query_instance)[0])
    actual_value = float(y_test.iloc[instance_index])

    if current_pred < actual_value:
        lower = current_pred * 1.1
        upper = actual_value
    else:
        lower = actual_value
        upper = current_pred * 0.9

    if lower >= upper:
        lower = min(current_pred, actual_value) * 0.95
        upper = max(current_pred, actual_value) * 1.05

    desired_range = [lower, upper]

    result = exp.generate_counterfactuals(
        query_instance,
        total_CFs=total_cfs,
        desired_range=desired_range,
    )
    for ex in result.cf_examples_list:
        ex.test_instance_df = _decode(ex.test_instance_df)
        ex.final_cfs_df = _decode(ex.final_cfs_df)
    return result


def infer_train_asset_name(test_asset_name: str) -> str:
    """Infer the paired train asset name from a test asset name ending with '-test'."""
    if test_asset_name.endswith("-test"):
        return f"{test_asset_name[:-5]}-train"
    raise ValueError("Could not infer train asset name. Set TRAIN_ASSET explicitly.")


def build_data_context(X_test: pd.DataFrame, y_test, target_column: str) -> str:
    """Build a statistical summary of the test data for the LLM system prompt."""
    df = X_test.copy()
    df[target_column] = y_test.values
    return (
        f"Dataset statistics:\n{df.describe(include='all').round(2).to_string()}\n\n"
        f"Target column: {target_column}\n"
        f"Test set size: {len(X_test)} rows\n"
    )
