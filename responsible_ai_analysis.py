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
from run_automl import detect_problem_type

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


def fill_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values and encode object/bool columns to float."""
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype == "bool":
            X[col] = X[col].fillna(X[col].mode()[0])
            X[col] = pd.factorize(X[col])[0].astype(float)
        else:
            X[col] = X[col].fillna(X[col].median())
    return X


def error_analysis(model, X_test, y_test, task_type="regression"):
    """Compute mean error per group or bin for every feature."""
    X_test = fill_missing_values(X_test)
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
    # Keep a copy of the original (unencoded) X_test so that categorical
    # features like Sex, Embarked etc. show their real labels ("male"/"female")
    # in the fairness output instead of the numeric codes (0.0, 1.0) produced
    # by fill_missing_values. The encoded copy is used only for model.predict().
    X_test_orig = X_test.copy()
    X_test = fill_missing_values(X_test)
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
        col = X_test_orig[feature]
        groups = col if col.dtype == "object" or col.nunique() <= 10 else pd.qcut(X_test[feature], q=n_bins, duplicates="drop")

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
):
    """Generate counterfactual examples for a single test instance using DiCE."""
    X_train = fill_missing_values(X_train)
    X_test = fill_missing_values(X_test)
    train_cf = X_train.copy()
    train_cf[target_column] = y_train.values

    continuous_features = list(X_train.select_dtypes(include=["number", "bool"]).columns)
    data = dice_ml.Data(
        dataframe=train_cf,
        continuous_features=continuous_features,
        outcome_name=target_column,
    )
    dice_model = dice_ml.Model(
        model=DiceModelAdapter(model),
        backend="sklearn",
        model_type="classifier" if task_type == "classification" else "regressor",
    )
    exp = dice_ml.Dice(data, dice_model, method="random")

    query_instance = X_test.iloc[[instance_index]]

    if task_type == "classification":
        return exp.generate_counterfactuals(
            query_instance,
            total_CFs=total_cfs,
            desired_class=desired_class,
        )

    # Auto compute desired_range from actual value of that specific row
    current_pred = float(model.predict(X_test.iloc[[instance_index]])[0])
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

    return exp.generate_counterfactuals(
        query_instance,
        total_CFs=total_cfs,
        desired_range=desired_range,
    )


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
