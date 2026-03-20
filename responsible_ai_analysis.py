"""Responsible AI analysis for registered Azure ML models."""

from __future__ import annotations

import importlib
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
import dice_ml

from ml_pipeline import get_ml_client
from register_model import register_best_model
from run_automl import detect_problem_type

client = get_ml_client()


class DiceModelAdapter:
    """Normalize Azure AutoML model outputs for DiCE."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions.to_numpy() if hasattr(predictions, "to_numpy") else predictions

    def predict_proba(self, X):
        probabilities = self.model.predict_proba(X)
        return probabilities.to_numpy() if hasattr(probabilities, "to_numpy") else probabilities

def _version_kwargs(version: str | None) -> dict:
    return {"version": version} if version else {"label": "latest"}


def load_model(model_name: str, version: str | None = None):
    """Download a registered Azure ML model and return the loaded pipeline."""
    if not version:
        raise ValueError("Model version is required. Pass the registered model version explicitly.")

    info = client.models.get(name=model_name, **_version_kwargs(version))
    path = Path("model_artifacts") / info.name / str(info.version)
    path.mkdir(parents=True, exist_ok=True)
    client.models.download(name=info.name, version=info.version, download_path=str(path))
    return pickle.load(open(next(path.rglob("model.pkl")), "rb"))


def load_test_data(asset_name: str, version: str | None = None) -> pd.DataFrame:
    """Load an Azure ML MLTable data asset into a DataFrame."""
    asset = client.data.get(name=asset_name, **_version_kwargs(version))
    return importlib.import_module("mltable").load(asset.path).to_pandas_dataframe()


def run_permutation_importance(
    model_name: str,
    test_asset_name: str,
    target_column: str,
    model_version: str | None = None,
    test_asset_version: str | None = None,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Load model and test data from Azure ML, return sorted permutation importance."""
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
    """Fill missing values: median for continuous, mode for categorical/low-cardinality."""
    X = X.copy()
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype == "object" or X[col].nunique() <= 10:
                X[col] = X[col].fillna(X[col].mode()[0])
            else:
                X[col] = X[col].fillna(X[col].median())
    return X


def error_analysis(model, X_test, y_test, task_type="regression"):
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
    """Compute per-group fairness metrics using fairlearn.metrics.MetricFrame."""
    import numpy as np
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
    from sklearn.metrics import accuracy_score, mean_absolute_error

    X_test = fill_missing_values(X_test)
    y_pred = model.predict(X_test)
    if hasattr(y_pred, "to_numpy"):
        y_pred = y_pred.to_numpy()

    if task_type == "classification":
        metrics = {
            "accuracy":           accuracy_score,
            "selection_rate":     selection_rate,
            "true_positive_rate": true_positive_rate,
        }
    else:
        metrics = {"mean_abs_error": mean_absolute_error}

    features = sensitive_features if sensitive_features is not None else list(X_test.columns)
    results = {}
    for feature in features:
        col = X_test[feature]
        groups = col if col.dtype == "object" or col.nunique() <= 10 else pd.qcut(col, q=n_bins, duplicates="drop")
        results[feature] = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=groups).by_group
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
    """Generate DiCE counterfactuals for one test instance."""
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
    actual_value = y_test.iloc[instance_index]

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
    """Infer the paired train asset name from a test asset name."""
    if test_asset_name.endswith("-test"):
        return f"{test_asset_name[:-5]}-train"
    raise ValueError("Could not infer train asset name. Set TRAIN_ASSET explicitly.")

if __name__ == "__main__":
    # If you have an AutoML job name, this will fetch registered model name/version from register_model.py
    JOB_NAME = os.environ.get("RA_JOB_NAME") or None  # e.g. "automl-job-xxxx"

    # Or set exact registered model values directly.
    MODEL_NAME = os.environ.get("RA_MODEL_NAME", "best-model-automl-b666b781bd0242b4_0")
    MODEL_VERSION = os.environ.get("RA_MODEL_VERSION", "1")
    TEST_ASSET = os.environ.get("RA_TEST_ASSET", "california_housing-test")
    TRAIN_ASSET = infer_train_asset_name(TEST_ASSET)
    TARGET_COLUMN = os.environ.get("RA_TARGET_COLUMN", "median_house_value")

    if JOB_NAME:
        print(f"RA_JOB_NAME is set ({JOB_NAME}); overriding MODEL_NAME/MODEL_VERSION from job registration.")
        reg = register_best_model(JOB_NAME)
        MODEL_NAME = reg["registered_model_name"]
        MODEL_VERSION = reg["registered_model_version"]

    print("\nEffective runtime config:")
    print("MODEL_NAME:", MODEL_NAME)
    print("MODEL_VERSION:", MODEL_VERSION)
    print("TEST_ASSET:", TEST_ASSET)
    print("TRAIN_ASSET:", TRAIN_ASSET)
    print("TARGET_COLUMN:", TARGET_COLUMN)

    # load_test_data — check asset loads correctly
    df = load_test_data(TEST_ASSET)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in test asset '{TEST_ASSET}'.")
    print("Test data shape:", df.shape)
    print(df.head())

    # load_model — check model downloads and deserialises
    model = load_model(MODEL_NAME, MODEL_VERSION)
    print("Model type:", type(model))

    X_test, y_test = df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]
    train_df = load_test_data(TRAIN_ASSET)
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in train asset '{TRAIN_ASSET}'.")
    X_train, y_train = train_df.drop(columns=[TARGET_COLUMN]), train_df[TARGET_COLUMN]
    task_info = detect_problem_type(df, TARGET_COLUMN)
    task_type = task_info["problem_type"].lower()

    # run_permutation_importance — full end-to-end
    importance_df = run_permutation_importance(
        model_name=MODEL_NAME,
        test_asset_name=TEST_ASSET,
        target_column=TARGET_COLUMN,
        model_version=MODEL_VERSION,
    )
    print(importance_df)

    print("\nError analysis:")
    error_results = error_analysis(model, X_test, y_test, task_type=task_type)
    for feature, errors in error_results.items():
        print(f"\n--- {feature} ---")
        print(errors)
    
    # Counterfactuals
    print("\nCounterfactuals:")
    cf = run_counterfactuals(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type=task_type,
        target_column=TARGET_COLUMN,
        instance_index=0,
        desired_class="opposite" if task_type == "classification" else None,
        total_cfs=5,         # ← request 5 counterfactuals
    )

    cf_example = cf.cf_examples_list[0]
    generated_count = 0 if cf_example.final_cfs_df is None else len(cf_example.final_cfs_df)
    print(f"Requested 5 counterfactuals, generated {generated_count}.")
    print("Original instance:")
    print(cf_example.test_instance_df)

    print("\nGenerated counterfactuals:")
    if cf_example.final_cfs_df is None or cf_example.final_cfs_df.empty:
        print("No counterfactuals were generated.")
    else:
        print(cf_example.final_cfs_df)  # ← already contains all 5 rows

    # Fairness analysis
    print("\nFairness analysis:")
    fairness_results = fairness_analysis(
        model=model,
        X_test=X_test,
        y_test=y_test,
        task_type=task_type,
    )
    for feature, metrics_df in fairness_results.items():
        print(f"\n--- {feature} ---")
        print(metrics_df.to_string())
