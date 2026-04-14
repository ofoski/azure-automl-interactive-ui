"""Tests for data cleaning and problem-type detection. No Azure credentials needed."""

import pandas as pd
import pytest
from pathlib import Path

from run_automl import detect_problem_type
from ml_pipeline.data import _prepare_automl_dataframe

DATA_DIR = Path(__file__).parent / "data"
TITANIC_PATH = DATA_DIR / "titanic_test.csv"
HOUSING_PATH = DATA_DIR / "california_housing_test.csv"


# --- Classification (Titanic) ---


def test_classification_detection_and_cleaning():
    """Tests logic using the Titanic (Classification) dataset."""
    df = pd.read_csv(TITANIC_PATH)
    target = "Survived"

    # 1. Task Detection — binary integer target must be classified as Classification
    result = detect_problem_type(df, target)
    assert result["problem_type"] == "Classification"

    # 2. Data Cleaning — feature explosion prevention
    # 'Name' is high-cardinality free text and must be dropped before training.
    clean_df = _prepare_automl_dataframe(df, target)

    assert "Name" not in clean_df.columns, (
        "High-cardinality 'Name' column was NOT dropped!"
    )
    assert target in clean_df.columns, "Target column was accidentally dropped!"


def test_high_cardinality_columns_removed():
    """Ticket and Cabin columns are also high-cardinality and must be dropped."""
    df = pd.read_csv(TITANIC_PATH)
    target = "Survived"

    clean_df = _prepare_automl_dataframe(df, target)

    # Ticket has near-unique values per row — should be treated as an ID column
    assert "Ticket" not in clean_df.columns, "'Ticket' ID-like column was NOT dropped!"


# --- Regression (California Housing) ---


def test_regression_detection():
    """Tests logic using the California Housing (Regression) dataset."""
    df = pd.read_csv(HOUSING_PATH)
    # Target column matches the CSV header exactly (lowercase)
    target = "median_house_value"

    result = detect_problem_type(df, target)
    assert result["problem_type"] == "Regression"
    assert "continuous" in result["reason"].lower()


# --- Responsible AI context builder ---


def test_responsible_ai_context_builder():
    """Ensures the statistical summary for the AI Agent is formatted correctly."""
    from responsible_ai_analysis import build_data_context

    df = pd.read_csv(TITANIC_PATH)
    target = "Survived"
    X = df.drop(columns=[target])
    y = df[target]

    context = build_data_context(X, y, target)

    assert isinstance(context, str), "build_data_context must return a string"
    # Match the exact keys produced by the function
    assert "Dataset statistics" in context, "Summary header missing from context"
    assert f"Target column: {target}" in context, (
        "Target column line missing from context"
    )
    assert "Test set size" in context, "Test set size line missing from context"


# --- Edge cases ---


def test_detect_problem_type_raises_on_all_null_target():
    """Must raise ValueError when the target column contains no non-null values."""
    df = pd.DataFrame({"feature": [1, 2, 3], "target": [None, None, None]})

    with pytest.raises(ValueError, match="no non-null values"):
        detect_problem_type(df, "target")


def test_detect_problem_type_string_target_is_classification():
    """A string target must always be Classification — this branch is not hit by the CSV tests."""
    df = pd.DataFrame({"label": ["cat", "dog", "cat", "bird", "dog"]})

    result = detect_problem_type(df, "label")
    assert result["problem_type"] == "Classification"
    assert "non-numeric" in result["reason"].lower()


def test_prepare_automl_drops_constant_columns():
    """Constant columns (same value in every row) contain no signal and must be dropped."""
    # 40 rows keeps unique_ratio at 0.10, safely below the 0.98 ID-detection threshold.
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1] * 10,
            "useful_feature": [10, 20, 30, 40] * 10,  # ratio 0.10 → kept
            "constant_feature": [99] * 40,  # ratio 0.025, unique_count=1 → dropped
        }
    )

    clean_df = _prepare_automl_dataframe(df, "target")

    assert "constant_feature" not in clean_df.columns, (
        "Constant column was NOT dropped!"
    )
    assert "useful_feature" in clean_df.columns, (
        "Useful column was incorrectly dropped!"
    )
    assert "target" in clean_df.columns, "Target column was accidentally dropped!"


def test_prepare_automl_drops_all_null_feature_columns():
    """All-NaN feature columns must be dropped — they would cause a divide-by-zero error."""
    # 40 rows keeps good_feature's unique_ratio at 0.10, safely below the 0.98 ID threshold.
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1] * 10,
            "good_feature": [1.0, 2.0, 3.0, 4.0] * 10,  # ratio 0.10 → kept
            "empty_feature": [None] * 40,  # drops to empty Series → dropped
        }
    )

    clean_df = _prepare_automl_dataframe(df, "target")

    assert "empty_feature" not in clean_df.columns, (
        "All-null feature column was NOT dropped!"
    )
    assert "good_feature" in clean_df.columns


def test_register_train_test_data_raises_on_missing_file():
    """Must raise FileNotFoundError before touching Azure when the CSV path does not exist.

    ml_client=None is intentional — the error is raised before the client is used.
    """
    from ml_pipeline.data import register_train_test_data

    with pytest.raises(FileNotFoundError):
        register_train_test_data(
            ml_client=None,
            local_csv_path="/nonexistent/path/data.csv",
            target_column="target",
        )
