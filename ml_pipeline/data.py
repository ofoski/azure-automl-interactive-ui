"""Register uploaded CSV data as train/test MLTable Data Assets."""

import tempfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data

_MAX_TEXT_UNIQUES = 50
_MIN_TEXT_WORDS = 3
_ID_UNIQUE_RATIO = 0.98


def _prepare_automl_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    keep_columns = [target_column]

    for column in df.columns:
        if column == target_column:
            continue

        series = df[column].dropna()
        if series.empty:
            continue

        unique_count = int(series.nunique())
        if unique_count <= 1:
            continue

        if unique_count / len(series) >= _ID_UNIQUE_RATIO:
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            avg_words = series.astype(str).str.split().str.len().mean()
            if unique_count > _MAX_TEXT_UNIQUES or avg_words >= _MIN_TEXT_WORDS:
                continue

        keep_columns.append(column)

    return df[keep_columns].copy()


def data_split_mltable(base_dir: Path, split_name: str, split_df: pd.DataFrame) -> Path:
    split_dir = base_dir / f"{split_name}_mltable"
    split_dir.mkdir(parents=True, exist_ok=True)

    csv_file = split_dir / f"{split_name}.csv"
    split_df.to_csv(csv_file, index=False)

    (split_dir / "MLTable").write_text(
        "paths:\n"
        f"  - file: ./{csv_file.name}\n"
        "transformations:\n"
        "  - read_delimited:\n"
        "      header: all_files_same_headers\n",
        encoding="utf-8",
    )
    return split_dir


def register_train_test_data(
    ml_client,
    local_csv_path: str,
    target_column: str,
    problem_type: str | None = None,
    name: str | None = None,
):
    """Split full uploaded data into train/test (80/20) and register both as MLTable assets."""
    csv_path = Path(local_csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV.")
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to split into train/test.")

    df = _prepare_automl_dataframe(df, target_column)

    stratify = None
    if problem_type == "Classification":
        stratify = df[target_column]

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=stratify,
    )

    base_name = name or csv_path.stem
    with tempfile.TemporaryDirectory(prefix="automl_assets_") as tmp:
        staging_dir = Path(tmp)
        train_dir = data_split_mltable(staging_dir, "train", train_df)
        test_dir = data_split_mltable(staging_dir, "test", test_df)

        train_asset = ml_client.data.create_or_update(
            Data(path=str(train_dir), type=AssetTypes.MLTABLE, name=f"{base_name}-train")
        )
        test_asset = ml_client.data.create_or_update(
            Data(path=str(test_dir), type=AssetTypes.MLTABLE, name=f"{base_name}-test")
        )

    return train_asset.id
