from pathlib import Path
import pandas as pd


def save_uploaded_file(uploaded_file, upload_dir="uploads") -> Path:
    """Save a Streamlit uploaded file to disk and return the path."""
    upload_path = Path(upload_dir)
    upload_path.mkdir(exist_ok=True)
    csv_path = upload_path / uploaded_file.name
    csv_path.write_bytes(uploaded_file.getvalue())
    return csv_path


def get_metric_mapping() -> dict:
    """Map UI metric labels to Azure ML metric names."""
    return {
        "Accuracy": "accuracy",
        "AUC": "auc_weighted",
        "F1": "f1_score_weighted",
        "Precision": "precision_score_weighted",
        "Recall": "recall_score_weighted",
        "RMSE": "normalized_root_mean_squared_error",
        "MAE": "normalized_mean_absolute_error",
        "R2": "r2_score",
    }


def safe_preview(df: pd.DataFrame, n=5) -> pd.DataFrame:
    """Return a preview DataFrame safe for Streamlit display."""
    df_preview = df.head(n).copy()
    for col in df_preview.columns:
        if df_preview[col].dtype == "object":
            df_preview[col] = df_preview[col].astype(str)
    return df_preview


def normalize_config_table(config_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize config table values to strings for Arrow compatibility."""
    config_df = config_df.copy()
    config_df["Value"] = config_df["Value"].astype(str)
    return config_df
