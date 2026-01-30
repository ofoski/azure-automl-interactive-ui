# ============================================================
# DATA PREPARATION & REGISTRATION
# ============================================================
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from pathlib import Path


def _ensure_mltable_from_csv(local_csv_path: str) -> str:
    """
    Create an MLTable folder structure for a CSV file.
    
    Args:
        local_csv_path: Path to the CSV file
    
    Returns:
        Path to the MLTable folder (as string)
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
    """
    csv_path = Path(local_csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    mltable_dir = csv_path.parent / f"{csv_path.stem}_mltable"
    mltable_dir.mkdir(parents=True, exist_ok=True)
    mltable_file = mltable_dir / "MLTable"

    if not mltable_file.exists():
        mltable_file.write_text(
            "paths:\n"
            f"  - file: ../{csv_path.name}\n"
            "transformations:\n"
            "  - read_delimited:\n"
            "      first_row_as_header: true\n",
            encoding="utf-8",
        )

    return str(mltable_dir)


def register_training_data(ml_client, local_csv_path: str, name: str = "training-data"):
    """
    Register a CSV file as an MLTable asset in Azure ML.
    
    Args:
        ml_client: Authenticated MLClient
        local_csv_path: Path to the CSV file
        name: Name for the registered data asset
    
    Returns:
        Registered data asset with .id property
    """
    mltable_path = _ensure_mltable_from_csv(local_csv_path)
    data = Data(
        path=mltable_path,
        type=AssetTypes.MLTABLE,
        name=name,
    )
    registered_data = ml_client.data.create_or_update(data)
    return registered_data
