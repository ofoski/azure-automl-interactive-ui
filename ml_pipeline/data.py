# ============================================================
# DATA PREPARATION & REGISTRATION
# ============================================================
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from pathlib import Path
import shutil


def register_training_data(ml_client, local_csv_path: str, name: str = "training-data"):
    """Register a local CSV as an MLTable data asset and return the registered asset."""
    csv_path = Path(local_csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    mltable_dir = csv_path.parent / f"{csv_path.stem}_mltable"
    mltable_dir.mkdir(parents=True, exist_ok=True)
    mltable_file = mltable_dir / "MLTable"
    csv_copy_path = mltable_dir / csv_path.name

    shutil.copy2(csv_path, csv_copy_path)

    mltable_content = (
        "paths:\n"
        f"  - file: ./{csv_path.name}\n"
        "transformations:\n"
        "  - read_delimited:\n"
        "      header: all_files_same_headers\n"
    )

    mltable_file.write_text(mltable_content, encoding="utf-8")

    data = Data(
        path=str(mltable_dir),
        type=AssetTypes.MLTABLE,
        name=name,
    )
    registered_data = ml_client.data.create_or_update(data)
    return registered_data
