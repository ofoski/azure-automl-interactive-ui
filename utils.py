from pathlib import Path
import shutil


def _clear_directory(path: Path) -> None:
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)


def save_uploaded_file(uploaded_file, upload_dir="uploads") -> Path:
    """Save a Streamlit uploaded file to disk and return the path."""
    upload_path = Path(upload_dir)
    upload_path.mkdir(exist_ok=True)

    # Keep only the latest upload and its generated artifacts.
    _clear_directory(upload_path)

    csv_path = upload_path / uploaded_file.name
    csv_path.write_bytes(uploaded_file.getvalue())
    return csv_path
