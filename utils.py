from pathlib import Path


def save_uploaded_file(uploaded_file, upload_dir="uploads") -> Path:
    """Save a Streamlit uploaded file to disk and return the path."""
    upload_path = Path(upload_dir)
    upload_path.mkdir(exist_ok=True)
    csv_path = upload_path / uploaded_file.name
    csv_path.write_bytes(uploaded_file.getvalue())
    return csv_path
