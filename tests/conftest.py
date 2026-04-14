# conftest.py — loaded automatically by pytest before any test runs.

import sys
import pytest
from unittest.mock import MagicMock

# run_automl.py imports streamlit at the top level.
# Stub it here so tests work without streamlit installed.
sys.modules.setdefault("streamlit", MagicMock())
sys.modules.setdefault("streamlit.components", MagicMock())
sys.modules.setdefault("streamlit.components.v1", MagicMock())


@pytest.fixture(autouse=True)
def mock_azure_client(monkeypatch):
    """Replace the Azure ML client with a fake for every test.

    autouse=True means this runs automatically — no test needs to request it.
    Patching both locations is needed because some modules import get_ml_client
    directly (binding the name at import time) while others call it via the package.
    """
    mock_client = MagicMock()
    monkeypatch.setattr(
        "ml_pipeline.client.get_ml_client", lambda **kwargs: mock_client
    )
    monkeypatch.setattr("ml_pipeline.get_ml_client", lambda **kwargs: mock_client)
    return mock_client
