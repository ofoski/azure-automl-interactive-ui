# ============================================================
# AUTOML PACKAGE
# ============================================================
from .client import get_ml_client
from .data import register_training_data
from .job import run_automl_job
from .config import COMPUTE_NAME, COMPUTE_VM_SIZE, COMPUTE_MIN_INSTANCES, COMPUTE_MAX_INSTANCES

__all__ = [
    "get_ml_client",
    "register_training_data",
    "run_automl_job",
    "COMPUTE_NAME",
    "COMPUTE_VM_SIZE",
    "COMPUTE_MIN_INSTANCES",
    "COMPUTE_MAX_INSTANCES",
]
