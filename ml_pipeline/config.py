# ============================================================
# SHARED CONFIGURATION
# ============================================================
from azure.ai.ml import automl


# Compute cluster configuration (values only; create the cluster when needed)
COMPUTE_NAME = "automl-cpu-cluster"
COMPUTE_VM_SIZE = "Standard_DS3_v2"
COMPUTE_MIN_INSTANCES = 0
COMPUTE_MAX_INSTANCES = 1


# AutoML job type mapping
job_type = {
    "Classification": automl.classification,
    "Regression": automl.regression,
}


PROBLEM_TYPE_ALIASES = {
    "classification": "Classification",
    "class": "Classification",
    "binary": "Classification",
    "multiclass": "Classification",
    "multi-class": "Classification",
    "regression": "Regression",
    "regress": "Regression",
    "continuous": "Regression",
}


def normalize_problem_type(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    return PROBLEM_TYPE_ALIASES.get(value.strip().lower())
