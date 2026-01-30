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
