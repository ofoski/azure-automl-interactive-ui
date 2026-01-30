# ============================================================
# AZURE ML CLIENT (authentication + workspace connection)
# ============================================================
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os


def get_ml_client() -> MLClient:
    """
    Create and return an authenticated MLClient for Azure ML workspace.
    
    Requires environment variables:
        - AZURE_SUBSCRIPTION_ID
        - AZURE_RESOURCE_GROUP
        - AZURE_WORKSPACE_NAME
    
    Returns:
        MLClient: Authenticated client for Azure ML operations
    """
    credential = DefaultAzureCredential()
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
    workspace = os.environ.get("AZURE_WORKSPACE_NAME")
    
    if not subscription_id or not resource_group or not workspace:
        raise EnvironmentError(
            "Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and AZURE_WORKSPACE_NAME"
        )
    
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )
