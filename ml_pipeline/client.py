# ============================================================
# AZURE ML CLIENT (authentication + workspace connection)
# ============================================================
from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
import os


def get_ml_client() -> MLClient:
    """Return an authenticated MLClient using workspace environment variables."""
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    credential = (
        InteractiveBrowserCredential(tenant_id=tenant_id)
        if tenant_id
        else InteractiveBrowserCredential()
    )
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
