"""Azure ML workspace client setup."""

import os
from functools import lru_cache

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

@lru_cache(maxsize=64)
def _create_credential(tenant_id: str | None):
    """Create a cached Azure credential for the given tenant."""
    return DefaultAzureCredential(
        exclude_interactive_browser_credential=False,
        interactive_browser_tenant_id=tenant_id,
    )


def _resolve_subscription_id(subscription_id: str | None) -> str:
    """Get subscription ID from the argument or environment variable."""
    if subscription_id:
        return subscription_id

    env_subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    if env_subscription_id:
        return env_subscription_id

    raise RuntimeError(
        "Subscription ID not provided. Set AZURE_SUBSCRIPTION_ID environment variable."
    )


def get_ml_client(
    subscription_id: str | None = None,
    resource_group: str | None = None,
    workspace_name: str | None = None,
) -> MLClient:
    """Create an authenticated MLClient using args or environment variables."""
    resolved_subscription_id = _resolve_subscription_id(subscription_id)
    resolved_resource_group = resource_group or os.environ.get("AZURE_RESOURCE_GROUP")
    resolved_workspace_name = workspace_name or os.environ.get("AZURE_WORKSPACE_NAME")
    if not resolved_resource_group:
        raise RuntimeError("Resource group not provided. Set AZURE_RESOURCE_GROUP environment variable.")
    if not resolved_workspace_name:
        raise RuntimeError("Workspace name not provided. Set AZURE_WORKSPACE_NAME environment variable.")
    resolved_tenant_id = os.environ.get("AZURE_TENANT_ID")

    return MLClient(
        credential=_create_credential(resolved_tenant_id),
        subscription_id=resolved_subscription_id,
        resource_group_name=resolved_resource_group,
        workspace_name=resolved_workspace_name,
    )