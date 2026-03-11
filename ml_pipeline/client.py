"""Azure ML client helpers."""

import os

import streamlit as st
from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential


def _resolve_subscription_id(subscription_id: str | None) -> str:
    if subscription_id:
        return subscription_id

    env_subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    if env_subscription_id:
        return env_subscription_id

    raise RuntimeError(
        "Subscription ID not provided. Set AZURE_SUBSCRIPTION_ID environment variable."
    )


@st.cache_resource
def _get_cached_credential() -> InteractiveBrowserCredential:
    """Get or create a cached credential. Called only once per session."""
    resolved_tenant_id = os.environ.get("AZURE_TENANT_ID")
    return InteractiveBrowserCredential(tenant_id=resolved_tenant_id)


def get_ml_client(
    subscription_id: str | None = None,
    resource_group: str | None = None,
    workspace_name: str | None = None,
) -> MLClient:
    """Create an MLClient from args or environment variables."""
    resolved_subscription_id = _resolve_subscription_id(subscription_id)
    resolved_resource_group = resource_group or os.environ.get("AZURE_RESOURCE_GROUP", "automl-demo-rg")
    resolved_workspace_name = workspace_name or os.environ.get("AZURE_WORKSPACE_NAME", "automl-demo-ws")
    credential = _get_cached_credential()

    return MLClient(
        credential=credential,
        subscription_id=resolved_subscription_id,
        resource_group_name=resolved_resource_group,
        workspace_name=resolved_workspace_name,
    )
