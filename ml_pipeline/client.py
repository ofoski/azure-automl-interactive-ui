# ============================================================
# AZURE ML CLIENT (automatic authentication + setup)
# ============================================================
import os
import re
from functools import lru_cache
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import InteractiveBrowserCredential, TokenCachePersistenceOptions
from azure.mgmt.resource import ResourceManagementClient

DEFAULT_RESOURCE_GROUP = "automl-demo-rg"
DEFAULT_WORKSPACE_NAME = "automl-demo-ws"
DEFAULT_LOCATION = "canadacentral"
TENANT_REGEX = re.compile(r"sts\.windows\.net/([0-9a-fA-F-]{36})/")
_TENANT_BY_SUBSCRIPTION: dict[str, str] = {}
TENANT_ID_FILE = Path(".azure_tenant_id")


@lru_cache(maxsize=1)
def _get_default_tenant_id() -> str | None:
    env_tenant = os.environ.get("AZURE_TENANT_ID")
    if env_tenant:
        return env_tenant
    try:
        tenant = TENANT_ID_FILE.read_text(encoding="utf-8").strip()
        return tenant or None
    except Exception:
        return None


def _save_tenant_id(tenant_id: str) -> None:
    try:
        TENANT_ID_FILE.write_text(str(tenant_id).strip(), encoding="utf-8")
    except Exception:
        pass


@lru_cache(maxsize=64)
def _create_credential(tenant_id: str | None):
    login_timeout = int(os.environ.get("AZURE_LOGIN_TIMEOUT_SECONDS", "1200"))
    cache_options = TokenCachePersistenceOptions(
        name="azure-automl-demo-cache",
        allow_unencrypted_storage=True,
    )

    if tenant_id:
        return InteractiveBrowserCredential(
            tenant_id=tenant_id,
            additionally_allowed_tenants=["*"],
            timeout=login_timeout,
            cache_persistence_options=cache_options,
        )
    return InteractiveBrowserCredential(
        additionally_allowed_tenants=["*"],
        timeout=login_timeout,
        cache_persistence_options=cache_options,
    )


def _extract_expected_tenant_id(error: Exception) -> str | None:
    text = str(error)
    matches = TENANT_REGEX.findall(text)
    if not matches:
        return None
    return matches[-1]


def get_credential(subscription_id: str | None = None):
    """
    Return shared cached credential so login happens once.
    Uses tenant discovered from previous tenant-mismatch errors for this subscription.
    """
    tenant_id = _get_default_tenant_id()
    if not tenant_id and subscription_id:
        tenant_id = _TENANT_BY_SUBSCRIPTION.get(subscription_id)
    return _create_credential(tenant_id)


def _resolve_subscription_id(subscription_id: str | None) -> str:
    if subscription_id:
        return subscription_id

    env_subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    if env_subscription_id:
        return env_subscription_id

    raise RuntimeError("Set AZURE_SUBSCRIPTION_ID in your environment.")


def _ensure_resource_group(
    credential,
    subscription_id: str,
    resource_group: str,
    location: str,
) -> None:
    resource_client = ResourceManagementClient(credential, subscription_id)
    resource_client.resource_groups.create_or_update(
        resource_group_name=resource_group,
        parameters={"location": location},
    )


def _ensure_workspace(
    credential,
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    location: str,
) -> None:
    workspace_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    try:
        workspace_client.workspaces.get(workspace_name)
    except Exception:
        workspace = Workspace(name=workspace_name, location=location)
        workspace_client.workspaces.begin_create(workspace).result()


def run_with_tenant_retry(subscription_id: str, operation):
    """
    Run an Azure SDK operation and retry once with tenant from error message if needed.
    """
    credential = get_credential(subscription_id)
    try:
        return operation(credential)
    except Exception as error:
        tenant_id = _extract_expected_tenant_id(error)
        if not tenant_id:
            raise
        _TENANT_BY_SUBSCRIPTION[subscription_id] = tenant_id
        _save_tenant_id(tenant_id)
        return operation(get_credential(subscription_id))


def _ensure_default_resources(subscription_id: str) -> None:
    run_with_tenant_retry(
        subscription_id,
        lambda cred: (
            _ensure_resource_group(
                credential=cred,
                subscription_id=subscription_id,
                resource_group=DEFAULT_RESOURCE_GROUP,
                location=DEFAULT_LOCATION,
            ),
            _ensure_workspace(
                credential=cred,
                subscription_id=subscription_id,
                resource_group=DEFAULT_RESOURCE_GROUP,
                workspace_name=DEFAULT_WORKSPACE_NAME,
                location=DEFAULT_LOCATION,
            ),
        ),
    )


@lru_cache(maxsize=16)
def get_ml_client(
    subscription_id: str | None = None,
    ensure_resources: bool = True,
) -> MLClient:
    """
    Return an authenticated MLClient and automatically create missing Azure resources.

    Subscription comes from:
    - function argument, then
    - AZURE_SUBSCRIPTION_ID

    Resource group/workspace/location are fixed in code.
    """
    resolved_subscription_id = _resolve_subscription_id(subscription_id)

    if ensure_resources:
        _ensure_default_resources(resolved_subscription_id)

    return MLClient(
        credential=get_credential(resolved_subscription_id),
        subscription_id=resolved_subscription_id,
        resource_group_name=DEFAULT_RESOURCE_GROUP,
        workspace_name=DEFAULT_WORKSPACE_NAME,
    )
