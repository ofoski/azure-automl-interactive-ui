import json
import os
import re
from functools import lru_cache
from time import sleep

from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from openai import AzureOpenAI

from ml_pipeline.client import (
    DEFAULT_RESOURCE_GROUP,
    get_ml_client,
    run_with_tenant_retry,
)

DEFAULT_OPENAI_ACCOUNT_NAME = "automlagentopenai"
DEFAULT_OPENAI_DEPLOYMENT_NAME = "automl-gpt"
DEFAULT_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-5-nano")
DEFAULT_OPENAI_MODEL_VERSION = os.environ.get("AZURE_OPENAI_MODEL_VERSION", "")
OPENAI_CANDIDATE_REGIONS = [
    os.environ.get("AZURE_OPENAI_LOCATION"),
    "canadacentral",
    "eastus",
    "eastus2",
]
MODEL_CANDIDATES = [
    (DEFAULT_OPENAI_MODEL_NAME, DEFAULT_OPENAI_MODEL_VERSION),
    ("gpt-5-nano", ""),
    ("gpt-4o-mini", "2024-07-18"),
    ("gpt-4o-mini", "2024-08-06"),
    ("gpt-4o", "2024-08-06"),
    ("gpt-35-turbo", "0125"),
]
SKU_CANDIDATES = ["GlobalStandard", "Standard"]


def _first_non_empty(values: list[str | None]) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _sanitize_name(value: str, default_value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9-]", "-", str(value or ""))
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    if not cleaned:
        cleaned = default_value
    return cleaned[:32].lower()


def _chat_completion_with_retry(client: AzureOpenAI, **kwargs):
    last_error = None
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as error:
            last_error = error
            message = str(error).lower()
            is_transient = any(
                token in message
                for token in ("connection error", "timed out", "timeout", "429", "rate limit")
            )
            if attempt < 3 and is_transient:
                sleep(2 * (attempt + 1))
                continue
            break
    raise RuntimeError(
        "Azure OpenAI request failed. Verify endpoint/deployment/network. "
        f"Details: {last_error}"
    )


def _wait_until_openai_ready(settings: dict[str, str]) -> None:
    """Best-effort warmup for newly created accounts/deployments."""
    client = AzureOpenAI(
        azure_endpoint=settings["endpoint"],
        api_key=settings["api_key"],
        api_version=settings["api_version"],
        timeout=30.0,
        max_retries=1,
    )
    last_error = None
    for attempt in range(8):
        try:
            client.chat.completions.create(
                model=settings["deployment"],
                temperature=0,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return
        except Exception as error:
            last_error = error
            sleep(5)
    raise RuntimeError(
        "Configured Azure OpenAI endpoint/deployment is not reachable from this machine. "
        f"Endpoint: {settings['endpoint']} Deployment: {settings['deployment']} Details: {last_error}"
    )


def _is_openai_reachable(settings: dict[str, str]) -> bool:
    try:
        client = AzureOpenAI(
            azure_endpoint=settings["endpoint"],
            api_key=settings["api_key"],
            api_version=settings["api_version"],
            timeout=10.0,
            max_retries=0,
        )
        client.chat.completions.create(
            model=settings["deployment"],
            temperature=0,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        return True
    except Exception:
        return False


def _ensure_openai_account(cog_client, location: str) -> str:
    base_name = _sanitize_name(
        os.environ.get("AZURE_OPENAI_ACCOUNT_NAME") or DEFAULT_OPENAI_ACCOUNT_NAME,
        DEFAULT_OPENAI_ACCOUNT_NAME,
    )
    location_suffix = _sanitize_name(location, "region").replace("-", "")
    account_name = _sanitize_name(f"{base_name}-{location_suffix}", DEFAULT_OPENAI_ACCOUNT_NAME)
    try:
        existing = cog_client.accounts.get(DEFAULT_RESOURCE_GROUP, account_name)
        existing_location = str(getattr(existing, "location", "")).lower()
        if existing_location == str(location).lower():
            return account_name
    except Exception:
        pass

    # If exact name exists in another region, append short hash-like suffix by region length.
    if len(account_name) > 24:
        account_name = account_name[:24]
    account_name = _sanitize_name(f"{account_name}-{len(location)}", DEFAULT_OPENAI_ACCOUNT_NAME)
    try:
        existing = cog_client.accounts.get(DEFAULT_RESOURCE_GROUP, account_name)
        existing_location = str(getattr(existing, "location", "")).lower()
        if existing_location == str(location).lower():
            return account_name
    except Exception:
        pass

    try:
        cog_client.accounts.get(DEFAULT_RESOURCE_GROUP, account_name)
        return account_name
    except Exception:
        pass

    # Create Azure OpenAI account.
    params = {
        "location": location,
        "kind": "OpenAI",
        "sku": {"name": "S0"},
        "properties": {
            "customSubDomainName": account_name,
            "publicNetworkAccess": "Enabled",
        },
    }
    cog_client.accounts.begin_create(DEFAULT_RESOURCE_GROUP, account_name, params).result()
    return account_name


def _ensure_openai_deployment(cog_client, account_name: str) -> str:
    deployment_name = _sanitize_name(
        os.environ.get("AZURE_OPENAI_DEPLOYMENT") or DEFAULT_OPENAI_DEPLOYMENT_NAME,
        DEFAULT_OPENAI_DEPLOYMENT_NAME,
    )

    # Try to check existing deployment first (management plane).
    try:
        existing = cog_client.deployments.get(DEFAULT_RESOURCE_GROUP, account_name, deployment_name)
        if existing:
            return deployment_name
    except Exception:
        pass

    # Best-effort deployment creation with model/SKU fallbacks.
    errors = []
    for model_name, model_version in MODEL_CANDIDATES:
        for sku_name in SKU_CANDIDATES:
            params = {
                "sku": {"name": sku_name, "capacity": 1},
                "properties": {
                    "model": {
                        "format": "OpenAI",
                        "name": model_name,
                    },
                    "versionUpgradeOption": "NoAutoUpgrade",
                    "raiPolicyName": "Microsoft.Default",
                },
            }
            if model_version:
                params["properties"]["model"]["version"] = model_version
            try:
                cog_client.deployments.begin_create_or_update(
                    DEFAULT_RESOURCE_GROUP,
                    account_name,
                    deployment_name,
                    params,
                ).result()
                return deployment_name
            except Exception as error:
                errors.append(f"{model_name}:{model_version}:{sku_name} -> {error}")

    raise RuntimeError(
        "Unable to create Azure OpenAI deployment with any supported model/SKU candidate. "
        + " | ".join(errors[-6:])
    )


def _discover_or_create_openai(ml_client) -> dict[str, str]:
    def _with_cog_retry(operation):
        return run_with_tenant_retry(
            ml_client.subscription_id,
            lambda cred: operation(CognitiveServicesManagementClient(cred, ml_client.subscription_id)),
        )

    discovery_errors = []

    for region in [value for value in OPENAI_CANDIDATE_REGIONS if value]:
        try:
            account_name = _with_cog_retry(lambda client: _ensure_openai_account(client, region))
            account = _with_cog_retry(
                lambda client: client.accounts.get(DEFAULT_RESOURCE_GROUP, account_name)
            )
            endpoint = getattr(getattr(account, "properties", None), "endpoint", None)
            keys = _with_cog_retry(
                lambda client: client.accounts.list_keys(DEFAULT_RESOURCE_GROUP, account_name)
            )
            api_key = _first_non_empty([getattr(keys, "key1", None), getattr(keys, "key2", None)])
            deployment = _with_cog_retry(
                lambda client: _ensure_openai_deployment(client, account_name)
            )

            if not endpoint or not api_key or not deployment:
                raise RuntimeError("Missing endpoint/key/deployment after provisioning.")

            return {
                "endpoint": endpoint,
                "api_key": api_key,
                "deployment": deployment,
            }
        except Exception as error:
            discovery_errors.append(f"{region}: {error}")

    raise RuntimeError(
        "Could not create/configure Azure OpenAI automatically. "
        "This usually means model access/quota is not enabled for this subscription. "
        "Details: " + " | ".join(discovery_errors)
    )


@lru_cache(maxsize=8)
def get_azure_openai_settings() -> dict[str, str]:
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

    if endpoint and api_key and deployment:
        env_settings = {
            "endpoint": endpoint,
            "api_key": api_key,
            "deployment": deployment,
            "api_version": api_version,
        }
        if _is_openai_reachable(env_settings):
            return env_settings

    ml_client = get_ml_client(ensure_resources=False)
    provisioned = _discover_or_create_openai(ml_client)
    endpoint = provisioned["endpoint"]
    api_key = provisioned["api_key"]
    deployment = provisioned["deployment"]

    return {
        "endpoint": endpoint,
        "api_key": api_key,
        "deployment": deployment,
        "api_version": api_version,
    }


@lru_cache(maxsize=8)
def get_azure_openai_client() -> AzureOpenAI:
    settings = get_azure_openai_settings()
    return AzureOpenAI(
        azure_endpoint=settings["endpoint"],
        api_key=settings["api_key"],
        api_version=settings["api_version"],
        timeout=45.0,
        max_retries=2,
    )


def ensure_azure_openai_ready() -> dict[str, str]:
    """Force provisioning/discovery and return active settings."""
    settings = get_azure_openai_settings()
    _wait_until_openai_ready(settings)
    return settings


def detect_problem_type_with_agent(target_column: str, non_null_count: int, dtype_text: str, sample_values: list) -> dict:
    client = get_azure_openai_client()
    settings = get_azure_openai_settings()
    prompt = (
        "You are an ML task classifier. "
        "Given target column metadata and sample values, return ONLY JSON with keys "
        "problem_type and reason. "
        "problem_type must be Classification or Regression."
    )
    user_payload = {
        "target_column": target_column,
        "non_null_count": non_null_count,
        "dtype": dtype_text,
        "sample_values": sample_values[:25],
    }
    response = _chat_completion_with_retry(
        client,
        model=settings["deployment"],
        temperature=0,
        max_tokens=180,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    problem_type = parsed.get("problem_type")
    if problem_type not in ("Classification", "Regression"):
        raise RuntimeError("Agent returned invalid problem_type.")
    reason = str(parsed.get("reason", "")).strip() or "No reason returned by agent."
    return {"problem_type": problem_type, "reason": reason, "source": "agent"}


def answer_results_question_with_agent(summary_payload: dict, question: str, history: list[dict] | None = None) -> str:
    client = get_azure_openai_client()
    settings = get_azure_openai_settings()
    history = history or []
    system_prompt = (
        "You are an AutoML assistant. Answer user questions about the experiment details. "
        "Be concrete and reference metrics/models in the provided context. "
        "If confusion_matrix or feature_importance are present in context, use them directly. "
        "If information is missing, say exactly what key is missing."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": f"Experiment context: {json.dumps(summary_payload)}"})
    for item in history[-6:]:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": question})

    response = _chat_completion_with_retry(
        client,
        model=settings["deployment"],
        temperature=0.2,
        max_tokens=500,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()
