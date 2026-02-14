import os
from openai import AzureOpenAI


def get_azure_openai_settings() -> dict[str, str | None]:
    return {
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    }


def get_missing_azure_openai_settings() -> list[str]:
    settings = get_azure_openai_settings()
    required_settings = {
        "AZURE_OPENAI_ENDPOINT": settings["endpoint"],
        "AZURE_OPENAI_API_KEY": settings["api_key"],
        "AZURE_OPENAI_DEPLOYMENT": settings["deployment"],
    }

    missing = []
    for name, value in required_settings.items():
        if not value:
            missing.append(name)

    return missing


def build_automl_system_prompt(
    *,
    columns_list,
) -> str:
    return (
        "You are an assistant that helps configure an AutoML job. "
        "Return ONLY valid JSON with keys: target_column, problem_type, message. "
        "Use null for unknown values. "
        "Always ask the user for the problem type if it is missing. "
        "Allowed problem_type values: Classification or Regression. "
        f"Available columns: {columns_list}. "
        "Do not ask the user to choose a metric. Metric is automatic. "
        "If user asks for unsupported task type, ask them to choose Classification or Regression. "
        "If the user provides a column not in the list, set target_column to null "
        "and ask them to choose from the list. "
        "Ask only one clear follow-up question in message."
    )


def chat_completion(messages, temperature=0.2, max_tokens=300) -> str:
    settings = get_azure_openai_settings()
    endpoint = settings["endpoint"]
    api_key = settings["api_key"]
    deployment = settings["deployment"]
    api_version = settings["api_version"]

    if not endpoint or not api_key or not deployment:
        missing = get_missing_azure_openai_settings()
        raise ValueError(
            "Missing Azure OpenAI settings: " + ", ".join(missing)
        )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content
