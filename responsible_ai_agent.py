"""Responsible AI Agent — Azure OpenAI function-calling backend.

This module exposes ``run_agent()`` for use by the Streamlit app (app.py).
All UI code lives in app.py; this file contains only the agent logic.

Prerequisites — set in PowerShell before running the app
---------------------------------------------------------
    $env:AZURE_OPENAI_ENDPOINT = "https://<resource>.openai.azure.com/"
    $env:AZURE_OPENAI_API_KEY  = "<your-api-key>"

The app is started with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os

from openai import AzureOpenAI

from responsible_ai_analysis import (
    build_data_context,
    error_analysis,
    fairness_analysis,
    run_counterfactuals,
    run_permutation_importance,
)

SYSTEM_PROMPT = (
    "You are a Responsible AI assistant for machine learning models trained with Azure AutoML.\n"
    "You have access to four analysis tools. Call the appropriate tool(s) based on the user "
    "question, then explain the results in plain language for a non-technical stakeholder.\n\n"

    "DATA CONTEXT:\n"
    "{data_context}\n\n"

    "DOMAIN CONTEXT:\n"
    "{domain_context}\n\n"

    "DATA UNDERSTANDING INSTRUCTIONS:\n"
    "Analyze the data context above and reason about:\n"
    "- What domain this dataset likely belongs to based on column names and value ranges\n"
    "- What the target column represents and what predicting it means in the real world\n"
    "- Which features are sensitive from a fairness perspective based on their names and distributions\n"
    "- Which features are actionable based on their nature and value ranges\n"
    "- What constitutes a large vs small error based on the target distribution\n"
    "- What value ranges are realistic for counterfactual changes based on min/max/mean/std\n"
    "Use this understanding to give domain-aware, specific, and meaningful answers.\n"
    "Ground all recommendations in the actual data statistics — "
    "do not suggest changes outside the realistic range of the data.\n\n"

    "TOOL GUIDANCE:\n"
    "- get_permutation_importance: returns features ranked by importance_mean. "
    "Higher = more influential. Values near 0 or negative mean negligible impact — "
    "describe these as not contributing, not hurting the model.\n"
    "- get_error_analysis: returns mean error per group or bin. "
    "Focus on features with large gaps between best and worst group. "
    "If a group has very few samples, note that its error rate may be unreliable.\n"
    "- get_fairness_analysis: returns performance metrics per group plus a gap score. "
    "For regression: MAE per group. "
    "For classification: accuracy, selection_rate, true_positive_rate per group. "
    "Gap above 0.1 for classification or above 20 percent of mean MAE for regression "
    "is a fairness concern. Always mention which group has the highest gap.\n"
    "- get_counterfactuals: returns minimal feature changes that would alter the prediction. "
    "Explain each change in plain terms such as increasing income from 30,000 to 45,000. "
    "Always separate actionable vs non-actionable changes in your explanation.\n\n"

    "TOOL CALLING STRATEGY:\n"
    "- For broad questions about model quality or full reports: "
    "call all four tools and synthesize findings into a cohesive summary.\n"
    "- For questions about worst performing group: call both get_error_analysis "
    "and get_fairness_analysis and combine the results.\n"
    "- For specific questions: call only the relevant tool.\n"
    "- If the user asks a question similar to a previous one: summarize briefly "
    "and refer back rather than repeating the full answer.\n\n"

    "ACTIONABILITY AND FEATURE REASONING:\n"
    "Before presenting any counterfactual change, reason about whether that change "
    "is realistically possible based on the data statistics and real world context.\n\n"

    "Features that are generally NOT actionable:\n"
    "- Immutable biological characteristics a person is born with\n"
    "- Identity and demographic attributes that cannot be changed\n"
    "- High cardinality identifier columns such as ID, ticket number, name — "
    "these are unique per row and carry no predictive meaning\n"
    "- Geographic coordinates such as latitude and longitude\n"
    "- Timestamps and dates of past events\n\n"

    "Features that are generally actionable:\n"
    "- Financial attributes such as income, savings, payments\n"
    "- Choice-based attributes such as product type, subscription, plan\n"
    "- Behavioral attributes such as spending, usage, activity frequency\n"
    "- Attributes that represent decisions a person can consciously make\n\n"

    "Borderline features:\n"
    "- Features like country, region, or zip code can be changed but require "
    "significant life decisions — mention the change but note it may not be "
    "practical for everyone.\n\n"

    "High cardinality features:\n"
    "- If a feature has many unique values such as IDs, names, or codes, "
    "flag it as not meaningful for analysis and do not draw conclusions from it.\n\n"

    "RESPONSE GUIDELINES:\n"
    "- Always call a tool before answering questions about model behaviour.\n"
    "- Summarise numerical results — do not paste raw tables.\n"
    "- Highlight the most important finding first.\n"
    "- For error analysis: note when a group has very few samples as results may be unreliable.\n"
    "- For counterfactuals: always end with practical recommendations using only "
    "actionable features. If no actionable changes exist, explicitly tell the user.\n"
    "- For fairness: use measured language — say the model shows a performance gap "
    "rather than the model is biased. Avoid definitive causal claims — say "
    "associated with rather than caused by.\n"
    "- When making recommendations: ground them in the data statistics and domain context. "
    "Avoid generic statements — be specific about what the data suggests.\n"
    "- Only use information returned by tools — do not guess or invent values.\n"
)

_TOOLS = [
    {"type": "function", "function": {
        "name": "get_permutation_importance",
        "description": (
            "Feature importance scores via permutation importance. "
            "Returns features ranked by importance_mean (higher = more influential). "
            "Use when the user asks which features matter most, drive predictions, "
            "or have the biggest impact on the model."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_error_analysis",
        "description": (
            "Average model error per feature group or bin. "
            "For regression: mean absolute error per bin. "
            "For classification: misclassification rate per group. "
            "Use when the user asks where the model makes the most mistakes, "
            "underperforms, or struggles with specific groups."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_fairness_analysis",
        "description": (
            "Per-group fairness metrics for each feature plus a gap score. "
            "For regression: MAE per group. "
            "For classification: accuracy, selection_rate, true_positive_rate per group. "
            "Gap = max difference between groups — larger gap means more unfair treatment. "
            "Use when the user asks about bias, fairness, discrimination, or group disparities."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "get_counterfactuals",
        "description": (
            "Minimal feature changes that would alter the model prediction for a specific instance. "
            "Shows original input and counterfactual alternatives. "
            "Use for what-if questions, how to improve an outcome, "
            "or what changes would flip a prediction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instance_index": {
                    "type": "integer",
                    "description": "Test instance index (0-based). Default 0."
                },
                "total_cfs": {
                    "type": "integer",
                    "description": "Number of counterfactuals to generate. Default 3."
                },
            },
            "required": [],
        },
    }},
]


def run_agent(
    user_message: str,
    context: dict,
    cache: dict,
    openai_endpoint: str = "",
    api_key: str = "",
    chat_history: list[dict] | None = None,
    deployment: str = "",
) -> str:
    """Run one turn of the RAI agent using Azure OpenAI function-calling.

    Parameters
    ----------
    user_message : str
        The user's question.
    context : dict
        Keys: model, X_test, y_test, X_train, y_train, task_type,
              model_name, model_version, target_column, test_asset.
    cache : dict
        Mutable dict for caching tool results between calls.  Pass a
        persistent dict (e.g. from ``st.session_state``) so results survive
        Streamlit reruns.
    openai_endpoint : str, optional
        Azure OpenAI endpoint URL.  Falls back to ``AZURE_OPENAI_ENDPOINT``.
    api_key : str, optional
        Azure OpenAI API key.  Falls back to ``AZURE_OPENAI_API_KEY``.
    chat_history : list[dict], optional
        Previous conversation turns in ``[{role, content}]`` format.
    """
    endpoint   = openai_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    key        = api_key         or os.environ.get("AZURE_OPENAI_API_KEY", "")
    model_name = deployment      or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")

    # Validate — catch the most common mis-configuration mistakes up front
    if not endpoint:
        return (
            "\u274c **AZURE_OPENAI_ENDPOINT is not set.**\n\n"
            "Set it in PowerShell before starting the app:\n"
            "```powershell\n"
            "$env:AZURE_OPENAI_ENDPOINT = 'https://<resource-name>.openai.azure.com/'\n"
            "```"
        )
    if not key:
        return (
            "\u274c **AZURE_OPENAI_API_KEY is not set.**\n\n"
            "Set it in PowerShell before starting the app:\n"
            "```powershell\n"
            "$env:AZURE_OPENAI_API_KEY = '<your-api-key>'\n"
            "```"
        )
    if "services.ai.azure.com" in endpoint or "/api/projects/" in endpoint:
        return (
            "\u274c **Wrong endpoint.**\n\n"
            "You set the **Foundry project URL** as the endpoint, but the agent needs the "
            "**Azure OpenAI endpoint** (ends in `.openai.azure.com/`).\n\n"
            "Find the correct value in **AI Foundry \u2192 your project \u2192 Overview \u2192 Azure OpenAI endpoint**, "
            "then update the env var:\n"
            "```powershell\n"
            "$env:AZURE_OPENAI_ENDPOINT = 'https://<resource-name>.openai.azure.com/'\n"
            "```"
        )
    if not model_name:
        return (
            "\u274c **AZURE_OPENAI_DEPLOYMENT is not set.**\n\n"
            "Set it to the deployment name you created in AI Foundry:\n"
            "```powershell\n"
            "$env:AZURE_OPENAI_DEPLOYMENT = 'gpt-4o-mini'\n"
            "```"
        )

    # ------------------------------------------------------------------
    # Tool closures — each one closes over `context` and `cache`
    # ------------------------------------------------------------------

    def get_permutation_importance() -> str:
        """Returns feature importance scores using permutation importance.
        Use when the user asks which features matter most or drive predictions."""
        if "importance" not in cache:
            df = run_permutation_importance(
                model_name=context["model_name"],
                test_asset_name=context["test_asset"],
                target_column=context["target_column"],
                model_version=context["model_version"],
            )
            cache["importance"] = df.to_string(index=False)
        return cache["importance"]

    def get_error_analysis() -> str:
        """Returns average model error per feature group or bin.
        Use when the user asks where the model makes the most mistakes or underperforms."""
        if "error" not in cache:
            results = error_analysis(
                model=context["model"],
                X_test=context["X_test"],
                y_test=context["y_test"],
                task_type=context["task_type"],
            )
            parts = [f"Feature: {f}\n{s.to_string()}" for f, s in results.items()]
            cache["error"] = "\n\n".join(parts)
        return cache["error"]

    def get_fairness_analysis() -> str:
        """Returns per-group fairness metrics for each feature.
        Use when the user asks about bias, fairness, or group disparities."""
        if "fairness" not in cache:
            results = fairness_analysis(
                model=context["model"],
                X_test=context["X_test"],
                y_test=context["y_test"],
                task_type=context["task_type"],
            )
            parts = [f"Feature: {f}\n{df.to_string()}" for f, df in results.items()]
            cache["fairness"] = "\n\n".join(parts)
        return cache["fairness"]

    def get_counterfactuals(instance_index: int = 0, total_cfs: int = 3) -> str:
        """Generates counterfactual examples showing minimal changes that would alter the prediction.
        Use when the user asks what-if questions or how to improve an outcome.
        :param instance_index: Index of the test instance (0-based). Default 0.
        :type instance_index: int
        :param total_cfs: Number of counterfactuals to generate. Default 3.
        :type total_cfs: int
        """
        key = f"cf_{instance_index}_{total_cfs}"
        if key not in cache:
            task_type = context["task_type"]
            cf = run_counterfactuals(
                model=context["model"],
                X_train=context["X_train"],
                y_train=context["y_train"],
                X_test=context["X_test"],
                y_test=context["y_test"],
                task_type=task_type,
                target_column=context["target_column"],
                instance_index=instance_index,
                desired_class="opposite" if task_type == "classification" else None,
                total_cfs=total_cfs,
            )
            ex = cf.cf_examples_list[0]
            cfs_str = (
                ex.final_cfs_df.to_string(index=False)
                if ex.final_cfs_df is not None
                else "None generated."
            )
            cache[key] = (
                f"Original instance (index {instance_index}):\n"
                f"{ex.test_instance_df.to_string(index=False)}\n\n"
                f"Counterfactuals:\n{cfs_str}"
            )
        return cache[key]

    # ------------------------------------------------------------------
    # Tool dispatch — closures already defined above, keyed by name
    # ------------------------------------------------------------------
    _dispatch = {
        "get_permutation_importance": lambda a: get_permutation_importance(),
        "get_error_analysis":         lambda a: get_error_analysis(),
        "get_fairness_analysis":      lambda a: get_fairness_analysis(),
        "get_counterfactuals":        lambda a: get_counterfactuals(
            a.get("instance_index", 0), a.get("total_cfs", 3)
        ),
    }

    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

        data_context = build_data_context(
            context["X_test"], context["y_test"], context["target_column"]
        )
        domain_context = (
            f"Model: {context['model_name']} v{context['model_version']}\n"
            f"Task type: {context['task_type']}\n"
            f"Target column: {context['target_column']}\n"
        )
        system_prompt = SYSTEM_PROMPT.format(
            data_context=data_context,
            domain_context=domain_context,
        )

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for turn in (chat_history or [])[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})

        for _ in range(5):
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return msg.content

            messages.append(msg)
            for tc in msg.tool_calls:
                fn_args = json.loads(tc.function.arguments or "{}")
                result  = _dispatch[tc.function.name](fn_args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return "Agent reached max tool-call rounds. Try rephrasing your question."

    except Exception as exc:
        err = str(exc)
        if "Connection error" in err or "ConnectionError" in err or "connection" in err.lower():
            return (
                f"\u274c **Connection error** — could not reach Azure OpenAI.\n\n"
                f"Most likely cause: the endpoint URL is wrong.\n"
                f"- Current endpoint: `{endpoint}`\n"
                f"- It must end with `.openai.azure.com/` (not a Foundry project URL).\n\n"
                f"Raw error: `{exc}`"
            )
        return f"Agent error: {exc}"


