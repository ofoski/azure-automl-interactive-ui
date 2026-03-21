"""Responsible AI agent powered by Azure OpenAI function-calling."""

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
    "- Which features are sensitive from a fairness perspective\n"
    "- Which features are actionable based on their nature and value ranges\n"
    "- What constitutes a large vs small error based on the target distribution\n"
    "- What value ranges are realistic for counterfactual changes based on min/max/mean/std\n"
    "Use this understanding to give domain-aware, specific, and meaningful answers.\n"
    "Ground all recommendations in the actual data statistics.\n\n"

    "TOOL GUIDANCE:\n"
    "- get_permutation_importance: returns features ranked by importance_mean. "
    "Higher = more influential. Values near 0 or negative mean negligible impact — "
    "describe these as not contributing, not hurting the model.\n"
    "- get_error_analysis: returns mean error per group or bin. "
    "Focus on features with large gaps between best and worst group.\n"
    "- get_fairness_analysis: returns performance metrics per group plus a gap score. "
    "For regression: MAE per group. "
    "For classification: accuracy, selection_rate, true_positive_rate per group. "
    "A notable gap between groups is a fairness concern — judge significance "
    "based on the data distribution rather than a fixed threshold. "
    "Always mention which group has the highest gap.\n\n"

    "METRIC DEFINITIONS — read carefully before interpreting any fairness result:\n"
    "- Accuracy: the percentage of predictions the model got correct for that group. "
    "Higher is better.\n"
    "- Selection rate: how often the model predicts the positive outcome for a group. "
    "This reflects only what the model predicted — it has nothing to do with "
    "actual real world decisions or outcomes. "
    "Never describe selection rate as approvals, rejections, or any real world action — "
    "it is purely a measure of the model's prediction behavior.\n"
    "- True positive rate: out of all cases in a group that actually had the positive outcome, "
    "how many the model correctly identified. "
    "A low true positive rate means the model is failing to detect many real positive cases "
    "in that group — this is a significant concern as it means real cases are being missed.\n"
    "- Performance gap: the difference between the best and worst performing group. "
    "A large gap indicates the model treats groups unequally — "
    "always report which group performs best and which performs worst.\n"
    "- When a group shows extreme values such as 0% or 100% for any metric: "
    "always check and report the sample size of that group. "
    "Report the group size alongside the metric so the user can judge reliability themselves.\n\n"

    "- get_counterfactuals: returns minimal feature changes that would alter the prediction. "
    "Explain each change in plain terms. "
    "For each change clearly state whether it is realistic and applicable or not. "
    "Some features are not realistic to change — "
    "for example biological characteristics, fixed demographic attributes, "
    "characteristics tied to past events, or attributes requiring significant life changes. "
    "For these: mention them and explain their impact but state they are "
    "not realistic or applicable to change. "
    "For applicable changes: the suggested value must be realistic — "
    "cross check against the data statistics and flag any suggestion "
    "that falls far outside the normal range of that feature as unreliable. "
    "If a suggestion seems counterintuitive or goes against common sense "
    "in the real world context, flag it clearly as unreliable and do not "
    "present it as a recommendation. "
    "If all suggested changes are either unrealistic or counterintuitive, "
    "explicitly tell the user that no actionable recommendations can be made "
    "from these results and suggest they try a different instance.\n\n"

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

    "Features that are generally not realistic or applicable to change:\n"
    "- Immutable biological characteristics a person is born with\n"
    "- Identity and demographic attributes that are fixed\n"
    "- High cardinality identifier columns such as ID, ticket number, name — "
    "these carry no predictive meaning\n"
    "- Geographic coordinates such as latitude and longitude\n"
    "- Timestamps and dates of past events\n"
    "- Features that require significant life changes — "
    "mention their impact but note they are not realistic to change\n\n"

    "High cardinality features:\n"
    "- If a feature has many unique values such as IDs, names, or codes, "
    "flag it as not meaningful for analysis.\n\n"

    "RESPONSE GUIDELINES:\n"
    "- For error analysis: always report group sample size when results seem unusual.\n"
    "- For fairness: use measured language — say the model shows a performance gap "
    "rather than the model is biased. Avoid definitive causal claims.\n"
    "- When a group performs poorly on a metric that aligns with its expected "
    "real world risk profile, note that this may reflect genuine risk differences "
    "rather than model bias — distinguish between the two carefully.\n"
    "- When making recommendations: ground them in the data statistics and domain context.\n"
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
                    "description": "Number of counterfactuals to generate. Default 5."
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
    """Handle one chat turn: validate config, call Azure OpenAI with tools, return response."""
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
        """Return cached or freshly computed permutation importance results."""
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
        """Return cached or freshly computed error analysis results."""
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
        """Return cached or freshly computed fairness analysis results."""
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

    def get_counterfactuals(instance_index: int = 0, total_cfs: int = 5) -> str:
        """Return cached or freshly generated counterfactual examples."""
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

    def dispatch(name: str, args: dict) -> str:
        """Route a tool call name to the matching function."""
        if name == "get_permutation_importance":
            return get_permutation_importance()
        if name == "get_error_analysis":
            return get_error_analysis()
        if name == "get_fairness_analysis":
            return get_fairness_analysis()
        if name == "get_counterfactuals":
            return get_counterfactuals(args.get("instance_index", 0), args.get("total_cfs", 5))
        raise ValueError(f"Unknown tool: {name}")

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
        for turn in (chat_history or [])[-10:]:
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
                result = dispatch(tc.function.name, fn_args)
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


