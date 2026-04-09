"""Responsible AI agent powered by LangChain + LangGraph + Azure OpenAI."""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

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

    "SCOPE RESTRICTION:\n"
    "Only answer questions related to:\n"
    "- The model's predictions and performance\n"
    "- The dataset the model was trained on\n"
    "- Responsible AI topics: fairness, explainability, errors, counterfactuals\n"
    "- General ML concepts that help the user understand the analysis\n"
    "If the user asks anything completely unrelated to the model or Responsible AI\n"
    "(e.g., coding help, web scraping, unrelated topics), respond with:\n"
    "'I am a Responsible AI assistant. I can only help with questions about\n"
    "your model, dataset, and Responsible AI analysis.'\n\n"

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

    "- get_fairness_analysis: returns performance metrics per group.\n"
    "FAIRNESS ANALYSIS RULES:\n"
    "1. SENSITIVE FEATURES ONLY:\n"
    "   Before calling get_fairness_analysis, look at the feature names in DATA CONTEXT.\n"
    "   Identify which features are human-sensitive by reasoning about their names\n"
    "   and value ranges — do not rely on exact keyword matching.\n"
    "   Human-sensitive features are those that describe human characteristics such as:\n"
    "   demographic attributes, socioeconomic status, biological traits, or identity.\n"
    "   If no such features exist: DO NOT call get_fairness_analysis at all.\n"
    "   Respond immediately with:\n"
    "   'No human-sensitive features detected — fairness analysis is not applicable\n"
    "   for this dataset. Consider using error analysis to explore where the model\n"
    "   underperforms across different data segments.'\n\n"
    "2. HOW TO DETERMINE IF THE MODEL IS FAIR:\n"
    "   Compare the metric values across all groups of each sensitive feature.\n"
    "   Always explicitly report:\n"
    "   - Best performing group\n"
    "   - Worst performing group\n"
    "   - Gap = worst - best\n"
    "   - If the gap is small → the model is fair for this feature.\n"
    "   - If the gap is large → the model shows a performance gap for that group.\n\n"
    "3. SAMPLE SIZE RULE:\n"
    "   Always report sample size (n) for each group.\n"
    "   If n < 30 → prepend ⚠️ Small sample — result may not be reliable.\n\n"
    "4. LANGUAGE RULES:\n"
    "   Never say 'biased' — say 'the model shows a performance gap.'\n"
    "   Always present both possibilities:\n"
    "   (a) the gap may reflect genuine differences in the data.\n"
    "   (b) the gap may indicate the model needs improvement for that group.\n"
    "   Let the user draw their own conclusion.\n\n"

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

    "- get_counterfactuals: returns minimal feature changes that would alter the prediction.\n"
    "COUNTERFACTUAL ANALYSIS RULES:\n"
    "The tool gives you a list of suggestions. Before presenting any of them, reason about\n"
    "whether a real person could actually act on that change, based on what each feature\n"
    "represents in the domain you identified from DATA CONTEXT.\n\n"
    "A feature change is not actionable if it involves:\n"
    "- A biological or physical trait a person is born with\n"
    "- A demographic or identity attribute that does not change over time\n"
    "- A past event — something that has already happened and cannot be undone\n"
    "- An identifier column that labels an entity but has no causal meaning\n"
    "For all other features, use domain reasoning: ask yourself whether a person could\n"
    "realistically take a concrete step to change that value in the context of this dataset.\n\n"
    "Also check that the suggested values are realistic: use the mean and std from\n"
    "DATA CONTEXT and discard any suggestion where a changed value falls outside\n"
    "mean ± 3 × std — those values do not occur in practice.\n\n"
    "After presenting valid suggestions, if you excluded any because they involved\n"
    "non-actionable features, add one brief note at the end explaining what was excluded\n"
    "and why it cannot be changed in real life.\n"
    "If nothing is valid, say: 'No actionable changes found — try a different row.'\n\n"

    "TOOL CALLING STRATEGY:\n"
    "- MODEL COMPARISON (compare trained models, which algorithm was best, training scores, "
    "fit times, algorithm names): answer DIRECTLY from the model comparison table in "
    "DOMAIN CONTEXT. Do NOT call any tools — the table already contains the answer.\n"
    "- For broad questions about model quality or full reports: "
    "call all four tools and synthesize findings into a cohesive summary.\n"
    "- For questions about worst performing group or where the model struggles: "
    "call get_error_analysis only — it covers all groups and segments.\n"
    "- For questions about fairness, bias, or discrimination: "
    "first check DATA CONTEXT for human-sensitive features. "
    "Only call get_fairness_analysis if such features exist.\n"
    "- For questions like 'does the model treat all groups equally': "
    "this is a fairness question. Apply the FAIRNESS ANALYSIS RULES — "
    "check for human-sensitive features first. "
    "If none exist, do not call any tool. Respond that fairness analysis requires human-sensitive "
    "features and suggest error analysis instead.\n"
    "- For specific questions: call only the relevant tool.\n"
    "- If the user asks a question similar to a previous one: summarize briefly "
    "and refer back rather than repeating the full answer.\n\n"

    "RESPONSE GUIDELINES:\n"
    "- CURRENCY FORMATTING: never use the $ symbol for money values in your response. "
    "It causes rendering errors. Write currency as plain numbers with commas, "
    "e.g. write 106,067 not $106,067.\n"
    "- For error analysis: always report group sample size when results seem unusual.\n"
    "- For fairness: use measured language — say the model shows a performance gap "
    "rather than the model is biased. Avoid definitive causal claims.\n"
    "- When a group performs poorly on a metric that aligns with its expected "
    "real world risk profile, note that this may reflect genuine risk differences "
    "rather than model bias — distinguish between the two carefully.\n"
    "- When making recommendations: ground them in the data statistics and domain context.\n\n"

    "INTERPRETING METRICS CORRECTLY:\n"
    "- A group showing low selection rate does not mean the model favors or "
    "discriminates against that group — it simply means the model rarely predicts "
    "the positive outcome for them. Always interpret selection rate in the context "
    "of what the positive outcome means for that specific dataset.\n"
    "- A group showing high selection rate does not automatically mean unfair treatment — "
    "it means the model frequently predicts the positive outcome for that group. "
    "Reason about whether this aligns with real world expectations before concluding bias.\n"
    "- Performance differences between groups do not always indicate bias — "
    "some groups may genuinely have different risk profiles or characteristics "
    "that the model correctly learned. "
    "Always reason about whether a performance gap reflects real world differences "
    "or unexpected model behavior before drawing conclusions.\n"
    "- Never label a performance difference as bias without reasoning about "
    "whether it makes sense in the real world context of the dataset.\n"
    "- When reporting findings always distinguish between: "
    "'the model shows a performance gap for this group' "
    "and 'this gap may reflect genuine differences in the data' — "
    "present both possibilities and let the user draw their own conclusions.\n"
)

# Shown to the user when the guardrail blocks a request.

_INJECTION_BLOCKED = (
    "⛔ **Blocked.** Your message appears to be a prompt injection attempt."
)

_OUT_OF_SCOPE = (
    "⚠️ **Out of scope.**\n\n"
    "I am a Responsible AI assistant. "
    "I can only help with questions about your model, dataset, fairness, "
    "errors, feature importance, and counterfactuals."
)


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

    if not endpoint:
        return "❌ AZURE_OPENAI_ENDPOINT is not set. Run: $env:AZURE_OPENAI_ENDPOINT = 'https://<resource>.openai.azure.com/'"
    if "services.ai.azure.com" in endpoint or "/api/projects/" in endpoint:
        return "❌ Wrong endpoint — set the Azure OpenAI endpoint (ends in .openai.azure.com/), not the Foundry project URL."
    if not key:
        return "❌ AZURE_OPENAI_API_KEY is not set. Run: $env:AZURE_OPENAI_API_KEY = '<your-key>'"
    if not model_name:
        return "❌ AZURE_OPENAI_DEPLOYMENT is not set. Run: $env:AZURE_OPENAI_DEPLOYMENT = '<your-deployment>'"

    # Each tool closes over `context` and `cache` so the agent can call them
    # with no arguments. Results are cached to avoid repeating expensive computations.

    @tool
    def get_permutation_importance() -> str:
        """Feature importance scores via permutation importance. Returns features ranked by importance_mean (higher = more influential). Use when the user asks which features matter most, drive predictions, or have the biggest impact on the model."""
        if "importance" not in cache:
            cache["importance"] = run_permutation_importance(
                context["model_name"], context["test_asset"], context["target_column"], context["model_version"]
            ).to_string(index=False)
        return cache["importance"]

    @tool
    def get_error_analysis() -> str:
        """Average model error per feature group or bin. For regression: mean absolute error per bin. For classification: misclassification rate per group. Use when the user asks where the model makes the most mistakes, underperforms, or struggles with specific groups."""
        if "error" not in cache:
            cache["error"] = "\n\n".join(
                f"Feature: {f}\n{s.to_string()}"
                for f, s in error_analysis(context["model"], context["X_test"], context["y_test"], context["task_type"]).items()
            )
        return cache["error"]

    @tool
    def get_fairness_analysis() -> str:
        """Per-group fairness metrics for each feature plus a gap score. For regression: MAE per group. For classification: accuracy, selection_rate, true_positive_rate per group. Gap = max difference between groups — larger gap means more unfair treatment. Use when the user asks about bias, fairness, discrimination, or group disparities."""
        if "fairness" not in cache:
            cache["fairness"] = "\n\n".join(
                f"Feature: {f}\n{df.to_string()}"
                for f, df in fairness_analysis(context["model"], context["X_test"], context["y_test"], context["task_type"]).items()
            )
        return cache["fairness"]

    @tool
    def get_counterfactuals(instance_index: int = 0, total_cfs: int = 5) -> str:
        """Minimal feature changes that would alter the model prediction for a specific row.
        Use for what-if questions or to find out what would need to change to flip a prediction.
        The user can specify a row with instance_index (0-based). 'row 10' means instance_index=10."""
        key = f"cf_{instance_index}_{total_cfs}"
        if key not in cache:
            ex = run_counterfactuals(
                context["model"], context["X_train"], context["y_train"],
                context["X_test"], context["y_test"], context["task_type"],
                context["target_column"], instance_index,
                "opposite" if context["task_type"] == "classification" else None,
                total_cfs,
            ).cf_examples_list[0]
            cache[key] = (
                f"Original instance (index {instance_index}):\n{ex.test_instance_df.to_string(index=False)}\n\n"
                f"Counterfactuals:\n"
                f"{ex.final_cfs_df.to_string(index=False) if ex.final_cfs_df is not None else 'None generated.'}"
            )
        return cache[key]

    tools = [get_permutation_importance, get_error_analysis, get_fairness_analysis, get_counterfactuals]

    try:
        # Connect to the Azure OpenAI deployment. temperature=0 keeps answers deterministic.
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            azure_deployment=model_name,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            temperature=0,
        )

        # Guardrail: allowlist check — only pass messages that are clearly on-topic.
        guard = llm.bind(max_tokens=3).invoke([
            SystemMessage(content="You are a topic classifier. Reply only YES or NO."),
            HumanMessage(content=(
                f"Is this message about machine learning model performance, fairness, errors, "
                f"feature importance, counterfactuals, or Responsible AI?\n\nMessage: {user_message}"
            )),
        ])

        if not guard.content.strip().upper().startswith("YES"):
            return _OUT_OF_SCOPE

        # Build the context strings that fill {data_context} and {domain_context}
        # placeholders inside SYSTEM_PROMPT. This gives the LLM live statistics
        # about the current dataset so it can give grounded, specific answers.
        data_context = build_data_context(context["X_test"], context["y_test"], context["target_column"])
        domain_context = (
            f"Model: {context['model_name']} v{context['model_version']}\n"
            f"Task type: {context['task_type']}\n"
            f"Target column: {context['target_column']}\n"
        )
        if context.get("model_comparison"):
            domain_context += f"\nAll models trained in this job (best=True marks the selected model):\n{context['model_comparison']}\n"
        system_prompt = SYSTEM_PROMPT.format(data_context=data_context, domain_context=domain_context)

        # create_react_agent wires the LLM to the tools in a loop:
        # think → pick tool → run tool → think again → … → write final answer.
        agent = create_react_agent(llm, tools)

        # Build the message list: system prompt + last 20 conversation turns + current question.
        # Keeping a rolling window limits token usage while still giving the LLM memory of the session.
        messages = [
            SystemMessage(content=system_prompt),
            *convert_to_messages((chat_history or [])[-20:]),
            HumanMessage(content=user_message),
        ]

        # Run the agent. recursion_limit=16 allows up to 8 tool calls before stopping.
        result = agent.invoke({"messages": messages}, {"recursion_limit": 16})
        output = result["messages"][-1].content

        # If the agent used all steps without producing a final answer, show a
        # friendly message instead of an empty or confusing internal response.
        if not output or "agent stopped" in output.lower():
            return (
                "\u26a0\ufe0f **The agent ran out of steps before finishing your request.**\n\n"
                "What happened: the agent calls analysis tools step by step. "
                "For very broad questions it can use up all 7 allowed steps without producing a final answer.\n\n"
                "**Try asking something more specific**, for example:\n"
                "- *'Which features matter most for predictions?'*\n"
                "- *'Where does the model make the most mistakes?'*\n"
                "- *'Is there a fairness gap between groups?'*\n"
                "- *'Compare the models trained in this job'*"
            )
        return output

    except Exception as exc:
        # Distinguish between content-policy blocks, connection issues, and
        # unexpected errors so the user gets a clear, actionable message.
        if "content_filter" in str(exc).lower() or "jailbreak" in str(exc).lower():
            return _INJECTION_BLOCKED
        if "connection" in str(exc).lower():
            return (
                f"❌ **Connection error** — could not reach Azure OpenAI.\n\n"
                f"Check that the endpoint is correct (must end with `.openai.azure.com/`):\n"
                f"`{endpoint}`"
            )
        return f"❌ Agent error: {exc}"


