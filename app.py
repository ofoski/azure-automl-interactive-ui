import json
import os
import streamlit as st
import pandas as pd
from llm_client import chat_completion
from run_automl import submit_automl_job
from utils import save_uploaded_file, get_metric_mapping, safe_preview, normalize_config_table

st.set_page_config(page_title="AutoML Demo", layout="wide")

st.title("Interactive AutoML Demo")
st.caption("Upload a dataset and configure your AutoML experiment")

# ============================================================
# 1. Dataset upload
# ============================================================
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    # Save uploaded file to disk
    csv_path = save_uploaded_file(uploaded_file)
    
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully")

    # ========================================================
    # 2. Input mode selection (left side)
    # ========================================================
    input_mode = st.sidebar.selectbox(
        "Configuration method",
        options=["Manual", "Assistant"],
        index=0
    )

    classification_metrics = ["Accuracy", "AUC", "F1", "Precision", "Recall"]
    regression_metrics = ["RMSE", "MAE", "R2"]

    if input_mode == "Manual":
        # ========================================================
        # Manual configuration
        # ========================================================
        st.subheader("Target configuration")

        target_column = st.selectbox(
            "Select the target column",
            options=df.columns.tolist()
        )

        st.write(f"**Selected target:** {target_column}")
        st.divider()

        target_series = df[target_column]

        unique_ratio = target_series.nunique() / len(target_series)

        if target_series.dtype in ["object", "bool"]:
            inferred_problem_type = "Classification"
        elif unique_ratio < 0.05:
            inferred_problem_type = "Classification"
        else:
            inferred_problem_type = "Regression"

        st.subheader("Inferred problem type")

        st.markdown(
            f"🧠 Based on the target column, this looks like **{inferred_problem_type}**."
        )

        st.caption(
            "This is an automatic guess based on the target column. "
            "Please confirm the correct problem type below."
        )

        final_problem_type = st.radio(
            "Please confirm the problem type ❓",
            options=["Classification", "Regression"],
            index=0 if inferred_problem_type == "Classification" else 1,
            help=(
                "Classification predicts discrete classes (e.g. 0/1 or categories).\n\n"
                "Regression predicts continuous numeric values (e.g. price, age)."
            )
        )

        st.success(f"Final problem type: **{final_problem_type}**")

        st.subheader("Evaluation metric")

        if final_problem_type == "Classification":
            available_metrics = classification_metrics
        else:
            available_metrics = regression_metrics

        selected_metric = st.selectbox(
            "Select the primary evaluation metric ❓",
            options=available_metrics,
            help="This metric will be used by AutoML to select the best model."
        )

        st.success(f"Selected metric: **{selected_metric}**")

        # ========================================================
        # Dataset inspection (manual only)
        # ========================================================
        with st.expander("Dataset overview"):
            col1, col2 = st.columns(2)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])

        with st.expander("Preview dataset"):
            st.dataframe(safe_preview(df), width="stretch")

        with st.expander("Show column names"):
            st.markdown(
                ", ".join(f"`{col}`" for col in df.columns)
            )

        if target_column and final_problem_type and selected_metric:
            # ========================================================
            # Review AutoML configuration
            # ========================================================
            st.subheader("Review configuration")

            config_df = pd.DataFrame(
                {
                    "Setting": [
                        "Target column",
                        "Problem type",
                        "Primary metric",
                        "Number of rows",
                        "Number of columns",
                    ],
                    "Value": [
                        target_column,
                        final_problem_type,
                        selected_metric,
                        df.shape[0],
                        df.shape[1],
                    ],
                }
            )

            config_df["Value"] = config_df["Value"].astype(str)
            st.table(normalize_config_table(config_df))

            st.info(
                "This configuration will be used to set up the Azure AutoML experiment. "
                "You will be able to adjust compute and cost limits in the next step."
            )

            # ========================================================
            # Launch AutoML
            # ========================================================
            st.subheader("Run AutoML")

            metric_mapping = get_metric_mapping()

            if st.button(
                "Start AutoML experiment 🚀",
                help="This will submit an AutoML job to Azure using the configuration above.",
                key="manual_run_automl",
            ):
                with st.spinner("Submitting AutoML job to Azure..."):
                    try:
                        azure_metric = metric_mapping[selected_metric]
                        data_name = f"data-{uploaded_file.name.replace('.csv', '')}"

                        job_name = submit_automl_job(
                            csv_path=str(csv_path),
                            target_column=target_column,
                            problem_type=final_problem_type,
                            primary_metric=azure_metric,
                            data_name=data_name,
                            experiment_name="streamlit-automl-demo"
                        )

                        st.success(f"✅ Job submitted successfully: {job_name}")
                        st.info("View your job in Azure ML Studio")

                    except Exception as e:
                        st.error(f"❌ Error submitting job: {e}")
                        st.exception(e)

    if input_mode == "Assistant":
        # ========================================================
        # Assistant configuration
        # ========================================================
        st.subheader("Assistant")

        if "assistant_messages" not in st.session_state:
            st.session_state.assistant_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "Hi! Which column should the model predict? "
                        "Please type the column name."
                    ),
                }
            ]

        if "assistant_config" not in st.session_state:
            st.session_state.assistant_config = {
                "target_column": None,
                "problem_type": None,
                "primary_metric": None,
            }

        for message in st.session_state.assistant_messages:
            st.chat_message(message["role"]).write(message["content"])

        secrets = st.secrets if hasattr(st, "secrets") else {}
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or secrets.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY") or secrets.get("AZURE_OPENAI_API_KEY")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or secrets.get("AZURE_OPENAI_DEPLOYMENT")

        if endpoint and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
            os.environ["AZURE_OPENAI_ENDPOINT"] = endpoint
        if api_key and not os.environ.get("AZURE_OPENAI_API_KEY"):
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
        if deployment and not os.environ.get("AZURE_OPENAI_DEPLOYMENT"):
            os.environ["AZURE_OPENAI_DEPLOYMENT"] = deployment

        env_missing = [
            name
            for name, value in [
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_API_KEY", api_key),
                ("AZURE_OPENAI_DEPLOYMENT", deployment),
            ]
            if not value
        ]

        if env_missing:
            st.warning(
                "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT as environment "
                "variables or Streamlit secrets."
            )
        else:
            user_input = st.chat_input("Type your answer")

            if user_input:
                st.session_state.assistant_messages.append(
                    {"role": "user", "content": user_input}
                )

                known_config = st.session_state.assistant_config
                columns_list = df.columns.tolist()

                system_prompt = (
                    "You are an assistant that helps configure an AutoML job. "
                    "Return ONLY valid JSON with keys: target_column, problem_type, "
                    "primary_metric, message. Use null for unknown values. "
                    "Allowed problem_type values: Classification, Regression. "
                    f"Available columns: {columns_list}. "
                    f"Classification metrics: {classification_metrics}. "
                    f"Regression metrics: {regression_metrics}. "
                    f"Known config: {known_config}. "
                    "If the user provides a column not in the list, set target_column to null "
                    "and ask them to choose from the list. "
                    "Ask only one clear follow-up question in message."
                )

                try:
                    response_text = chat_completion(
                        [{"role": "system", "content": system_prompt}]
                        + st.session_state.assistant_messages[-6:]
                    )

                    parsed = json.loads(response_text)
                except Exception as e:
                    parsed = None
                    response_text = (
                        "I had trouble understanding that. "
                        "Please tell me the target column name exactly as shown."
                    )

                if parsed:
                    candidate_target = parsed.get("target_column")
                    candidate_problem_type = parsed.get("problem_type")
                    candidate_metric = parsed.get("primary_metric")

                    if isinstance(candidate_target, str) and candidate_target in df.columns:
                        known_config["target_column"] = candidate_target

                    if isinstance(candidate_problem_type, str):
                        normalized = candidate_problem_type.strip().lower()
                        if normalized == "classification":
                            known_config["problem_type"] = "Classification"
                        elif normalized == "regression":
                            known_config["problem_type"] = "Regression"

                    if isinstance(candidate_metric, str):
                        metric_map = {
                            m.lower(): m for m in (classification_metrics + regression_metrics)
                        }
                        normalized_metric = candidate_metric.strip().lower()
                        if normalized_metric in metric_map:
                            known_config["primary_metric"] = metric_map[normalized_metric]

                    assistant_message = parsed.get("message")
                    if not assistant_message:
                        assistant_message = "Thanks. What else should I know?"
                else:
                    assistant_message = response_text

                if known_config.get("target_column") and not known_config.get("problem_type"):
                    target_series = df[known_config["target_column"]]
                    unique_ratio = target_series.nunique() / len(target_series)

                    if target_series.dtype in ["object", "bool"]:
                        known_config["problem_type"] = "Classification"
                    elif unique_ratio < 0.05:
                        known_config["problem_type"] = "Classification"
                    else:
                        known_config["problem_type"] = "Regression"

                if known_config.get("primary_metric") and known_config.get("problem_type"):
                    if known_config["problem_type"] == "Classification" and known_config["primary_metric"] not in classification_metrics:
                        known_config["primary_metric"] = None
                    if known_config["problem_type"] == "Regression" and known_config["primary_metric"] not in regression_metrics:
                        known_config["primary_metric"] = None

                st.session_state.assistant_messages.append(
                    {"role": "assistant", "content": assistant_message}
                )
                st.rerun()

        assistant_config = st.session_state.assistant_config
        target_column = assistant_config.get("target_column")
        final_problem_type = assistant_config.get("problem_type")
        selected_metric = assistant_config.get("primary_metric")

        if target_column and final_problem_type and selected_metric:
            # ========================================================
            # Review AutoML configuration
            # ========================================================
            st.subheader("Review configuration")

            config_df = pd.DataFrame(
                {
                    "Setting": [
                        "Target column",
                        "Problem type",
                        "Primary metric",
                        "Number of rows",
                        "Number of columns",
                    ],
                    "Value": [
                        target_column,
                        final_problem_type,
                        selected_metric,
                        df.shape[0],
                        df.shape[1],
                    ],
                }
            )

            config_df["Value"] = config_df["Value"].astype(str)
            st.table(normalize_config_table(config_df))

            st.info(
                "This configuration will be used to set up the Azure AutoML experiment. "
                "You will be able to adjust compute and cost limits in the next step."
            )

            # ========================================================
            # Launch AutoML
            # ========================================================
            st.subheader("Run AutoML")

            metric_mapping = get_metric_mapping()

            if st.button(
                "Start AutoML experiment 🚀",
                help="This will submit an AutoML job to Azure using the configuration above.",
                key="assistant_run_automl",
            ):
                with st.spinner("Submitting AutoML job to Azure..."):
                    try:
                        azure_metric = metric_mapping[selected_metric]
                        data_name = f"data-{uploaded_file.name.replace('.csv', '')}"

                        job_name = submit_automl_job(
                            csv_path=str(csv_path),
                            target_column=target_column,
                            problem_type=final_problem_type,
                            primary_metric=azure_metric,
                            data_name=data_name,
                            experiment_name="streamlit-automl-demo"
                        )

                        st.success(f"✅ Job submitted successfully: {job_name}")
                        st.info("View your job in Azure ML Studio")

                    except Exception as e:
                        st.error(f"❌ Error submitting job: {e}")
                        st.exception(e)