import json
import streamlit as st
import pandas as pd
from llm_client import (
    chat_completion,
    get_missing_azure_openai_settings,
    build_automl_system_prompt,
)
from run_automl import submit_automl_job
from utils import save_uploaded_file
from ml_pipeline.config import normalize_problem_type

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

    default_metric_by_problem_type = {
        "Classification": "Accuracy",
        "Regression": "RMSE",
    }
    azure_metric_by_problem_type = {
        "Classification": "accuracy",
        "Regression": "normalized_root_mean_squared_error",
    }

    # ========================================================
    # Assistant configuration
    # ========================================================
    st.subheader("Assistant")

    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [
            {
                "role": "assistant",
                "content": "Hi! Which column should the model predict?",
            }
        ]

    if "assistant_config" not in st.session_state:
        st.session_state.assistant_config = {
            "target_column": None,
            "problem_type": None,
        }

    for message in st.session_state.assistant_messages:
        st.chat_message(message["role"]).write(message["content"])

    env_missing = get_missing_azure_openai_settings()

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
            stripped_input = user_input.strip()

            if not known_config.get("target_column") and stripped_input in df.columns:
                known_config["target_column"] = stripped_input
                assistant_message = "Great. Is this Classification or Regression?"
                st.session_state.assistant_messages.append(
                    {"role": "assistant", "content": assistant_message}
                )
                st.rerun()

            if known_config.get("target_column") and not known_config.get("problem_type"):
                normalized_problem_type = normalize_problem_type(stripped_input)
                if normalized_problem_type:
                    known_config["problem_type"] = normalized_problem_type
                    assistant_message = "Perfect. I have everything needed to run AutoML."
                    st.session_state.assistant_messages.append(
                        {"role": "assistant", "content": assistant_message}
                    )
                    st.rerun()

            system_prompt = build_automl_system_prompt(
                columns_list=df.columns.tolist(),
            )

            try:
                response_text = chat_completion(
                    [{"role": "system", "content": system_prompt}]
                    + st.session_state.assistant_messages[-6:]
                )
                parsed = json.loads(response_text)
            except Exception:
                parsed = None
                response_text = (
                    "I had trouble understanding that. "
                    "Please tell me the target column name exactly as shown."
                )

            if parsed:
                candidate_target = parsed.get("target_column")
                candidate_problem_type = parsed.get("problem_type")

                if isinstance(candidate_target, str) and candidate_target in df.columns:
                    known_config["target_column"] = candidate_target

                if isinstance(candidate_problem_type, str):
                    known_config["problem_type"] = normalize_problem_type(candidate_problem_type)

                assistant_message = parsed.get("message") or "Thanks. What else should I know?"
            else:
                assistant_message = response_text

            if known_config.get("target_column") and not known_config.get("problem_type"):
                assistant_message = "Great. Is this Classification or Regression?"

            st.session_state.assistant_messages.append(
                {"role": "assistant", "content": assistant_message}
            )
            st.rerun()

    assistant_config = st.session_state.assistant_config
    target_column = assistant_config.get("target_column")
    final_problem_type = assistant_config.get("problem_type")
    selected_metric = default_metric_by_problem_type.get(final_problem_type)

    if target_column and final_problem_type:
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

        st.table(config_df.astype(str))

        st.info(
            "This configuration will be used to set up the Azure AutoML experiment. "
            "You will be able to adjust compute and cost limits in the next step."
        )

        # ========================================================
        # Launch AutoML
        # ========================================================
        st.subheader("Run AutoML")

        if st.button(
            "Start AutoML experiment 🚀",
            help="This will submit an AutoML job to Azure using the configuration above.",
            key="assistant_run_automl",
        ):
            with st.spinner("Submitting AutoML job to Azure..."):
                try:
                    azure_metric = azure_metric_by_problem_type[final_problem_type]
                    data_name = f"data-{uploaded_file.name.replace('.csv', '')}"

                    job_name = submit_automl_job(
                        csv_path=str(csv_path),
                        target_column=target_column,
                        problem_type=final_problem_type,
                        primary_metric=azure_metric,
                        data_name=data_name,
                        experiment_name="streamlit-automl-demo",
                    )

                    st.success(f"✅ Job submitted successfully: {job_name}")
                    st.info("View your job in Azure ML Studio")

                except Exception as e:
                    st.error(f"❌ Error submitting job: {e}")
                    st.exception(e)