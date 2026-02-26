import pandas as pd
import streamlit as st

from llm_client import (
    answer_results_question_with_agent,
    detect_problem_type_with_agent,
    ensure_azure_openai_ready,
)
from ml_pipeline.client import DEFAULT_LOCATION, DEFAULT_RESOURCE_GROUP, DEFAULT_WORKSPACE_NAME
from run_automl import DEFAULT_VM_SIZE, get_automl_job_details, submit_automl_job
from utils import save_uploaded_file


st.set_page_config(page_title="AutoML Demo", layout="wide")

st.title("Interactive AutoML Demo")
st.caption("Upload a dataset and run Azure AutoML with serverless compute + agent guidance")

TERMINAL_STATUSES = {"Completed", "Failed", "Canceled", "Cancelled"}
METRIC_BY_PROBLEM_TYPE = {
    "Classification": ("Accuracy", "accuracy"),
    "Regression": ("RMSE", "normalized_root_mean_squared_error"),
}


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    csv_path = save_uploaded_file(uploaded_file)
    dataframe = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully")

    st.subheader("Training setup")
    st.caption(
        "Azure setup is automatic using AZURE_SUBSCRIPTION_ID. "
        f"Resource Group: {DEFAULT_RESOURCE_GROUP}, Workspace: {DEFAULT_WORKSPACE_NAME}, "
        f"Region: {DEFAULT_LOCATION}."
    )
    if st.session_state.get("agent_ready") is not True:
        with st.spinner("Preparing Azure OpenAI agent automatically..."):
            try:
                ensure_azure_openai_ready()
                st.session_state["agent_ready"] = True
            except Exception as error:
                st.error(
                    "Agent is required and could not be prepared automatically. "
                    f"Details: {error}"
                )
                st.stop()

    target_column = st.selectbox("Target column", options=dataframe.columns.tolist(), index=0)
    detect_key = f"detection::{target_column}"
    if st.session_state.get("detection_target_column") != target_column:
        st.session_state["detection_target_column"] = target_column
        st.session_state.pop(detect_key, None)

    detection = st.session_state.get(detect_key)
    if detection is None:
        target_values = dataframe[target_column].dropna().head(60).tolist()
        with st.spinner("Agent is detecting problem type..."):
            try:
                detection = detect_problem_type_with_agent(
                    target_column=target_column,
                    non_null_count=int(dataframe[target_column].notna().sum()),
                    dtype_text=str(dataframe[target_column].dtype),
                    sample_values=target_values,
                )
                st.session_state[detect_key] = detection
            except Exception as error:
                st.error(
                    "Agent detection failed. Please verify Azure OpenAI connectivity and deployment. "
                    f"Details: {error}"
                )
                st.stop()

    problem_type = detection["problem_type"]
    detection_reason = f"[agent] {detection['reason']}"

    st.info(f"Detected problem type: {problem_type}")
    st.caption(f"Agent reason: {detection_reason}")

    metric_label, azure_metric = METRIC_BY_PROBLEM_TYPE[problem_type]

    config_table = pd.DataFrame(
        {
            "Setting": [
                "Target column",
                "Problem type",
                "Primary metric",
                "Compute",
                "Rows",
                "Columns",
            ],
            "Value": [
                target_column,
                problem_type,
                metric_label,
                f"Serverless CPU ({DEFAULT_VM_SIZE})",
                dataframe.shape[0],
                dataframe.shape[1],
            ],
        }
    )
    st.table(config_table.astype(str))

    if st.button("Start AutoML experiment", use_container_width=True):
        with st.spinner("Submitting AutoML job to Azure..."):
            try:
                data_name = f"data-{uploaded_file.name.replace('.csv', '')}"
                job_name = submit_automl_job(
                    csv_path=str(csv_path),
                    target_column=target_column,
                    problem_type=problem_type,
                    primary_metric=azure_metric,
                    data_name=data_name,
                    vm_size=DEFAULT_VM_SIZE,
                    experiment_name="streamlit-automl-demo",
                )
                st.session_state["latest_automl_job_name"] = job_name
                st.session_state["latest_problem_type"] = problem_type
                st.session_state["latest_target_column"] = target_column
                st.success(f"Job submitted successfully: {job_name}")
            except Exception as error:
                st.error(f"Error submitting job: {error}")

latest_job_name = st.session_state.get("latest_automl_job_name")
if latest_job_name:
    st.subheader("Last result")

    if st.button("Refresh job status"):
        st.rerun()

    try:
        job_details = get_automl_job_details(latest_job_name)
        latest_problem_type = st.session_state.get("latest_problem_type", "Unknown")
        latest_target_column = st.session_state.get("latest_target_column", "Unknown")

        status_text = str(job_details.get("status", "Unknown"))
        primary_metric = str(job_details.get("primary_metric") or "N/A").upper()
        best_score = job_details.get("best_metric_value")
        best_score_text = f"{best_score:.5f}" if isinstance(best_score, (int, float)) else "N/A"

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", status_text)
        col2.metric("Primary metric", primary_metric)
        col3.metric("Best score", best_score_text)

        if status_text in TERMINAL_STATUSES:
            best_algorithm = job_details.get("best_algorithm")
            best_run_name = job_details.get("best_run_name")
            if best_algorithm or best_run_name:
                st.caption(f"Best model ready: {best_algorithm or best_run_name}")

            feature_importance = job_details.get("feature_importance")
            confusion_matrix = job_details.get("confusion_matrix")

            st.subheader("Ask Agent About Results")
            if "results_chat_history" not in st.session_state:
                st.session_state["results_chat_history"] = []
            for message in st.session_state["results_chat_history"]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            user_question = st.chat_input("Ask about models, metrics, errors, or next steps")
            if user_question:
                st.session_state["results_chat_history"].append(
                    {"role": "user", "content": user_question}
                )
                with st.chat_message("user"):
                    st.write(user_question)
                with st.chat_message("assistant"):
                    with st.spinner("Agent is answering..."):
                        try:
                            answer = answer_results_question_with_agent(
                                {
                                    "job_name": latest_job_name,
                                    "status": status_text,
                                    "problem_type": latest_problem_type,
                                    "target_column": latest_target_column,
                                    "primary_metric": primary_metric,
                                    "best_score": best_score_text,
                                    "best_algorithm": job_details.get("best_algorithm"),
                                    "top_models": job_details.get("top_models"),
                                    "all_scored_models": job_details.get("all_scored_models"),
                                    "feature_importance": feature_importance,
                                    "feature_importance_source_run": job_details.get("feature_importance_source_run"),
                                    "confusion_matrix": confusion_matrix,
                                    "confusion_matrix_source_run": job_details.get("confusion_matrix_source_run"),
                                    "raw_job_details": job_details,
                                },
                                question=user_question,
                                history=st.session_state["results_chat_history"],
                            )
                        except Exception as error:
                            answer = f"Could not answer right now: {error}"
                        st.write(answer)
                st.session_state["results_chat_history"].append(
                    {"role": "assistant", "content": answer}
                )
        else:
            st.info("Training is still in progress. Refresh to see results.")

    except Exception as error:
        st.warning(f"Unable to fetch job details right now: {error}")
