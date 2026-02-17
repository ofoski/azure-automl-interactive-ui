import pandas as pd
import streamlit as st

from run_automl import get_automl_job_details, submit_automl_job
from utils import save_uploaded_file


st.set_page_config(page_title="AutoML Demo", layout="wide")

st.title("Interactive AutoML Demo")
st.caption("Upload a dataset and run Azure AutoML with a simple serverless configuration")

METRIC_BY_PROBLEM_TYPE = {
    "Classification": ("Accuracy", "accuracy"),
    "Regression": ("RMSE", "normalized_root_mean_squared_error"),
}
DEFAULT_VM_SIZE = "Standard_DS11_v2"
TERMINAL_STATUSES = {"Completed", "Failed", "Canceled", "Cancelled"}


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    csv_path = save_uploaded_file(uploaded_file)
    dataframe = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully")

    st.subheader("Training setup")

    target_column = st.selectbox("Target column", options=dataframe.columns.tolist(), index=0)
    problem_type = st.radio("Problem type", options=["Classification", "Regression"], horizontal=True)

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

    if st.button("Start AutoML experiment 🚀", use_container_width=True):
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
                st.write(f"**Best model**: {best_algorithm or best_run_name}")

            top_models = job_details.get("top_models") or []
            if top_models:
                leaderboard_rows = []
                for index, model in enumerate(top_models, start=1):
                    leaderboard_rows.append(
                        {
                            "Rank": index,
                            "Model": model.get("algorithm") or model.get("run_name") or "N/A",
                            "Score": f"{model.get('score'):.5f}",
                        }
                    )
                st.subheader("Top models")
                st.dataframe(pd.DataFrame(leaderboard_rows), use_container_width=True, hide_index=True)

            feature_importance = job_details.get("feature_importance")
            if isinstance(feature_importance, dict) and feature_importance:
                fi_df = pd.DataFrame(
                    {
                        "Feature": list(feature_importance.keys()),
                        "Importance": list(feature_importance.values()),
                    }
                )
                st.subheader("Feature importance")
                st.dataframe(fi_df, use_container_width=True, hide_index=True)
            else:
                st.caption("Feature importance is not available for this run.")
        else:
            st.info("Training is still in progress. Refresh to see results.")

    except Exception as error:
        st.warning(f"Unable to fetch job details right now: {error}")
