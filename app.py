import streamlit as st
import pandas as pd
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
    # 2. Target selection (what the model should predict)
    # ========================================================
    st.subheader("Target configuration")

    target_column = st.selectbox(
        "Select the target column",
        options=df.columns.tolist()
    )

    st.write(f"**Selected target:** {target_column}")
    st.divider()

    # ========================================================
    # 3. Dataset inspection (optional, for user understanding)
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

    # ========================================================
    # 4. Problem type inference (heuristic suggestion)
    # ========================================================
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

    # ========================================================
    # 5. Problem type confirmation (user decision)
    # ========================================================
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

    # ========================================================
    # 6. Evaluation metric selection (AutoML configuration)
    # ========================================================
    st.subheader("Evaluation metric")

    if final_problem_type == "Classification":
        available_metrics = [
            "Accuracy",
            "AUC",
            "F1",
            "Precision",
            "Recall"
        ]
    else:
        available_metrics = [
            "RMSE",
            "MAE",
            "R2"
        ]

    selected_metric = st.selectbox(
        "Select the primary evaluation metric ❓",
        options=available_metrics,
        help="This metric will be used by AutoML to select the best model."
    )

    st.success(f"Selected metric: **{selected_metric}**")

    # ========================================================
    # 7. Review AutoML configuration
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
    # 8. Internal AutoML configuration 
    # ========================================================

    automl_config = {
        "target_column": target_column,
        "problem_type": final_problem_type,
        "primary_metric": selected_metric,
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
    }

    # ========================================================
    # 9. Launch AutoML (next step)
    # ========================================================

    st.subheader("Run AutoML")

    # Metric name mapping (Azure expects lowercase)
    metric_mapping = get_metric_mapping()

    if st.button(
        "Start AutoML experiment 🚀",
        help="This will submit an AutoML job to Azure using the configuration above."
    ):
        with st.spinner("Submitting AutoML job to Azure..."):
            try:
                # Submit job using the separated script
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