"""
Azure AutoML Demo - Streamlit App
==================================
This app guides you through training an AutoML model:
1. Upload CSV data
2. Select target column
3. Auto-detect task type (Classification or Regression)
4. Run AutoML
5. View best model results

"""

import pandas as pd
import streamlit as st
import logging

from get_best_model import get_best_model_details
from run_automl import (
    DEFAULT_VM_SIZE,
    detect_problem_type,
    get_primary_metric,
    register_best_model,
    submit_automl_job,
)
from utils import save_uploaded_file

# Keep terminal output concise.
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Page configuration
st.set_page_config(page_title="Azure AutoML Demo", layout="wide", initial_sidebar_state="collapsed")
st.title("🤖 Azure AutoML Trainer")
st.caption("Upload CSV, select target, run AutoML, inspect best model.")

# Define terminal statuses for checking if job is complete
TERMINAL_STATUSES = {"Completed", "Failed", "Canceled", "Cancelled"}

# ============================================================
# SECTION 1: UPLOAD DATA
# ============================================================
st.header("📥 Step 1: Upload Training Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload your training data. First row should be column names."
)

if uploaded_file is None:
    st.info("👈 Please upload a CSV file to get started.")
    st.stop()

# Save and load file
csv_path = save_uploaded_file(uploaded_file)
df = pd.read_csv(uploaded_file)

st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
st.caption(f"File: `{uploaded_file.name}`")

with st.expander("Preview data"):
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================
# SECTION 2: SELECT TARGET & DETECT TASK TYPE
# ============================================================
st.header("🎯 Step 2: Select Target & Auto-Detect Task")

target_column = st.selectbox(
    "Select the target column (what you want to predict)",
    options=df.columns.tolist(),
    help="This is the column you want AutoML to learn to predict."
)

if target_column:
    # Detect problem type
    detection = detect_problem_type(df, target_column)
    problem_type = detection["problem_type"]
    reason = detection["reason"]
    primary_metric = get_primary_metric(problem_type)
    
    # Display detection results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Detected Task", problem_type)
    with col2:
        st.metric("Primary Metric", primary_metric)
    with col3:
        st.metric("Data Rows", df.shape[0])
    
    st.info(f"**Reason:** {reason}")
    
    # Show training configuration
    st.subheader("Training Configuration")
    config_df = pd.DataFrame({
        "Setting": [
            "Target Column",
            "Problem Type",
            "Primary Metric",
            "Compute (VM Size)",
            "Columns (Features)",
        ],
        "Value": [
            target_column,
            problem_type,
            primary_metric,
            DEFAULT_VM_SIZE,
            df.shape[1] - 1,  # excluding target
        ]
    })
    st.dataframe(config_df, use_container_width=True, hide_index=True)
    
# ============================================================
# SECTION 3: SUBMIT AUTOML JOB
# ============================================================
st.header("⚙️ Step 3: Run AutoML")

if st.button("▶️ Start AutoML Training", use_container_width=True, type="primary", key="run_button"):
    with st.spinner("📤 Submitting AutoML job to Azure..."):
        try:
            # Submit job
            job_name = submit_automl_job(
                csv_path=str(csv_path),
                target_column=target_column,
                problem_type=problem_type,
                data_name=f"data-{uploaded_file.name.replace('.csv', '')}",
                vm_size=DEFAULT_VM_SIZE,
            )
            
            # Save to session for retrieval later
            st.session_state["latest_automl_job_name"] = job_name
            
            st.success(f"✅ Job submitted successfully! Job name: `{job_name}`")
            st.info("Training started. Use Refresh to update status.")
            
        except Exception as error:
            error_msg = str(error)
            st.error(f"❌ Job submission failed: {error_msg}")
            
            # Helpful hints for common errors
            if "soft-deleted" in error_msg.lower():
                st.warning(
                    "Soft-deleted workspace found. Use a different workspace name or wait for recovery window."
                )
            elif "workspace" in error_msg.lower() and "not found" in error_msg.lower():
                st.warning(
                    "Workspace not found. Check workspace/resource-group names in Azure."
                )
            elif "authentication" in error_msg.lower():
                st.warning(
                    "Authentication failed. Sign in again with the Azure browser prompt."
                )

# ============================================================
# SECTION 4: VIEW AUTOML RESULTS
# ============================================================

latest_job_name = st.session_state.get("latest_automl_job_name")
if latest_job_name:
    st.header("📊 Results")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Job: `{latest_job_name}`")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    try:
        # Fetch job details
        details = get_best_model_details(latest_job_name)
        
        status = details.get("status", "Unknown")
        primary_metric_name = details.get("primary_metric", "N/A")
        best_model = details.get("best_model")

        if best_model and status in TERMINAL_STATUSES:
            best_model = register_best_model(latest_job_name, best_model)
        
        # Display job status
        st.subheader("Job Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status_icon = "✅" if status in TERMINAL_STATUSES else "⏳"
            st.metric("Status", f"{status_icon} {status}")
        with col2:
            st.metric("Primary Metric", primary_metric_name.title())
        with col3:
            if best_model and isinstance(best_model.get("score"), (int, float)):
                st.metric("Best Score", f"{best_model['score']:.4f}")
            else:
                st.metric("Best Score", "N/A")
        
        # Display best model details
        if best_model:
            st.subheader("🥇 Best Model Details")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Algorithm", best_model.get("algorithm") or "N/A")
            with col2:
                st.metric("Model Name", (best_model.get("model_name") or "N/A")[:30])
            with col3:
                st.metric("Run ID", (best_model.get("run_id") or "N/A")[:15])
            with col4:
                score = best_model.get("score")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                st.metric(primary_metric_name.title(), score_str)

            registered_name = best_model.get("registered_model_name") or "N/A"
            registered_version = best_model.get("registered_model_version") or "N/A"
            st.caption(f"Registered model: `{registered_name}` (v{registered_version})")
            
            st.subheader("📈 Used Metric")
            st.metric(primary_metric_name.title(), score_str)
        
        # Status info
        if status not in TERMINAL_STATUSES:
            st.info(f"Training in progress: {status}. Click Refresh.")
    
    except Exception as error:
        st.warning(f"⚠️ Could not fetch job details: {str(error)}")
        st.caption("Job may still be initializing. Click Refresh.")
