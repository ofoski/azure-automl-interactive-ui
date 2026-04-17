"""
Streamlit app for Azure AutoML model training and Responsible AI analysis.

Three tabs:
  Train Model   — upload data, submit an AutoML job, register the best model.
  Analyse Model — view metrics for any registered model in the workspace.
  Chat with AI  — ask the Responsible AI agent questions about a loaded model.
"""

import pandas as pd
import streamlit as st
import logging
import os
import shutil
from pathlib import Path


from ml_pipeline import get_ml_client
from training.register_model import register_best_model
from training.run_automl import (
    detect_problem_type,
    get_primary_metric,
    submit_automl_job,
)

os.environ.setdefault("TQDM_DISABLE", "1")

logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Azure AutoML - Responsible AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{padding:2rem 1.4rem 1rem !important;}
h1,h2,h3,h4{margin:0.1rem 0 0.2rem !important;}
h1{font-size:1.15rem !important;}
h2{font-size:0.95rem !important;}
h3,h4{font-size:0.85rem !important;}
div[data-testid="stMetricValue"]{font-size:0.95rem !important;}
div[data-testid="stMetricLabel"]{font-size:0.68rem !important;}
.stAlert p{font-size:0.8rem;margin:0;}
.stAlert{padding:0.2rem 0.6rem !important;margin:0.1rem 0 !important;}
section[data-testid="stVerticalBlock"]{gap:0.3rem !important;}
.stExpander{margin:0.15rem 0 !important;}
div[data-testid="stHorizontalBlock"]{gap:0.5rem !important;}
.stChatMessage{padding:0.3rem 0.5rem !important;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Azure AutoML — Responsible AI")
st.markdown("Train models with Azure AutoML and analyze them for **fairness, bias, and performance**")

TERMINAL_STATUSES = {"Completed", "Failed", "Canceled", "Cancelled"}


def _fetch_registered_models(ml_client) -> dict:
    """Return {display_label: {name, version}} for all registered models."""
    result: dict = {}
    for _c in ml_client.models.list():
        _cname = getattr(_c, "name", None)
        if not _cname:
            continue
        _lv = getattr(_c, "latest_version", None)
        if not _lv:
            try:
                _vers = list(ml_client.models.list(name=_cname))
                if _vers:
                    _lv = str(max(_vers, key=lambda v: int(v.version) if str(v.version).isdigit() else 0).version)
            except Exception:
                pass
        if _lv:
            result[f"{_cname} (v{_lv})"] = {"name": _cname, "version": str(_lv)}
    return result

# Restore job from URL on reload
_qjob = st.query_params.get("job")
if _qjob and "latest_automl_job_name" not in st.session_state:
    st.session_state["latest_automl_job_name"] = str(_qjob)

_latest_job = st.session_state.get("latest_automl_job_name")
tab1, tab2, tab3 = st.tabs(["📊 Train Model", "🔍 Analyze Model", "💬 Chat with AI Agent"])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1: TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    _up_col, _tgt_col, _btn_col = st.columns([2, 2, 1])

    with _up_col:
        uploaded_file = st.file_uploader(
            "📂 Upload Training Dataset", type=["csv"],
            help="Upload a CSV file with your training data. The file should include all features and the target column.",
        )
        if uploaded_file:
            if uploaded_file.size > 100 * 1024 * 1024:
                st.warning("⚠️ File exceeds 100 MB. Consider a smaller sample.")
            _upload_dir = Path("uploads")
            _upload_dir.mkdir(exist_ok=True)
            for _item in _upload_dir.iterdir():
                shutil.rmtree(_item, ignore_errors=True) if _item.is_dir() else _item.unlink(missing_ok=True)
            _csv_path = _upload_dir / uploaded_file.name
            _csv_path.write_bytes(uploaded_file.getvalue())
            _df_up = pd.read_csv(_csv_path)
            st.session_state["_csv_path"] = str(_csv_path)
            st.success(f"✅ Loaded {_df_up.shape[0]} rows and {_df_up.shape[1]} columns")
            with st.expander("👀 Preview Data"):
                st.dataframe(_df_up.head(), use_container_width=True)
            with st.expander("📋 Column Information"):
                _info = pd.DataFrame({"dtype": _df_up.dtypes, "missing": _df_up.isna().sum()})
                st.dataframe(_info, use_container_width=True)
                _nums = _df_up.select_dtypes("number")
                if not _nums.empty:
                    st.dataframe(_nums.describe().T[["min", "max", "mean"]], use_container_width=True)

    _csv_ready = st.session_state.get("_csv_path")
    _df_ready  = pd.read_csv(_csv_ready) if _csv_ready else None

    with _tgt_col:
        if _df_ready is not None:
            _tgt = st.selectbox(
                "Target column", _df_ready.columns.tolist(),
                key="target_col_sel", label_visibility="collapsed",
            )
            if _tgt:
                _det  = detect_problem_type(_df_ready, _tgt)
                _prob = _det["problem_type"]
                st.session_state["_problem_type"] = _prob
                st.session_state["_primary_metric"] = get_primary_metric(_prob)
                st.caption(f"Task: **{_prob}** | metric: **{get_primary_metric(_prob)}**")
        else:
            st.caption("Upload a CSV first.")

    with _btn_col:
        _can_train = bool(
            _csv_ready
            and st.session_state.get("target_col_sel")
            and st.session_state.get("_problem_type")
        )
        if st.button(
            "▶️ Train", type="primary",
            disabled=not _can_train, use_container_width=True, key="run_button",
        ):
            try:
                with st.spinner("Submitting…"):
                    _job = submit_automl_job(
                        csv_path=_csv_ready,
                        target_column=st.session_state["target_col_sel"],
                        problem_type=st.session_state["_problem_type"],
                    )
                st.session_state["latest_automl_job_name"] = _job
                st.session_state["rai_target_column"] = st.session_state["target_col_sel"]
                st.session_state["rai_data_asset_name"] = Path(_csv_ready).stem
                st.query_params["job"] = _job
                _latest_job = _job
                st.success(f"✅ Submitted: `{_job}` — check Job Results below for status.")
                st.rerun()
            except Exception as _e:
                st.error(str(_e))

    # ── Job Results ─────────────────────────────────────────────────────
    _job_label = (" — " + _latest_job) if _latest_job else ""
    with st.expander(
        f"📊 Job Results{_job_label}",
        expanded=bool(_latest_job),
    ):
        if not _latest_job:
            st.caption("No active job. Start a new training run above.")
        else:
            _rc1, _rc2 = st.columns([5, 1])
            _rc1.caption(f"`{_latest_job}`")
            with _rc2:
                if st.button("🔄 Refresh", use_container_width=True, key="refresh_btn"):
                    st.rerun()

            try:
                _job_obj = get_ml_client().jobs.get(_latest_job)
                _status  = getattr(_job_obj, "status", "Unknown")
            except Exception as _e:
                st.warning(f"Could not fetch job: {_e}")
                _status = None

            if _status:
                _sicon = "✅" if _status in TERMINAL_STATUSES else "⏳"
                st.metric("Status", f"{_sicon} {_status}")

                _reg_key     = f"registered_model:{_latest_job}"
                _reg_err_key = f"registration_error:{_latest_job}"

                if _status == "Completed" and not st.session_state.get(_reg_key):
                    try:
                        _ml       = get_ml_client()
                        _tags     = getattr(_ml.jobs.get(_latest_job), "tags", {}) or {}
                        _best_run = _tags.get("automl_best_child_run_id")
                        _exp_name = f"best-model-{_best_run}" if _best_run else None
                        try:
                            _existing = next(_ml.models.list(name=_exp_name), None) if _exp_name else None
                        except Exception:
                            _existing = None
                        if _existing:
                            st.session_state[_reg_key] = {
                                "run_id": _best_run, "score": None,
                                "registered_model_name": getattr(_existing, "name", _exp_name),
                                "registered_model_version": str(getattr(_existing, "version", "1")),
                                "model_id": f"azureml:{_existing.name}:{_existing.version}",
                                "source_path": f"azureml://jobs/{_latest_job}/outputs/best_model",
                            }
                        else:
                            with st.spinner("Registering model…"):
                                st.session_state[_reg_key] = register_best_model(_latest_job)
                        st.session_state.pop(_reg_err_key, None)
                    except Exception as _e:
                        st.session_state[_reg_err_key] = str(_e)

                if st.session_state.get(_reg_err_key):
                    st.error(f"Registration failed: {st.session_state[_reg_err_key]}")
                    if st.button("🔁 Retry", key="retry_register"):
                        st.session_state.pop(_reg_key, None)
                        st.session_state.pop(_reg_err_key, None)
                        st.rerun()

                _reg = st.session_state.get(_reg_key)
                if _reg:
                    st.success(
                        f"`{_reg.get('registered_model_name')}` "
                        f"v{_reg.get('registered_model_version')} registered"
                    )

                if _status not in TERMINAL_STATUSES:
                    st.info("Still running — click Refresh.")
                elif _status in {"Failed", "Canceled", "Cancelled"}:
                    st.warning(f"Job finished: {_status}")

# ─────────────────────────────────────────────────────────────────────────
# TAB 2: ANALYZE MODEL
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 📊 Model Performance Dashboard")
    if st.button("Fetch registered models", key="analyze_fetch"):
        with st.spinner("Fetching…"):
            try:
                st.session_state["analyze_models"] = _fetch_registered_models(get_ml_client())
                st.rerun()
            except Exception as _exc:
                st.error(f"Could not fetch: {_exc}")

    _a_models = st.session_state.get("analyze_models", {})
    if _a_models:
        _a_sel = st.selectbox("Select model", list(_a_models.keys()), key="analyze_model_sel")
        _a_model = _a_models[_a_sel]
        with st.spinner("Loading model info…"):
            try:
                from training.model_utils import extract_child_metrics
                _ml_c2 = get_ml_client()
                _model_obj = _ml_c2.models.get(_a_model["name"], version=_a_model["version"])
                _mtags = getattr(_model_obj, "tags", {}) or {}
                # Prefer stored tag; fall back to deriving from model name: best-model-{parent}_{n}
                _parent_job = _mtags.get("parent_job_name")
                if not _parent_job and _a_model["name"].startswith("best-model-"):
                    _child_run_id = _a_model["name"][len("best-model-"):]
                    _parent_job = _child_run_id.rsplit("_", 1)[0]
                _job_meta = None
                if _parent_job:
                    try:
                        _job_meta = _ml_c2.jobs.get(_parent_job)
                    except Exception:
                        pass
                _c1, _c2, _c3, _c4 = st.columns(4)
                _c1.metric("Model", _a_model["name"])
                _c2.metric("Version", _a_model["version"])
                _c3.metric("Job Status", getattr(_job_meta, "status", "N/A") if _job_meta else "N/A")
                _c4.metric("Parent Job", (_parent_job[:20] + "…") if _parent_job and len(_parent_job) > 20 else (_parent_job or "N/A"))
                if _parent_job:
                    st.markdown("#### 📊 All Models Tried")
                    _child_df = extract_child_metrics(_ml_c2, _parent_job)
                    if _child_df is not None and not _child_df.empty:
                        st.dataframe(_child_df, use_container_width=True)
                        st.download_button("⬇️ Download CSV", _child_df.to_csv(index=False).encode(), "model_comparison.csv", "text/csv")
                    else:
                        st.info("No child model metrics available for this job.")
                else:
                    st.info("No parent job linked to this model.")
            except Exception as _ae:
                st.error(f"Error loading model info: {_ae}")
    else:
        st.info("Click **Fetch registered models** to load the model list.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 3: CHAT WITH AI AGENT
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 🔍 Responsible AI Agent")
    _ai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    _ai_api_key  = os.environ.get("AZURE_OPENAI_API_KEY", "")

    with st.expander("Model & data", expanded=True):
        _fc1, _fc2, _fc3 = st.columns([2, 1, 1])

        with _fc1:
            if st.button("Fetch registered models", key="rai_fetch", use_container_width=True):
                with st.spinner("Fetching…"):
                    try:
                        st.session_state["rai_available_models"] = _fetch_registered_models(get_ml_client())
                        st.rerun()
                    except Exception as _exc:
                        st.error(f"Could not fetch: {_exc}")

            _opt: dict = dict(st.session_state.get("rai_available_models", {}))
            _wf_reg = st.session_state.get(f"registered_model:{_latest_job}") if _latest_job else None
            if _wf_reg:
                _wl = f"{_wf_reg['registered_model_name']} (v{_wf_reg['registered_model_version']})"
                if _wl not in _opt:
                    _opt = {_wl: {"name": _wf_reg["registered_model_name"], "version": str(_wf_reg["registered_model_version"])}, **_opt}

            if _opt:
                _sel           = st.selectbox("Model", list(_opt.keys()), key="rai_model_sel", label_visibility="collapsed")
                _model_name    = _opt[_sel]["name"]
                _model_version = str(_opt[_sel]["version"])
            else:
                st.caption("No models yet — click **Fetch registered models**.")
                _model_name = _model_version = None

        with _fc2:
            _target_col = st.text_input(
                "Target column",
                value=st.session_state.get("rai_target_column", ""),
                key="rai_target_in", placeholder="e.g. survived",
            )
        with _fc3:
            _data_asset = st.text_input(
                "Data asset base",
                value=st.session_state.get("rai_data_asset_name", ""),
                key="rai_asset_in", placeholder="e.g. titanic",
            )

    _ready = bool(_model_name and _target_col and _data_asset)

    if not _ready:
        st.info("Fetch models and fill in **target column** + **data asset base** to continue.")
    else:
        _mc1, _mc2, _mc3 = st.columns(3)
        _mc1.metric("Model", f"{_model_name} v{_model_version}")
        _mc2.metric("Target", _target_col)
        _mc3.metric("Test asset", f"{_data_asset}-test")

        _rai_key = f"rai_loaded:{_model_name}:{_model_version}"
        if _rai_key not in st.session_state:
            if st.button("Load model & data", key="rai_load", type="primary"):
                with st.spinner("Loading from Azure ML…"):
                    try:
                        from responsible_ai.responsible_ai_analysis import load_model, load_test_data
                        _m        = load_model(_model_name, _model_version)
                        _df_test  = load_test_data(f"{_data_asset}-test")
                        _df_train = load_test_data(f"{_data_asset}-train")
                        _task_t   = detect_problem_type(_df_test, _target_col)["problem_type"].lower()
                        st.session_state[_rai_key] = {
                            "model":     _m,
                            "X_test":    _df_test.drop(columns=[_target_col]),
                            "y_test":    _df_test[_target_col],
                            "X_train":   _df_train.drop(columns=[_target_col]),
                            "y_train":   _df_train[_target_col],
                            "task_type": _task_t,
                        }
                        st.session_state.pop("rai_cache", None)
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Failed to load: {_e}")
        else:
            _rai = st.session_state[_rai_key]
            st.success(f"✅ Ready — task: **{_rai['task_type']}** | test rows: **{len(_rai['X_test'])}**")

            def _run_agent(user_msg: str) -> str:
                from responsible_ai.responsible_ai_agent import run_agent as _rai_fn
                from training.model_utils import extract_child_metrics
                try:
                    _pjob = get_ml_client().models.get(_model_name, version=_model_version).tags.get("parent_job_name") or (_model_name[len("best-model-"):].rsplit("_", 1)[0] if _model_name.startswith("best-model-") else None)
                    _cmp_str = extract_child_metrics(get_ml_client(), _pjob).to_string(index=False) if _pjob else ""
                except Exception:
                    _cmp_str = ""
                _ctx = {
                    "model":            _rai["model"],
                    "X_test":           _rai["X_test"],
                    "y_test":           _rai["y_test"],
                    "X_train":          _rai["X_train"],
                    "y_train":          _rai["y_train"],
                    "task_type":        _rai["task_type"],
                    "model_name":       _model_name,
                    "model_version":    _model_version,
                    "target_column":    _target_col,
                    "test_asset":       f"{_data_asset}-test",
                    "model_comparison": _cmp_str,
                }
                return _rai_fn(
                    user_message=user_msg,
                    context=_ctx,
                    cache=st.session_state.setdefault("rai_cache", {}),
                    openai_endpoint=_ai_endpoint,
                    api_key=_ai_api_key,
                    chat_history=st.session_state.get("rai_chat", []),
                )

            if "rai_chat" not in st.session_state:
                st.session_state["rai_chat"] = []

            for _turn in st.session_state["rai_chat"]:
                with st.chat_message(_turn["role"]):
                    st.markdown(_turn["content"])

            _user_q = st.chat_input("Ask about your model — e.g. 'Which features matter most?'")
            if _user_q:
                st.session_state["rai_chat"].append({"role": "user", "content": _user_q})
                with st.chat_message("user"):
                    st.markdown(_user_q)
                _answer = _run_agent(_user_q)
                st.session_state["rai_chat"].append({"role": "assistant", "content": _answer})
                with st.chat_message("assistant"):
                    st.markdown(_answer)

            if st.session_state["rai_chat"] and st.button("🗑️ Clear chat", key="rai_clear"):
                st.session_state["rai_chat"] = []
                st.rerun()
