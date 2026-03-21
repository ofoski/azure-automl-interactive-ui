"""
Azure AutoML Demo - Streamlit App
==================================
Single-page layout:
  A. New Training Job  (collapsed expander — upload + target + submit)
  B. Job Results       (expander — only relevant when a job exists)
  C. Responsible AI Agent (main area — model picker + chat)
"""

import pandas as pd
import streamlit as st
import logging
import os

from ml_pipeline import get_ml_client
from register_model import register_best_model
from run_automl import (
    DEFAULT_VM_SIZE,
    detect_problem_type,
    get_primary_metric,
    submit_automl_job,
)
from utils import save_uploaded_file

os.environ.setdefault("TQDM_DISABLE", "1")

logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Azure AutoML Demo", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{padding:0.6rem 1.4rem 1rem !important;}
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

st.markdown("#### \U0001f916 Azure AutoML \u2014 Responsible AI")

TERMINAL_STATUSES = {"Completed", "Failed", "Canceled", "Cancelled"}

# Restore job from URL on reload
_qjob = st.query_params.get("job")
if _qjob and "latest_automl_job_name" not in st.session_state:
    st.session_state["latest_automl_job_name"] = str(_qjob)

# ─────────────────────────────────────────────────────────────────────────
# SECTION A: NEW TRAINING JOB  (collapsed — just a quick-access panel)
# ─────────────────────────────────────────────────────────────────────────
with st.expander("\u25b6 New AutoML Training Job", expanded=False):
    _up_col, _tgt_col, _btn_col = st.columns([2, 2, 1])

    with _up_col:
        uploaded_file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        if uploaded_file:
            _csv_path = save_uploaded_file(uploaded_file)
            _df_up = pd.read_csv(_csv_path)
            st.session_state["_csv_path"] = str(_csv_path)
            st.caption(f"`{uploaded_file.name}` \u2014 {_df_up.shape[0]} rows \u00d7 {_df_up.shape[1]} cols")

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
            "\u25b6\ufe0f Train", type="primary",
            disabled=not _can_train, use_container_width=True, key="run_button",
        ):
            from pathlib import Path as _Path
            with st.spinner("Submitting\u2026"):
                try:
                    _job = submit_automl_job(
                        csv_path=_csv_ready,
                        target_column=st.session_state["target_col_sel"],
                        problem_type=st.session_state["_problem_type"],
                        vm_size=DEFAULT_VM_SIZE,
                    )
                    st.session_state["latest_automl_job_name"] = _job
                    st.session_state["rai_target_column"] = st.session_state["target_col_sel"]
                    st.session_state["rai_data_asset_name"] = _Path(_csv_ready).stem
                    st.query_params["job"] = _job
                    st.success(f"Submitted: `{_job}`")
                except Exception as _e:
                    st.error(str(_e))

# ─────────────────────────────────────────────────────────────────────────
# SECTION B: JOB RESULTS  (auto-expands when a job is in session state)
# ─────────────────────────────────────────────────────────────────────────
_latest_job = st.session_state.get("latest_automl_job_name")

_job_label = (" \u2014 " + _latest_job) if _latest_job else ""

with st.expander(
    f"\U0001f4ca Job Results{_job_label}",
    expanded=bool(_latest_job),
):
    if not _latest_job:
        st.caption("No active job. Start a new training run above.")
    else:
        _rc1, _rc2 = st.columns([5, 1])
        _rc1.caption(f"`{_latest_job}`")
        with _rc2:
            if st.button("\U0001f504 Refresh", use_container_width=True, key="refresh_btn"):
                st.rerun()

        try:
            _job_obj = get_ml_client().jobs.get(_latest_job)
            _status  = getattr(_job_obj, "status", "Unknown")
        except Exception as _e:
            st.warning(f"Could not fetch job: {_e}")
            _status = None

        if _status:
            _sicon = "\u2705" if _status in TERMINAL_STATUSES else "\u23f3"
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
                        with st.spinner("Registering model\u2026"):
                            st.session_state[_reg_key] = register_best_model(_latest_job)
                    st.session_state.pop(_reg_err_key, None)
                except Exception as _e:
                    st.session_state[_reg_err_key] = str(_e)

            if st.session_state.get(_reg_err_key):
                st.error(f"Registration failed: {st.session_state[_reg_err_key]}")
                if st.button("\U0001f501 Retry", key="retry_register"):
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
                st.info("Still running \u2014 click Refresh.")
            elif _status in {"Failed", "Canceled", "Cancelled"}:
                st.warning(f"Job finished: {_status}")

# ─────────────────────────────────────────────────────────────────────────
# SECTION C: RESPONSIBLE AI AGENT
# ─────────────────────────────────────────────────────────────────────────
st.markdown("#### \U0001f50d Responsible AI Agent")

# ── Model + context picker ────────────────────────────────────────────────
# Read Azure OpenAI credentials from environment only — not shown in UI
_ai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
_ai_api_key  = os.environ.get("AZURE_OPENAI_API_KEY", "")

with st.expander("Model & data", expanded=True):
    _fc1, _fc2, _fc3 = st.columns([2, 1, 1])

    with _fc1:
        if st.button("Fetch registered models", key="rai_fetch", use_container_width=True):
            with st.spinner("Fetching\u2026"):
                try:
                    _ml_c = get_ml_client()
                    # models.list() returns ModelContainer objects (one per model name).
                    # Each container may carry latest_version; if not, query per-model.
                    _containers = list(_ml_c.models.list())
                    _best: dict = {}
                    for _c in _containers:
                        _cname = getattr(_c, "name", None)
                        if not _cname:
                            continue
                        _latest_v = getattr(_c, "latest_version", None)
                        if not _latest_v:
                            try:
                                _vers = list(_ml_c.models.list(name=_cname))
                                if not _vers:
                                    continue
                                def _vkey(v):
                                    try:
                                        return int(v.version)
                                    except Exception:
                                        return 0
                                _latest_v = str(max(_vers, key=_vkey).version)
                            except Exception:
                                continue
                        _best[_cname] = {"name": _cname, "version": str(_latest_v)}
                    st.session_state["rai_available_models"] = {
                        f"{v['name']} (v{v['version']})": v for v in _best.values()
                    }
                    st.rerun()
                except Exception as _exc:
                    st.error(f"Could not fetch: {_exc}")

        _opt: dict = dict(st.session_state.get("rai_available_models", {}))
        _wf_reg = st.session_state.get(f"registered_model:{_latest_job}") if _latest_job else None
        if _wf_reg:
            _wl = f"{_wf_reg['registered_model_name']} (v{_wf_reg['registered_model_version']})"
            if _wl not in _opt:
                _opt = {
                    _wl: {
                        "name":    _wf_reg["registered_model_name"],
                        "version": str(_wf_reg["registered_model_version"]),
                    },
                    **_opt,
                }

        if _opt:
            _sel           = st.selectbox("Model", list(_opt.keys()), key="rai_model_sel", label_visibility="collapsed")
            _model_name    = _opt[_sel]["name"]
            _model_version = str(_opt[_sel]["version"])
        else:
            st.caption("No models yet \u2014 click **Fetch registered models**.")
            _model_name = _model_version = None

    with _fc2:
        _target_col = st.text_input(
            "Target column",
            value=st.session_state.get("rai_target_column", ""),
            key="rai_target_in",
            placeholder="e.g. survived",
        )
    with _fc3:
        _data_asset = st.text_input(
            "Data asset base",
            value=st.session_state.get("rai_data_asset_name", ""),
            key="rai_asset_in",
            placeholder="e.g. titanic",
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
            with st.spinner("Loading from Azure ML\u2026"):
                try:
                    from responsible_ai_analysis import load_model, load_test_data
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
        st.success(
            f"\u2705 Ready \u2014 task: **{_rai['task_type']}** | "
            f"test rows: **{len(_rai['X_test'])}**"
        )

        def _run_agent(user_msg: str) -> str:
            from responsible_ai_agent import run_agent as _rai_fn
            _ctx = {
                "model":        _rai["model"],
                "X_test":       _rai["X_test"],
                "y_test":       _rai["y_test"],
                "X_train":      _rai["X_train"],
                "y_train":      _rai["y_train"],
                "task_type":    _rai["task_type"],
                "model_name":   _model_name,
                "model_version": _model_version,
                "target_column": _target_col,
                "test_asset":   f"{_data_asset}-test",
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

        _user_q = st.chat_input("Ask about your model \u2014 e.g. 'Which features matter most?'")
        if _user_q:
            st.session_state["rai_chat"].append({"role": "user", "content": _user_q})
            with st.chat_message("user"):
                st.markdown(_user_q)
            _answer = _run_agent(_user_q)
            st.session_state["rai_chat"].append({"role": "assistant", "content": _answer})
            with st.chat_message("assistant"):
                st.markdown(_answer)

        if st.session_state["rai_chat"] and st.button("\U0001f5d1\ufe0f Clear chat", key="rai_clear"):
            st.session_state["rai_chat"] = []
            st.rerun()
