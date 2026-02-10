# Copilot instructions for azure-automl-demo

## Project overview
- Streamlit UI in [app.py](app.py) lets a user upload a CSV, pick a target, and submit an Azure AutoML job.
- AutoML submission is orchestrated via [run_automl.py](run_automl.py) → package [ml_pipeline](ml_pipeline/__init__.py).
- Azure ML integration uses `azure.ai.ml` and `azure.identity` with `DefaultAzureCredential` in [ml_pipeline/client.py](ml_pipeline/client.py).

## Core data flow
1. Streamlit uploads CSV → saved locally via `save_uploaded_file()` in [utils.py](utils.py).
2. File is registered as MLTable by `register_training_data()` in [ml_pipeline/data.py](ml_pipeline/data.py) (creates a *_mltable folder and MLTable file).
3. AutoML job created in `run_automl_job()` in [ml_pipeline/job.py](ml_pipeline/job.py) and submitted via MLClient.

## Configuration & conventions
- Azure workspace config is **env-var driven**: `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_WORKSPACE_NAME` (required by `get_ml_client()`).
- UI metric labels map to Azure metric names in `get_metric_mapping()` in [utils.py](utils.py); keep UI labels in sync with Azure names.
- Problem types are normalized to title case strings (`Classification`/`Regression`) and mapped to Azure AutoML builders in [ml_pipeline/config.py](ml_pipeline/config.py).
- Default AutoML limits live in [ml_pipeline/job.py](ml_pipeline/job.py): `max_trials=3`, `timeout_minutes=15` unless overridden.

## Important folders
- [data](data/): sample data (e.g., Titanic CSV).
- [uploads](uploads/): saved user uploads; local files are referenced when registering MLTable assets.
- [ml_pipeline](ml_pipeline/): Azure ML client, data registration, and job submission helpers.

## Where to make changes
- UI/UX or validation changes: [app.py](app.py).
- AutoML config, limits, or job creation behavior: [ml_pipeline/job.py](ml_pipeline/job.py).
- Data asset registration/MLTable behavior: [ml_pipeline/data.py](ml_pipeline/data.py).
- Metric naming or UI mapping: [utils.py](utils.py).
