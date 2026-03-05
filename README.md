# Azure AutoML Demo

Simple Streamlit app for Azure AutoML.

## What it does
- Upload a CSV file
- Select the target column
- Auto-detect classification or regression
- Submit an Azure AutoML job
- Show the best model and metric

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required environment variables
```bash
AZURE_SUBSCRIPTION_ID
AZURE_RESOURCE_GROUP
AZURE_WORKSPACE_NAME
```

## Main files
- `app.py`: Streamlit UI
- `run_automl.py`: submit job and register best model
- `get_best_model.py`: fetch best model details
- `ml_pipeline/client.py`: Azure ML client
- `ml_pipeline/data.py`: register CSV as MLTable
- `ml_pipeline/job.py`: create/submit AutoML job

