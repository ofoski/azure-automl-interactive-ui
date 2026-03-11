# Azure AutoML Demo

Simple Streamlit app for Azure AutoML.

## What it does
- Upload a CSV file
- Select the target column
- Auto-detect classification or regression
- Split uploaded data into train/test (80/20)
- Save both splits as Azure ML MLTable Data Assets
- Submit an Azure AutoML job using the train split
- Show the best model and metric

## Data Processing Workflow
- The uploaded CSV is split into 80% train and 20% test.
- Both train and test are converted to MLTable format and registered as Data Assets.
- AutoML training uses only the train Data Asset.
- The test Data Asset is saved for later evaluation and analysis in notebook or Python scripts.

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

