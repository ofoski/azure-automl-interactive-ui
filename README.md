# Azure AutoML — Responsible AI Demo

A Streamlit app that trains machine learning models using **Azure AutoML** and lets you interrogate them using a **Responsible AI agent** powered by LangChain, LangGraph, and Azure OpenAI.

---

## What it does

### 1. Train a model (optional — skip if you already have one)
- Upload a CSV file
- Select the target column
- The app auto-detects whether it's a **classification** or **regression** problem
- Splits the data into 80% train / 20% test and registers both as Azure ML Data Assets
- Submits an **Azure AutoML** job to find the best model
- Automatically registers the winning model in Azure ML

### 2. Analyse any registered model
- Pick any model already registered in your Azure ML workspace
- Load it together with its test dataset
- Chat with the **Responsible AI Agent** (powered by GPT-4o-mini) to ask questions like:
  - *"Which features matter most?"*
  - *"Where does the model make the most mistakes?"*
  - *"Is the model fair across different groups?"*
  - *"What would need to change for this person to get a different outcome?"*

The agent runs four analysis tools under the hood:
| Tool | What it answers |
|---|---|
| Permutation importance | Which features drive predictions |
| Error analysis | Where the model underperforms |
| Fairness analysis | Whether the model treats groups equally |
| Counterfactuals | What minimal changes would flip a prediction |

---

## Project structure

```
app.py                      # Streamlit UI (single page)
responsible_ai_agent.py     # LangChain/LangGraph ReAct agent with 4 analysis tools
responsible_ai_analysis.py  # Analysis tools (importance, fairness, errors, counterfactuals)
register_model.py           # Register best AutoML model in Azure ML
run_automl.py               # Submit AutoML training job
model_utils.py              # Extract child model metrics from AutoML jobs
ml_pipeline/
    client.py               # Azure ML client (reads from env vars)
    data.py                 # Register CSV as MLTable data asset
    job.py                  # Create and submit AutoML job
data/                       # Sample datasets
app_requirements.txt        # Pinned dependencies (install with --no-deps)
```

---

## Quick start

See [SETUP.md](SETUP.md) for full setup instructions.

```powershell
# 1. Install dependencies
python -m venv lastenv
lastenv\Scripts\pip.exe install --no-deps -r app_requirements.txt

# 2. Set environment variables (see SETUP.md)
$env:AZURE_SUBSCRIPTION_ID   = "..."
$env:AZURE_RESOURCE_GROUP    = "..."
$env:AZURE_WORKSPACE_NAME    = "..."
$env:AZURE_TENANT_ID         = "..."
$env:AZURE_OPENAI_ENDPOINT   = "..."
$env:AZURE_OPENAI_API_KEY    = "..."
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"

# 3. Run
.\lastenv\Scripts\Activate.ps1
streamlit run app.py
```

---

## Requirements

- Python 3.10+
- An **Azure ML workspace**
- An **Azure OpenAI resource** with a GPT-4o-mini (or similar) deployment

> **Install note:** Use `pip install --no-deps -r app_requirements.txt` (not bare `pip install -r`) to avoid version conflicts between Azure ML packages.

