![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Azure ML](https://img.shields.io/badge/Azure-Machine%20Learning-0078D4?logo=microsoftazure&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-6B4FBB?logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078D4?logo=microsoftazure&logoColor=white)

# 🤖 Azure AutoML — Responsible AI Demo

A Streamlit app that trains machine learning models using **Azure AutoML** and lets you interrogate them using a **Responsible AI agent** powered by LangChain, LangGraph, and Azure OpenAI.

---

## Why This Project

Most ML projects stop at accuracy — they ship a model but leave users unable to understand, question, or challenge its decisions. This project goes further by wrapping any trained model in a conversational Responsible AI agent that explains predictions, surfaces where the model fails, and checks for unfair treatment across groups. Any non-technical person can load a model and ask questions in plain English — no code required.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML training | Azure AutoML — tries many algorithms automatically and registers the best model |
| AI agent | LangChain + LangGraph — ReAct reasoning loop that selects and calls analysis tools |
| RAI analysis | scikit-learn, fairlearn, DiCE-ML — importance, error, fairness, and counterfactual tools |
| UI | Streamlit — single-page app with three tabs |
| LLM | Azure OpenAI (GPT-4.1) — guardrail checks, context interpretation, plain-language answers |

---

## 🧠 How it works

### Step 1 — Train a model
You upload a CSV and pick a target column. The app identifies the prediction task from the target column, splits your data 80/20, uploads both splits to Azure ML as data assets, then submits an **AutoML job**. AutoML tries many algorithm and preprocessing combinations automatically and selects the best one. When the job finishes, the winning model is registered in your Azure ML workspace. See the project structure for the source files involved.

> Skip this step if you already have a registered model.

### Step 2 — Analyse the model
You pick any registered model from your workspace, enter the target column name and data asset name, and click **Load**. The app downloads the model from Azure ML and loads the test dataset. From this point you can chat with the Responsible AI Agent. Model downloading and data loading are handled in the source code.

### Step 3 — Chat with the Responsible AI Agent
You type a question in plain English. Here is what happens under the hood:

1. **Guardrail check** — the message is screened before any tools are called. Off-topic or unsafe messages are blocked immediately. See [Guardrails and Safety](#%EF%B8%8F-guardrails-and-safety) for details.

2. **Context injection** — the agent is given a statistical summary of your dataset (column names, value ranges, min/max/mean/std of every feature) and basic model info. This is what lets it give answers grounded in your actual data rather than generic responses.

3. **ReAct loop** — the agent decides which tool(s) to call based on your question, calls them, reads the numbers, then writes a plain-language answer. It can call multiple tools in sequence. For example, "give me a full analysis" triggers all four tools one by one.

4. **Caching** — tool results are cached for the whole session. If you ask two different questions that both need feature importance, the computation only runs once.

See the 📊 Demo section below for example questions and what each tool does.

---

## ⚖️ Guardrails and Safety

The agent includes the following protections:

- **Prompt injection detection** — if the model's content filter or a jailbreak pattern is triggered, the response is blocked and the user is notified.
- **Out-of-scope blocking** — every message passes through a guardrail LLM call before any tools are invoked. Questions unrelated to the model or Responsible AI are rejected immediately.
- **Human-sensitive feature detection** — fairness analysis is only run when the dataset contains features that describe human characteristics. If none are found, the agent declines and suggests error analysis instead.
- **3-sigma realism filtering** — counterfactual suggestions are discarded if any changed value falls outside three standard deviations of the training distribution, preventing unrealistic recommendations.

---

## 📊 Demo

Example questions you can ask the agent after loading a model:

| Question | What the agent does |
|---|---|
| "Which features matter most for predictions?" | Runs permutation importance and ranks all features by their impact on the model's score |
| "Where does the model make the most mistakes?" | Runs error analysis and identifies the groups or value ranges with the highest average error |
| "Is the model fair across different groups?" | Checks for human-sensitive features, runs fairness analysis if found, and reports the performance gap between best and worst group |
| "What would need to change for this person to get a different outcome?" | Generates counterfactuals for one row — minimal, realistic feature changes that would flip the prediction |
| "Compare the models trained in this job" | Reads the AutoML model comparison table directly and summarises which algorithm performed best and why |

---

## 📁 Project structure

```
app.py                      # Streamlit UI — three tabs: Train Model, Analyse Model, Chat with AI Agent
responsible_ai_agent.py     # The AI agent — guardrail, system prompt, tool definitions, ReAct loop
responsible_ai_analysis.py  # The four analysis functions + model/data loading from Azure ML
register_model.py           # Finds the best child run from a finished AutoML job and registers it
run_automl.py               # Submits an AutoML training job and detects the prediction task
model_utils.py              # Reads the model comparison table from AutoML job tags for the UI
ml_pipeline/
    client.py               # Creates the Azure ML client from environment variables
    data.py                 # Splits a CSV and registers train/test splits as Azure ML data assets
    job.py                  # Builds and submits the AutoML job configuration
data/                       # Sample datasets
app_requirements.txt        # Pinned dependencies (install with --no-deps)
```

---

## 🚀 Quick start

See [SETUP.md](SETUP.md) for full setup instructions including Azure App Registration and access control.

### Option A — Local (pip install)

```powershell
# 1. Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\pip.exe install --no-deps -r app_requirements.txt

# 2. Create your .env file (see SETUP.md for all required values)
copy .env.example .env
# Edit .env and fill in your values

# 3. Load .env into the shell, then run (must be the same terminal)
Get-Content .env | Where-Object { $_ -match '^\w' } | ForEach-Object { $k,$v=$_ -split '=',2; Set-Item "env:$k" $v }
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

### Option B — Docker

```powershell
# 1. Create your .env file (service principal credentials required)
copy .env.example .env
# Edit .env and fill in all values including AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID

# 2. Build and run (first time — allow ~35 minutes)
docker compose up --build

# 3. Subsequent runs (fast)
docker compose up
```

App available at [http://localhost:8501](http://localhost:8501)

---

## 📋 Requirements

- Python 3.10+
- Docker Desktop (Option B)
- An **Azure ML workspace**
- An **Azure OpenAI resource** with a GPT-4 deployment
- An **Azure App Registration** (service principal) with Contributor role on the resource group — required for Docker, optional for local

> **Install note:** Use `pip install --no-deps -r app_requirements.txt` (not bare `pip install -r`) to avoid version conflicts between Azure ML packages.

