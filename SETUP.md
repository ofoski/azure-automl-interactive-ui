# Setup Guide

This guide walks you through two ways to run the app: **Option A — local pip install** and **Option B — Docker**. Both use a `.env` file for credentials.

---

## Prerequisites

**Both options:**
- **Azure subscription** with an Azure ML workspace
- **Azure AI Foundry** with an Azure OpenAI resource and a model deployment
- An **Azure App Registration** (service principal) — required for Docker, recommended for local

**Option A only:**
- **Python 3.10 or later** — [download here](https://www.python.org/downloads/)

**Option B only:**
- **Docker Desktop** — [download here](https://www.docker.com/products/docker-desktop/) — must be running before you use `docker compose`

---

## Step 1 — Create an Azure ML workspace (one-time)

All AutoML training, model registration, and data assets live inside an Azure ML workspace. If you already have one, skip to Step 2.

1. Go to [portal.azure.com](https://portal.azure.com) → search for **Azure Machine Learning** → select it
2. Click **+ Create** → **New workspace**
3. Fill in the form:

| Field | Guidance |
|---|---|
| **Subscription** | Select your Azure subscription |
| **Resource group** | Create a new one (e.g. `automl-demo-rg`) or select an existing group |
| **Workspace name** | e.g. `automl-demo-ws` — this becomes `AZURE_WORKSPACE_NAME` |
| **Region** | Choose a region near you that supports AutoML compute (e.g. `East US`, `West Europe`) |

4. Leave Storage account, Key vault, and Application Insights on their defaults — Azure creates them automatically
5. Click **Review + create** → **Create** and wait ~2–3 minutes for deployment
6. Click **Go to resource** — you are now inside your workspace

---

## Step 2 — Deploy a GPT model (one-time)

You need a GPT deployment in **Azure AI Foundry** for the Responsible AI agent to work.

1. Go to [Azure AI Foundry](https://ai.azure.com) → select your project
2. Left menu → **Models + endpoints** → **+ Deploy model** → **Deploy base model**
3. Select `gpt-4.1` (or `gpt-4o-mini` as a cheaper alternative)
4. Set a **Deployment name** — e.g. `gpt-4.1`
5. Click **Deploy** and wait ~1 minute

---

## Step 3 — Create an Azure App Registration (service principal)

This creates a non-human identity the app can use to authenticate to Azure programmatically. **Required for Docker** (no browser available inside a container). Recommended for local too — it avoids relying on interactive browser login every session.

1. Go to [portal.azure.com](https://portal.azure.com) → search for **Microsoft Entra ID** → select it
2. Left menu → **App registrations** → **+ New registration**
3. Give it a name (e.g. `automl-demo-app`) → click **Register**
4. You are now on the app's overview page. Copy these two values:

| What you need | Where to find it |
|---|---|
| **Tenant ID** (`AZURE_TENANT_ID`) | Overview → "Directory (tenant) ID" |
| **Client ID** (`AZURE_CLIENT_ID`) | Overview → "Application (client) ID" |

5. Left menu → **Certificates & secrets** → **Client secrets** tab → **+ New client secret**
6. Add a description and choose an expiry → click **Add**
7. **Immediately copy the value from the `Value` column** (not the `Secret ID` column). This is your `AZURE_CLIENT_SECRET`. It will never be shown again after you leave this page.

---

## Step 4 — Grant the service principal Contributor access

The app needs permission to read from and write to your Azure ML workspace. Assign Contributor at the **resource group** level so it can access the workspace, storage, and any other resources within it.

1. Go to [portal.azure.com](https://portal.azure.com) → navigate to your **Resource Group** (the one containing your ML workspace)
2. Left menu → **Access control (IAM)** → **+ Add** → **Add role assignment**
3. On the **Role** tab: search for and select **AzureML Data Scientist** → click **Next**
4. On the **Members** tab: select **User, group, or service principal** → click **+ Select members**
5. Search for the app registration name you created in Step 3 (e.g. `automl-demo-app`) → select it → click **Select** → click **Review + assign**

> **Why resource group level?** Assigning at the workspace level alone is not sufficient — the service principal also needs access to the storage account and other resources in the same group.

---

## Step 5 — Collect your credentials

### From Azure ML workspace
Go to [portal.azure.com](https://portal.azure.com) → your ML workspace → **Overview**

| What you need | Where to find it |
|---|---|
| `AZURE_SUBSCRIPTION_ID` | Overview → "Subscription ID" |
| `AZURE_RESOURCE_GROUP` | Overview → "Resource group" |
| `AZURE_WORKSPACE_NAME` | Overview → "Name" |

### From Azure AI Foundry
Go to [ai.azure.com](https://ai.azure.com) → your project → **Overview**

| What you need | Where to find it |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Overview → "Azure OpenAI endpoint" (ends in `.openai.azure.com/`) |
| `AZURE_OPENAI_API_KEY` | Overview → "API keys" → Key 1 |
| `AZURE_OPENAI_DEPLOYMENT` | Models + endpoints → your deployment name from Step 1 |

---

## Step 6 — Create your `.env` file

Copy the example file and fill in all values:

```powershell
copy .env.example .env
```

Open `.env` in a text editor and replace each placeholder:

```
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_WORKSPACE_NAME=your-workspace-name
AZURE_TENANT_ID=your-tenant-id              # from Step 2
AZURE_CLIENT_ID=your-client-id              # from Step 2
AZURE_CLIENT_SECRET=your-client-secret      # from Step 2 — the Value column, not Secret ID
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-openai-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
```

> **Never commit `.env` to Git.** It is already listed in `.gitignore`. Treat it like a password file.

---

## Option A — Local pip install

### Install dependencies

```powershell
python -m venv venv
venv\Scripts\pip.exe install --no-deps -r app_requirements.txt
```

> **Why `--no-deps`?** Some Azure ML packages have conflicting transitive dependencies. `--no-deps` installs exactly the pinned versions in `app_requirements.txt` without automatic resolution. All transitive dependencies are already listed in the file.

### Run the app

The app reads credentials from environment variables. You must load the `.env` file into your shell session before starting Streamlit — both commands must run in the **same terminal**.

```powershell
# 1. Load .env into the current shell session (one-liner)
Get-Content .env | Where-Object { $_ -match '^\w' } | ForEach-Object { $k,$v=$_ -split '=',2; Set-Item "env:$k" $v }

# 2. Activate venv and run
.\venv\Scripts\Activate.ps1
streamlit run app.py
```



> **Important:** If you open a new terminal and run `streamlit run app.py` without loading `.env` first, the app will start but all Azure calls will fail with missing credentials errors.

> **Azure authentication (local):** The app uses `DefaultAzureCredential` which picks up the service principal credentials from the environment. If `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID` are all set, no browser login is needed.

---

## Option B — Docker

### First-time build and run

Make sure Docker Desktop is running, then:

```powershell
docker compose up --build
```

The first build downloads and installs all dependencies — expect approximately 30–40 minutes. Subsequent starts reuse the built image and take under 10 seconds.

```powershell
docker compose up
```

App is available at [http://localhost:8501](http://localhost:8501).

> **Service principal required.** Docker containers have no browser, so interactive login is not possible. `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID` must all be set in your `.env` file. The app will fail immediately at startup if any of these are missing when running inside Docker.

### Stop the container

```powershell
docker compose down
```

---

## Verify your `.env` is complete

```powershell
@("AZURE_SUBSCRIPTION_ID","AZURE_RESOURCE_GROUP","AZURE_WORKSPACE_NAME",
  "AZURE_TENANT_ID","AZURE_CLIENT_ID","AZURE_CLIENT_SECRET",
  "AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_KEY","AZURE_OPENAI_DEPLOYMENT") |
  ForEach-Object {
    $line = Get-Content .env | Where-Object { $_ -match "^$_=" }
    if ($line -and ($line -split "=",2)[1].Trim()) {
      Write-Host "OK      $_"
    } else {
      Write-Host "MISSING $_"
    }
  }
```

All nine should show `OK` before starting the app.

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'azure.ai.ml'` | pip installed into system Python, not the venv | Use `venv\Scripts\pip.exe install --no-deps -r app_requirements.txt` |
| `ImportError: cannot import name 'FieldInstanceResolutionError' from 'marshmallow'` | marshmallow 4.x installed | Re-install: `venv\Scripts\pip.exe install --no-deps -r app_requirements.txt` |
| `DeploymentNotFound` | Wrong deployment name | Check exact name in AI Foundry → Models + endpoints |
| `Resource not found (404)` | Wrong endpoint URL | Use the `.openai.azure.com/` endpoint, not the Foundry project URL |
| `404 — Could not find deployment to match model` | API key from different resource than endpoint | Endpoint, API key, and deployment must come from the **same** Azure OpenAI resource |
| `ClientSecretCredential authentication failed` | Wrong tenant ID, client ID, or secret value | Re-check Step 3 — ensure you copied the secret **Value**, not the **Secret ID** |
| `AuthorizationFailed` | Service principal lacks permissions | Re-check Step 3 — role must be assigned at **resource group** level, not just workspace |
| `Subscription ID not provided` | Missing `.env` value | Check `AZURE_SUBSCRIPTION_ID` in `.env` |
| `Workspace not found` | Wrong resource group or workspace name | Cross-check with Azure Portal |
| Docker container exits immediately | Missing service principal credentials | All three of `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` must be in `.env` |
