# Setup Guide

This guide walks you through setting up the project from scratch.

---

## Prerequisites

- **Python 3.10 or later** — [download here](https://www.python.org/downloads/)
- **Azure subscription** — with an Azure ML workspace created
- **Azure AI Foundry** — with an Azure OpenAI resource and a model deployment

---

## Step 1 — Install dependencies

**Windows (recommended):** Create a virtual environment, then use the venv's pip directly:

```powershell
python -m venv lastenv
lastenv\Scripts\pip.exe install --no-deps -r app_requirements.txt
```

> **Why `--no-deps`?** Some Azure ML packages have conflicting transitive dependencies (e.g. `psutil`). Using `--no-deps` installs exactly the pinned versions in `app_requirements.txt` without automatic dependency resolution, which avoids version conflicts. All required packages including transitive dependencies are already listed in the file.

**macOS / Linux:**
```bash
python3 -m venv lastenv
source lastenv/bin/activate
pip install --no-deps -r app_requirements.txt
```

---

## Step 2 — Deploy a GPT model (one-time)

You need a GPT deployment in **Azure AI Foundry** for the Responsible AI agent to work.

1. Go to [Azure AI Foundry](https://ai.azure.com) → select your project
2. Left menu → **Models + endpoints** → **+ Deploy model** → **Deploy base model**
3. Select `gpt-4o-mini` (recommended — cheap and reliable)
4. Set a **Deployment name** — e.g. `gpt-4o-mini`
5. Click **Deploy** and wait ~1 minute

---

## Step 3 — Collect your credentials

You need values from two places:

### From Azure ML workspace (Azure Portal)
Go to [portal.azure.com](https://portal.azure.com) → search for your ML workspace → **Overview**

| What you need | Where to find it |
|---|---|
| Subscription ID | Overview page → "Subscription ID" |
| Resource group | Overview page → "Resource group" |
| Workspace name | Overview page → "Name" |

### From Azure AI Foundry
Go to [ai.azure.com](https://ai.azure.com) → your project → **Overview**

| What you need | Where to find it |
|---|---|
| Azure OpenAI endpoint | Overview → "Azure OpenAI endpoint" (ends in `.openai.azure.com/`) |
| API key | Overview → "API keys" → Key 1 |
| Deployment name | Models + endpoints → your deployment name (what you typed in Step 2) |

---

## Step 4 — Set environment variables

Run these in PowerShell **before** starting the app. Replace each `<...>` with your actual value.

```powershell
# Azure ML workspace
$env:AZURE_SUBSCRIPTION_ID   = "<your-subscription-id>"
$env:AZURE_RESOURCE_GROUP    = "<your-resource-group>"
$env:AZURE_WORKSPACE_NAME    = "<your-workspace-name>"

# Azure OpenAI (for the RAI agent)
$env:AZURE_OPENAI_ENDPOINT   = "https://<resource-name>.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY    = "<your-api-key>"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"   # the deployment name from Step 2
```

> **These are session variables.** They are only set for the current PowerShell window and are never saved to disk or committed to Git. You need to re-run these commands each time you open a new terminal.

> **Tip:** To make env vars persist across terminal sessions, append them to `lastenv\Scripts\Activate.ps1`. They will be set automatically every time you activate the venv.

> **Azure authentication:** When the app connects to Azure for the first time, a browser window will open automatically asking you to sign in. Tenant ID is resolved automatically from your login — no need to set it manually.

---

## Verify all variables are set

```powershell
@("AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_KEY","AZURE_OPENAI_DEPLOYMENT",
  "AZURE_SUBSCRIPTION_ID","AZURE_RESOURCE_GROUP","AZURE_WORKSPACE_NAME") |
  ForEach-Object {
    $val = [System.Environment]::GetEnvironmentVariable($_)
    if ($val) { Write-Host "OK      $_" } else { Write-Host "MISSING $_" }
  }
```

All six should show `OK`.

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'azure.ai.ml'` | `pip` installed into system Python, not the venv | Use `lastenv\Scripts\pip.exe install --no-deps -r app_requirements.txt` |
| `ImportError: cannot import name 'FieldInstanceResolutionError' from 'marshmallow'` | marshmallow 4.x installed instead of 3.x | Re-install with `lastenv\Scripts\pip.exe install --no-deps -r app_requirements.txt` |
| `DeploymentNotFound` | Wrong deployment name | Check the exact name in AI Foundry → Models + endpoints |
| `Resource not found (404)` | Wrong endpoint URL | Use the `.openai.azure.com/` endpoint, not the Foundry project URL |
| `Subscription ID not provided` | Env var not set | Re-run Step 4 |
| `Workspace not found` | Wrong resource group or workspace name | Double-check in Azure Portal |
| `Authentication failed` | Not signed in | A browser window should open automatically — sign in with your Azure account |
