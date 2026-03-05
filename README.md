# 🤖 Azure AutoML Trainer

A simple, clean Streamlit app for training machine learning models using **Azure AutoML**. 

Upload your CSV, select a target, and let Azure ML find the best model for your data.

---

## Features

✅ **Simple UI** - Upload CSV → Select target → Auto-detect task → Run training  
✅ **Auto-Detect Task Type** - Automatically detects Classification vs Regression  
✅ **Azure AutoML** - Uses official Python SDK v2  
✅ **Top Models** - Shows top 5 trained models ranked by score  
✅ **Best Model Metrics** - Displays accuracy, precision, recall, F1, RMSE, etc.  
✅ **Production Ready** - Clean, documented code ready to fork/modify  

---

## Quick Start

### 1. Setup (5 minutes)

See **[SETUP.md](SETUP.md)** for detailed instructions. Quick version:

```bash
# Create Azure ML workspace in Portal (manual - takes 5 min)
# https://portal.azure.com

# Set environment variables
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="automl-demo-rg"
export AZURE_WORKSPACE_NAME="automl-demo-ws"

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### 2. Usage

1. Open the app (http://localhost:8501)
2. Upload a CSV file
3. Select the target column to predict
4. Click "Start AutoML Training"
5. View results as training progresses

---

## Project Structure

```
.
├── app.py                  # Streamlit UI (step 1-4)
├── run_automl.py          # AutoML helpers (problem detection, metrics extraction)
├── utils.py               # File upload utilities
├── ml_pipeline/
│   ├── client.py          # Azure authentication
│   ├── data.py            # Data registration as MLTable
│   └── job.py             # AutoML job submission
├── data/                  # Sample datasets
├── SETUP.md               # Detailed setup guide
└── README.md             # This file
```

---

## How It Works

### Step 1: Upload & Analyze
- User uploads CSV file
- App shows data preview
- Detects data types and column count

### Step 2: Auto-Detect Task Type
- Checks if target is numeric or categorical
- Checks cardinality (unique values)
- Returns: **Classification** or **Regression**

Example logic:
```python
if target is non-numeric:
    return "Classification"
if target has <= 20 unique integer values:
    return "Classification"
if target has < 5% unique values:
    return "Classification"
else:
    return "Regression"
```

### Step 3: Submit to Azure AutoML
- Registers CSV as MLTable data asset
- Submits AutoML job with appropriate metric
- Returns job name for tracking

### Step 4: Get Results
- Fetches all trained models
- Extracts metrics (accuracy, precision, RMSE, etc.)
- Displays top 5 models + best model details

---

## Metrics Displayed

### Classification Models
- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- Weighted Accuracy

### Regression Models
- R² Score
- Normalized RMSE
- RMSE
- MAE (Mean Absolute Error)
- MAPE

---

## Configuration

All configuration is in environment variables:

```bash
AZURE_SUBSCRIPTION_ID      # Your Azure subscription
AZURE_RESOURCE_GROUP       # Resource group name (default: automl-demo-rg)
AZURE_WORKSPACE_NAME       # Workspace name (default: automl-demo-ws)
AZURE_LOCATION            # Region (default: canadacentral)
```

To change training limits, edit [ml_pipeline/job.py](ml_pipeline/job.py):

```python
DEFAULT_MAX_TRIALS = 3                    # Number of models to try
DEFAULT_TRIAL_TIMEOUT_MINUTES = 5         # Time per model
DEFAULT_TIMEOUT_MINUTES = 15              # Total timeout
```

---

## Code Quality

- ✅ **Step-by-step docstrings** - Each function explained with clear steps
- ✅ **Type hints** - Full type annotations for clarity
- ✅ **No magic** - No hidden auto-creation or complex logic
- ✅ **Clean architecture** - Clear separation of concerns
- ✅ **Error handling** - User-friendly error messages

---

## Sample Data

A Titanic dataset is included for testing:

```bash
streamlit run app.py

# Upload: data/titanic.csv
# Target: Survived
# Expected: Classification task detected
```

---

## Troubleshooting

### "Set AZURE_SUBSCRIPTION_ID in your environment"
→ See **[SETUP.md](SETUP.md)** Step 4

### "Workspace not found"
→ Verify resource group & workspace names in Azure Portal

### "Soft-deleted workspace exists"
→ Use a different workspace name or wait 48 hours

### "Authentication failed"
→ Follow the browser prompt to login with your Azure account

See [SETUP.md](SETUP.md#troubleshooting) for more help.

---

## Why Manual Setup Over Auto-Creation?

This project uses **manual resource creation** instead of auto-deployment because:

| Aspect | Why |
|--------|-----|
| **Simpler Code** | No deployment scripts = fewer bugs |
| **Transparent** | Users see exactly what they're creating |
| **Fewer Conflicts** | No soft-delete state issues |
| **Better Learning** | Understand the architecture |
| **Flexible** | Reuse existing workspaces |

---

## Requirements

- Python 3.9+
- Azure subscription (free tier eligible)
- Azure ML Workspace (manually created)

Dependencies installed via `pip install -r requirements.txt`:
- `streamlit` - Web UI
- `pandas` - Data handling  
- `azure-ai-ml==1.x` - Azure ML SDK v2
- `azure-identity` - Authentication

---

## Development

To modify this project:

1. Change AutoML parameters → [ml_pipeline/job.py](ml_pipeline/job.py)
2. Add data preprocessing → [ml_pipeline/data.py](ml_pipeline/data.py)
3. Modify UI → [app.py](app.py)
4. Add new metrics → [run_automl.py](run_automl.py)

---

## License

This project is provided as-is for educational purposes.

---

## Learn More

- [Azure ML Documentation](https://learn.microsoft.com/azure/machine-learning/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python SDK v2 Reference](https://learn.microsoft.com/python/api/overview/azure/ai-ml-readme)

---

## Contributing

Found a bug or have an idea? Contributions welcome!

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Happy training! 🚀**

Need help? See [SETUP.md](SETUP.md) or the troubleshooting section above.

