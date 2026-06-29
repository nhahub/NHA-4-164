# Enhanced Pipeline for Churn Prediction (DEPI)

A full end-to-end MLOps pipeline for predicting customer churn using the IBM Telco Customer Churn dataset (7,043 rows, 21 features).

---

## Project Structure

```
Enhanced Pipeline for Churn Prediction (DEPI)/
├── requirements.txt
├── README.md
├── milestone1-ml-pipeline/
│   ├── Milestone_1_Data_Collection_&_Feature_Engineering.ipynb
│   ├── artifacts/                  ← auto-generated when you run M1
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   ├── y_test.csv
│   │   ├── minmax_scaler.pkl
│   │   └── feature_columns.json
│   └── EDA_plots/
└── milestone2-ml-pipeline/
    ├── Milestone_2_AI_Model_Integration.ipynb
    ├── models/                     ← auto-generated when you run M2
    │   ├── xgb_churn_model.pkl
    │   ├── rf_churn_model.pkl
    │   └── best_churn_model.pkl
    └── evaluation_plots/
```

---

## Prerequisites

- Python 3.11
- Conda (recommended) or any Python virtual environment
- VS Code with the **Jupyter** extension installed
- A Kaggle account (for dataset download via `kagglehub`)

---

## Setup — Do This Once

### 1. Clone the repository
```bash
git clone https://github.com/nhahub/NHA-4-164.git
cd NHA-4-164
cd "Enhanced Pipeline for Churn Prediction (DEPI)"
```

### 2. Create and activate the conda environment
```bash
conda create -n bigdata_depi python=3.11
conda activate bigdata_depi
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

### 4. Register the kernel so VS Code can see it
```bash
python -m ipykernel install --user --name bigdata_depi --display-name "Python (bigdata_depi)"
```

### 5. Set up Kaggle credentials (for dataset download in Milestone 1)
- Go to https://www.kaggle.com → Account → API → Create New Token
- This downloads a `kaggle.json` file
- Place it at:
  - **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`
  - **Linux/Mac:** `~/.kaggle/kaggle.json`

---

## Running the Notebooks

> ⚠️ **Order matters** — always run Milestone 1 first. Milestone 2 loads the artifacts that Milestone 1 produces.

---

### Milestone 1 — Data Collection & Feature Engineering

**What it does:**
- Downloads the Telco Churn dataset via `kagglehub`
- Cleans and preprocesses data (handles missing values, encodes features)
- Runs 8 SQL queries on an in-memory SQLite database to engineer features
- Performs Exploratory Data Analysis (EDA) with 9 visualizations
- Applies MinMaxScaler to continuous features (no data leakage — fit on train only)
- Handles class imbalance using SMOTE
- Exports all artifacts to `milestone1-ml-pipeline/artifacts/`

**Steps:**
1. Open VS Code → `File > Open Folder` → select the project folder
2. Open `milestone1-ml-pipeline/Milestone_1_Data_Collection_&_Feature_Engineering.ipynb`
3. Click **Select Kernel** (top right) → choose **Python (bigdata_depi)**
4. Click **Run All** (`Ctrl+Alt+R`)

**Expected outputs in `milestone1-ml-pipeline/artifacts/`:**

| File | Description |
|---|---|
| `X_train.csv` | SMOTE-balanced, scaled training features (8,278 rows) |
| `y_train.csv` | Training labels — balanced 50/50 after SMOTE |
| `X_test.csv` | Scaled test features (1,409 rows, real distribution) |
| `y_test.csv` | Test labels (imbalanced, untouched — realistic) |
| `minmax_scaler.pkl` | Fitted MinMaxScaler — reuse in inference to avoid leakage |
| `feature_columns.json` | Final list of 24 feature column names after encoding |

---

### Milestone 2 — AI Model Integration

**What it does:**
- Loads Milestone 1 artifacts directly (no re-running the pipeline)
- Trains XGBoost and Random Forest as deployment candidates
- Trains Logistic Regression as a benchmark comparison model
- Runs full evaluation: accuracy, precision, recall, F1, ROC-AUC
- Generates confusion matrices, ROC curves, and a model comparison chart
- Auto-selects the best model by ROC-AUC and saves it as `best_churn_model.pkl`

**Steps:**
1. Open `milestone2-ml-pipeline/Milestone_2_AI_Model_Integration.ipynb`
2. Select the same kernel: **Python (bigdata_depi)**
3. In **Cell 2**, confirm `ARTIFACTS_DIR` points to Milestone 1's artifacts:
   ```python
   ARTIFACTS_DIR = '../milestone1-ml-pipeline/artifacts'
   ```
4. Click **Run All**

**Expected outputs in `milestone2-ml-pipeline/`:**

| File | Description |
|---|---|
| `models/xgb_churn_model.pkl` | Trained XGBoost model |
| `models/rf_churn_model.pkl` | Trained Random Forest model |
| `models/best_churn_model.pkl` | Best model by ROC-AUC (used by the API in Milestone 3) |
| `evaluation_plots/evaluation_report.csv` | Full metrics table for all 3 models |
| `evaluation_plots/confusion_matrices.png` | Confusion matrices side-by-side |
| `evaluation_plots/model_comparison.png` | Bar chart comparing all metrics |
| `evaluation_plots/roc_curves.png` | ROC curves for all 3 models |

**Evaluation Results (on 1,409-row test set):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.7601 | 0.5350 | 0.7353 | 0.6194 | **0.8395** ✅ |
| Logistic Regression *(benchmark)* | 0.7516 | 0.5227 | 0.7380 | 0.6120 | 0.8289 |
| XGBoost | 0.7700 | 0.5539 | 0.6872 | 0.6134 | 0.8277 |

**Random Forest wins** on ROC-AUC and is saved as `best_churn_model.pkl` for deployment.

---

## Notes for the Team

- **Never run Milestone 2 before Milestone 1** — it will fail with a file-not-found error since the artifacts won't exist yet.
- **Do not manually edit** any file inside `artifacts/` — these are auto-generated by Milestone 1. If something looks wrong, re-run Milestone 1 clean.
- The `minmax_scaler.pkl` file is critical for inference — it must be the same scaler used during training. Never re-fit it on new data.
- `best_churn_model.pkl` is what the Milestone 3 API loads. If you retrain the models and a different one wins, this file updates automatically.

---

## Dataset

**IBM Telco Customer Churn** — loaded automatically via `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
```
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn