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
│   ├── artifacts/
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   ├── y_test.csv
│   │   ├── minmax_scaler.pkl
│   │   └── feature_columns.json
│   └── EDA_plots/
├── milestone2-ml-pipeline/
│   ├── Milestone_2_AI_Model_Integration.ipynb
│   ├── models/
│   │   ├── xgb_churn_model.pkl
│   │   ├── rf_churn_model.pkl
│   │   └── best_churn_model.pkl
│   └── evaluation_plots/
└── milestone3-api-deployment/
    └── milestone3/
        ├── Dockerfile
        ├── docker-compose.yml
        ├── requirements.txt
        ├── train_and_save_model.py
        ├── run_pipeline.py
        ├── app/
        │   ├── main.py
        │   ├── api/
        │   ├── core/
        │   ├── models/
        │   ├── schemas/
        │   ├── services/
        │   └── data/predictions/prediction_logs.csv   ← Milestone 4 monitoring log
        └── dags/
            └── churn_inference_dag.py
```

---

## Prerequisites

- Python 3.11
- Conda
- VS Code with the **Jupyter** extension installed
- Docker Desktop
- A Kaggle account

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
pip install -r requirements.txt
python -m ipykernel install --user --name bigdata_depi --display-name "Python (bigdata_depi)"
```

### 3. Set up Kaggle credentials
- Go to https://www.kaggle.com → Account → API → Create New Token
- Place `kaggle.json` at `C:\Users\<YourUsername>\.kaggle\kaggle.json`

---

## Milestone 1 — Data Collection & Feature Engineering

**What it does:**
- Downloads the Telco Churn dataset via `kagglehub`
- Cleans and preprocesses data (handles missing values, type conversions)
- Runs 8 SQL queries on an in-memory SQLite database to engineer features
- Performs Exploratory Data Analysis (EDA) with visualizations
- Applies MinMaxScaler to continuous features (fit on train only — no data leakage)
- Handles class imbalance using SMOTE
- Exports all artifacts for downstream milestones

**How to run:**
1. Open `milestone1-ml-pipeline/Milestone_1_Data_Collection_&_Feature_Engineering.ipynb`
2. Select kernel: **Python (bigdata_depi)**
3. Click **Run All**

**Outputs in `milestone1-ml-pipeline/artifacts/`:**

| File | Description |
|---|---|
| `X_train.csv` | SMOTE-balanced, scaled training features (8,278 rows) |
| `y_train.csv` | Training labels — balanced 50/50 after SMOTE |
| `X_test.csv` | Scaled test features (1,409 rows) |
| `y_test.csv` | Test labels (real distribution, untouched) |
| `minmax_scaler.pkl` | Fitted MinMaxScaler — must be reused at inference time |
| `feature_columns.json` | Final 24 feature column names and order |

---

## Milestone 2 — AI Model Integration

**What it does:**
- Loads Milestone 1 artifacts (no re-running the pipeline)
- Trains XGBoost and Random Forest as deployment candidates
- Trains Logistic Regression as a benchmark comparison model
- Full evaluation: accuracy, precision, recall, F1, ROC-AUC
- Auto-selects best model by ROC-AUC → saved as `best_churn_model.pkl`

**How to run:**
1. Open `milestone2-ml-pipeline/Milestone_2_AI_Model_Integration.ipynb`
2. Select kernel: **Python (bigdata_depi)**
3. Confirm Cell 2: `ARTIFACTS_DIR = '../milestone1-ml-pipeline/artifacts'`
4. Click **Run All**

**Evaluation Results (test set, 1,409 rows):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | 0.7601 | 0.5350 | 0.7353 | 0.6194 | **0.8395 ✅** |
| Logistic Regression *(benchmark)* | 0.7516 | 0.5227 | 0.7380 | 0.6120 | 0.8289 |
| XGBoost | 0.7700 | 0.5539 | 0.6872 | 0.6134 | 0.8277 |

Random Forest wins on ROC-AUC and is deployed as `best_churn_model.pkl`.

---

## Milestone 3 — Pipeline Automation & Deployment

**What it does:**
- Deploys the best model as a REST API using FastAPI
- Automates daily batch predictions using an Apache Airflow DAG
- Runs everything in Docker containers for consistency

### Running with Docker (recommended)

```bash
cd milestone3-api-deployment/milestone3

# First time only — retrain model inside Docker for version consistency
docker compose build
docker compose run --rm web-api python train_and_save_model.py

# Start the full stack
docker compose up
```

### What's available after startup

| Service | URL | Description |
|---|---|---|
| Prediction GUI | `http://localhost:8000` | Dashboard to test predictions |
| API Docs (Swagger) | `http://localhost:8000/docs` | Interactive API documentation |
| Health Check | `http://localhost:8000/health` | Confirms all 3 artifacts loaded |
| Airflow UI | `http://localhost:8082` | Trigger and monitor the daily DAG |

### Airflow login
Username: `admin` — password is printed in terminal on first startup:
```
standalone | Login with username: admin  password: XXXXXXXXXX
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — verifies model, scaler, feature columns |
| POST | `/api/v1/churn/predict` | Predict churn for a customer |
| GET | `/docs` | Swagger UI |

### Example prediction request
```bash
curl -X POST http://localhost:8000/api/v1/churn/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0,
    "Contract": "Month-to-month",
    "total_services": 1,
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "PaperlessBilling": 1,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0
  }'
```

Expected response:
```json
{
  "success": true,
  "churn_prediction": 1,
  "churn_probability": 0.8738,
  "model_version": "best_churn_model.pkl"
}
```

### Airflow DAG — Daily Batch Pipeline

The DAG `telco_churn_daily_inference_pipeline` runs `@daily` with 3 tasks:

```
Task 1: mock_raw_daily_data
    → Simulates pulling fresh customer records from a database

Task 2: perform_feature_engineering
    → Encodes features, calculates total_services (8 services),
      applies MinMaxScaler from Milestone 1

Task 3: execute_batch_predictions
    → Loads best_churn_model.pkl, runs predictions,
      saves results with risk levels (HIGH/MEDIUM/LOW)
```

Results saved to: `data/predictions/daily_results.csv`

### Running without Docker (local testing)
```bash
conda activate bigdata_depi
pip install fastapi uvicorn pydantic pydantic-settings joblib scikit-learn==1.8.0 xgboost pandas
cd milestone3-api-deployment/milestone3
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run DAG locally (second terminal)
python run_pipeline.py
```

---

## Milestone 4 — Monitoring & Reporting

**What it does:**
- Logs every prediction automatically to a CSV file for drift monitoring
- Provides data for Power BI dashboard showing churn trends over time

### Prediction Logging (automatic)

Every call to `/api/v1/churn/predict` automatically appends a row to:
```
milestone3-api-deployment/milestone3/app/data/predictions/prediction_logs.csv
```

Logged fields per prediction:

| Column | Description |
|---|---|
| `timestamp` | When the prediction was made |
| `tenure`, `MonthlyCharges`, `TotalCharges` | Core numeric features |
| `Contract`, `InternetService`, `PaymentMethod` | Key categorical features |
| `total_services` | Number of subscribed services |
| `SeniorCitizen`, `Partner`, `Dependents` | Demographics |
| `PaperlessBilling`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Add-on services |
| `churn_prediction` | 0 = stays, 1 = churns |
| `churn_probability` | Model confidence (0.0–1.0) |

### How to connect to Power BI

1. Open **Power BI Desktop**
2. Click **Get Data** → **Text/CSV**
3. Navigate to `prediction_logs.csv`
4. Build these charts:
   - **Line chart** — `churn_probability` over `timestamp` (drift monitoring)
   - **Bar chart** — average churn rate by `Contract` type
   - **Pie chart** — churn distribution by `InternetService`
   - **KPI card** — overall churn rate today vs. yesterday

### Model Drift Detection

If the average `churn_probability` in `prediction_logs.csv` shifts significantly over time (e.g. average was 0.35 last month, now it's 0.55), this signals potential model drift — the model may need retraining on newer data.

---

## Run Order Summary

```
Step 1 → Run Milestone 1 notebook        (produces artifacts/)
Step 2 → Run Milestone 2 notebook        (produces models/ + evaluation_plots/)
Step 3 → docker compose build            (builds Docker image)
Step 4 → docker compose run train        (retrains models inside Docker)
Step 5 → docker compose up               (starts API + Airflow)
Step 6 → Open http://localhost:8000      (test predictions via GUI)
Step 7 → Open http://localhost:8082      (trigger/monitor DAG)
Step 8 → Connect prediction_logs.csv     (to Power BI for dashboard)
```

---

## Notes for the Team

- **Run milestones in order** — each milestone depends on artifacts from the previous one
- **Never re-fit the scaler** — `minmax_scaler.pkl` must be the same one used during training
- **Model version consistency** — always retrain inside Docker (`train_and_save_model.py`) to avoid scikit-learn version mismatches
- **Prediction logs grow over time** — refresh Power BI periodically to see updated drift metrics
- `best_churn_model.pkl` auto-updates when you retrain — no config changes needed

---

## Dataset

**IBM Telco Customer Churn** — loaded automatically via `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
```
Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn