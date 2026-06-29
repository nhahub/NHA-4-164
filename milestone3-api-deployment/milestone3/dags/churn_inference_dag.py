"""
Airflow DAG: telco_churn_daily_inference_pipeline

Automates the full daily batch churn prediction pipeline:

Task 1 → mock_raw_data_task:
    Simulates pulling fresh customer records from a database.
    In production this would be a database query or API call.

Task 2 → feature_engineering_task:
    Replicates the exact feature engineering from Milestone 1:
    - Encodes categorical fields (Contract, InternetService, PaymentMethod)
    - Calculates total_services the SAME way as Milestone 1 SQL Query 8
      (8 services: 6 add-ons + PhoneService + InternetService)
    - Applies MinMaxScaler (loaded from Milestone 1 artifact) to the 3
      continuous columns — this was missing before and caused wrong predictions

Task 3 → batch_inference_task:
    Loads best_churn_model.pkl (winner from Milestone 2 evaluation),
    runs predictions on the engineered batch, saves results CSV.

Flow: mock_data >> feature_engineering >> batch_predictions
"""

import os
import json
import joblib
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Airflow import with graceful fallback so the file can be
# tested locally without a running Airflow instance
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    class DAG:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class _MockOp:
        def __init__(self, *args, **kwargs): pass
        def __rshift__(self, other): return other
    def PythonOperator(**kwargs): return _MockOp()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airflow.dag.churn_inference")

# ── Default Airflow task arguments ────────────────────────────────────────────
default_args = {
    "owner": "mlops_engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

# ── File paths (Docker volume mounts) ────────────────────────────────────────
RAW_DATA_PATH        = "/app/data/inputs/raw_daily_data.csv"
ENGINEERED_DATA_PATH = "/app/data/inputs/engineered_daily_data.csv"
RESULTS_PATH         = "/app/data/predictions/daily_results.csv"
_LOCAL_RESULTS_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "predictions", "daily_results.csv")

# Milestone 1 artifacts
SCALER_PATH          = "/app/app/models/minmax_scaler.pkl"
FEATURE_COLS_PATH    = "/app/app/models/feature_columns.json"

# Milestone 2 best model — NOT hardcoded to XGBoost anymore
MODEL_PATH           = "/app/app/models/best_churn_model.pkl"

# Fallback paths for local testing outside Docker
def _resolve(primary, *fallbacks):
    """Return the first path that actually exists."""
    for p in [primary, *fallbacks]:
        if os.path.exists(p):
            return p
    return primary  # return primary so the error message is meaningful

def get_model_path():
    return _resolve(
        MODEL_PATH,
        os.path.join(os.path.dirname(__file__), "..", "app", "models", "best_churn_model.pkl")
    )

def get_scaler_path():
    return _resolve(
        SCALER_PATH,
        os.path.join(os.path.dirname(__file__), "..", "app", "models", "minmax_scaler.pkl")
    )

def get_feature_cols_path():
    return _resolve(
        FEATURE_COLS_PATH,
        os.path.join(os.path.dirname(__file__), "..", "app", "models", "feature_columns.json")
    )


# ── TASK 1: Raw Data Extraction ───────────────────────────────────────────────

def mock_raw_data_task():
    """
    Simulates pulling fresh daily customer records from a source database.

    In production this would be replaced with:
        df = pd.read_sql("SELECT * FROM customers WHERE date = TODAY", conn)

    Saves raw records to RAW_DATA_PATH for Task 2 to pick up.
    Note: raw records use string values for categorical fields
    (e.g. Contract="One year") — encoding happens in Task 2,
    exactly like Milestone 1 did it.
    """
    logger.info("Task 1: Extracting raw customer records...")
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    raw_df = pd.DataFrame([
        {
            "CustomerID": "1001-A", "tenure": 12,
            "MonthlyCharges": 70.05, "TotalCharges": 840.60,
            "Contract": "One year", "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check", "PhoneService": "Yes",
            "OnlineSecurity": 1, "OnlineBackup": 0, "DeviceProtection": 1,
            "TechSupport": 1, "StreamingTV": 0, "StreamingMovies": 0,
            "SeniorCitizen": 0, "Partner": 1, "Dependents": 0, "PaperlessBilling": 1
        },
        {
            "CustomerID": "2002-B", "tenure": 3,
            "MonthlyCharges": 45.15, "TotalCharges": 135.45,
            "Contract": "Month-to-month", "InternetService": "DSL",
            "PaymentMethod": "Mailed check", "PhoneService": "No",
            "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
            "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 1,
            "SeniorCitizen": 1, "Partner": 0, "Dependents": 0, "PaperlessBilling": 1
        },
        {
            "CustomerID": "3003-C", "tenure": 72,
            "MonthlyCharges": 115.80, "TotalCharges": 8337.60,
            "Contract": "Two year", "InternetService": "Fiber optic",
            "PaymentMethod": "Bank transfer (automatic)", "PhoneService": "Yes",
            "OnlineSecurity": 1, "OnlineBackup": 1, "DeviceProtection": 1,
            "TechSupport": 1, "StreamingTV": 1, "StreamingMovies": 1,
            "SeniorCitizen": 0, "Partner": 1, "Dependents": 1, "PaperlessBilling": 0
        }
    ])

    raw_df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Task 1 complete: {len(raw_df)} records saved to {RAW_DATA_PATH}")


# ── TASK 2: Feature Engineering ───────────────────────────────────────────────

def feature_engineering_task():
    """
    Replicates the EXACT feature engineering pipeline from Milestone 1.

    Key fix: total_services now counts the same 8 services as Milestone 1
    SQL Query 8 (previously only counted 6, missing PhoneService and
    InternetService — this caused a feature mismatch at inference time).

    Also applies MinMaxScaler to tenure, MonthlyCharges, TotalCharges —
    this is the critical step that was completely missing from the original
    DAG, causing the model to receive raw unscaled numbers.
    """
    logger.info("Task 2: Running feature engineering pipeline...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data not found at: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    # ── 2a. total_services (matches Milestone 1 SQL Query 8 exactly) ─────────
    # Counts 8 services:
    # 6 add-on services + PhoneService + has any InternetService
    add_on_services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["total_services"] = (
        df[add_on_services].sum(axis=1)                              # 0–6
        + (df["PhoneService"] == "Yes").astype(int)                  # +1 if phone
        + (df["InternetService"] != "No").astype(int)                # +1 if internet
    )
    logger.info(f"total_services range: {df['total_services'].min()}–{df['total_services'].max()}")

    # ── 2b. One-hot encode categorical fields (matches Milestone 1) ───────────
    df = pd.get_dummies(
        df,
        columns=["InternetService", "Contract", "PaymentMethod"],
        drop_first=False
    )

    # Drop columns not used by the model
    drop_cols = ["CustomerID", "PhoneService", "MultipleLines",
                 "gender", "Churn"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── 2c. Load feature column list from Milestone 1 ─────────────────────────
    feature_cols_path = get_feature_cols_path()
    with open(feature_cols_path) as f:
        expected_cols = json.load(f)

    # Add any missing columns as 0 (e.g. a payment method not in today's batch)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Keep only the model's expected columns in the correct order
    df = df[expected_cols]

    # ── 2d. Apply MinMaxScaler from Milestone 1 ───────────────────────────────
    # THE CRITICAL FIX: scale the 3 continuous columns the same way they
    # were scaled during training. Without this, tenure=12 goes into the
    # model as 12.0 instead of ~0.16, which breaks all predictions.
    scaler_path = get_scaler_path()
    scaler = joblib.load(scaler_path)
    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[continuous_cols] = scaler.transform(df[continuous_cols])

    os.makedirs(os.path.dirname(ENGINEERED_DATA_PATH), exist_ok=True)
    df.to_csv(ENGINEERED_DATA_PATH, index=False)
    logger.info(f"Task 2 complete: engineered data saved to {ENGINEERED_DATA_PATH}")


# ── TASK 3: Batch Inference ───────────────────────────────────────────────────

def batch_inference_task():
    """
    Loads best_churn_model.pkl (selected by Milestone 2 evaluation),
    runs it on the engineered daily batch, and saves a prediction report.

    Because Task 2 already applied the scaler and aligned columns,
    this task is purely: load → predict → save. Clean and simple.
    """
    logger.info("Task 3: Running batch inference...")

    if not os.path.exists(ENGINEERED_DATA_PATH):
        raise FileNotFoundError(f"Engineered data not found at: {ENGINEERED_DATA_PATH}")

    model_path = get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    logger.info(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    df = pd.read_csv(ENGINEERED_DATA_PATH)

    logger.info(f"Running predictions on {len(df)} records...")
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    # Reload raw data just for CustomerID to include in results
    raw_df = pd.read_csv(RAW_DATA_PATH)

    results_df = pd.DataFrame({
        "CustomerID":         raw_df["CustomerID"],
        "churn_prediction":   preds,
        "churn_probability":  [round(float(p), 4) for p in probs],
        "risk_level":         ["HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.4 else "LOW"
                               for p in probs],
        "model_used":         os.path.basename(model_path),
        "prediction_timestamp": datetime.now().isoformat()
    })

    # Save to Docker path if running in container, else local path
    save_path = RESULTS_PATH if os.path.exists('/app') else os.path.abspath(_LOCAL_RESULTS_PATH)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)

    # Summary log
    high_risk = (results_df["risk_level"] == "HIGH").sum()
    logger.info(
        f"Task 3 complete: {len(results_df)} predictions saved. "
        f"High risk: {high_risk}/{len(results_df)}"
    )
    logger.info(f"Results saved to: {RESULTS_PATH}")


# ── AIRFLOW DAG DEFINITION ────────────────────────────────────────────────────

with DAG(
    dag_id="telco_churn_daily_inference_pipeline",
    default_args=default_args,
    description="Daily automated customer churn batch prediction pipeline",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "churn", "batch", "depi"],
) as dag:

    mock_data = PythonOperator(
        task_id="mock_raw_daily_data",
        python_callable=mock_raw_data_task
    )

    feature_engineering = PythonOperator(
        task_id="perform_feature_engineering",
        python_callable=feature_engineering_task
    )

    batch_predictions = PythonOperator(
        task_id="execute_batch_predictions",
        python_callable=batch_inference_task
    )

    # Task dependency: Task1 → Task2 → Task3
    mock_data >> feature_engineering >> batch_predictions
