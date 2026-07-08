"""
Airflow DAG: telco_churn_daily_inference_pipeline

Automates the full daily batch churn prediction pipeline:

Task 1 → mock_raw_data_task:
    Simulates pulling fresh customer records from a database.
    FIX: previously returned the same 3 hardcoded rows every run. Now
    samples real, held-out customers from Milestone 1's X_test.csv (never
    used in training) and reverses the scaling/encoding Milestone 1 applied,
    so Task 2 can re-process them exactly like fresh incoming data. Which
    customers get sampled is seeded by the DAG's logical date, so a given
    date always reproduces the same batch (safe for retries/backfills)
    while different dates genuinely differ.

Task 2 → feature_engineering_task:
    Replicates the exact feature engineering from Milestone 1:
    - Encodes categorical fields (Contract, InternetService, PaymentMethod)
    - Calculates total_services the SAME way as Milestone 1 SQL Query 8
      (8 services: 6 add-ons + PhoneService + InternetService)
    - Applies MinMaxScaler (loaded from Milestone 1 artifact) to the 3
      continuous columns

Task 3 → batch_inference_task:
    Loads best_churn_model.pkl (winner from Milestone 2 evaluation),
    runs predictions on the engineered batch, saves results.
    FIX: previously overwrote a single fixed results file every run, so
    there was no history to plot drift against. Now keeps a dated snapshot
    per run AND a rolling history file. The history file is an idempotent
    upsert by run_date: re-running the same date replaces that date's rows
    only, so retries/backfills never create duplicates, and every other
    date's history is left untouched.

Flow: mock_data >> feature_engineering >> batch_predictions
"""

import os
import json
import hashlib
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
RESULTS_DIR          = "/app/data/predictions"
HISTORY_PATH         = "/app/data/predictions/prediction_history.csv"
_LOCAL_RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "predictions")
_LOCAL_HISTORY_PATH  = os.path.join(_LOCAL_RESULTS_DIR, "prediction_history.csv")

# Milestone 1 artifacts (mounted into the container)
SCALER_PATH          = "/app/app/models/minmax_scaler.pkl"
FEATURE_COLS_PATH    = "/app/app/models/feature_columns.json"
X_TEST_PATH          = "/app/app/models/X_test.csv"

# Milestone 2 best model
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

def get_x_test_path():
    return _resolve(
        X_TEST_PATH,
        os.path.join(os.path.dirname(__file__), "..", "app", "models", "X_test.csv")
    )


# ── TASK 1: Raw Data Extraction ───────────────────────────────────────────────

def mock_raw_data_task(**context):
    """
    Simulates pulling fresh daily customer records from a source database.

    FIX (gap 1): instead of 3 hardcoded rows repeated forever, this samples
    real customers from Milestone 1's held-out X_test.csv (never seen during
    training — the closest thing this project has to "new" customers).

    X_test.csv is already scaled and one-hot encoded, so we reverse both
    steps to hand Task 2 the same kind of raw, human-readable row it would
    get from a real source system (e.g. Contract="One year" instead of
    Contract_One year=1, tenure=12 instead of tenure=0.16).

    Sampling is seeded by the DAG's logical date (`ds`): the same date
    always samples the same rows (safe to retry/backfill), while different
    dates sample different customers.

    Note: X_test.csv doesn't retain the original PhoneService column (it
    was dropped upstream in Milestone 1), so it's approximated here as
    "Yes" for every row — true for ~90% of the source dataset. This can
    shift total_services by 1 for the small minority of customers who
    didn't have phone service; everything else is the real customer record.
    """
    logger.info("Task 1: Sampling real held-out customers from Milestone 1...")
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    ds = context.get("ds") or datetime.now().strftime("%Y-%m-%d")
    seed = int(hashlib.md5(ds.encode()).hexdigest(), 16) % (2**32)

    x_test = pd.read_csv(get_x_test_path())
    scaler = joblib.load(get_scaler_path())

    n_records = np.random.RandomState(seed).randint(4, 8)
    sample = x_test.sample(n=n_records, random_state=seed).reset_index(drop=True)

    # Reverse the scaling Milestone 1 applied, back to real-world values
    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    sample[continuous_cols] = scaler.inverse_transform(sample[continuous_cols]).round(2)
    sample["tenure"] = sample["tenure"].round().astype(int)

    # Reverse the one-hot encoding back to single categorical columns
    def decode_onehot(df, prefix, options):
        cols = [f"{prefix}_{opt}" for opt in options]
        return df[cols].idxmax(axis=1).str.replace(f"{prefix}_", "", regex=False)

    sample["InternetService"] = decode_onehot(sample, "InternetService", ["DSL", "Fiber optic", "No"])
    sample["Contract"] = decode_onehot(sample, "Contract", ["Month-to-month", "One year", "Two year"])
    sample["PaymentMethod"] = decode_onehot(
        sample, "PaymentMethod",
        ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
    )

    # Not preserved in X_test.csv — see docstring note above
    sample["PhoneService"] = "Yes"
    sample["CustomerID"] = [f"{ds}-{i + 1:02d}" for i in range(len(sample))]

    raw_cols = [
        "CustomerID", "tenure", "MonthlyCharges", "TotalCharges",
        "Contract", "InternetService", "PaymentMethod", "PhoneService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
        "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling",
    ]
    raw_df = sample[raw_cols]
    raw_df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(
        f"Task 1 complete: {len(raw_df)} real held-out customers sampled "
        f"to {RAW_DATA_PATH} (date={ds}, seed={seed})"
    )


# ── TASK 2: Feature Engineering ───────────────────────────────────────────────

def feature_engineering_task():
    """
    Replicates the EXACT feature engineering pipeline from Milestone 1.

    Counts total_services the same way as Milestone 1 SQL Query 8
    (6 add-ons + PhoneService + has-any-InternetService), one-hot encodes
    the categorical fields, aligns to the exact 24-column order Milestone 1
    exported, then applies the same MinMaxScaler to the 3 continuous columns.
    """
    logger.info("Task 2: Running feature engineering pipeline...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data not found at: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    # ── 2a. total_services (matches Milestone 1 SQL Query 8 exactly) ─────────
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

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[expected_cols]

    # ── 2d. Apply MinMaxScaler from Milestone 1 ───────────────────────────────
    scaler_path = get_scaler_path()
    scaler = joblib.load(scaler_path)
    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[continuous_cols] = scaler.transform(df[continuous_cols])

    os.makedirs(os.path.dirname(ENGINEERED_DATA_PATH), exist_ok=True)
    df.to_csv(ENGINEERED_DATA_PATH, index=False)
    logger.info(f"Task 2 complete: engineered data saved to {ENGINEERED_DATA_PATH}")


# ── TASK 3: Batch Inference ───────────────────────────────────────────────────

def batch_inference_task(**context):
    """
    Loads best_churn_model.pkl, runs it on the engineered daily batch,
    and saves results.

    FIX (gap 2): writes two things instead of one fixed, overwritten file:
      1. A dated snapshot (daily_results_<ds>.csv) — one immutable file
         per run, safe to overwrite on retry since it only ever describes
         that one date.
      2. A rolling history file (prediction_history.csv) that a dashboard
         can watch grow over time — but as an IDEMPOTENT UPSERT by
         run_date: if this date's rows already exist in history (e.g. the
         task is retried, or someone re-triggers the same date), they are
         replaced, not duplicated. Every other date's rows are untouched.
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

    raw_df = pd.read_csv(RAW_DATA_PATH)
    ds = context.get("ds") or datetime.now().strftime("%Y-%m-%d")

    results_df = pd.DataFrame({
        "run_date":           ds,
        "CustomerID":         raw_df["CustomerID"],
        "churn_prediction":   preds,
        "churn_probability":  [round(float(p), 4) for p in probs],
        "risk_level":         ["HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.4 else "LOW"
                               for p in probs],
        "model_used":         os.path.basename(model_path),
        "prediction_timestamp": datetime.now().isoformat()
    })

    in_container = os.path.exists('/app')
    results_dir = RESULTS_DIR if in_container else os.path.abspath(_LOCAL_RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Dated snapshot — always safe to overwrite, describes only this date
    snapshot_path = os.path.join(results_dir, f"daily_results_{ds}.csv")
    results_df.to_csv(snapshot_path, index=False)

    # 2. Rolling history — idempotent upsert by run_date
    history_path = HISTORY_PATH if in_container else os.path.abspath(_LOCAL_HISTORY_PATH)
    if os.path.isfile(history_path):
        history_df = pd.read_csv(history_path)
        history_df = history_df[history_df["run_date"].astype(str) != str(ds)]
        combined = pd.concat([history_df, results_df], ignore_index=True)
    else:
        combined = results_df
    combined.to_csv(history_path, index=False)

    high_risk = (results_df["risk_level"] == "HIGH").sum()
    logger.info(
        f"Task 3 complete: {len(results_df)} predictions saved. "
        f"High risk: {high_risk}/{len(results_df)}"
    )
    logger.info(f"Snapshot saved to: {snapshot_path}")
    logger.info(f"History upserted at: {history_path} ({len(combined)} total rows)")


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

    mock_data >> feature_engineering >> batch_predictions