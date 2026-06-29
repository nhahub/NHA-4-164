import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the Churn Prediction API.
    All paths are overridable via environment variables or a .env file,
    which means Docker / Airflow can mount different paths without
    changing any source code.
    """
    PROJECT_NAME: str = "Telco Customer Churn Prediction API"
    API_V1_STR: str = "/api/v1"

    # ── Model (from Milestone 2) ──────────────────────────────────────────────
    # Points to best_churn_model.pkl — the winner selected automatically
    # by Milestone 2's ROC-AUC comparison (Random Forest in our case).
    # Previously this was hardcoded to xgb_churn_model.pkl which ignored
    # the whole evaluation step we did in M2.
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        "app/models/best_churn_model.pkl"
    )

    # ── Scaler (from Milestone 1) ─────────────────────────────────────────────
    # The MinMaxScaler fitted ONLY on the training set during Milestone 1.
    # Must be applied to tenure, MonthlyCharges, TotalCharges before every
    # prediction — the model was trained on 0-1 scaled values, so sending
    # raw values (e.g. tenure=12, MonthlyCharges=70) gives completely wrong
    # predictions. This was the biggest bug in the original API.
    SCALER_PATH: str = os.getenv(
        "SCALER_PATH",
        "app/models/minmax_scaler.pkl"
    )

    # ── Feature Columns (from Milestone 1) ───────────────────────────────────
    # The exact 24-column list and order exported by Milestone 1.
    # Instead of hardcoding the column list inside inference.py (which would
    # silently break if M1 ever changes), we load it from this file so M1
    # is always the single source of truth for feature structure.
    FEATURE_COLUMNS_PATH: str = os.getenv(
        "FEATURE_COLUMNS_PATH",
        "app/models/feature_columns.json"
    )

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
