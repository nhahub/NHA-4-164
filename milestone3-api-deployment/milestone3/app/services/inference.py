import os
import csv
import json
import joblib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import ModelLoadError, ModelInferenceError
from app.schemas.churn import ChurnInput


class ChurnInferenceService:
    """
    ML Inference Service responsible for:
    1. Loading the best model from Milestone 2
    2. Loading the scaler from Milestone 1
    3. Loading the feature column list from Milestone 1
    4. Preprocessing incoming requests correctly (with scaling)
    5. Running predictions and logging every result
    """

    def __init__(self):
        self.model_path = settings.MODEL_PATH
        self.scaler_path = settings.SCALER_PATH
        self.feature_columns_path = settings.FEATURE_COLUMNS_PATH

        # These are loaded lazily (on first request, not at import time)
        # so the app starts up even if the files are being mounted
        self._model = None
        self._scaler = None
        self._feature_columns = None

        # The three continuous columns that MUST be scaled before inference.
        # These are the exact columns MinMaxScaler was fit on in Milestone 1.
        # Everything else (binary 0/1 and one-hot columns) is already in 0-1
        # range naturally and does NOT need scaling.
        self.continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    # ── Lazy Loaders ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Loads best_churn_model.pkl from Milestone 2 into memory.
        Uses pickle (same format joblib uses internally).
        Only loads once — subsequent calls are instant no-ops.
        """
        if self._model is not None:
            return
        if not os.path.exists(self.model_path):
            raise ModelLoadError(f"Model not found at: {self.model_path}")
        try:
            logger.info(f"Loading model from: {self.model_path}")
            self._model = joblib.load(self.model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}")

    def _load_scaler(self) -> None:
        """
        Loads minmax_scaler.pkl from Milestone 1.
        This scaler was fitted ONLY on X_train to avoid data leakage.
        We must use this exact same scaler at inference time — never refit it.
        """
        if self._scaler is not None:
            return
        if not os.path.exists(self.scaler_path):
            raise ModelLoadError(f"Scaler not found at: {self.scaler_path}")
        try:
            logger.info(f"Loading scaler from: {self.scaler_path}")
            self._scaler = joblib.load(self.scaler_path)
            logger.info("Scaler loaded successfully.")
        except Exception as e:
            raise ModelLoadError(f"Failed to load scaler: {str(e)}")

    def _load_feature_columns(self) -> None:
        """
        Loads feature_columns.json from Milestone 1.
        This is the authoritative list of 24 column names in the exact order
        the model expects. Previously this was hardcoded in inference.py —
        now M1 is the single source of truth. If M1 changes its features,
        the API automatically stays in sync.
        """
        if self._feature_columns is not None:
            return
        if not os.path.exists(self.feature_columns_path):
            raise ModelLoadError(
                f"Feature columns file not found at: {self.feature_columns_path}"
            )
        try:
            with open(self.feature_columns_path, "r") as f:
                self._feature_columns = json.load(f)
            logger.info(
                f"Feature columns loaded: {len(self._feature_columns)} columns"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load feature columns: {str(e)}")

    def _ensure_all_loaded(self) -> None:
        """Convenience method to load all three artifacts at once."""
        self._load_model()
        self._load_scaler()
        self._load_feature_columns()

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess_features(self, data: ChurnInput) -> pd.DataFrame:
        """
        Converts a ChurnInput request into the exact 24-column DataFrame
        the model expects, with scaling applied to continuous features.

        Steps:
        1. Start with a zero-vector of all 24 features
        2. Fill in binary/integer features directly (already 0-1 range)
        3. One-hot encode the 3 categorical fields (Contract,
           InternetService, PaymentMethod)
        4. Apply MinMaxScaler to tenure, MonthlyCharges, TotalCharges
           — this is the step that was MISSING before and causing wrong predictions
        5. Enforce exact column order from feature_columns.json
        """
        try:
            # Step 1: zero-vector for all 24 columns
            row = {col: 0.0 for col in self._feature_columns}

            # Step 2: fill binary / integer features directly
            # (these are already in 0-1 range — no scaling needed)
            row["SeniorCitizen"]    = float(data.SeniorCitizen)
            row["Partner"]          = float(data.Partner)
            row["Dependents"]       = float(data.Dependents)
            row["OnlineSecurity"]   = float(data.OnlineSecurity)
            row["OnlineBackup"]     = float(data.OnlineBackup)
            row["DeviceProtection"] = float(data.DeviceProtection)
            row["TechSupport"]      = float(data.TechSupport)
            row["StreamingTV"]      = float(data.StreamingTV)
            row["StreamingMovies"]  = float(data.StreamingMovies)
            row["PaperlessBilling"] = float(data.PaperlessBilling)
            row["total_services"]   = float(data.total_services)

            # Step 3: one-hot encode the 3 categorical fields
            # e.g. Contract="One year" → Contract_One year = 1.0, others = 0.0
            contract_key = f"Contract_{data.Contract.value}"
            if contract_key in row:
                row[contract_key] = 1.0

            internet_key = f"InternetService_{data.InternetService.value}"
            if internet_key in row:
                row[internet_key] = 1.0

            payment_key = f"PaymentMethod_{data.PaymentMethod.value}"
            if payment_key in row:
                row[payment_key] = 1.0

            # Step 4: put continuous features in BEFORE scaling
            # (raw values from the user: e.g. tenure=12, MonthlyCharges=70.05)
            row["tenure"]          = float(data.tenure)
            row["MonthlyCharges"]  = float(data.MonthlyCharges)
            row["TotalCharges"]    = float(data.TotalCharges)

            # Step 5: build DataFrame and enforce exact column order
            df = pd.DataFrame([row])[self._feature_columns]

            # ── THE CRITICAL FIX ─────────────────────────────────────────────
            # Apply the MinMaxScaler (loaded from Milestone 1) to the 3
            # continuous columns. The model was trained on scaled values
            # (e.g. tenure 0–72 → 0.0–1.0), so we must transform the raw
            # input values the same way before passing to the model.
            # WITHOUT this step, the model receives e.g. tenure=12 instead
            # of ~0.16, MonthlyCharges=70 instead of ~0.57 — completely
            # different numbers than anything it saw during training.
            df[self.continuous_cols] = self._scaler.transform(
                df[self.continuous_cols]
            )

            return df

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise ModelInferenceError(f"Failed to preprocess features: {str(e)}")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, data: ChurnInput) -> Tuple[int, float]:
        """
        Main inference entry point.
        1. Ensures all artifacts are loaded
        2. Preprocesses and scales input
        3. Runs model prediction
        4. Logs result to CSV for Milestone 4 monitoring
        Returns: (churn_prediction: 0 or 1, churn_probability: 0.0–1.0)
        """
        self._ensure_all_loaded()

        features_df = self.preprocess_features(data)

        try:
            proba = self._model.predict_proba(features_df)
            churn_prob = float(proba[0][1])
            churn_pred = int(self._model.predict(features_df)[0])

            logger.info(
                f"Prediction: churn={churn_pred}, probability={churn_prob:.4f}"
            )

            self._log_prediction(data, churn_pred, churn_prob)
            return churn_pred, churn_prob

        except Exception as e:
            logger.exception(f"Inference failed: {str(e)}")
            raise ModelInferenceError(f"Model prediction failed: {str(e)}")

    # ── Logging (Milestone 4 prep) ────────────────────────────────────────────

    def _log_prediction(
        self, data: ChurnInput, churn_pred: int, churn_prob: float
    ) -> None:
        """
        Appends every prediction to a CSV log file.
        This is the foundation for Milestone 4 monitoring:
        - Track prediction distribution over time (drift detection)
        - See which customer profiles trigger high churn risk
        - Feed into a dashboard (Power BI / Grafana)
        Logs all 17 input fields (not just 5 like before) so the log
        is actually useful for analysis.
        Logging failure never breaks a prediction response.
        """
        try:
            log_dir = os.path.join(
                os.path.dirname(self.model_path), "..", "data", "predictions"
            )
            log_dir = os.path.abspath(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "prediction_logs.csv")
            file_exists = os.path.isfile(log_file)

            with open(log_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    # Write header with all 17 input fields + prediction output
                    writer.writerow([
                        "timestamp",
                        "tenure", "MonthlyCharges", "TotalCharges",
                        "Contract", "InternetService", "PaymentMethod",
                        "total_services", "SeniorCitizen", "Partner",
                        "Dependents", "PaperlessBilling", "OnlineSecurity",
                        "OnlineBackup", "DeviceProtection", "TechSupport",
                        "StreamingTV", "StreamingMovies",
                        "churn_prediction", "churn_probability"
                    ])
                writer.writerow([
                    datetime.now(),
                    data.tenure, data.MonthlyCharges, data.TotalCharges,
                    data.Contract.value, data.InternetService.value,
                    data.PaymentMethod.value, data.total_services,
                    data.SeniorCitizen, data.Partner, data.Dependents,
                    data.PaperlessBilling, data.OnlineSecurity,
                    data.OnlineBackup, data.DeviceProtection,
                    data.TechSupport, data.StreamingTV, data.StreamingMovies,
                    churn_pred, churn_prob
                ])
        except Exception as e:
            logger.error(f"Prediction logging failed (non-fatal): {str(e)}")


# Singleton — one instance shared across all API requests
inference_service = ChurnInferenceService()
