import os
from fastapi import APIRouter, status
from app.schemas.churn import ChurnInput, ChurnOutput
from app.services.inference import inference_service
from app.core.logging import logger

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict:
    """
    Health check endpoint.
    Verifies that all three artifacts (model, scaler, feature columns)
    load correctly. If any one fails, returns 'degraded' status so
    Docker / Airflow health probes can detect the issue.
    """
    try:
        inference_service._ensure_all_loaded()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_file": os.path.basename(inference_service.model_path),
            "scaler_loaded": True,
            "feature_columns": len(inference_service._feature_columns),
            "message": "All artifacts loaded. API is ready for predictions."
        }
    except Exception as e:
        logger.exception(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": str(e)
        }


@router.post("/predict", response_model=ChurnOutput, status_code=status.HTTP_200_OK)
def predict_churn(payload: ChurnInput) -> ChurnOutput:
    """
    Churn prediction endpoint.
    Accepts all 17 customer feature fields, applies MinMaxScaler from
    Milestone 1, runs the best model from Milestone 2, and returns the
    churn prediction + probability.

    The model_version field in the response tells you exactly which model
    file was used — so if best_churn_model.pkl switches from XGBoost to
    Random Forest after retraining, the response reflects that automatically.
    """
    logger.info("Prediction request received.")
    churn_pred, churn_prob = inference_service.predict(payload)

    # Report which model file actually ran the prediction
    model_version = os.path.basename(inference_service.model_path)

    return ChurnOutput(
        success=True,
        churn_prediction=churn_pred,
        churn_probability=round(churn_prob, 4),
        model_version=model_version
    )
