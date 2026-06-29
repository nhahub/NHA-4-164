from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.logging import logger

class ModelLoadError(Exception):
    """
    Raised when the pre-trained XGBoost model fails to load correctly from the system.
    """
    def __init__(self, message: str = "ML Model binary failed to load."):
        self.message = message
        super().__init__(self.message)


class ModelInferenceError(Exception):
    """
    Raised when a feature preprocessing or model.predict execution fails.
    """
    def __init__(self, message: str = "Error executing model prediction."):
        self.message = message
        super().__init__(self.message)


async def model_load_exception_handler(request: Request, exc: ModelLoadError) -> JSONResponse:
    """
    JSON response handler for model loading failures (500 Internal Server Error).
    """
    logger.critical(f"ModelLoadError caught: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "ModelLoadError",
            "message": exc.message,
            "detail": "The predictive model binary is either missing, corrupted, or incompatible with the environment."
        }
    )


async def model_inference_exception_handler(request: Request, exc: ModelInferenceError) -> JSONResponse:
    """
    JSON response handler for model execution failures (500 Internal Server Error).
    """
    logger.error(f"ModelInferenceError caught: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "ModelInferenceError",
            "message": exc.message,
            "detail": "Failed to run model prediction on the provided inputs. Please verify features and types."
        }
    )
