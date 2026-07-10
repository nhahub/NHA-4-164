import os
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import (
    ModelLoadError,
    ModelInferenceError,
    model_load_exception_handler,
    model_inference_exception_handler,
)
from app.api.router import api_router
from app.services.inference import inference_service

# Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production MLOps pipeline for Customer Churn classification.",
    version="1.0.0",
    docs_url=None,  # Disable default OpenAPI Swagger docs (custom themed one below)
)

# CORS Policy configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom exception handlers (previously unused/dead code)
app.add_exception_handler(ModelLoadError, model_load_exception_handler)
app.add_exception_handler(ModelInferenceError, model_inference_exception_handler)

# Mount the modular API router (this was missing before -> root cause of the 5-field bug)
app.include_router(api_router, prefix=settings.API_V1_STR)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Serves custom themed Swagger UI docs."""
    response = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )
    html_content = response.body.decode("utf-8")
    themed_html = html_content.replace(
        "</head>",
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-themes@3.0.1/themes/3.x/theme-material.css"></head>',
    )
    return HTMLResponse(content=themed_html, status_code=response.status_code)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_gui():
    """Serves the custom single-page dashboard & predictor GUI."""
    gui_path = os.path.join(CURRENT_DIR, "core", "gui.html")
    if not os.path.exists(gui_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GUI HTML asset not found locally.",
        )
    with open(gui_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


@app.get("/health")
async def check_health():
    """Top-level health check (also available at /api/v1/churn/health)."""
    try:
        inference_service._load_model()
        model_file = os.path.basename(inference_service.model_path)
        model_type = type(inference_service._model).__name__   # e.g. "RandomForestClassifier"
        return {
            "status": "healthy",
            "message": f"Churn model ({model_type}, file: {model_file}) is fully loaded and operational.",
            "model_path": inference_service.model_path,
        }
    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        return {"status": "degraded", "error": f"Model load failure: {str(e)}"}
