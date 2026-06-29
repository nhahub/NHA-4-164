# pyrefly: ignore [missing-import]
from fastapi import APIRouter
from app.api.v1.endpoints import churn

api_router = APIRouter()

# Register churn prediction endpoint router
api_router.include_router(churn.router, prefix="/churn", tags=["Churn Prediction Pipeline"])
