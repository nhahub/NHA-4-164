from enum import Enum
from pydantic import BaseModel, Field


class ContractEnum(str, Enum):
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR       = "One year"
    TWO_YEAR       = "Two year"


class InternetServiceEnum(str, Enum):
    DSL        = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO         = "No"


class PaymentMethodEnum(str, Enum):
    BANK_TRANSFER    = "Bank transfer (automatic)"
    CREDIT_CARD      = "Credit card (automatic)"
    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK     = "Mailed check"


class ChurnInput(BaseModel):
    """
    Request schema for the churn prediction endpoint.
    All 17 real customer fields — no hardcoded defaults that bias predictions.

    Required fields (5): the most predictive features per EDA in Milestone 1.
    Optional fields (12): default to 0/No but should always be provided for
    accurate predictions. The GUI collects all 17 automatically.
    """
    # ── Required fields ───────────────────────────────────────────────────────
    tenure:         int   = Field(..., ge=0,   description="Months with the company (0–72)")
    MonthlyCharges: float = Field(..., gt=0.0, description="Monthly bill amount ($)")
    TotalCharges:   float = Field(..., ge=0.0, description="Total billed to date ($)")
    Contract:       ContractEnum = Field(...,  description="Contract type")
    total_services: int   = Field(..., ge=0, le=8, description="Number of services subscribed (0–8)")

    # ── Optional fields (default 0 / No) ─────────────────────────────────────
    InternetService: InternetServiceEnum = Field(
        InternetServiceEnum.NO, description="Internet service type"
    )
    PaymentMethod: PaymentMethodEnum = Field(
        PaymentMethodEnum.MAILED_CHECK, description="Payment method"
    )
    SeniorCitizen:    int = Field(0, ge=0, le=1, description="1 = senior citizen")
    Partner:          int = Field(0, ge=0, le=1, description="1 = has partner")
    Dependents:       int = Field(0, ge=0, le=1, description="1 = has dependents")
    PaperlessBilling: int = Field(0, ge=0, le=1, description="1 = paperless billing")
    OnlineSecurity:   int = Field(0, ge=0, le=1, description="1 = subscribed")
    OnlineBackup:     int = Field(0, ge=0, le=1, description="1 = subscribed")
    DeviceProtection: int = Field(0, ge=0, le=1, description="1 = subscribed")
    TechSupport:      int = Field(0, ge=0, le=1, description="1 = subscribed")
    StreamingTV:      int = Field(0, ge=0, le=1, description="1 = subscribed")
    StreamingMovies:  int = Field(0, ge=0, le=1, description="1 = subscribed")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 70.05,
                "TotalCharges": 840.60,
                "Contract": "One year",
                "total_services": 4,
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
                "SeniorCitizen": 0,
                "Partner": 1,
                "Dependents": 0,
                "PaperlessBilling": 1,
                "OnlineSecurity": 1,
                "OnlineBackup": 0,
                "DeviceProtection": 1,
                "TechSupport": 1,
                "StreamingTV": 0,
                "StreamingMovies": 0
            }
        }
    }


class ChurnOutput(BaseModel):
    """
    Response schema for the churn prediction endpoint.
    model_version now reflects which model file is actually loaded,
    not a hardcoded string that would be wrong when RF replaces XGBoost.
    """
    success:           bool  = Field(True)
    churn_prediction:  int   = Field(..., description="0 = stays, 1 = churns")
    churn_probability: float = Field(..., description="Churn probability (0.0–1.0)")
    model_version:     str   = Field(..., description="Name of the model file used")
