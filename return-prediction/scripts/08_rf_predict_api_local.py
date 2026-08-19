# Local FastAPI service for ecommerce return-risk prediction
#
# Loads model artifacts created by 06_rf_train_sklearn_local.py:
#   /home/arvind/ecommerce_rf/models/rf_transaction_return_sklearn.pkl
#   /home/arvind/ecommerce_rf/models/rf_transaction_return_sklearn_columns.pkl
#
# Starts a local HTTP API with:
#   GET  /health
#   GET  /model-info
#   POST /predict
#   POST /predict-batch
#
# Install dependencies once inside spark_env:
#   python -m pip install fastapi "uvicorn[standard]"
#
# Run:
#   source ~/spark_env/bin/activate
#   uvicorn 08_rf_predict_api_local:app --app-dir ~/ecommerce_rf/scripts \
#       --host 0.0.0.0 --port 8000
#
# Open interactive API documentation in Windows browser:
#   http://localhost:8000/docs
#
# Note:
#   This API uses the scikit-learn balanced Random Forest selected for
#   high-recall return detection. It is a local demonstration service,
#   not a production deployment.

import os
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# -----------------------------------------------------------------------
# 1. Project paths and model settings
# -----------------------------------------------------------------------
PROJECT_DIR = "/home/arvind/ecommerce_rf"
MODELS_DIR = f"{PROJECT_DIR}/models"
MODEL_PKL_PATH = f"{MODELS_DIR}/rf_transaction_return_sklearn.pkl"
FEATURE_COLUMNS_PKL_PATH = (
    f"{MODELS_DIR}/rf_transaction_return_sklearn_columns.pkl"
)

MODEL_NAME = "scikit-learn Random Forest (balanced)"
MODEL_THRESHOLD = 0.50

# These are metadata columns added only while creating the pandas feature
# matrix in stage 06. They must never be supplied as prediction inputs.
INTERNAL_COLUMNS = {"__split__"}

# -----------------------------------------------------------------------
# 2. Input schemas
# extra="allow" permits a full raw transaction record. The API selects only
# fields that appear in the saved feature schema after one-hot encoding.
# -----------------------------------------------------------------------
class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    quarter: float | None = Field(default=None, examples=[3])
    customer_age: float | None = Field(default=None, examples=[31])
    unit_price: float | None = Field(default=None, examples=[499.0])
    quantity: float | None = Field(default=None, examples=[2])
    discount_percent: float | None = Field(default=None, examples=[10.0])
    discount_amount: float | None = Field(default=None, examples=[99.8])
    shipping_cost: float | None = Field(default=None, examples=[50.0])
    tax_amount: float | None = Field(default=None, examples=[85.3])
    order_amount: float | None = Field(default=None, examples=[933.5])
    delivery_days: float | None = Field(default=None, examples=[4])
    review_rating: float | None = Field(default=None, examples=[4.0])
    customer_lifetime_value: float | None = Field(default=None, examples=[1500.0])
    profit_margin_percent: float | None = Field(default=None, examples=[22.0])
    profit_amount: float | None = Field(default=None, examples=[205.0])
    high_value_order: float | None = Field(default=None, examples=[0])
    order_year: float | None = Field(default=None, examples=[2025])
    order_month: float | None = Field(default=None, examples=[8])
    order_day: float | None = Field(default=None, examples=[18])
    order_dow: float | None = Field(default=None, examples=[2])

    customer_gender: str | None = Field(default=None, examples=["Male"])
    country: str | None = Field(default=None, examples=["India"])
    customer_segment: str | None = Field(default=None, examples=["Regular"])
    product_category: str | None = Field(default=None, examples=["Electronics"])
    product_subcategory: str | None = Field(default=None, examples=["Mobile Phones"])
    brand: str | None = Field(default=None, examples=["Samsung"])
    payment_method: str | None = Field(default=None, examples=["Credit Card"])
    device_type: str | None = Field(default=None, examples=["Mobile"])
    traffic_source: str | None = Field(default=None, examples=["Organic Search"])
    membership_status: str | None = Field(default=None, examples=["Gold"])
    shipping_method: str | None = Field(default=None, examples=["Standard"])
    warehouse_region: str | None = Field(default=None, examples=["North"])
    season: str | None = Field(default=None, examples=["Summer"])
    holiday_season: str | None = Field(default=None, examples=["No"])
    day_of_week: str | None = Field(default=None, examples=["Monday"])


class BatchPredictionInput(BaseModel):
    transactions: list[TransactionInput]


class PredictionOutput(BaseModel):
    prediction: int
    return_probability: float
    threshold: float
    risk_label: str


class BatchPredictionOutput(BaseModel):
    model_name: str
    prediction_count: int
    predictions: list[PredictionOutput]


# -----------------------------------------------------------------------
# 3. Load persisted model artifacts once when the API process starts
# -----------------------------------------------------------------------
def load_artifacts() -> tuple[Any, list[str]]:
    if not os.path.isfile(MODEL_PKL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PKL_PATH}. "
            "Run 06_rf_train_sklearn_local.py first."
        )

    if not os.path.isfile(FEATURE_COLUMNS_PKL_PATH):
        raise FileNotFoundError(
            f"Feature-column file not found: {FEATURE_COLUMNS_PKL_PATH}. "
            "Run 06_rf_train_sklearn_local.py first."
        )

    loaded_model = joblib.load(MODEL_PKL_PATH)
    loaded_columns = joblib.load(FEATURE_COLUMNS_PKL_PATH)

    if not isinstance(loaded_columns, list) or not loaded_columns:
        raise ValueError("Saved model feature-column artifact is invalid or empty.")

    return loaded_model, loaded_columns


try:
    model, feature_columns = load_artifacts()
    startup_error = None
except Exception as error:
    model = None
    feature_columns = []
    startup_error = str(error)

# -----------------------------------------------------------------------
# 4. Create FastAPI application
# -----------------------------------------------------------------------
app = FastAPI(
    title="Ecommerce Return-Risk Prediction API",
    version="1.0.0",
    description=(
        "Local demonstration API that scores ecommerce orders with the "
        "class-balanced scikit-learn Random Forest model."
    ),
)


def require_model() -> None:
    if startup_error is not None or model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Prediction model is unavailable.",
                "error": startup_error,
            },
        )


def prepare_features(transactions: list[TransactionInput]) -> pd.DataFrame:
    """Encode raw transaction dictionaries and align them to training columns."""
    records = [item.model_dump(exclude_none=True) for item in transactions]

    if not records:
        raise HTTPException(status_code=422, detail="At least one transaction is required.")

    raw_df = pd.DataFrame(records)

    # Match stage 06: pandas one-hot encoding creates columns like
    # country_India, product_category_Electronics, etc.
    encoded_df = pd.get_dummies(raw_df, dummy_na=False)

    # Ensure all trained feature columns exist. Unknown categories produce
    # unseen dummy names, which are intentionally discarded below. Their
    # known-category indicator columns remain zero.
    aligned_df = encoded_df.reindex(columns=feature_columns, fill_value=0)

    # Numeric nulls must be replaced because scikit-learn RF does not accept
    # NaN values. Zero is a simple demonstration default; production systems
    # should use training-set imputation statistics saved with the model.
    aligned_df = aligned_df.fillna(0).astype(float)

    return aligned_df


def risk_label(probability: float) -> str:
    if probability >= 0.75:
        return "High return risk"
    if probability >= MODEL_THRESHOLD:
        return "Likely return"
    if probability >= 0.30:
        return "Moderate return risk"
    return "Low return risk"


def make_predictions(transactions: list[TransactionInput]) -> list[PredictionOutput]:
    require_model()

    feature_df = prepare_features(transactions)
    probabilities = model.predict_proba(feature_df)[:, 1]

    outputs = []
    for probability in probabilities:
        probability = float(probability)
        prediction = int(probability >= MODEL_THRESHOLD)
        outputs.append(
            PredictionOutput(
                prediction=prediction,
                return_probability=round(probability, 6),
                threshold=MODEL_THRESHOLD,
                risk_label=risk_label(probability),
            )
        )

    return outputs


# -----------------------------------------------------------------------
# 5. API endpoints
# -----------------------------------------------------------------------
@app.get("/", tags=["Service"])
def root() -> dict[str, str]:
    return {
        "service": "Ecommerce Return-Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Service"])
def health() -> dict[str, Any]:
    return {
        "status": "healthy" if startup_error is None else "unavailable",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "feature_count": len(feature_columns),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "error": startup_error,
    }


@app.get("/model-info", tags=["Service"])
def model_info() -> dict[str, Any]:
    require_model()
    return {
        "model_name": MODEL_NAME,
        "threshold": MODEL_THRESHOLD,
        "feature_count": len(feature_columns),
        "model_path": MODEL_PKL_PATH,
        "feature_columns_path": FEATURE_COLUMNS_PKL_PATH,
        "note": (
            "The model is class-balanced and optimized for high-recall "
            "return detection."
        ),
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(transaction: TransactionInput) -> PredictionOutput:
    return make_predictions([transaction])[0]


@app.post(
    "/predict-batch",
    response_model=BatchPredictionOutput,
    tags=["Prediction"],
)
def predict_batch(payload: BatchPredictionInput) -> BatchPredictionOutput:
    predictions = make_predictions(payload.transactions)
    return BatchPredictionOutput(
        model_name=MODEL_NAME,
        prediction_count=len(predictions),
        predictions=predictions,
    )
