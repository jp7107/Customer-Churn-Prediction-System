"""FastAPI application for Telco Customer Churn Prediction.

Cognirex Solutions AI/ML Technical Assessment
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.churn_prediction.config import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH
from src.churn_prediction.inference import ChurnPredictor
from src.churn_prediction.logging_config import setup_logger

logger = setup_logger("churn_api")

# Global predictor instance
predictor: ChurnPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load ML model on startup."""
    global predictor
    logger.info("Initializing API and loading model artifacts...")
    try:
        predictor = ChurnPredictor.get_instance(
            model_path=DEFAULT_MODEL_PATH,
            metadata_path=DEFAULT_METADATA_PATH,
        )
        logger.info("Model pipeline loaded successfully during startup.")
    except Exception as exc:
        logger.error("Failed to load model pipeline at startup: %s", exc)
        predictor = None
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Cognirex Solutions AI/ML Technical Assessment — Telco Customer Churn Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return clean 422 JSON response for client validation errors."""
    errors = exc.errors()
    clean_errors = [{"loc": err["loc"], "msg": err["msg"], "type": err["type"]} for err in errors]
    logger.warning("Validation error on path %s: %s", request.url.path, clean_errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Input validation error", "errors": clean_errors},
    )


@app.get("/health", response_model=HealthResponse, summary="Check API health and model status")
def health_check() -> HealthResponse:
    """Return API health status and whether model is loaded."""
    is_loaded = predictor is not None and predictor.pipeline is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=predictor.model_name if is_loaded and predictor else None,
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, summary="Retrieve model metadata")
def get_model_info() -> ModelInfoResponse:
    """Return model metadata, threshold, and evaluation summary."""
    if predictor is None or predictor.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train the model pipeline first.",
        )
    meta = predictor.metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", "Trained Churn Model"),
        algorithm=meta.get("algorithm", "ClassifierPipeline"),
        selected_threshold=predictor.decision_threshold,
        default_threshold=meta.get("default_threshold", 0.50),
        target_definition="1 = Churn (Customer Cancels Service), 0 = Stay (Customer Retained)",
        evaluation_metrics=meta.get("test_metrics_optimal_threshold", {}),
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict churn for a single customer")
def predict_churn(customer: CustomerFeatures) -> PredictionResponse:
    """Accept single customer JSON payload and return churn prediction and probability."""
    if predictor is None or predictor.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable. Train model before calling inference.",
        )

    try:
        payload = customer.model_dump()
        customer_id = payload.get("customerID")
        result = predictor.predict_single(payload)
        
        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            churn_probability=result["churn_probability"],
            decision_threshold=result["decision_threshold"],
            is_churn=result["is_churn"],
            customerID=customer_id,
        )
    except Exception as exc:
        logger.error("Inference failure on single prediction: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during model prediction.",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Predict churn for multiple customers")
def predict_churn_batch(batch: BatchPredictionRequest) -> BatchPredictionResponse:
    """Accept multiple customer records and return batch predictions."""
    if predictor is None or predictor.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable. Train model before calling inference.",
        )

    try:
        payloads = [c.model_dump() for c in batch.customers]
        results = predictor.predict_records(payloads)
        
        responses: List[PredictionResponse] = []
        for p, r in zip(payloads, results):
            responses.append(
                PredictionResponse(
                    prediction=r["prediction"],
                    probability=r["probability"],
                    churn_probability=r["churn_probability"],
                    decision_threshold=r["decision_threshold"],
                    is_churn=r["is_churn"],
                    customerID=p.get("customerID"),
                )
            )

        return BatchPredictionResponse(
            total_records=len(responses),
            predictions=responses,
        )
    except Exception as exc:
        logger.error("Inference failure on batch prediction: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during batch model prediction.",
        )
