"""FastAPI application for Telco Customer Churn Prediction.

Cognirex Solutions AI/ML Technical Assessment
"""

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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

# Rate limiting: Track requests per IP
request_counts: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 100  # Max requests per window
RATE_LIMIT_WINDOW = 60  # Window in seconds

# Metrics tracking
api_metrics = {
    "total_requests": 0,
    "total_predictions": 0,
    "total_errors": 0,
    "avg_latency_ms": 0.0
}
latency_samples: List[float] = []

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_and_metrics_middleware(request: Request, call_next):
    """Rate limiting and latency tracking middleware."""
    start_time = time.time()
    
    # Rate limiting check
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    # Clean old requests outside the time window
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check rate limit
    if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds."
            }
        )
    
    # Record this request
    request_counts[client_ip].append(current_time)
    api_metrics["total_requests"] += 1
    
    # Process request
    try:
        response = await call_next(request)
    except Exception as exc:
        api_metrics["total_errors"] += 1
        logger.error(f"Request failed: {exc}", exc_info=True)
        raise
    
    # Track latency
    latency_ms = (time.time() - start_time) * 1000
    latency_samples.append(latency_ms)
    
    # Keep only last 1000 samples for average
    if len(latency_samples) > 1000:
        latency_samples.pop(0)
    
    api_metrics["avg_latency_ms"] = sum(latency_samples) / len(latency_samples)
    
    # Add headers
    response.headers["X-Response-Time"] = f"{latency_ms:.2f}ms"
    response.headers["X-RateLimit-Remaining"] = str(RATE_LIMIT_REQUESTS - len(request_counts[client_ip]))
    
    return response


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


@app.get("/metrics", summary="Get API performance metrics")
def get_metrics() -> Dict[str, Any]:
    """Return API usage and performance metrics."""
    return {
        "total_requests": api_metrics["total_requests"],
        "total_predictions": api_metrics["total_predictions"],
        "total_errors": api_metrics["total_errors"],
        "error_rate": api_metrics["total_errors"] / max(api_metrics["total_requests"], 1),
        "avg_latency_ms": round(api_metrics["avg_latency_ms"], 2),
        "active_rate_limited_ips": len(request_counts),
    }



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
        
        api_metrics["total_predictions"] += 1
        
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
        
        api_metrics["total_predictions"] += len(payloads)
        
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

