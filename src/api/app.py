"""
============================================================================
Ash-Vigil: Mental Health Risk Detection Service
The Alphabet Cartel - https://discord.gg/alphabetcartel | alphabetcartel.org
============================================================================

MISSION - NEVER TO BE VIOLATED:
    Watch    → Scan messages for subtle crisis signals that generic models miss
    Amplify  → Boost ensemble confidence when specialized risk patterns emerge
    Catch    → Detect planning signals, passive ideation, and minority stress indicators
    Protect  → Safeguard our LGBTQIA+ community through vigilant pattern detection

============================================================================
FastAPI Application - REST API endpoints for risk detection
----------------------------------------------------------------------------
FILE VERSION: v5.0-2-1.3-1
LAST MODIFIED: 2026-01-26
PHASE: Phase 2 - Zero-Shot Classification Support
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
"""

import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from src.managers import (
    create_config_manager,
    create_logging_config_manager,
    create_model_manager,
)


# =============================================================================
# Request/Response Models
# =============================================================================


class AnalyzeRequest(BaseModel):
    """Request model for /analyze endpoint."""

    text: str = Field(
        ..., min_length=1, max_length=10000, description="Text to analyze"
    )
    request_id: Optional[str] = Field(
        None, description="Optional request ID for tracking"
    )


class AnalyzeResponse(BaseModel):
    """Response model for /analyze endpoint."""

    request_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_label: str
    matched_label: str = Field(..., description="The candidate label that best matched")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    inference_time_ms: float
    timestamp: str


class EvaluatePhrase(BaseModel):
    """Single phrase for bulk evaluation."""

    id: str = Field(..., description="Unique identifier for the phrase")
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")


class EvaluateRequest(BaseModel):
    """Request model for /evaluate endpoint (bulk testing)."""

    phrases: List[EvaluatePhrase] = Field(
        ..., min_items=1, max_items=1000, description="Phrases to evaluate"
    )
    include_timing: bool = Field(
        default=True, description="Include per-phrase timing data"
    )


class EvaluateResult(BaseModel):
    """Single result from bulk evaluation."""

    id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    matched_label: Optional[str] = Field(None, description="The candidate label that best matched")
    inference_time_ms: Optional[float] = None
    error: Optional[str] = None


class EvaluateResponse(BaseModel):
    """Response model for /evaluate endpoint."""

    model_name: str
    model_type: str
    results: List[EvaluateResult]
    total_phrases: int
    successful_phrases: int
    failed_phrases: int
    total_time_ms: float
    average_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str
    model_loaded: bool
    model_name: str
    model_type: str
    candidate_labels_count: Optional[int] = None
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[float] = None
    uptime_seconds: float
    version: str


# =============================================================================
# Application Setup
# =============================================================================

# Initialize managers using factory functions (Clean Architecture Rule #1)
config_manager = create_config_manager()
config = config_manager.config

logging_manager = create_logging_config_manager(config)
logger = logging_manager.get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Ash-Vigil",
    description="Mental Health Risk Detection Service for the Ash Ecosystem",
    version="5.0.2",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global state
model_manager = None
start_time: float = time.time()


# =============================================================================
# Lifecycle Events
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model_manager

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  Ash-Vigil Starting Up")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    try:
        # Create model manager using factory function
        model_manager = create_model_manager(config)
        await model_manager.load_model()

        # Perform warmup inference to eliminate first-run latency
        await model_manager.warmup()

        logger.success("✅ Ash-Vigil ready to serve requests")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Ash-Vigil shutting down...")
    if model_manager:
        await model_manager.cleanup()


# =============================================================================
# Endpoints
# =============================================================================


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze text for mental health risk indicators.

    Returns a risk score (0.0-1.0), risk label, and confidence level.

    This endpoint is used for real-time single-message analysis.
    """
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = request.request_id or str(uuid.uuid4())

    logger.debug(f"[{request_id}] Analyzing text ({len(request.text)} chars)")

    start = time.perf_counter()
    result = await model_manager.predict(request.text)
    inference_time = (time.perf_counter() - start) * 1000

    logger.info(
        f"[{request_id}] Risk: {result['risk_label']} ({result['risk_score']:.3f}) in {inference_time:.1f}ms"
    )

    return AnalyzeResponse(
        request_id=request_id,
        risk_score=result["risk_score"],
        risk_label=result["risk_label"],
        matched_label=result.get("matched_label", "unknown"),
        confidence=result["confidence"],
        model_version=model_manager.model_name,
        inference_time_ms=round(inference_time, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_phrases(request: EvaluateRequest) -> EvaluateResponse:
    """
    Bulk evaluate multiple phrases for testing and benchmarking.

    This endpoint is designed for Ash-Thrash to evaluate model performance
    across test phrase sets. Individual phrase failures don't fail the batch.

    Returns:
        Results for each phrase with optional timing data.
    """
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"📊 Evaluating {len(request.phrases)} phrases")

    results: List[EvaluateResult] = []
    total_start = time.perf_counter()
    successful = 0
    failed = 0

    for phrase in request.phrases:
        phrase_start = time.perf_counter()

        try:
            result = await model_manager.predict(phrase.text)
            phrase_time = (time.perf_counter() - phrase_start) * 1000

            results.append(
                EvaluateResult(
                    id=phrase.id,
                    risk_score=result["risk_score"],
                    risk_label=result["risk_label"],
                    confidence=result["confidence"],
                    matched_label=result.get("matched_label"),
                    inference_time_ms=round(phrase_time, 2) if request.include_timing else None,
                    error=None,
                )
            )
            successful += 1

        except Exception as e:
            phrase_time = (time.perf_counter() - phrase_start) * 1000
            logger.warning(f"⚠️ Evaluation failed for phrase {phrase.id}: {e}")

            results.append(
                EvaluateResult(
                    id=phrase.id,
                    risk_score=0.0,
                    risk_label="error",
                    confidence=0.0,
                    matched_label=None,
                    inference_time_ms=round(phrase_time, 2) if request.include_timing else None,
                    error=str(e),
                )
            )
            failed += 1

    total_time = (time.perf_counter() - total_start) * 1000
    avg_time = total_time / len(request.phrases) if request.phrases else 0

    logger.success(
        f"✅ Evaluation complete: {successful}/{len(request.phrases)} successful "
        f"in {total_time:.1f}ms (avg: {avg_time:.1f}ms)"
    )

    return EvaluateResponse(
        model_name=model_manager.model_name,
        model_type=model_manager.model_type,
        results=results,
        total_phrases=len(request.phrases),
        successful_phrases=successful,
        failed_phrases=failed,
        total_time_ms=round(total_time, 2),
        average_time_ms=round(avg_time, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns model status, GPU availability, and uptime.
    """
    import torch

    gpu_memory = None
    gpu_available = False
    gpu_name = None
    candidate_labels_count = None

    if model_manager:
        gpu_available = model_manager.gpu_available
        gpu_memory = model_manager.get_gpu_memory_used()

        if gpu_available and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

        if model_manager.model_type == "zero-shot":
            candidate_labels_count = len(model_manager.candidate_labels)

    model_config = config.get("model", {})

    return HealthResponse(
        status="healthy" if model_manager and model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded if model_manager else False,
        model_name=model_config.get("name", "unknown"),
        model_type=model_config.get("type", "unknown"),
        candidate_labels_count=candidate_labels_count,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_memory,
        uptime_seconds=round(time.time() - start_time, 2),
        version="5.0.2",
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """
    Prometheus-compatible metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    import torch

    uptime = time.time() - start_time
    model_loaded = 1 if (model_manager and model_manager.is_loaded) else 0
    gpu_available = 1 if (model_manager and model_manager.gpu_available) else 0

    gpu_memory_mb = 0
    if model_manager:
        mem = model_manager.get_gpu_memory_used()
        if mem is not None:
            gpu_memory_mb = mem

    metrics_text = f"""# HELP ash_vigil_uptime_seconds Uptime in seconds
# TYPE ash_vigil_uptime_seconds gauge
ash_vigil_uptime_seconds {uptime}

# HELP ash_vigil_model_loaded Whether the model is loaded (1=yes, 0=no)
# TYPE ash_vigil_model_loaded gauge
ash_vigil_model_loaded {model_loaded}

# HELP ash_vigil_gpu_available Whether GPU is available (1=yes, 0=no)
# TYPE ash_vigil_gpu_available gauge
ash_vigil_gpu_available {gpu_available}

# HELP ash_vigil_gpu_memory_mb GPU memory used in megabytes
# TYPE ash_vigil_gpu_memory_mb gauge
ash_vigil_gpu_memory_mb {gpu_memory_mb}
"""

    return metrics_text


@app.get("/labels")
async def get_labels():
    """
    Get the candidate labels used for zero-shot classification.

    Returns the configured labels grouped by risk category.
    """
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if model_manager.model_type != "zero-shot":
        return {
            "model_type": model_manager.model_type,
            "message": "This endpoint is only available for zero-shot models",
        }

    zero_shot_config = config.get("zero_shot", {})
    labels_config = zero_shot_config.get("candidate_labels", {})

    return {
        "model_type": "zero-shot",
        "model_name": model_manager.model_name,
        "hypothesis_template": zero_shot_config.get("hypothesis_template", "This text is from a {}"),
        "candidate_labels": labels_config,
        "total_labels": len(model_manager.candidate_labels),
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    model_info = {}
    if model_manager:
        model_info = model_manager.get_model_info()

    return {
        "service": "Ash-Vigil",
        "description": "Mental Health Risk Detection Service",
        "version": "5.0.2",
        "ecosystem": "Ash Crisis Detection",
        "community": "The Alphabet Cartel",
        "model": model_info,
        "links": {
            "discord": "https://discord.gg/alphabetcartel",
            "website": "https://alphabetcartel.org",
            "repository": "https://github.com/the-alphabet-cartel/ash-vigil",
        },
        "endpoints": {
            "analyze": "POST /analyze - Single text analysis",
            "evaluate": "POST /evaluate - Bulk phrase evaluation",
            "labels": "GET /labels - View zero-shot candidate labels",
            "health": "GET /health - Health check",
            "metrics": "GET /metrics - Prometheus metrics",
            "docs": "GET /docs - OpenAPI documentation",
        },
    }
