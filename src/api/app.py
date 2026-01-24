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
Fast API APP
----------------------------------------------------------------------------
FILE VERSION: v5.0-1-1.0-1
LAST MODIFIED: 2026-01-24
PHASE: Phase 1 - {Phase Description}
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.managers.config_manager import ConfigManager
from src.managers.logging_config_manager import LoggingConfigManager
from src.managers.model_manager import ModelManager


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
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    inference_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str
    model_loaded: bool
    model_name: str
    gpu_available: bool
    gpu_memory_used_mb: Optional[float]
    uptime_seconds: float


# =============================================================================
# Application Setup
# =============================================================================

# Initialize managers
config_manager = ConfigManager()
config = config_manager.config

logging_manager = LoggingConfigManager(config)
logger = logging_manager.get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Ash-Vigil",
    description="Mental Health Risk Detection Service for the Ash Ecosystem",
    version="5.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global state
model_manager: Optional[ModelManager] = None
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
        model_manager = ModelManager(config)
        await model_manager.load_model()
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
    """
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = request.request_id or str(uuid.uuid4())

    logger.debug(f"[{request_id}] Analyzing text ({len(request.text)} chars)")

    start = time.perf_counter()
    result = await model_manager.predict(request.text)
    inference_time = (time.perf_counter() - start) * 1000

    logger.info(
        f"[{request_id}] Risk: {result['label']} ({result['score']:.3f}) in {inference_time:.1f}ms"
    )

    return AnalyzeResponse(
        request_id=request_id,
        risk_score=result["score"],
        risk_label=result["label"],
        confidence=result["confidence"],
        model_version=model_manager.model_name,
        inference_time_ms=round(inference_time, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns model status, GPU availability, and uptime.
    """
    gpu_memory = None
    gpu_available = False

    if model_manager:
        gpu_available = model_manager.gpu_available
        gpu_memory = model_manager.get_gpu_memory_used()

    return HealthResponse(
        status="healthy" if model_manager and model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded if model_manager else False,
        model_name=config.get("model", {}).get("name", "unknown"),
        gpu_available=gpu_available,
        gpu_memory_used_mb=gpu_memory,
        uptime_seconds=round(time.time() - start_time, 2),
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint.
    """
    # Basic metrics for now - can be expanded with prometheus_client
    uptime = time.time() - start_time
    model_loaded = 1 if (model_manager and model_manager.is_loaded) else 0
    gpu_available = 1 if (model_manager and model_manager.gpu_available) else 0

    metrics_text = f"""# HELP ash_vigil_uptime_seconds Uptime in seconds
# TYPE ash_vigil_uptime_seconds gauge
ash_vigil_uptime_seconds {uptime}

# HELP ash_vigil_model_loaded Whether the model is loaded
# TYPE ash_vigil_model_loaded gauge
ash_vigil_model_loaded {model_loaded}

# HELP ash_vigil_gpu_available Whether GPU is available
# TYPE ash_vigil_gpu_available gauge
ash_vigil_gpu_available {gpu_available}
"""

    return metrics_text


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Ash-Vigil",
        "description": "Mental Health Risk Detection Service",
        "version": "5.0.1",
        "ecosystem": "Ash Crisis Detection",
        "community": "The Alphabet Cartel",
        "endpoints": {
            "analyze": "POST /analyze",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "docs": "GET /docs",
        },
    }
