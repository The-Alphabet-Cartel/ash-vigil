# =============================================================================
# Ash-Vigil Dockerfile
# Mental Health Risk Detection Service
# =============================================================================
# The Alphabet Cartel - https://discord.gg/alphabetcartel | https://alphabetcartel.org
# =============================================================================
#
# Build: docker build -t ash-vigil .
# Run:   docker run --gpus all -p 30882:30882 ash-vigil
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Build arguments
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# Default user/group IDs (can be overridden at runtime via PUID/PGID)
ARG DEFAULT_UID=1000
ARG DEFAULT_GID=1000

# Labels
LABEL maintainer="PapaBearDoes <github.com/PapaBearDoes>"
LABEL org.opencontainers.image.title="Ash-Vigil"
LABEL org.opencontainers.image.description="Mental Health Risk Detection Service for Ash Ecosystem"
LABEL org.opencontainers.image.version="5.0.0"
LABEL org.opencontainers.image.vendor="The Alphabet Cartel"
LABEL org.opencontainers.image.url="https://github.com/the-alphabet-cartel/ash-vigil"
LABEL org.opencontainers.image.source="https://github.com/the-alphabet-cartel/ash-vigil"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    # Application
    VIGIL_API_HOST=0.0.0.0 \
    VIGIL_API_PORT=30882 \
    VIGIL_LOG_LEVEL=INFO \
    # Model settings
    VIGIL_MODEL_NAME=ourafla/mental-health-bert-finetuned \
    VIGIL_MODEL_DEVICE=cuda \
    # HuggingFace cache
    HF_HOME=/app/models-cache \
    TRANSFORMERS_CACHE=/app/models-cache \
    # CUDA
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # Default PUID/PGID (LinuxServer.io style)
    PUID=${DEFAULT_UID} \
    PGID=${DEFAULT_GID}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directories (owned by root initially, entrypoint fixes ownership)
RUN mkdir -p /app/config /app/models-cache /app/logs

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/dist-packages

# Copy application code
COPY src/ /app/src/
COPY main.py /app/
COPY docker-entrypoint.py /app/

# Make entrypoint executable
RUN chmod +x /app/docker-entrypoint.py 2>/dev/null || true

# NOTE: We do NOT switch to non-root user here.
# The entrypoint.py handles:
# 1. Creating user with PUID/PGID
# 2. Fixing ownership of /app directories
# 3. Dropping privileges before starting the server

# Expose port
EXPOSE 30882

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:30882/health || exit 1

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command - uses entrypoint for user setup and model initialization
CMD ["python", "/app/docker-entrypoint.py"]
