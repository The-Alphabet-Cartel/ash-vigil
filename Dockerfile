# =============================================================================
# Ash-Vigil Dockerfile
# Mental Health Risk Detection Service
# =============================================================================
# The Alphabet Cartel - https://discord.gg/alphabetcartel | https://alphabetcartel.org
# =============================================================================
#
# Build: docker build -t ash-vigil .
# Run:   docker run --gpus all -p 30890:30890 ash-vigil
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

# Default user/group IDs (can be overridden at runtime via PUID/PGID)
ARG DEFAULT_UID=1000
ARG DEFAULT_GID=1000

LABEL maintainer="PapaBearDoes <github.com/PapaBearDoes>"
LABEL org.opencontainers.image.source="https://github.com/the-alphabet-cartel/ash-vigil"
LABEL org.opencontainers.image.description="Mental Health Risk Detection Service for Ash Ecosystem"

# Environment defaults
ENV VIGIL_API_HOST=0.0.0.0 \
    VIGIL_API_PORT=30890 \
    VIGIL_MODEL_NAME=ourafla/mental-health-bert-finetuned \
    VIGIL_MODEL_DEVICE=cuda \
    VIGIL_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models-cache \
    TRANSFORMERS_CACHE=/app/models-cache \
    # Default PUID/PGID (LinuxServer.io style)
    PUID=${DEFAULT_UID} \
    PGID=${DEFAULT_GID}

WORKDIR /app

# Install runtime dependencies
# - tini: PID 1 signal handling
# - curl: health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (will be modified at runtime by entrypoint)
RUN groupadd -g ${PGID} ash-vigil && \
    useradd -u ${PUID} -g ${PGID} -d /app -s /bin/bash ash-vigil

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY src/ /app/src/
COPY main.py /app/
COPY docker-entrypoint.py /app/

# Create directories for runtime data
RUN mkdir -p /app/logs /app/models-cache && \
    chown -R ${PUID}:${PGID} /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:30890/health || exit 1

EXPOSE 30882

# NOTE: Do NOT use USER directive - entrypoint handles PUID/PGID at runtime
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/app/docker-entrypoint.py"]
