# Ash-Vigil

---

**Ash-Vigil**: Mental Health Risk Detection Service
**Ash Ecosystem**: Crisis Detection for The Alphabet Cartel
**The Alphabet Cartel** - https://discord.gg/alphabetcartel | https://alphabetcartel.org

---

## Overview

Ash-Vigil is a specialized mental health risk detection service that provides the 5th model in the Ash-NLP ensemble. It uses a model specifically trained on crisis/suicide detection data to catch patterns that generic NLP models miss.

### Mission

```
Watch    → Scan messages for subtle crisis signals that generic models miss
Amplify  → Boost ensemble confidence when specialized risk patterns emerge
Catch    → Detect planning signals, passive ideation, and minority stress indicators
Protect  → Safeguard our LGBTQIA+ community through vigilant pattern detection
```

## Architecture

```
Lofn (Ash-NLP)                    Bacchus (Ash-Vigil)
┌──────────────────┐               ┌───────────────────┐
│ BART + Sentiment │               │ Mental Health     │
│ + Emotions       │──── HTTP ────>│ Risk Detector     │
│ + Irony          │               │ (GPU-accelerated) │
└──────────────────┘               └───────────────────┘
```

## Features

- **GPU-Accelerated Inference**: Runs on NVIDIA GPU for fast inference (<100ms)
- **Model Pre-Download**: Models are cached during container startup
- **Warmup Inference**: First-run latency eliminated via warmup
- **Bulk Evaluation**: `/evaluate` endpoint for Ash-Thrash testing
- **Soft Amplification**: Boosts ensemble scores without overriding irony dampening
- **Circuit Breaker Ready**: Graceful degradation when unavailable
- **Clean Architecture**: Follows Ash ecosystem Charter v5.2.3

## Quick Start

### Prerequisites

- Docker Desktop with WSL2 backend
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

### Running

```bash
# Clone the repository
git clone https://github.com/the-alphabet-cartel/ash-vigil.git
cd ash-vigil

# Copy environment template
cp .env.template .env

# Create secrets directory and add HuggingFace token (if needed)
mkdir -p secrets
echo "your-hf-token" > secrets/huggingface_token

# Start the service
docker-compose up -d

# Check health
curl http://localhost:30882/health
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze single text for mental health risk |
| `/evaluate` | POST | Bulk phrase evaluation for testing |
| `/health` | GET | Health check with GPU status |
| `/metrics` | GET | Prometheus-compatible metrics |
| `/docs` | GET | OpenAPI/Swagger documentation |

### POST /analyze

Single message analysis for real-time use.

```json
// Request
{
    "text": "I've been hoarding my medication",
    "request_id": "optional-uuid"
}

// Response
{
    "request_id": "uuid",
    "risk_score": 0.87,
    "risk_label": "high_risk",
    "confidence": 0.92,
    "model_version": "ourafla/mental-health-bert-finetuned",
    "inference_time_ms": 23.45,
    "timestamp": "2026-01-26T16:30:00Z"
}
```

### POST /evaluate

Bulk phrase evaluation for Ash-Thrash testing.

```json
// Request
{
    "phrases": [
        {"id": "test_001", "text": "I feel hopeless"},
        {"id": "test_002", "text": "Having a great day!"}
    ],
    "include_timing": true
}

// Response
{
    "model_name": "ourafla/mental-health-bert-finetuned",
    "model_version": "1.0.0",
    "results": [
        {"id": "test_001", "risk_score": 0.78, "risk_label": "high_risk", ...},
        {"id": "test_002", "risk_score": 0.05, "risk_label": "safe", ...}
    ],
    "total_phrases": 2,
    "successful_phrases": 2,
    "failed_phrases": 0,
    "total_time_ms": 47.2,
    "average_time_ms": 23.6,
    "timestamp": "2026-01-26T16:30:00Z"
}
```

## Configuration

Environment variables (set in `.env` or docker-compose):

| Variable | Default | Description |
|----------|---------|-------------|
| `VIGIL_HOST` | `0.0.0.0` | API bind address |
| `VIGIL_PORT` | `30882` | API port |
| `VIGIL_MODEL_NAME` | `ourafla/mental-health-bert-finetuned` | HuggingFace model |
| `VIGIL_MODEL_DEVICE` | `cuda` | Device (cuda/cpu) |
| `VIGIL_MODEL_CACHE_DIR` | `/app/models-cache` | Model cache directory |
| `VIGIL_LOG_LEVEL` | `INFO` | Logging level |
| `VIGIL_LOG_FORMAT` | `human` | Log format (human/json) |
| `PUID` | `1000` | User ID for file permissions |
| `PGID` | `1000` | Group ID for file permissions |
| `FORCE_COLOR` | `1` | Force colorized logging |

## Infrastructure

| Spec | Value |
|------|-------|
| **Host** | Bacchus (10.20.30.14) |
| **Port** | 30882 |
| **GPU** | NVIDIA RTX 5060 (8GB VRAM) |
| **Container** | Docker Desktop (WSL2) |
| **Python** | 3.12 |

## Documentation

- **API Reference**: [docs/api/reference.md](docs/api/reference.md)
- **Development Roadmap**: [docs/v5.0/roadmap.md](docs/v5.0/roadmap.md)
- **Clean Architecture Charter**: [docs/standards/clean_architecture_charter.md](docs/standards/clean_architecture_charter.md)

## Project Structure

```
ash-vigil/
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI application
│   ├── config/
│   │   └── default.json        # Default configuration
│   └── managers/
│       ├── config_manager.py   # Configuration loading
│       ├── logging_config_manager.py  # Logging setup
│       ├── model_manager.py    # ML model management
│       └── secrets_manager.py  # Docker secrets handling
├── docs/
│   ├── api/
│   │   └── reference.md        # API documentation
│   ├── standards/
│   │   └── clean_architecture_charter.md
│   └── v5.0/
│       ├── roadmap.md
│       └── phase1/planning.md
├── main.py                     # Application entry point
├── docker-entrypoint.py        # Container startup script
├── Dockerfile                  # Container build
├── docker-compose.yml          # Service orchestration
├── requirements.txt            # Python dependencies
└── .env.template               # Environment template
```

## License

Apache License 2.0 - See [LICENSE](LICENSE)

---

**Built with care for chosen family** 🏳️‍🌈
