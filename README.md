# Ash-Vigil

============================================================================
**Ash-Vigil**: Mental Health Risk Detection Service
**Ash Ecosystem**: Crisis Detection for The Alphabet Cartel
**The Alphabet Cartel** - https://discord.gg/alphabetcartel | https://alphabetcartel.org
============================================================================

## Overview

Ash-Vigil is a specialized mental health risk detection service that provides the 5th model in the Ash-NLP ensemble. It uses a model specifically trained on crisis/suicide detection data to catch patterns that generic NLP models miss.

## Architecture

```
Lofn (Ash-NLP)                    Bacchus (Ash-Vigil)
┌─────────────────┐               ┌─────────────────┐
│ BART + Sentiment│               │ Mental Health   │
│ + Emotions      │──── HTTP ────>│ Risk Detector   │
│ + Irony         │               │ (GPU-accelerated)│
└─────────────────┘               └─────────────────┘
```

## Features

- **GPU-Accelerated Inference**: Runs on NVIDIA GPU for fast inference
- **Soft Amplification**: Boosts ensemble scores without overriding irony dampening
- **Circuit Breaker**: Graceful degradation when unavailable
- **Clean Architecture**: Follows Ash ecosystem Charter v5.2.2

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

# Start the service
docker-compose up -d

# Check health
curl http://localhost:30890/health
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze text for mental health risk |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### POST /analyze

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
    "risk_label": "suicide_risk",
    "confidence": 0.92,
    "model_version": "ourafla/mental-health-bert-finetuned",
    "inference_time_ms": 45,
    "timestamp": "2026-01-24T16:30:00Z"
}
```

## Configuration

Environment variables (set in `.env` or docker-compose):

| Variable | Default | Description |
|----------|---------|-------------|
| `VIGIL_API_HOST` | `0.0.0.0` | API bind address |
| `VIGIL_API_PORT` | `30890` | API port |
| `VIGIL_MODEL_NAME` | `ourafla/mental-health-bert-finetuned` | HuggingFace model |
| `VIGIL_MODEL_DEVICE` | `cuda` | Device (cuda/cpu) |
| `VIGIL_LOG_LEVEL` | `INFO` | Logging level |
| `PUID` | `1000` | User ID for file permissions |
| `PGID` | `1000` | Group ID for file permissions |

## Infrastructure

| Spec | Value |
|------|-------|
| **Host** | Bacchus (10.20.30.14) |
| **Port** | 30890 |
| **GPU** | NVIDIA RTX 5060 (8GB VRAM) |
| **Container** | Docker Desktop (WSL2) |

## Documentation

- [Planning Document](https://github.com/the-alphabet-cartel/ash/blob/main/docs/v5.0/ash-vigil/planning.md)
- [Development Roadmap](https://github.com/the-alphabet-cartel/ash/blob/main/docs/v5.0/ash-vigil/roadmap.md)
- [Clean Architecture Charter](https://github.com/the-alphabet-cartel/ash/blob/main/docs/standards/clean_architecture_charter.md)

## License

Apache License 2.0 - See [LICENSE](LICENSE)

---

**Built with care for chosen family** 🏳️‍🌈
