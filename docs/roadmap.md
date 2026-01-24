---
title: "Ash-Vigil - Development Roadmap"
description: "Development roadmap for the Ash-Vigil mental health risk detection service"
category: roadmap
tags:
  - ash-vigil
  - roadmap
  - planning
author: "PapaBearDoes"
version: "5.0"
last_updated: "2026-01-24"
---
# Ash-Vigil: Development Roadmap

============================================================================
**Ash**: Crisis Detection Ecosystem for The Alphabet Cartel
**The Alphabet Cartel** - https://discord.gg/alphabetcartel | https://alphabetcartel.org
============================================================================

**Document Version**: v5.0.1
**Created**: 2026-01-24
**Status**: 📋 Planning
**Repository**: https://github.com/the-alphabet-cartel/ash-vigil

---

## Table of Contents

1. [Mission Statement](#-mission-statement)
2. [Component Overview](#-component-overview)
3. [Infrastructure](#-infrastructure)
4. [Phase Summary](#-phase-summary)
5. [Phase 1: Evaluation Infrastructure](#-phase-1-evaluation-infrastructure)
6. [Phase 2: Model Evaluation](#-phase-2-model-evaluation)
7. [Phase 3: Ash-Vigil Service](#-phase-3-ash-vigil-service)
8. [Phase 4: Ash-NLP Integration](#-phase-4-ash-nlp-integration)
9. [Phase 5: Full System Testing](#-phase-5-full-system-testing)
10. [Phase 6: Production Deployment](#-phase-6-production-deployment)
11. [Success Criteria](#-success-criteria)
12. [Change Log](#-change-log)

---

## 🎯 Mission Statement

```
ASH-VIGIL MISSION:
    Detect   → Identify crisis patterns that generic models miss
    Amplify  → Boost detection confidence for at-risk messages
    Protect  → Close the 63% false negative gap in crisis detection
```

Ash-Vigil serves as the vigilant guardian that catches what others miss - the subtle signals of crisis that generic NLP models overlook.

---

## 📦 Component Overview

### What is Ash-Vigil?

Ash-Vigil is a specialized mental health risk detection service that provides a 5th model for the Ash-NLP ensemble. It runs on dedicated GPU hardware (Bacchus) and communicates with Ash-NLP via HTTP.

### Role in Ecosystem

| Capability | Description |
|------------|-------------|
| **Detect** | Identify suicide/crisis risk patterns in text |
| **Amplify** | Boost ensemble scores when risk is detected |
| **Flag** | Mark messages for CRT review when risk is high |
| **Fallback** | Graceful degradation when unavailable |

### Key Specifications

| Spec | Value |
|------|-------|
| **Host** | Bacchus (10.20.30.14) |
| **Port** | 30885 |
| **GPU** | NVIDIA RTX 5060 (8GB VRAM) |
| **Container** | Docker Desktop (WSL2) |
| **Model** | TBD after Phase 2 evaluation |
| **License** | Apache 2.0 (ecosystem standard) |

---

## 🖥️ Infrastructure

### Bacchus Server

| Spec | Value |
|------|-------|
| **OS** | Windows 11 Pro |
| **CPU** | AMD Ryzen 7 7700X |
| **RAM** | 128GB |
| **GPU** | NVIDIA RTX 5060 (8GB VRAM) |
| **Container Runtime** | Docker Desktop (WSL2 backend) |
| **GPU Status** | ✅ Verified working (2026-01-24) |

### Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Lofn (10.20.30.253)                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Ash-NLP (:30880)                    │  │
│  │  ┌──────┐ ┌─────────┐ ┌────────┐ ┌─────┐             │  │
│  │  │ BART │ │Sentiment│ │Emotions│ │Irony│             │  │
│  │  └──────┘ └─────────┘ └────────┘ └─────┘             │  │
│  │                    │                                  │  │
│  │           ┌────────▼────────┐                        │  │
│  │           │ Decision Engine │◄───── HTTP ────┐       │  │
│  │           └─────────────────┘                │       │  │
│  └──────────────────────────────────────────────│───────┘  │
└─────────────────────────────────────────────────│──────────┘
                                                  │
                                    Local Network (~2-10ms)
                                                  │
┌─────────────────────────────────────────────────│──────────┐
│                    Bacchus (10.20.30.14)        ▼          │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                  Ash-Vigil (:30885)                   │ │
│  │         Mental Health Risk Detection API              │ │
│  │                                                       │ │
│  │  Model: [TBD - candidate evaluation in Phase 2]      │ │
│  │  GPU: RTX 5060 (CUDA 13.1)                           │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

---

## 📊 Phase Summary

| Phase | Name | Status | Estimated | Dependencies |
|-------|------|--------|-----------|--------------|
| 1 | Evaluation Infrastructure | 📋 Planning | 8-12 hours | Ash-Thrash test phrases |
| 2 | Model Evaluation | 📋 Planning | 4-6 hours | Phase 1 |
| 3 | Ash-Vigil Service | 📋 Planning | 12-16 hours | Phase 2 decision |
| 4 | Ash-NLP Integration | 📋 Planning | 8-12 hours | Phase 3 |
| 5 | Full System Testing | 📋 Planning | 6-8 hours | Phase 4 |
| 6 | Production Deployment | 📋 Planning | 4-6 hours | Phase 5 |

**Total Estimated**: 42-60 hours

---

## 📋 Phase 1: Evaluation Infrastructure

**Status**: 📋 Planning
**Estimated**: 8-12 hours
**Component**: Ash-Thrash

### Objective

Create infrastructure in Ash-Thrash to evaluate candidate mental health detection models independently of Ash-NLP.

### Deliverables

| Deliverable | Description |
|-------------|-------------|
| `model_evaluator.py` | Standalone model evaluation module |
| `candidate_models.json` | Configuration for candidate models |
| Comparison reports | HTML/JSON reports comparing model performance |
| Expanded test phrases | LGBTQIA+ (100), Gaming (100) - ✅ Complete |

### Tasks

- [ ] 1.1 Create `ash-thrash/src/evaluators/` directory
- [ ] 1.2 Implement `model_evaluator.py` with HuggingFace transformers
- [ ] 1.3 Create `candidate_models.json` configuration
- [ ] 1.4 Implement comparison report generation
- [ ] 1.5 Expand remaining specialty test phrases to 50 each
- [ ] 1.6 Test evaluation pipeline with dummy model

### File Structure

```
ash-thrash/
├── src/
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── model_evaluator.py      # Core evaluation logic
│   │   └── report_generator.py     # Comparison reports
│   └── config/
│       └── candidate_models.json   # Model configurations
```

### Success Criteria

- [ ] Can load any HuggingFace model for evaluation
- [ ] Generates per-category accuracy metrics
- [ ] Produces side-by-side comparison reports
- [ ] Works on Bacchus GPU (where models will run)

---

## 📋 Phase 2: Model Evaluation

**Status**: 📋 Planning
**Estimated**: 4-6 hours
**Dependencies**: Phase 1 complete

### Objective

Evaluate candidate models against Ash-Thrash test phrases to select the best performer for integration.

### Candidate Models

| Model | Priority | Rationale |
|-------|----------|-----------|
| `ourafla/mental-health-bert-finetuned` | Primary | Trained on r/SuicideWatch, Apache 2.0 |
| `dsuram/distilbert-mentalhealth-classifier` | Secondary | Smaller, MIT license |
| `paulagarciaserrano/roberta-depression-detection` | Tertiary | Depression-specific |

### Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **LGBTQIA+ Accuracy** | >70% | Must catch community-specific patterns |
| **Planning Signals** | >75% | Must catch "bought the rope" patterns |
| **Medication/Overdose** | >80% | Must catch pharmaceutical patterns |
| **Gaming False Positives** | <10% | Must not over-trigger on gaming |
| **Inference Latency** | <100ms | Must be fast enough for real-time |

### Tasks

- [ ] 2.1 Run evaluation on `ourafla/mental-health-bert-finetuned`
- [ ] 2.2 Run evaluation on `dsuram/distilbert-mentalhealth-classifier`
- [ ] 2.3 Generate comparison report
- [ ] 2.4 **DECISION POINT**: Select model for integration
- [ ] 2.5 Document selection rationale

### Decision Gate

**Before proceeding to Phase 3**, we must have:
1. Evaluation results for at least 2 candidate models
2. Clear winner based on accuracy metrics
3. Acceptable inference latency (<100ms)
4. Documented decision with rationale

---

## 📋 Phase 3: Ash-Vigil Service

**Status**: 📋 Planning
**Estimated**: 12-16 hours
**Dependencies**: Phase 2 decision

### Objective

Create the Ash-Vigil service following Clean Architecture Charter v5.2.2 standards.

### Repository Structure

```
ash-vigil/
├── .github/
│   └── workflows/
│       └── docker-publish.yml      # GHCR publishing
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                  # FastAPI application
│   ├── config/
│   │   └── default.json            # Default configuration
│   ├── managers/
│   │   ├── __init__.py
│   │   ├── config_manager.py       # Configuration management
│   │   ├── logging_config_manager.py
│   │   ├── model_manager.py        # Model loading/inference
│   │   └── secrets_manager.py      # Docker secrets
│   └── __init__.py
├── tests/
├── logs/
├── models-cache/                   # HuggingFace model cache
├── .env.template
├── .gitignore
├── docker-compose.yml
├── docker-entrypoint.py            # Pure Python + tini
├── Dockerfile
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze text for risk |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### Tasks

- [ ] 3.1 Create GitHub repository `the-alphabet-cartel/ash-vigil`
- [ ] 3.2 Initialize repository with Clean Architecture structure
- [ ] 3.3 Implement ConfigManager following Charter v5.2.2
- [ ] 3.4 Implement LoggingConfigManager with colorization
- [ ] 3.5 Implement ModelManager for model loading/inference
- [ ] 3.6 Implement FastAPI application with endpoints
- [ ] 3.7 Create Dockerfile with GPU support
- [ ] 3.8 Create docker-compose.yml
- [ ] 3.9 Implement docker-entrypoint.py (Pure Python + tini)
- [ ] 3.10 Create GitHub Actions workflow for GHCR
- [ ] 3.11 Deploy to Bacchus and verify GPU inference
- [ ] 3.12 Test all endpoints

### Docker Compose

```yaml
services:
  ash-vigil:
    build: .
    container_name: ash-vigil
    image: ghcr.io/the-alphabet-cartel/ash-vigil:latest
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    ports:
      - "30885:30885"
    
    environment:
      - VIGIL_API_HOST=0.0.0.0
      - VIGIL_API_PORT=30885
      - VIGIL_MODEL_DEVICE=cuda
      - NVIDIA_VISIBLE_DEVICES=all
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}
    
    volumes:
      - ./logs:/app/logs
      - ./models-cache:/app/models-cache
    
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:30885/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
    
    restart: unless-stopped
```

### Success Criteria

- [ ] Service starts and loads model on GPU
- [ ] `/health` returns healthy status
- [ ] `/analyze` returns risk scores
- [ ] Inference latency <100ms
- [ ] GHCR image builds successfully

---

## 📋 Phase 4: Ash-NLP Integration

**Status**: 📋 Planning
**Estimated**: 8-12 hours
**Dependencies**: Phase 3 complete

### Objective

Integrate Ash-Vigil as a soft amplifier in the Ash-NLP Decision Engine.

### Components Modified

| Component | Changes |
|-----------|---------|
| `ash-nlp/src/ensemble/decision_engine.py` | Add Vigil client, risk amplification |
| `ash-nlp/src/clients/vigil_client.py` | New HTTP client for Ash-Vigil |
| `ash-nlp/config/default.json` | Add Vigil configuration |

### Amplification Algorithm

```python
# Soft amplification (doesn't override irony dampening)
base_score = weighted_ensemble_score()  # BART + Sentiment + Emotions
risk_signal = vigil_result.risk_score   # 0.0-1.0

if risk_signal > 0.8 and base_score < 0.6:
    amplified_score = max(base_score + 0.35, 0.55)
    requires_review = True
elif risk_signal > 0.6 and base_score < 0.5:
    amplified_score = base_score + (risk_signal * 0.3)
elif risk_signal > 0.4:
    amplified_score = base_score + (risk_signal * 0.1)
else:
    amplified_score = base_score

# Irony dampening STILL APPLIES
final_score = amplified_score * irony_dampening
```

### Tasks

- [ ] 4.1 Create `ash-nlp/src/clients/vigil_client.py`
- [ ] 4.2 Implement circuit breaker for Vigil calls
- [ ] 4.3 Add Vigil configuration to `default.json`
- [ ] 4.4 Modify DecisionEngine to call Vigil
- [ ] 4.5 Implement risk amplification logic
- [ ] 4.6 Add `vigil_status` to response schema
- [ ] 4.7 Update Ash-NLP tests
- [ ] 4.8 Deploy updated Ash-NLP

### Configuration Schema

```json
{
    "vigil": {
        "enabled": true,
        "host": "10.20.30.14",
        "port": 30885,
        "timeout_ms": 500,
        "retry_attempts": 1,
        "circuit_breaker": {
            "failure_threshold": 3,
            "recovery_timeout_seconds": 30
        }
    },
    "risk_amplification": {
        "critical_threshold": 0.8,
        "high_threshold": 0.6,
        "moderate_threshold": 0.4,
        "critical_boost": 0.35,
        "high_boost_multiplier": 0.3,
        "moderate_boost_multiplier": 0.1
    }
}
```

### Success Criteria

- [ ] Ash-NLP calls Ash-Vigil for each analysis
- [ ] Risk amplification correctly boosts scores
- [ ] Irony dampening still works after amplification
- [ ] Circuit breaker prevents cascade failures
- [ ] Fallback mode works when Vigil unavailable

---

## 📋 Phase 5: Full System Testing

**Status**: 📋 Planning
**Estimated**: 6-8 hours
**Dependencies**: Phase 4 complete

### Objective

Run comprehensive Ash-Thrash testing against the integrated system and tune parameters.

### Test Categories

| Category | Baseline | Target | Focus |
|----------|----------|--------|-------|
| definite_high | 31% | >80% | Must improve significantly |
| specialty_lgbtqia | 20% | >70% | Critical community patterns |
| specialty_gaming | 40% | >90% | Must reduce false positives |
| planning_signals | ~0% | >75% | New detection capability |
| medication_overdose | 31% | >80% | New detection capability |

### Tasks

- [ ] 5.1 Run full Ash-Thrash suite (baseline without Vigil)
- [ ] 5.2 Run full Ash-Thrash suite (with Vigil enabled)
- [ ] 5.3 Generate comparison report
- [ ] 5.4 Tune amplification parameters if needed
- [ ] 5.5 Verify gaming false positives are dampened
- [ ] 5.6 Performance benchmark (latency impact)
- [ ] 5.7 Document final accuracy metrics

### Success Criteria

- [ ] Overall false negative rate <30% (from 63%)
- [ ] LGBTQIA+ accuracy >70% (from 20%)
- [ ] Gaming false positives <10% (from 60%)
- [ ] Total latency increase <200ms
- [ ] No regression in existing detection

---

## 📋 Phase 6: Production Deployment

**Status**: 📋 Planning
**Estimated**: 4-6 hours
**Dependencies**: Phase 5 complete

### Objective

Deploy Ash-Vigil to production and integrate with ecosystem monitoring.

### Tasks

- [ ] 6.1 Final configuration review
- [ ] 6.2 Update Ash (Core) to monitor Ash-Vigil health
- [ ] 6.3 Add Ash-Vigil to ecosystem architecture diagram
- [ ] 6.4 Deploy Ash-Vigil to Bacchus (production)
- [ ] 6.5 Deploy updated Ash-NLP to Lofn
- [ ] 6.6 Monitor initial production traffic
- [ ] 6.7 Collect CRT feedback on alert quality
- [ ] 6.8 Update all documentation

### Ecosystem Updates

| Document | Update |
|----------|--------|
| `ash/docs/v5.0/roadmap.md` | Add Ash-Vigil to component list |
| `ash/README.md` | Add Ash-Vigil to architecture |
| `ash-nlp/README.md` | Document Vigil integration |
| Clean Architecture Charter | Add Ash-Vigil patterns |

### Success Criteria

- [ ] Ash-Vigil running stable in production
- [ ] Ash (Core) monitoring Vigil health
- [ ] CRT receiving improved quality alerts
- [ ] Documentation complete
- [ ] No production incidents

---

## ✅ Success Criteria

### Overall Project Success

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| False Negative Rate | <30% | Ash-Thrash full suite |
| LGBTQIA+ Detection | >70% | specialty_lgbtqia category |
| Planning Signal Detection | >75% | New test category |
| Gaming False Positives | <10% | specialty_gaming category |
| Latency Impact | <200ms | End-to-end benchmark |
| Availability | >99% | Uptime with fallback |

### Per-Phase Gates

Each phase must meet its success criteria before proceeding to the next.

---

## 📝 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-01-24 | v5.0.1 | Initial roadmap created | PapaBearDoes + Claude |

---

**Built with care for chosen family** 🏳️‍🌈
