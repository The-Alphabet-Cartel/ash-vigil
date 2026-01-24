---
title: "Ash-Vigil - Mental Health Risk Detection Service"
description: "Planning document for the 5th model risk amplifier service running on Bacchus"
category: planning
tags:
  - ash-vigil
  - risk-detection
  - mental-health
  - machine-learning
author: "PapaBearDoes"
version: "5.0"
last_updated: "2026-01-24"
---
# Ash-Vigil: Mental Health Risk Detection Service

============================================================================
**Ash**: Crisis Detection Ecosystem for The Alphabet Cartel
**The Alphabet Cartel** - https://discord.gg/alphabetcartel | https://alphabetcartel.org
============================================================================

**Document Version**: v5.0.1
**Created**: 2026-01-24
**Status**: 📋 Planning
**Repository**: https://github.com/the-alphabet-cartel/ash-vigil (to be created)

---

## Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Problem Statement](#-problem-statement)
3. [Solution: 5th Model Risk Amplifier](#-solution-5th-model-risk-amplifier)
4. [Architecture](#-architecture)
5. [Infrastructure](#-infrastructure)
6. [API Schema](#-api-schema)
7. [Model Selection](#-model-selection)
8. [Integration with Ash-NLP](#-integration-with-ash-nlp)
9. [Fallback Behavior](#-fallback-behavior)
10. [Testing Strategy](#-testing-strategy)
11. [Implementation Phases](#-implementation-phases)
12. [Future Considerations](#-future-considerations)
13. [Change Log](#-change-log)

---

## 📋 Executive Summary

### What is Ash-Vigil?

Ash-Vigil is a specialized mental health risk detection service that runs as the 5th model in the Ash-NLP ensemble. Unlike the existing four models (BART, Sentiment, Emotions, Irony), Ash-Vigil uses a model specifically trained on crisis/suicide detection data to catch patterns that generic NLP models miss.

### Why Ash-Vigil?

Ash-Thrash v5.0 testing revealed **critical detection gaps** in Ash-NLP:

| Issue | Current Accuracy | Risk |
|-------|------------------|------|
| LGBTQIA+ Risk Factors | **20%** | Family rejection marked "safe" |
| Medication/Overdose | **31%** | "Take all these pills" marked "safe" |
| Planning Signals | **~0%** | "Bought the rope today" marked "safe" |
| Overall False Negative Rate | **63%** | Life-safety critical |

These gaps exist because BART zero-shot classification works by semantic similarity to generic labels like "suicide ideation" - it doesn't understand planning signals, passive ideation, or community-specific risk factors.

### Why Not Regex Patterns?

Pattern matching was initially considered but rejected:
- **Maintenance nightmare**: Endless game of whack-a-mole with edge cases
- **False positives**: Over time, trying to catch every phrase leads to over-triggering
- **No generalization**: Can't recognize novel phrasings of known concepts
- **Human review exists**: CRT already reviews alerts; we need better signal, not more rules

A trained model can **generalize** to new patterns without constant updates.

### Key Decision

**Start with pre-trained model (Option A)**, evaluate via Ash-Thrash, then consider fine-tuning for LGBTQIA+-specific patterns as future enhancement.

---

## 🚨 Problem Statement

### Root Cause Analysis

The current Ash-NLP ensemble uses BART zero-shot classification as its primary model (50% weight). BART works by comparing input text to candidate labels semantically.

**Current Crisis Labels**:
- Primary: "suicide ideation", "self-harm", "domestic violence", "panic attack", "severe depression", "substance abuse crisis"
- Secondary: "emotional distress", "anxiety", "grief", "relationship crisis", "identity crisis", "isolation"

**The Problem**: These labels are too generic to catch:

| Pattern Type | Example | Why BART Misses It |
|--------------|---------|-------------------|
| Planning Signals | "I bought the rope today" | No label captures acquisition/preparation |
| Passive Ideation | "I wish I'd never wake up" | "suicide ideation" implies active intent |
| Burdensomeness | "Everyone better off without me" | Closer to "relationship crisis" than crisis labels |
| LGBTQIA+ Risk | "My family disowned me for being gay" | "identity crisis" doesn't convey rejection trauma |
| Medication Intent | "I've been hoarding my medication" | No pharmaceutical-specific label |

### Impact

A 63% false negative rate on a crisis detection system means **real crises are being missed**. This directly contradicts the Ash mission:

> **Protect** → Safeguard our LGBTQIA+ chosen family through early crisis response

---

## 💡 Solution: 5th Model Risk Amplifier

### Concept

Add a dedicated mental health risk detection model that acts as a **soft amplifier** in the ensemble scoring:

```
Message → [BART Zero-Shot]    ←50%─┐
          [Sentiment]         ←25%─┤
          [Emotions]          ←10%─┼→ [Weighted Score] → [Risk Amplification] → [Irony Dampening] → Severity
          [Risk Detector]     ←NEW─┤         ↑                    ↑
                                   │         │                    │
          [Irony]            ─────→┘         │                    │
                              (dampening)────┘                    │
                                   └──────────────────────────────┘
```

### Why "Soft Amplifier"?

Gaming language requires careful handling. The risk model should:
- **Boost** scores when it detects risk patterns BART missed
- **NOT override** irony dampening (gaming hyperbole should still be dampened)
- **Flag for review** high-confidence detections regardless of final score

### Amplification Logic

```python
# Soft amplification algorithm
base_score = weighted_ensemble_score()  # BART + Sentiment + Emotions (0.85 weight)
risk_signal = risk_detector_result       # 0.0-1.0 from Ash-Vigil

if risk_signal > 0.8 and base_score < 0.6:
    # Critical risk detected that BART missed entirely
    # Override toward HIGH minimum
    amplified_score = max(base_score + 0.35, 0.55)
    requires_review = True  # Always flag for CRT
elif risk_signal > 0.6 and base_score < 0.5:
    # Significant risk, boost moderately
    amplified_score = base_score + (risk_signal * 0.3)
elif risk_signal > 0.4:
    # Modest risk signal
    amplified_score = base_score + (risk_signal * 0.1)
else:
    amplified_score = base_score

# Irony dampening STILL APPLIES after amplification
final_score = amplified_score * irony_dampening
```

---

## 🏗️ Architecture

### Distributed Architecture

Ash-Vigil runs on Bacchus (Windows 11 + RTX 3050), separate from the main Ash-NLP on Lofn.

```
┌──────────────────────────────────────────────────────────────────┐
│                         Lofn (10.20.30.253)                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                        Ash-NLP (:30880)                    │  │
│  │   ┌──────┐ ┌───────────┐ ┌──────────┐ ┌──────┐            │  │
│  │   │ BART │ │ Sentiment │ │ Emotions │ │Irony │            │  │
│  │   └──────┘ └───────────┘ └──────────┘ └──────┘            │  │
│  │        │         │            │          │                 │  │
│  │        └─────────┴─────┬──────┴──────────┘                 │  │
│  │                        ↓                                   │  │
│  │              ┌─────────────────┐                           │  │
│  │              │ Decision Engine │←── HTTP ──┐               │  │
│  │              └─────────────────┘           │               │  │
│  └────────────────────────────────────────────│───────────────┘  │
└───────────────────────────────────────────────│──────────────────┘
                                                │
                                   Local Network (~2-10ms)
                                                │
┌───────────────────────────────────────────────│──────────────────┐
│                         Bacchus (10.20.30.14) ↓                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Docker Desktop (WSL2 Backend)                 │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │                 Ash-Vigil (:30885)                   │  │  │
│  │  │        Mental Health Risk Detection API              │  │  │
│  │  │                                                      │  │  │
│  │  │  Model: ourafla/mental-health-bert-finetuned        │  │  │
│  │  │  (or other candidate after evaluation)               │  │  │
│  │  │                                                      │  │  │
│  │  │  POST /analyze → { risk_score, risk_label, conf }   │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                       RTX 3050 (8GB VRAM)                        │
│                   (GPU verified working 2026-01-24)              │
└──────────────────────────────────────────────────────────────────┘
```

### Benefits of Distributed Architecture

| Benefit | Description |
|---------|-------------|
| **VRAM Isolation** | Lofn's 12GB stays dedicated to current 4 models |
| **Failure Isolation** | If Ash-Vigil crashes, core detection continues |
| **Independent Updates** | Can update/swap risk models without touching Ash-NLP |
| **Future Scaling** | Easy to add more specialized models on Bacchus |
| **Testing Flexibility** | Can point Ash-Thrash directly at Bacchus for isolated testing |

---

## 🖥️ Infrastructure

### Bacchus Configuration

| Spec | Value |
|------|-------|
| **Host OS** | Windows 11 Pro |
| **CPU** | AMD Ryzen 7 7700X |
| **GPU** | NVIDIA RTX 3050 (8GB VRAM) |
| **RAM** | 128GB |
| **IP** | 10.20.30.14 |
| **Container Runtime** | Docker Desktop (WSL2 backend) |

### GPU Verification

GPU passthrough to Docker verified working (2026-01-24):

```
docker run --rm --gpus all nvidia/cuda:12.6.0-runtime-ubuntu22.04 nvidia-smi
```

Output confirmed:
- NVIDIA GeForce RTX 5060 detected (note: 3050 upgraded to 5060)
- Driver Version: 591.74
- CUDA Version: 13.1

### Port Allocation

| Port | Service | Host | Description |
|------|---------|------|-------------|
| 30880 | Ash-NLP | Lofn | NLP API endpoint |
| 30881 | Ash-Bot | Lofn | Bot health endpoint |
| 30883 | Ash-Dash | Lofn | Dashboard web UI |
| 30884 | MinIO API | Syn | Object storage API |
| **30885** | **Ash-Vigil** | **Bacchus** | **Risk Detection API** |
| 30886 | MinIO Console | Syn | Object storage UI |
| 30887 | Ash (Core) | Lofn | Ecosystem Health API |
| 30888 | Ash-Thrash | Lofn | Testing suite API |

**Note**: Port 30885 was previously allocated to MinIO Console. MinIO Console moves to 30886.

---

## 📡 API Schema

### Endpoints

#### POST /analyze

Analyze text for mental health risk indicators.

**Request**:
```json
{
    "text": "I've been hoarding my medication",
    "request_id": "uuid-optional"
}
```

**Response**:
```json
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

**Risk Labels** (model-dependent):
- `suicide_risk` - Active or passive suicidal ideation
- `self_harm_risk` - Self-harm indicators
- `crisis_distress` - Severe emotional distress
- `moderate_concern` - Elevated concern, not crisis
- `low_risk` - Minimal risk indicators
- `safe` - No risk indicators detected

#### GET /health

Health check endpoint.

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_name": "ourafla/mental-health-bert-finetuned",
    "gpu_available": true,
    "gpu_memory_used_mb": 1200,
    "uptime_seconds": 3600
}
```

#### GET /metrics

Prometheus-compatible metrics endpoint.

---

## 🧠 Model Selection

### Candidate Models

Based on Hugging Face search (2026-01-24):

| Model | Downloads | Training Data | Size | License |
|-------|-----------|---------------|------|---------|
| **ourafla/mental-health-bert-finetuned** | 428 | Suicide-Watch, Reddit MH, Social media | 109.5M | Apache 2.0 ✅ |
| mental/mental-bert-base-uncased | 222K | Mental health domain corpus | BERT-base | CC-BY-NC-4.0 ⚠️ |
| paulagarciaserrano/roberta-depression-detection | 24.7K | Depression-specific | RoBERTa | Not specified |
| dsuram/distilbert-mentalhealth-classifier | 1.1K | Mental health classification | DistilBERT | MIT ✅ |

### Recommended Starting Point

**`ourafla/mental-health-bert-finetuned`**

Rationale:
- Fine-tuned from MentalBERT (BERT pre-trained on mental health text)
- Trained on **Suicide-Watch subreddit** - contains exactly the patterns we're missing
- Apache 2.0 license (commercial-friendly, though we're non-commercial)
- Reasonable size for inference

### Evaluation Required

Before integration, run candidates through Ash-Thrash to verify they catch what BART misses:

```
Ash-Thrash Candidate Evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Category                    BART    Candidate A  Candidate B
──────────────────────────────────────────────────────────────
definite_high               31%     ??%          ??%
specialty_lgbtqia           20%     ??%          ??%
specialty_gaming            40%     ??%          ??%
planning_signals            ~0%     ??%          ??%
medication_overdose         31%     ??%          ??%
```

---

## 🔗 Integration with Ash-NLP

### Decision Engine Modifications

File: `ash-nlp/src/ensemble/decision_engine.py`

```python
class DecisionEngine:
    def __init__(self, config, vigil_client=None):
        self.vigil_client = vigil_client  # Optional Ash-Vigil client
        self.vigil_enabled = vigil_client is not None
        self.vigil_timeout = config.get("vigil_timeout_ms", 500)
    
    async def analyze(self, text: str) -> AnalysisResult:
        # Existing ensemble analysis
        base_result = await self._ensemble_analyze(text)
        
        # Risk amplification (if Ash-Vigil available)
        if self.vigil_enabled:
            try:
                vigil_result = await self.vigil_client.analyze(
                    text, 
                    timeout=self.vigil_timeout
                )
                base_result = self._apply_risk_amplification(
                    base_result, 
                    vigil_result
                )
            except VigilUnavailableError:
                base_result.vigil_status = "unavailable"
                base_result.requires_review = True  # Flag for CRT
        
        return base_result
```

### Configuration

File: `ash-nlp/config/default.json`

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

---

## 🔄 Fallback Behavior

### When Ash-Vigil is Unavailable

Per project requirements: **Continue with base scoring and add warning for CRT review.**

```python
class AnalysisResult:
    # ... existing fields ...
    vigil_status: str  # "ok", "unavailable", "timeout", "error"
    requires_review: bool  # True if Vigil unavailable

# In response JSON
{
    "severity": "medium",
    "confidence": 0.65,
    "vigil_status": "unavailable",
    "requires_review": true,
    "review_reason": "Risk amplification unavailable - manual review recommended"
}
```

### Circuit Breaker Pattern

If Ash-Vigil fails repeatedly, stop calling it temporarily:

```python
class VigilCircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=30):
        self.failures = 0
        self.last_failure = None
        self.state = "closed"  # closed, open, half-open
    
    def should_call(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        return True  # half-open: try one request
    
    def record_success(self):
        self.failures = 0
        self.state = "closed"
    
    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"
```

---

## 🧪 Testing Strategy

### Phase 1: Standalone Model Evaluation

Create `ash-thrash/src/evaluators/model_evaluator.py`:

```python
class ModelEvaluator:
    """Evaluate candidate models against Ash-Thrash test phrases."""
    
    def __init__(self, model_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    async def evaluate_phrases(self, phrases: List[TestPhrase]) -> EvaluationReport:
        results = []
        for phrase in phrases:
            prediction = await self._predict(phrase.message)
            results.append(EvaluationResult(
                phrase=phrase,
                predicted_score=prediction.score,
                predicted_label=prediction.label,
                correct=self._is_correct(prediction, phrase.expected_priority)
            ))
        return EvaluationReport(results)
```

### Phase 2: Comparative Analysis

Run same test phrases through:
1. Current BART (baseline)
2. Candidate Model A
3. Candidate Model B (if evaluating multiple)

Generate comparison report showing accuracy per category.

### Phase 3: Integration Testing

After selecting best candidate:
1. Deploy Ash-Vigil on Bacchus
2. Run full Ash-Thrash suite against Ash-NLP with Ash-Vigil enabled
3. Compare results to baseline (Ash-NLP without Ash-Vigil)
4. Verify gaming false positives are still dampened

### Test Phrase Expansion

Per requirements, expand specialty test phrases:
- Gaming context: 15 → 100 phrases ✅ (completed 2026-01-24)
- LGBTQIA+ specific: 20 → 100 phrases ✅ (completed 2026-01-24)
- Other specialties: → 50 phrases each (pending)

---

## 📅 Implementation Phases

### Phase 1: Evaluation Infrastructure (Ash-Thrash)

**Status**: 📋 Planning
**Estimated**: 8-12 hours

- [ ] Create `model_evaluator.py` module
- [ ] Create `candidate_models.json` configuration
- [ ] Implement standalone model testing
- [ ] Generate comparison reports
- [ ] Expand test phrase files (in progress)

### Phase 2: Model Evaluation

**Status**: 📋 Planning
**Estimated**: 4-6 hours

- [ ] Evaluate `ourafla/mental-health-bert-finetuned`
- [ ] Evaluate `dsuram/distilbert-mentalhealth-classifier`
- [ ] Generate comparison report
- [ ] **Decision point**: Select model for integration

### Phase 3: Ash-Vigil Service

**Status**: 📋 Planning
**Estimated**: 12-16 hours

- [ ] Create `ash-vigil` repository
- [ ] Implement FastAPI service following Clean Architecture
- [ ] Create Dockerfile with GPU support
- [ ] Create docker-compose.yml
- [ ] Implement health/metrics endpoints
- [ ] Create GitHub Actions for GHCR publishing
- [ ] Deploy to Bacchus

### Phase 4: Ash-NLP Integration

**Status**: 📋 Planning
**Estimated**: 8-12 hours

- [ ] Add Vigil client to Ash-NLP
- [ ] Implement risk amplification logic
- [ ] Implement circuit breaker
- [ ] Update configuration schema
- [ ] Add `vigil_status` to response schema

### Phase 5: Full System Testing

**Status**: 📋 Planning
**Estimated**: 6-8 hours

- [ ] Run complete Ash-Thrash suite
- [ ] Tune amplification parameters
- [ ] Verify gaming dampening still works
- [ ] Performance benchmarking
- [ ] Update documentation

### Phase 6: Production Deployment

**Status**: 📋 Planning
**Estimated**: 4-6 hours

- [ ] Final configuration review
- [ ] Deploy to production
- [ ] Monitor initial results
- [ ] CRT feedback collection

**Total Estimated**: 42-60 hours

---

## 🔮 Future Considerations

### Fine-Tuning for LGBTQIA+ Patterns

**Status**: ⚪ Someday
**Complexity**: 🟥 Very High

After initial deployment, consider fine-tuning the model on:
- LGBTQIA+-specific crisis language
- Community-specific slang and expressions
- Minority stress patterns

**Requirements**:
- Curated training dataset (ethically sourced, anonymized)
- Compute resources for fine-tuning
- Evaluation framework to prevent regression

### Additional Specialized Models

Consider adding more specialized models:
- Gaming context detector (reduce false positives)
- LGBTQIA+ risk factor detector
- Planning/preparation signal detector

Each could run as additional service on Bacchus.

### Model Quantization

If inference latency becomes an issue:
- Quantize model to INT8
- Use ONNX runtime for faster inference
- Consider smaller distilled models

---

## 📝 Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2026-01-24 | v5.0.1 | Initial planning document created | PapaBearDoes + Claude |

---

**Built with care for chosen family** 🏳️‍🌈
