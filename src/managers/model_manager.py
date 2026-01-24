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
Manages the loading, inference, and lifecycle of the mental health risk detection model.
----------------------------------------------------------------------------
FILE VERSION: v5.0-1-1.0-1
LAST MODIFIED: 2026-01-24
PHASE: Phase 1 - {Phase Description}
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
Supports GPU acceleration via CUDA.
"""

import asyncio
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.managers.logging_config_manager import LoggingConfigManager


class ModelManager:
    """
    Manages the mental health risk detection model.

    Handles model loading, inference, GPU management, and cleanup.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelManager.

        Args:
            config: Configuration dictionary containing model settings
        """
        self._config = config
        self._model_config = config.get("model", {})
        self._risk_labels = config.get("risk_labels", {})

        # Get logger
        logging_manager = LoggingConfigManager(config)
        self._logger = logging_manager.get_logger(__name__)

        # Model state
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._is_loaded = False
        self._device = None

        # Model info
        self.model_name = self._model_config.get(
            "name", "ourafla/mental-health-bert-finetuned"
        )
        self._max_length = self._model_config.get("max_length", 512)
        self._requested_device = self._model_config.get("device", "cuda")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._is_loaded

    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available."""
        return torch.cuda.is_available()

    @property
    def device(self) -> str:
        """Get the device the model is running on."""
        return str(self._device) if self._device else "unknown"

    async def load_model(self) -> None:
        """
        Load the model and tokenizer.

        Uses GPU if available and requested, falls back to CPU.
        """
        self._logger.info(f"Loading model: {self.model_name}")

        # Determine device
        if self._requested_device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            self._logger.info(
                f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
        else:
            self._device = torch.device("cpu")
            if self._requested_device == "cuda":
                self._logger.warning("⚠️ CUDA requested but not available, using CPU")
            else:
                self._logger.info("Using CPU for inference")

        # Load in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync)

        self._is_loaded = True
        self._logger.success(f"✅ Model loaded successfully on {self._device}")

    def _load_model_sync(self) -> None:
        """Synchronous model loading (run in executor)."""
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self._device)

        # Create pipeline for easier inference
        self._pipeline = pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            device=0 if self._device.type == "cuda" else -1,
            max_length=self._max_length,
            truncation=True,
        )

    async def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on input text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with score, label, and confidence
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run inference in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._predict_sync, text)

        return result

    def _predict_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous prediction (run in executor)."""
        # Run pipeline
        outputs = self._pipeline(text)

        if outputs and len(outputs) > 0:
            result = outputs[0]
            label = result.get("label", "unknown")
            score = result.get("score", 0.0)

            # Normalize the result
            normalized = self._normalize_result(label, score)
            return normalized

        return {
            "label": "unknown",
            "score": 0.0,
            "confidence": 0.0,
            "raw_label": "unknown",
            "raw_score": 0.0,
        }

    def _normalize_result(self, label: str, score: float) -> Dict[str, Any]:
        """
        Normalize model output to standard format.

        Maps model-specific labels to standard risk categories.

        Args:
            label: Raw label from model
            score: Raw confidence score

        Returns:
            Normalized result dictionary
        """
        # Map label to risk category
        label_lower = label.lower()

        # Check against configured risk labels
        high_risk_labels = self._risk_labels.get("high_risk", [])
        moderate_risk_labels = self._risk_labels.get("moderate_risk", [])
        low_risk_labels = self._risk_labels.get("low_risk", [])
        safe_labels = self._risk_labels.get("safe", [])

        if any(l.lower() in label_lower for l in high_risk_labels):
            risk_score = score  # High confidence = high risk
            risk_label = "high_risk"
        elif any(l.lower() in label_lower for l in moderate_risk_labels):
            risk_score = score * 0.7  # Scale moderate risk
            risk_label = "moderate_risk"
        elif any(l.lower() in label_lower for l in low_risk_labels):
            risk_score = score * 0.4  # Scale low risk
            risk_label = "low_risk"
        elif any(l.lower() in label_lower for l in safe_labels):
            risk_score = 1.0 - score  # Invert for safe (low score = safe)
            risk_label = "safe"
        else:
            # Unknown label - use raw score and label
            risk_score = score
            risk_label = label_lower

        return {
            "label": risk_label,
            "score": round(risk_score, 4),
            "confidence": round(score, 4),
            "raw_label": label,
            "raw_score": round(score, 4),
        }

    def get_gpu_memory_used(self) -> Optional[float]:
        """
        Get GPU memory usage in MB.

        Returns:
            Memory used in MB, or None if GPU not available
        """
        if not torch.cuda.is_available():
            return None

        try:
            memory_allocated = torch.cuda.memory_allocated(0)
            return round(memory_allocated / (1024 * 1024), 2)
        except Exception:
            return None

    async def cleanup(self) -> None:
        """Clean up model resources."""
        self._logger.info("Cleaning up model resources...")

        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False
        self._logger.info("Model resources cleaned up")
