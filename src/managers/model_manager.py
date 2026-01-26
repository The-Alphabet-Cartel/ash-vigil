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
Model Manager - Manages ML model loading, inference, and lifecycle
----------------------------------------------------------------------------
FILE VERSION: v5.0-1-1.2-1
LAST MODIFIED: 2026-01-26
PHASE: Phase 1 - Service Completion
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
"""

import asyncio
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.managers.logging_config_manager import create_logging_config_manager
from src.managers.secrets_manager import create_secrets_manager


class ModelManager:
    """
    Manages the mental health risk detection model.

    Handles model loading, inference, GPU management, and cleanup.
    Supports GPU acceleration via CUDA and gated model access via HuggingFace token.

    Usage:
        model_manager = create_model_manager(config)
        await model_manager.load_model()
        result = await model_manager.predict("some text")
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

        # Get logger using factory function
        logging_manager = create_logging_config_manager(config)
        self._logger = logging_manager.get_logger(__name__)

        # Initialize secrets manager and configure HuggingFace
        self._secrets_manager = create_secrets_manager()
        self._hf_token_configured = self._secrets_manager.configure_huggingface()

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
        self._cache_dir = self._model_config.get("cache_dir", "/app/models-cache")

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

    @property
    def hf_token_available(self) -> bool:
        """Check if HuggingFace token is configured for gated models."""
        return self._hf_token_configured

    async def load_model(self) -> None:
        """
        Load the model and tokenizer.

        Uses GPU if available and requested, falls back to CPU.
        HuggingFace token is automatically used if available for gated models.
        """
        self._logger.info(f"Loading model: {self.model_name}")
        self._logger.info(f"Cache directory: {self._cache_dir}")

        if self._hf_token_configured:
            self._logger.info("✅ HuggingFace token configured (gated model access enabled)")
        else:
            self._logger.info("ℹ️ No HuggingFace token (public models only)")

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
        # Get token for gated models (transformers will also check HF_TOKEN env var)
        token = self._secrets_manager.get_huggingface_token()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=token,
            cache_dir=self._cache_dir,
        )

        # Load model
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            token=token,
            cache_dir=self._cache_dir,
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

    def predict_batch_sync(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Synchronous batch prediction for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of result dictionaries
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = []
        for text in texts:
            result = self._predict_sync(text)
            results.append(result)

        return results

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

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "name": self.model_name,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "max_length": self._max_length,
            "cache_dir": self._cache_dir,
            "gpu_available": self.gpu_available,
            "hf_token_configured": self._hf_token_configured,
        }

    async def warmup(self, warmup_text: str = "This is a warmup inference.") -> float:
        """
        Perform a warmup inference to eliminate first-run latency.

        Args:
            warmup_text: Text to use for warmup inference

        Returns:
            Warmup inference time in milliseconds
        """
        import time

        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self._logger.info("🔥 Running warmup inference...")

        start = time.perf_counter()
        await self.predict(warmup_text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._logger.success(f"✅ Warmup complete in {elapsed_ms:.1f}ms")
        return elapsed_ms

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


# =============================================================================
# Factory Function
# =============================================================================


def create_model_manager(config: Dict[str, Any]) -> ModelManager:
    """
    Factory function to create a ModelManager instance.

    Following Clean Architecture Rule #1: Factory Functions.

    Args:
        config: Configuration dictionary

    Returns:
        Configured ModelManager instance

    Example:
        >>> config = create_config_manager().config
        >>> model_manager = create_model_manager(config)
        >>> await model_manager.load_model()
        >>> result = await model_manager.predict("some text")
    """
    return ModelManager(config)


# =============================================================================
# Export public interface
# =============================================================================

__all__ = [
    "ModelManager",
    "create_model_manager",
]
