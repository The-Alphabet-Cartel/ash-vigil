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
FILE VERSION: v5.1-6-6.3-1
LAST MODIFIED: 2026-02-14
PHASE: Phase 6.3 - Risk Score Formula Rebalancing
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import pipeline

from src.managers.logging_config_manager import create_logging_config_manager
from src.managers.secrets_manager import create_secrets_manager


class ModelManager:
    """
    Manages the mental health risk detection model.

    Supports two model types:
    - zero-shot: Uses NLI-based zero-shot classification with custom labels
    - classifier: Traditional fine-tuned text classification

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
        self._zero_shot_config = config.get("zero_shot", {})
        self._thresholds = config.get("thresholds", {})

        # Get logger using factory function
        logging_manager = create_logging_config_manager(config)
        self._logger = logging_manager.get_logger(__name__)

        # Initialize secrets manager and configure HuggingFace
        self._secrets_manager = create_secrets_manager()
        self._hf_token_configured = self._secrets_manager.configure_huggingface()

        # Model state
        self._pipeline = None
        self._is_loaded = False
        self._device = None

        # Model info
        self.model_name = self._model_config.get("name", "facebook/bart-large-mnli")
        self.model_type = self._model_config.get("type", "zero-shot")
        self._max_length = self._model_config.get("max_length", 512)
        self._requested_device = self._model_config.get("device", "cuda")
        self._cache_dir = self._model_config.get("cache_dir", "/app/models-cache")

        # Zero-shot specific setup
        self._candidate_labels: List[str] = []
        self._label_to_risk: Dict[str, str] = {}
        self._hypothesis_template: str = "This text is from a {}"
        self._multi_label: bool = False

        if self.model_type == "zero-shot":
            self._setup_zero_shot_labels()

    def _setup_zero_shot_labels(self) -> None:
        """
        Set up candidate labels and their risk mappings from config.

        Flattens the categorized labels into a single list and builds
        a reverse mapping from label -> risk category.
        """
        labels_config = self._zero_shot_config.get("candidate_labels", {})

        for risk_category in ["high_risk", "moderate_risk", "low_risk", "safe"]:
            labels = labels_config.get(risk_category, [])
            for label in labels:
                self._candidate_labels.append(label)
                self._label_to_risk[label] = risk_category

        self._hypothesis_template = self._zero_shot_config.get(
            "hypothesis_template", "This text is from a {}"
        )
        self._multi_label = self._zero_shot_config.get("multi_label", False)

        self._logger.info(f"📋 Zero-shot labels configured: {len(self._candidate_labels)} labels")
        for risk_cat in ["high_risk", "moderate_risk", "low_risk", "safe"]:
            count = sum(1 for l, r in self._label_to_risk.items() if r == risk_cat)
            self._logger.debug(f"   {risk_cat}: {count} labels")

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

    @property
    def candidate_labels(self) -> List[str]:
        """Get the candidate labels for zero-shot classification."""
        return self._candidate_labels.copy()

    async def load_model(self) -> None:
        """
        Load the model and create inference pipeline.

        Uses GPU if available and requested, falls back to CPU.
        HuggingFace token is automatically used if available for gated models.
        """
        self._logger.info(f"Loading model: {self.model_name}")
        self._logger.info(f"Model type: {self.model_type}")
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
        # Get token for gated models
        token = self._secrets_manager.get_huggingface_token()

        # Determine pipeline type based on model type
        if self.model_type == "zero-shot":
            pipeline_type = "zero-shot-classification"
            self._logger.info(f"📋 Creating zero-shot pipeline with {len(self._candidate_labels)} candidate labels")
        else:
            pipeline_type = "text-classification"
            self._logger.info("📋 Creating text-classification pipeline")

        # Create pipeline
        self._pipeline = pipeline(
            pipeline_type,
            model=self.model_name,
            device=0 if self._device.type == "cuda" else -1,
            token=token,
            model_kwargs={"cache_dir": self._cache_dir},
        )

    async def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on input text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with risk_label, risk_score, confidence, and details
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run inference in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._predict_sync, text)

        return result

    def _predict_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous prediction (run in executor)."""
        if self.model_type == "zero-shot":
            return self._predict_zero_shot(text)
        else:
            return self._predict_classifier(text)

    def _predict_zero_shot(self, text: str) -> Dict[str, Any]:
        """
        Run zero-shot classification inference.

        Args:
            text: Input text to analyze

        Returns:
            Normalized result dictionary
        """
        # Run zero-shot classification
        result = self._pipeline(
            text,
            candidate_labels=self._candidate_labels,
            hypothesis_template=self._hypothesis_template,
            multi_label=self._multi_label,
        )

        # Extract results
        # Result format: {"sequence": text, "labels": [...], "scores": [...]}
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        # Get risk category for top label
        risk_category = self._label_to_risk.get(top_label, "unknown")

        # Calculate risk score based on category and confidence
        risk_score = self._calculate_risk_score(risk_category, top_score, result)

        # Determine final risk label based on score thresholds
        final_risk_label = self._score_to_risk_label(risk_score)

        return {
            "risk_label": final_risk_label,
            "risk_score": round(risk_score, 4),
            "confidence": round(top_score, 4),
            "matched_label": top_label,
            "matched_category": risk_category,
            "all_scores": {
                label: round(score, 4)
                for label, score in zip(result["labels"][:5], result["scores"][:5])
            },
        }

    def _calculate_risk_score(
        self, risk_category: str, top_score: float, full_result: Dict
    ) -> float:
        """
        Calculate overall risk score from zero-shot results using ratio-based formula.

        Uses a two-signal approach: risk signal vs safe signal. Low_risk is grouped
        with safe (someone "venting frustration" is not a crisis). The final score
        is the difference, naturally producing scores near zero for safe content
        and only elevating when genuine high/moderate risk signals dominate.

        Formula (Option B - Ratio-Based):
            risk_signal  = high_risk_max * 1.0 + moderate_risk_max * 0.5
            safe_signal  = safe_max * 1.0 + low_risk_max * 0.4
            risk_score   = max(0.0, risk_signal - safe_signal)

        Args:
            risk_category: Risk category of top label
            top_score: Confidence score of top label
            full_result: Full zero-shot result with all labels/scores

        Returns:
            Normalized risk score between 0 and 1
        """
        # Build score aggregation by risk category (max score per category)
        category_scores: Dict[str, float] = {
            "high_risk": 0.0,
            "moderate_risk": 0.0,
            "low_risk": 0.0,
            "safe": 0.0,
        }

        for label, score in zip(full_result["labels"], full_result["scores"]):
            cat = self._label_to_risk.get(label, "unknown")
            if cat in category_scores:
                category_scores[cat] = max(category_scores[cat], score)

        # Ratio-based formula: group signals into risk vs safe camps
        risk_signal = (
            category_scores["high_risk"] * 1.0
            + category_scores["moderate_risk"] * 0.5
        )
        safe_signal = (
            category_scores["safe"] * 1.0
            + category_scores["low_risk"] * 0.4
        )

        risk_score = max(0.0, risk_signal - safe_signal)

        # Clamp to [0, 1]
        return min(1.0, risk_score)

    def _score_to_risk_label(self, risk_score: float) -> str:
        """
        Convert risk score to risk label using thresholds.

        Args:
            risk_score: Calculated risk score (0-1)

        Returns:
            Risk label string
        """
        high_threshold = self._thresholds.get("high_risk_min", 0.6)
        moderate_threshold = self._thresholds.get("moderate_risk_min", 0.4)
        low_threshold = self._thresholds.get("low_risk_min", 0.25)

        if risk_score >= high_threshold:
            return "high_risk"
        elif risk_score >= moderate_threshold:
            return "moderate_risk"
        elif risk_score >= low_threshold:
            return "low_risk"
        else:
            return "safe"

    def _predict_classifier(self, text: str) -> Dict[str, Any]:
        """
        Run traditional classifier inference (for backwards compatibility).

        Args:
            text: Input text to analyze

        Returns:
            Normalized result dictionary
        """
        outputs = self._pipeline(text)

        if outputs and len(outputs) > 0:
            result = outputs[0]
            raw_label = result.get("label", "unknown")
            score = result.get("score", 0.0)

            # Simple mapping for classifier models
            risk_label = self._map_classifier_label(raw_label)
            risk_score = score if risk_label != "safe" else (1.0 - score)

            return {
                "risk_label": risk_label,
                "risk_score": round(risk_score, 4),
                "confidence": round(score, 4),
                "matched_label": raw_label,
                "matched_category": risk_label,
                "all_scores": {raw_label: round(score, 4)},
            }

        return {
            "risk_label": "unknown",
            "risk_score": 0.0,
            "confidence": 0.0,
            "matched_label": "unknown",
            "matched_category": "unknown",
            "all_scores": {},
        }

    def _map_classifier_label(self, raw_label: str) -> str:
        """Map classifier output label to risk category."""
        label_lower = raw_label.lower()

        high_risk_keywords = ["suicid", "self-harm", "crisis"]
        moderate_risk_keywords = ["depress", "anxiety", "stress", "distress"]
        safe_keywords = ["normal", "neutral", "safe", "positive"]

        for kw in high_risk_keywords:
            if kw in label_lower:
                return "high_risk"
        for kw in moderate_risk_keywords:
            if kw in label_lower:
                return "moderate_risk"
        for kw in safe_keywords:
            if kw in label_lower:
                return "safe"

        return "low_risk"

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
        info = {
            "name": self.model_name,
            "type": self.model_type,
            "device": self.device,
            "is_loaded": self._is_loaded,
            "max_length": self._max_length,
            "cache_dir": self._cache_dir,
            "gpu_available": self.gpu_available,
            "hf_token_configured": self._hf_token_configured,
        }

        if self.model_type == "zero-shot":
            info["candidate_labels_count"] = len(self._candidate_labels)
            info["hypothesis_template"] = self._hypothesis_template

        return info

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
