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
Ash-Vigil Managers Module - Factory functions and manager classes
----------------------------------------------------------------------------
FILE VERSION: v5.0-1-1.2-1
LAST MODIFIED: 2026-01-26
PHASE: Phase 1 - Service Completion
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
"""

# Import classes and factory functions
from .config_manager import ConfigManager, create_config_manager
from .logging_config_manager import LoggingConfigManager, create_logging_config_manager
from .model_manager import ModelManager, create_model_manager
from .secrets_manager import SecretsManager, create_secrets_manager

__all__ = [
    # Config Manager
    "ConfigManager",
    "create_config_manager",
    # Logging Config Manager
    "LoggingConfigManager",
    "create_logging_config_manager",
    # Model Manager
    "ModelManager",
    "create_model_manager",
    # Secrets Manager
    "SecretsManager",
    "create_secrets_manager",
]
