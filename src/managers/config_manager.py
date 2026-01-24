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
Configuration management following Clean Architecture Charter v5.2.2.
----------------------------------------------------------------------------
FILE VERSION: v5.0-1-1.0-1
LAST MODIFIED: 2026-01-24
PHASE: Phase 1 - {Phase Description}
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
Loads configuration from JSON files with environment variable overrides.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional


class ConfigManager:
    """
    Configuration manager for Ash-Vigil.

    Loads configuration from JSON files and resolves environment variable
    placeholders following the pattern: ${ENV_VAR_NAME}

    Priority (highest to lowest):
    1. Environment variables
    2. Configuration file values
    3. Default values in configuration

    Usage:
        config_manager = ConfigManager()
        port = config_manager.get("api", "port")
    """

    # Environment variable pattern: ${VAR_NAME}
    ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Optional path to configuration file.
                         Defaults to src/config/default.json
        """
        self._config_path = config_path or self._get_default_config_path()
        self._raw_config: dict = {}
        self._config: dict = {}

        self._load_config()

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Try relative to this file first
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config" / "default.json"

        if config_path.exists():
            return str(config_path)

        # Fallback to /app/src/config (Docker)
        docker_path = Path("/app/src/config/default.json")
        if docker_path.exists():
            return str(docker_path)

        # Last resort
        return str(config_path)

    def _load_config(self) -> None:
        """Load and process the configuration file."""
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._raw_config = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Configuration file not found: {self._config_path}")
            print("    Using empty configuration with environment defaults")
            self._raw_config = {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

        # Process the configuration, resolving environment variables
        self._config = self._process_config(self._raw_config)

    def _process_config(self, config: dict, defaults_key: str = "defaults") -> dict:
        """
        Process configuration recursively, resolving environment variables.

        Args:
            config: Configuration dictionary to process
            defaults_key: Key name for default values

        Returns:
            Processed configuration with resolved values
        """
        processed = {}
        defaults = config.get(defaults_key, {})

        for key, value in config.items():
            # Skip metadata and defaults
            if key.startswith("_") or key == defaults_key or key == "validation":
                continue

            if isinstance(value, dict):
                # Recurse into nested dictionaries
                processed[key] = self._process_config(value)
            elif isinstance(value, str):
                # Resolve environment variable or use default
                resolved = self._resolve_value(value, defaults.get(key))
                processed[key] = resolved
            else:
                processed[key] = value

        return processed

    def _resolve_value(self, value: str, default: Any = None) -> Any:
        """
        Resolve a configuration value, checking for environment variables.

        Args:
            value: The value string, potentially containing ${ENV_VAR}
            default: Default value if environment variable not set

        Returns:
            Resolved value (from env var, default, or original)
        """
        match = self.ENV_PATTERN.match(value)

        if match:
            env_var = match.group(1)
            env_value = os.environ.get(env_var)

            if env_value is not None:
                # Try to convert to appropriate type based on default
                return self._convert_type(env_value, default)
            elif default is not None:
                return default
            else:
                return None

        return value

    def _convert_type(self, value: str, reference: Any) -> Any:
        """
        Convert a string value to the appropriate type based on reference.

        Args:
            value: String value to convert
            reference: Reference value for type inference

        Returns:
            Converted value
        """
        if reference is None:
            return value

        if isinstance(reference, bool):
            return value.lower() in ("true", "1", "yes", "on")
        elif isinstance(reference, int):
            try:
                return int(value)
            except ValueError:
                return value
        elif isinstance(reference, float):
            try:
                return float(value)
            except ValueError:
                return value

        return value

    @property
    def config(self) -> dict:
        """Get the processed configuration dictionary."""
        return self._config

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.

        Args:
            *keys: Key path (e.g., "api", "port")
            default: Default value if not found

        Returns:
            Configuration value or default

        Example:
            port = config_manager.get("api", "port", default=30885)
        """
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default

            if value is None:
                return default

        return value

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
