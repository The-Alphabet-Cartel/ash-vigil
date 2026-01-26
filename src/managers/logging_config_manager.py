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
Logging Configuration Manager - Standardized colorized logging (Charter v5.2.3)
----------------------------------------------------------------------------
FILE VERSION: v5.0-1-1.2-1
LAST MODIFIED: 2026-01-26
PHASE: Phase 1 - Service Completion
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
"""

import logging
import os
import sys
from typing import Any, Dict, Optional


# =============================================================================
# Custom SUCCESS Log Level
# =============================================================================

SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def _success_method(self, message, *args, **kwargs):
    """Log a SUCCESS level message."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Add success method to Logger class
logging.Logger.success = _success_method


# =============================================================================
# ANSI Color Codes (Charter v5.2.3 Compliant)
# =============================================================================


class Colors:
    """ANSI color codes for log formatting per Charter v5.2.3 Rule #9."""

    RESET = "\033[0m"

    # Log level colors (Charter v5.2.3 standard)
    CRITICAL = "\033[1;91m"  # Bright Red (Bold)
    ERROR = "\033[91m"       # Red
    WARNING = "\033[93m"     # Yellow
    SUCCESS = "\033[92m"     # Green
    INFO = "\033[96m"        # Cyan
    DEBUG = "\033[90m"       # Gray

    # Additional formatting
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Emoji symbols for visual identification (Charter v5.2.3 standard)
    SYMBOLS = {
        "CRITICAL": "🚨",
        "ERROR": "❌",
        "WARNING": "⚠️",
        "SUCCESS": "✅",
        "INFO": "ℹ️",
        "DEBUG": "🔍",
    }


# =============================================================================
# Color Formatter
# =============================================================================


class ColorFormatter(logging.Formatter):
    """
    Logging formatter with ANSI color support.

    Applies colors based on log level following Charter v5.2.3 color scheme.
    """

    LEVEL_COLORS = {
        logging.CRITICAL: Colors.CRITICAL,
        logging.ERROR: Colors.ERROR,
        logging.WARNING: Colors.WARNING,
        SUCCESS_LEVEL: Colors.SUCCESS,
        logging.INFO: Colors.INFO,
        logging.DEBUG: Colors.DEBUG,
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
        use_symbols: bool = True,
    ):
        """
        Initialize the color formatter.

        Args:
            fmt: Log format string
            datefmt: Date format string
            use_colors: Whether to apply ANSI colors
            use_symbols: Whether to include emoji symbols
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
        self.use_symbols = use_symbols

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Get the color for this level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)

        # Add symbol if enabled
        if self.use_symbols:
            symbol = Colors.SYMBOLS.get(record.levelname, "")
            record.symbol = f"{symbol} " if symbol else ""
        else:
            record.symbol = ""

        # Format the message
        formatted = super().format(record)

        # Apply colors if enabled
        if self.use_colors:
            formatted = f"{color}{formatted}{Colors.RESET}"

        return formatted


# =============================================================================
# Logging Configuration Manager
# =============================================================================


class LoggingConfigManager:
    """
    Logging configuration manager for Ash-Vigil.

    Configures logging with ANSI colorization following Charter v5.2.3 Rule #9.

    Required Color Scheme:
    - CRITICAL: Bright Red Bold (🚨)
    - ERROR: Red (❌)
    - WARNING: Yellow (⚠️)
    - SUCCESS: Green (✅) - Custom level 25
    - INFO: Cyan (ℹ️)
    - DEBUG: Gray (🔍)

    Usage:
        logging_manager = create_logging_config_manager(config)
        logger = logging_manager.get_logger(__name__)
        logger.info("Starting up...")
        logger.success("Operation completed!")
    """

    # Default log format (Charter v5.2.3 compliant)
    DEFAULT_FORMAT = "[%(asctime)s] %(levelname)-8s | %(name)-30s | %(symbol)s%(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        log_level: str = "INFO",
        log_format: str = "human",
        log_file: Optional[str] = None,
        console_enabled: bool = True,
        app_name: str = "ash-vigil",
    ):
        """
        Initialize logging configuration.

        Args:
            config: Configuration dictionary with logging settings (optional)
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Log format (human or json)
            log_file: Path to log file (optional)
            console_enabled: Whether to enable console output
            app_name: Application name for logger hierarchy
        """
        self._config = config or {}
        self._app_name = app_name
        self._configured = False

        # Extract settings from config or use parameters
        logging_config = self._config.get("logging", {})
        self._log_level = logging_config.get("level", log_level).upper()
        self._log_format = logging_config.get("format", log_format)
        self._log_file = logging_config.get("file", log_file)
        self._console_enabled = logging_config.get("console", console_enabled)
        self._colorize = logging_config.get("colorize", True)

        self._configure_logging()

    def _should_colorize(self) -> bool:
        """
        Determine if output should be colorized.

        Returns True if:
        - FORCE_COLOR environment variable is set, OR
        - stdout is a TTY (terminal)
        """
        # Check explicit colorize setting
        if not self._colorize:
            return False

        # FORCE_COLOR takes precedence (for Docker containers)
        force_color = os.environ.get("FORCE_COLOR", "").lower()
        if force_color in ("1", "true", "yes"):
            return True

        # Check if stdout is a TTY
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def _configure_logging(self) -> None:
        """Configure the logging system."""
        if self._configured:
            return

        # Determine if we should colorize
        use_colors = self._should_colorize()

        # Convert level name to level number
        level = getattr(logging, self._log_level, logging.INFO)

        # Create formatter
        if self._log_format == "json":
            # JSON format (no colors)
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}',
                datefmt=self.DEFAULT_DATE_FORMAT,
            )
        else:
            # Human format with optional colors (Charter v5.2.3 compliant)
            formatter = ColorFormatter(
                fmt=self.DEFAULT_FORMAT,
                datefmt=self.DEFAULT_DATE_FORMAT,
                use_colors=use_colors,
                use_symbols=use_colors,
            )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler if enabled
        if self._console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # Add file handler if configured
        if self._log_file:
            try:
                file_handler = logging.FileHandler(self._log_file)
                file_handler.setLevel(level)
                # File logs always use non-colored format
                file_formatter = logging.Formatter(
                    "[%(asctime)s] %(levelname)-8s | %(name)-30s | %(message)s",
                    datefmt=self.DEFAULT_DATE_FORMAT,
                )
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
            except (IOError, OSError) as e:
                print(f"⚠️  Could not create log file {self._log_file}: {e}")

        # Reduce noise from third-party libraries
        noisy_loggers = [
            "uvicorn",
            "uvicorn.access",
            "uvicorn.error",
            "transformers",
            "torch",
            "httpx",
            "httpcore",
            "asyncio",
        ]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        self._configured = True

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Configured logger instance with success() method
        """
        # Create hierarchical logger name
        if not name.startswith(self._app_name):
            full_name = f"{self._app_name}.{name}"
        else:
            full_name = name

        return logging.getLogger(full_name)

    @property
    def log_level(self) -> str:
        """Get the configured log level."""
        return self._log_level

    @property
    def is_colorized(self) -> bool:
        """Check if output is colorized."""
        return self._should_colorize()


# =============================================================================
# Factory Function
# =============================================================================


def create_logging_config_manager(
    config: Optional[Dict[str, Any]] = None,
    log_level: str = "INFO",
    log_format: str = "human",
    log_file: Optional[str] = None,
    console_enabled: bool = True,
    app_name: str = "ash-vigil",
) -> LoggingConfigManager:
    """
    Factory function to create a LoggingConfigManager instance.

    Following Clean Architecture Rule #1: Factory Functions.

    Args:
        config: Configuration dictionary (optional)
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (human or json)
        log_file: Path to log file (optional)
        console_enabled: Whether to enable console output
        app_name: Application name for logger hierarchy

    Returns:
        Configured LoggingConfigManager instance

    Example:
        >>> logging_manager = create_logging_config_manager(config)
        >>> logger = logging_manager.get_logger(__name__)
        >>> logger.info("Starting up...")
        >>> logger.success("Operation completed!")
    """
    return LoggingConfigManager(
        config=config,
        log_level=log_level,
        log_format=log_format,
        log_file=log_file,
        console_enabled=console_enabled,
        app_name=app_name,
    )


# =============================================================================
# Export public interface
# =============================================================================

__all__ = [
    "LoggingConfigManager",
    "create_logging_config_manager",
    "SUCCESS_LEVEL",
    "Colors",
    "ColorFormatter",
]
