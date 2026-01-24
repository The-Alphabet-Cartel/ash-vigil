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
Standardized logging following Clean Architecture Charter.
----------------------------------------------------------------------------
FILE VERSION: v5.0-1-1.0-1
LAST MODIFIED: 2026-01-24
PHASE: Phase 1 - {Phase Description}
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
Provides ANSI colorized output with custom SUCCESS level.
"""

import logging
import os
import sys
from typing import Optional


# =============================================================================
# Custom SUCCESS Log Level
# =============================================================================

SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def success(self, message, *args, **kwargs):
    """Log a SUCCESS level message."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Add success method to Logger class
logging.Logger.success = success


# =============================================================================
# ANSI Color Codes (Charter v5.2 Compliant)
# =============================================================================


class Colors:
    """ANSI color codes for log formatting."""

    RESET = "\033[0m"

    # Log level colors (Charter v5.2 Rule #9)
    CRITICAL = "\033[1;91m"  # Bright Red (Bold)
    ERROR = "\033[91m"  # Red
    WARNING = "\033[93m"  # Yellow
    SUCCESS = "\033[92m"  # Green
    INFO = "\033[96m"  # Cyan
    DEBUG = "\033[90m"  # Gray

    # Additional formatting
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Emoji symbols for visual identification
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

    Applies colors based on log level following Charter v5.2 color scheme.
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

    Configures logging with ANSI colorization following Charter v5.2.2.

    Usage:
        logging_manager = LoggingConfigManager(config)
        logger = logging_manager.get_logger(__name__)
        logger.info("Starting up...")
        logger.success("Operation completed!")
    """

    # Default log format
    DEFAULT_FORMAT = "%(asctime)s | %(symbol)s%(levelname)-8s | %(name)s | %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, config: dict):
        """
        Initialize logging configuration.

        Args:
            config: Configuration dictionary with logging settings
        """
        self._config = config
        self._configured = False

        self._configure_logging()

    def _should_colorize(self) -> bool:
        """
        Determine if output should be colorized.

        Returns True if:
        - FORCE_COLOR environment variable is set, OR
        - stdout is a TTY (terminal)
        """
        # FORCE_COLOR takes precedence (for Docker containers)
        if os.environ.get("FORCE_COLOR"):
            return True

        # Check if stdout is a TTY
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def _configure_logging(self) -> None:
        """Configure the logging system."""
        if self._configured:
            return

        logging_config = self._config.get("logging", {})

        # Get configuration values
        level_name = logging_config.get("level", "INFO").upper()
        log_format = logging_config.get("format", "standard")
        colorize_config = logging_config.get("colorize", True)

        # Determine if we should colorize
        use_colors = colorize_config and self._should_colorize()

        # Convert level name to level number
        level = getattr(logging, level_name, logging.INFO)

        # Create formatter
        if log_format == "json":
            # JSON format (no colors)
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}',
                datefmt=self.DEFAULT_DATE_FORMAT,
            )
        else:
            # Standard format with optional colors
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

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Reduce noise from third-party libraries
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)

        self._configured = True

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)
