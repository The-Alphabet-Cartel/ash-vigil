#!/usr/bin/env python3
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
This is the main entry point for the Ash-Vigil service.
----------------------------------------------------------------------------
FILE VERSION: v5.0-1-1.0-1
LAST MODIFIED: 2026-01-24
PHASE: Phase 1 - {Phase Description}
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
It initializes the FastAPI application and starts the uvicorn server.

Usage:
    python main.py

Or via Docker:
    docker-compose up -d
"""

import uvicorn

from src.managers.config_manager import ConfigManager
from src.managers.logging_config_manager import LoggingConfigManager


def main():
    """Main entry point for Ash-Vigil service."""
    # Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.config

    # Initialize logging
    logging_manager = LoggingConfigManager(config)
    logger = logging_manager.get_logger(__name__)

    # Get API configuration
    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 30890)

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  🛡️  Ash-Vigil - Mental Health Risk Detection Service")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Start the server
    uvicorn.run(
        "src.api.app:app", host=host, port=port, log_level="info", access_log=True
    )


if __name__ == "__main__":
    main()
