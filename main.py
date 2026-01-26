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
Main Entry Point - Initializes and starts the Ash-Vigil FastAPI service
----------------------------------------------------------------------------
FILE VERSION: v5.0-1-1.2-1
LAST MODIFIED: 2026-01-26
PHASE: Phase 1 - Service Completion
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================

Usage:
    python main.py

Or via Docker:
    docker-compose up -d
"""

import uvicorn

from src.managers import create_config_manager, create_logging_config_manager


def main():
    """Main entry point for Ash-Vigil service."""
    # Initialize configuration using factory function
    config_manager = create_config_manager()
    config = config_manager.config

    # Initialize logging using factory function
    logging_manager = create_logging_config_manager(config)
    logger = logging_manager.get_logger(__name__)

    # Get API configuration
    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 30882)

    # Print startup info
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  🛡️  Ash-Vigil - Mental Health Risk Detection Service")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Start the server
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        log_level="warning",  # Reduce uvicorn noise, we have our own logging
        access_log=False,
    )


if __name__ == "__main__":
    main()
