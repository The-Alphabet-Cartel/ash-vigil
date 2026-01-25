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
Docker Entrypoint for Ash-Vigil Service
----------------------------------------------------------------------------
FILE VERSION: v5.0-1-1.0-2
LAST MODIFIED: 2026-01-24
PHASE: Phase 1 - Skeleton Setup
CLEAN ARCHITECTURE: Rule #13 - Standardized Docker Entrypoint
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
DESCRIPTION:
    Python-based Docker entrypoint that:
    1. Sets up user/group based on PUID/PGID environment variables
    2. Fixes ownership of application directories
    3. Drops privileges to the configured user
    4. Starts the FastAPI server via uvicorn

    This approach follows the project's "No Bash Scripting" philosophy
    while enabling LinuxServer.io-style user configuration.

USAGE:
    # Called automatically by Docker
    # Or manually:
    python docker-entrypoint.py

    # With custom PUID/PGID:
    PUID=1000 PGID=1000 python docker-entrypoint.py
============================================================================
"""

import grp
import os
import pwd
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# =============================================================================
# COMPONENT CONFIGURATION
# =============================================================================
COMPONENT_NAME = "ash-vigil"
COMPONENT_EMOJI = "👁️"

# Default user/group (should match Dockerfile ARG defaults)
DEFAULT_UID = 1000
DEFAULT_GID = 1000
APP_USER = "ash-vigil"
APP_GROUP = "ash-vigil"

# Application directory
APP_HOME = Path("/app")

# Directories that need write access
WRITABLE_DIRECTORIES = [
    "/app/logs",
    "/app/models-cache",
]

# Default command if none provided
DEFAULT_COMMAND = [
    "python",
    "-m",
    "uvicorn",
    "src.api.app:app",
    "--host",
    os.environ.get("VIGIL_API_HOST", "0.0.0.0"),
    "--port",
    os.environ.get("VIGIL_API_PORT", "30882"),
]

# =============================================================================
__version__ = "v5.0-1-1.0-2"


# =============================================================================
# Colorized Logging
# =============================================================================
class Colors:
    """ANSI escape codes for Charter v5.2 compliant colorization."""

    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CRITICAL = "\033[1;91m"  # Bright Red Bold
    ERROR = "\033[91m"  # Bright Red
    WARNING = "\033[93m"  # Bright Yellow
    INFO = "\033[96m"  # Bright Cyan
    DEBUG = "\033[90m"  # Gray
    SUCCESS = "\033[92m"  # Bright Green
    TIMESTAMP = "\033[90m"  # Gray
    HEADER = "\033[95m"  # Magenta


def _should_use_colors() -> bool:
    """Check if colors should be used based on FORCE_COLOR or TTY."""
    force_color = os.environ.get("FORCE_COLOR", "").lower() in ("1", "true", "yes")
    return force_color or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())


_USE_COLORS = _should_use_colors()


def _format_log(level: str, message: str, color: str) -> str:
    """Format a log message with Charter v5.2 colorization."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if _USE_COLORS:
        return (
            f"{Colors.TIMESTAMP}[{timestamp}]{Colors.RESET} "
            f"{color}{level.ljust(8)}{Colors.RESET} "
            f"{Colors.DIM}|{Colors.RESET} "
            f"{color}{message}{Colors.RESET}"
        )
    return f"[{timestamp}] {level.ljust(8)} | {message}"


def log_info(message: str) -> None:
    """Log an info message."""
    print(_format_log("INFO", message, Colors.INFO), flush=True)


def log_success(message: str) -> None:
    """Log a success message."""
    print(_format_log("SUCCESS", message, Colors.SUCCESS), flush=True)


def log_warning(message: str) -> None:
    """Log a warning message."""
    print(_format_log("WARNING", message, Colors.WARNING), flush=True)


def log_error(message: str) -> None:
    """Log an error message."""
    print(_format_log("ERROR", message, Colors.ERROR), file=sys.stderr, flush=True)


def print_startup_banner() -> None:
    """Print the ASCII art startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                       ║
║       █████╗ ███████╗██╗  ██╗        ██╗   ██╗██╗ ██████╗ ██╗██╗                      ║
║      ██╔══██╗██╔════╝██║  ██║        ██║   ██║██║██╔════╝ ██║██║                      ║
║      ███████║███████╗███████║ █████╗ ██║   ██║██║██║  ███╗██║██║                      ║
║      ██╔══██║╚════██║██╔══██║ ╚════╝ ╚██╗ ██╔╝██║██║   ██║██║██║                      ║
║      ██║  ██║███████║██║  ██║         ╚████╔╝ ██║╚██████╔╝██║███████╗                 ║
║      ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝          ╚═══╝  ╚═╝ ╚═════╝ ╚═╝╚══════╝                 ║
║                                                                                       ║
║                      Mental Health Risk Detection Service v5.0                        ║
║                                                                                       ║
║                   The Alphabet Cartel - https://discord.gg/alphabetcartel             ║
║                                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
"""
    if _USE_COLORS:
        print(f"{Colors.HEADER}{banner}{Colors.RESET}")
    else:
        print(banner)


def print_header() -> None:
    """Print the startup header."""
    border = "━" * 60
    if _USE_COLORS:
        print(f"\n{Colors.HEADER}{border}{Colors.RESET}")
        print(
            f"{Colors.HEADER}  {COMPONENT_EMOJI} {COMPONENT_NAME.upper()} Container Entrypoint{Colors.RESET}"
        )
        print(f"{Colors.HEADER}{border}{Colors.RESET}\n")
    else:
        print(f"\n{border}")
        print(f"  {COMPONENT_EMOJI} {COMPONENT_NAME.upper()} Container Entrypoint")
        print(f"{border}\n")


# =============================================================================
# User/Group Management
# =============================================================================
def get_puid_pgid() -> Tuple[int, int]:
    """Get PUID and PGID from environment variables with validation."""
    puid_str = os.environ.get("PUID", str(DEFAULT_UID))
    pgid_str = os.environ.get("PGID", str(DEFAULT_GID))

    try:
        puid = int(puid_str)
    except ValueError:
        log_warning(f"Invalid PUID '{puid_str}', using default {DEFAULT_UID}")
        puid = DEFAULT_UID

    try:
        pgid = int(pgid_str)
    except ValueError:
        log_warning(f"Invalid PGID '{pgid_str}', using default {DEFAULT_GID}")
        pgid = DEFAULT_GID

    # Validate ranges
    if not 0 <= puid <= 65534:
        log_warning(f"PUID {puid} out of range, using default {DEFAULT_UID}")
        puid = DEFAULT_UID

    if not 0 <= pgid <= 65534:
        log_warning(f"PGID {pgid} out of range, using default {DEFAULT_GID}")
        pgid = DEFAULT_GID

    return puid, pgid


def is_root() -> bool:
    """Check if the current process is running as root."""
    return os.geteuid() == 0


def get_user_info(username: str) -> Tuple[Optional[int], Optional[int]]:
    """Get current UID and GID for a user."""
    try:
        user_info = pwd.getpwnam(username)
        return user_info.pw_uid, user_info.pw_gid
    except KeyError:
        return None, None


def get_group_gid(groupname: str) -> Optional[int]:
    """Get GID for a group by name."""
    try:
        group_info = grp.getgrnam(groupname)
        return group_info.gr_gid
    except KeyError:
        return None


def create_or_modify_group(groupname: str, gid: int) -> bool:
    """Create a group or modify existing group's GID."""
    current_gid = get_group_gid(groupname)

    if current_gid is None:
        try:
            subprocess.run(
                ["groupadd", "-o", "--gid", str(gid), groupname],
                check=True,
                capture_output=True,
            )
            log_success(f"Created group '{groupname}' with GID {gid}")
            return True
        except subprocess.CalledProcessError as e:
            log_error(f"Failed to create group: {e.stderr.decode() if e.stderr else e}")
            return False
    elif current_gid != gid:
        try:
            subprocess.run(
                ["groupmod", "-o", "-g", str(gid), groupname],
                check=True,
                capture_output=True,
            )
            log_success(f"Modified group '{groupname}' GID: {current_gid} → {gid}")
            return True
        except subprocess.CalledProcessError as e:
            log_error(f"Failed to modify group: {e.stderr.decode() if e.stderr else e}")
            return False
    else:
        log_info(f"Group '{groupname}' already has GID {gid}")
        return True


def create_or_modify_user(username: str, uid: int, gid: int) -> bool:
    """Create a user or modify existing user's UID/GID."""
    current_uid, current_gid = get_user_info(username)

    if current_uid is None:
        try:
            subprocess.run(
                [
                    "useradd",
                    "--uid",
                    str(uid),
                    "--gid",
                    str(gid),
                    "--shell",
                    "/bin/bash",
                    "--create-home",
                    "--no-log-init",
                    username,
                ],
                check=True,
                capture_output=True,
            )
            log_success(f"Created user '{username}' with UID {uid}")
            return True
        except subprocess.CalledProcessError as e:
            log_error(f"Failed to create user: {e.stderr.decode() if e.stderr else e}")
            return False
    else:
        needs_update = current_uid != uid or current_gid != gid
        if needs_update:
            try:
                subprocess.run(
                    ["usermod", "-o", "-u", str(uid), "-g", str(gid), username],
                    check=True,
                    capture_output=True,
                )
                log_success(
                    f"Modified user '{username}': UID {current_uid}→{uid}, GID {current_gid}→{gid}"
                )
                return True
            except subprocess.CalledProcessError as e:
                log_error(
                    f"Failed to modify user: {e.stderr.decode() if e.stderr else e}"
                )
                return False
        else:
            log_info(f"User '{username}' already has UID {uid}, GID {gid}")
            return True


# =============================================================================
# Permission Management
# =============================================================================
def fix_ownership(uid: int, gid: int, directories: Optional[List[str]] = None) -> None:
    """Fix ownership of application directories."""
    dirs_to_fix = directories or WRITABLE_DIRECTORIES

    if not dirs_to_fix:
        return

    log_info("Fixing directory ownership...")

    for dir_path in dirs_to_fix:
        path = Path(dir_path)
        try:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                log_info(f"  Created: {dir_path}")

            for item in path.rglob("*"):
                try:
                    os.chown(item, uid, gid)
                except PermissionError:
                    pass
            os.chown(path, uid, gid)
            log_success(f"  ✓ {dir_path}")

        except Exception as e:
            log_warning(f"  Could not fix {dir_path}: {e}")


# =============================================================================
# Privilege Management
# =============================================================================
def setup_user_and_permissions(puid: int, pgid: int) -> bool:
    """Main setup function - creates user/group and fixes permissions."""
    log_info(f"PUID: {puid}")
    log_info(f"PGID: {pgid}")

    if not is_root():
        log_warning("Not running as root - skipping user/group setup")
        log_info(f"Running as UID={os.getuid()}, GID={os.getgid()}")
        return True

    if not create_or_modify_group(APP_GROUP, pgid):
        return False

    if not create_or_modify_user(APP_USER, puid, pgid):
        return False

    fix_ownership(puid, pgid)

    log_success("User and permissions configured")
    return True


def drop_privileges(puid: int, pgid: int) -> None:
    """Drop privileges from root to the specified user/group."""
    if not is_root():
        return

    try:
        try:
            user_info = pwd.getpwnam(APP_USER)
            home_dir = user_info.pw_dir
        except KeyError:
            home_dir = str(APP_HOME)

        os.environ["HOME"] = home_dir
        os.environ["USER"] = APP_USER
        os.environ["LOGNAME"] = APP_USER

        os.setgroups([])
        os.setgid(pgid)
        os.setuid(puid)

        log_success(f"Dropped privileges to UID={os.getuid()}, GID={os.getgid()}")

    except Exception as e:
        log_error(f"Failed to drop privileges: {e}")
        raise


# =============================================================================
# Application Execution
# =============================================================================
def execute_command(command: List[str]) -> None:
    """Execute the application command (replaces current process)."""
    log_info("Starting Ash-Vigil...")
    log_info(f"Command: {' '.join(command)}")
    if _USE_COLORS:
        print(f"{Colors.HEADER}{'━' * 60}{Colors.RESET}\n")
    else:
        print(f"{'━' * 60}\n")

    os.execvp(command[0], command)


# =============================================================================
# Main Entry Point
# =============================================================================
def main() -> int:
    """Main entrypoint function."""
    print_startup_banner()
    print_header()

    puid, pgid = get_puid_pgid()

    if not setup_user_and_permissions(puid, pgid):
        log_error("Failed to setup user - exiting")
        return 1

    drop_privileges(puid, pgid)

    if len(sys.argv) > 1:
        command = sys.argv[1:]
    else:
        command = DEFAULT_COMMAND

    execute_command(command)

    return 0


if __name__ == "__main__":
    sys.exit(main())
