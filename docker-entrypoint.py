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
Pure Python entrypoint for Docker container initialization.
----------------------------------------------------------------------------
FILE VERSION: v5.0-1-1.0-1
LAST MODIFIED: 2026-01-24
PHASE: Phase 1 - {Phase Description}
CLEAN ARCHITECTURE: Compliant
Repository: https://github.com/the-alphabet-cartel/ash-vigil
============================================================================
Handles PUID/PGID user switching (LinuxServer.io style) without bash scripting.

This script:
1. Reads PUID/PGID from environment (defaults to 1000)
2. Modifies the 'vigil' user/group to match
3. Fixes ownership of writable directories
4. Drops privileges and executes the main application

Usage with tini in Dockerfile:
    ENTRYPOINT ["/usr/bin/tini", "--", "python", "/app/docker-entrypoint.py"]
"""

import os
import pwd
import grp
import subprocess
import sys


def get_env_int(name: str, default: int) -> int:
    """Get integer environment variable with default."""
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def user_exists(username: str) -> bool:
    """Check if a user exists."""
    try:
        pwd.getpwnam(username)
        return True
    except KeyError:
        return False


def group_exists(groupname: str) -> bool:
    """Check if a group exists."""
    try:
        grp.getgrnam(groupname)
        return True
    except KeyError:
        return False


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and optionally check for errors."""
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def main():
    """Main entrypoint logic."""
    # Get target PUID/PGID from environment
    target_uid = get_env_int("PUID", 1000)
    target_gid = get_env_int("PGID", 1000)

    username = "vigil"
    groupname = "vigil"

    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Ash-Vigil Container Initialization")
    print(f"  Target UID: {target_uid} | Target GID: {target_gid}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Check if running as root (required for user modification)
    if os.geteuid() != 0:
        print("⚠️  Not running as root - skipping user modification")
        print("   Container will run as current user")
    else:
        # Modify group if it exists and GID differs
        if group_exists(groupname):
            current_gid = grp.getgrnam(groupname).gr_gid
            if current_gid != target_gid:
                print(
                    f"📝 Modifying group '{groupname}' GID: {current_gid} → {target_gid}"
                )
                run_command(["groupmod", "-g", str(target_gid), groupname])

        # Modify user if it exists and UID differs
        if user_exists(username):
            current_uid = pwd.getpwnam(username).pw_uid
            if current_uid != target_uid:
                print(
                    f"📝 Modifying user '{username}' UID: {current_uid} → {target_uid}"
                )
                run_command(["usermod", "-u", str(target_uid), username])

        # Fix ownership of writable directories
        writable_dirs = ["/app/logs", "/app/models-cache"]
        for directory in writable_dirs:
            if os.path.exists(directory):
                print(f"📁 Fixing ownership: {directory}")
                run_command(["chown", "-R", f"{target_uid}:{target_gid}", directory])

        # Drop privileges and execute main application
        print(f"🔐 Dropping privileges to {username} ({target_uid}:{target_gid})")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Set environment for the new user
        os.environ["HOME"] = f"/app"
        os.environ["USER"] = username

        # Change to target group and user
        os.setgid(target_gid)
        os.setuid(target_uid)

    # Execute the main application
    print("🚀 Starting Ash-Vigil...")
    os.chdir("/app")

    # Replace current process with the main application
    os.execvp(
        "python",
        [
            "python",
            "-m",
            "uvicorn",
            "src.api.app:app",
            "--host",
            os.environ.get("VIGIL_API_HOST", "0.0.0.0"),
            "--port",
            os.environ.get("VIGIL_API_PORT", "30882"),
        ],
    )


if __name__ == "__main__":
    main()
