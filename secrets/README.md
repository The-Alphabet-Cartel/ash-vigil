# Ash-Vigil Secrets Directory

============================================================================
**The Alphabet Cartel** - https://discord.gg/alphabetcartel | https://alphabetcartel.org
============================================================================

## Overview

This directory contains Docker Secrets for the Ash-Vigil service. Secrets are sensitive values that should never be committed to version control.

## ⚠️ SECURITY WARNING

- **NEVER** commit actual secret files to Git
- Only `README.md` and `.gitkeep` should be tracked
- All other files in this directory are ignored via `.gitignore`

## Available Secrets

| Secret File | Required | Description |
|-------------|----------|-------------|
| `discord_alert_token` | Optional | Discord webhook URL for system alerts |

## Setup Instructions

### 1. Create Secret Files

```bash
# Navigate to secrets directory
cd secrets/

# Create the optional Discord alert webhook
echo "https://discord.com/api/webhooks/your-webhook-url" > discord_alert_token

# Set proper permissions (Linux/macOS)
chmod 600 discord_alert_token
```

### 2. Verify Permissions

Secrets should only be readable by the owner:

```bash
ls -la secrets/
# Should show: -rw------- for secret files
```

### 3. Docker Compose Integration

Secrets are automatically mounted in the container via `docker-compose.yml`:

```yaml
secrets:
  discord_alert_token:
    file: ./secrets/discord_alert_token
```

Inside the container, secrets are available at `/run/secrets/<secret_name>`.

## Rotation

To rotate a secret:

1. Update the secret file with the new value
2. Restart the container: `docker compose restart ash-vigil`

## Troubleshooting

### Permission Denied
```bash
chmod 600 secrets/*
```

### Secret Not Found in Container
Ensure the secret is defined in both the `secrets:` section and the service's `secrets:` list in `docker-compose.yml`.

---

**Built with care for chosen family** 🏳️‍🌈
