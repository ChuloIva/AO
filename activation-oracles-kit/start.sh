#!/bin/bash
set -e

echo "========================================"
echo "Activation Oracles Kit - RunPod Startup"
echo "========================================"

# Pull latest changes from GitHub (optional, for development)
cd /workspace/AO
echo "Pulling latest changes from GitHub..."
git pull origin main || echo "Could not pull latest changes (using cached version)"

# Navigate to the kit directory
cd /workspace/AO/activation-oracles-kit

# Start TUI app via ttyd web terminal
echo "Starting TUI on port 7681..."
echo "Access via: HTTP Service [Port 7681]"
echo "========================================"
exec ttyd -p 7681 -W \
    -t fontSize=14 \
    -t theme='{"background": "#1a1a2e", "foreground": "#eaeaea"}' \
    python tui_app.py
