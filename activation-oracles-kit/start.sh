#!/bin/bash
set -e

echo "========================================"
echo "Activation Oracles Kit - RunPod Startup"
echo "========================================"

# Pull latest changes from GitHub (optional, for development)
cd /workspace/AO
echo "Pulling latest changes from GitHub..."
git pull origin main || echo "Could not pull latest changes (using cached version)"

# Check if OpenRouter API key is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "WARNING: OPENROUTER_API_KEY environment variable not set!"
    echo "You'll need to enter it manually in the Setup tab."
else
    echo "OpenRouter API key detected."
fi

# Navigate to the kit directory
cd /workspace/AO/activation-oracles-kit

# Start Streamlit
echo "Starting Streamlit on port 8501..."
echo "========================================"
streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
