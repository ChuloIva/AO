# RunPod Deployment Guide

This guide shows you how to deploy the Activation Oracles TUI on RunPod with automatic setup and GPU support.

## What You Get

A browser-based terminal running the Token Oracle Chat TUI:
- Select tasks and generate multi-persona responses
- Browse tokens with keyboard navigation
- Query the oracle about model activations
- Full terminal experience in your browser via ttyd

---

## Deployment Options

### Option 1: Docker Hub (Recommended - One-Click Deploy)

Build once, deploy anywhere.

#### Step 1: Build and Push Docker Image

```bash
# Navigate to the activation-oracles-kit directory
cd activation-oracles-kit

# Build for Linux/AMD64 (required for RunPod)
docker build --platform linux/amd64 -t yourusername/activation-oracles:latest .

# Push to Docker Hub (requires docker login)
docker push yourusername/activation-oracles:latest
```

#### Step 2: Deploy on RunPod

1. Go to [RunPod.io](https://runpod.io) and sign in
2. Click **"Deploy"** → **"Pods"**
3. Select **"GPU Cloud"** or **"Community Cloud"**
4. Choose a GPU (Recommended: RTX 4090, A4000, or higher with 16GB+ VRAM)
5. Click **"Customize Deployment"**
6. Under **"Docker Image"**, enter: `yourusername/activation-oracles:latest`
7. Set **"Docker Command"** to: `/workspace/start.sh`
8. Set **"Expose HTTP Ports"** to: `7681`
9. Click **"Deploy"**

#### Step 3: Access Your App

1. Wait for the pod to start (1-2 minutes for model loading)
2. Click **"Connect"** → **"HTTP Service [Port 7681]"**
3. The TUI opens in your browser - use keyboard to navigate!

---

### Option 2: Manual Setup (For Testing)

Quick setup for testing, but not persistent.

```bash
# SSH into your RunPod pod terminal

# Clone the repository
git clone https://github.com/ChuloIva/AO.git
cd AO/activation_oracles
pip install -e .

# Install kit requirements
cd ../activation-oracles-kit
pip install -r requirements.txt

# Install ttyd
apt-get update && apt-get install -y ttyd

# Run TUI via ttyd
ttyd -p 7681 -W python tui_app.py
```

Then connect via HTTP Service [Port 7681].

---

## TUI Controls

Once connected, use these keyboard shortcuts:

### Token Browser
| Key | Action |
|-----|--------|
| `←` `→` or `h` `l` | Navigate tokens |
| `↑` `↓` or `j` `k` | Jump by line |
| `Space` | Select/deselect token |
| `1` `2` `3` | Toggle layers 10, 18, 25 |
| `Q` | Go to oracle chat |
| `N` | New task |
| `ESC` | Quit |

### Oracle Chat
| Command | Action |
|---------|--------|
| `/t` | Back to token browser |
| `/c` | Clear chat history |
| `/n` | New task |
| `/q` | Quit |
| `1-4` | Preset questions |

---

## Recommended GPU Specs

| Model | VRAM Required | Recommended GPUs |
|-------|---------------|------------------|
| Qwen3-4B | 8-12 GB | RTX 4090, A5000, A6000 |

The model runs in full precision for best oracle accuracy.

---

## Troubleshooting

### Port 7681 not accessible
- Check that you exposed port 7681 in pod settings
- Wait 1-2 minutes after pod starts for model to load
- Check pod logs for errors

### Keyboard not working in browser
- Try refreshing the page
- Use Chrome or Firefox (Safari may have issues)
- Check that you clicked inside the terminal window

### Model fails to load
- Ensure GPU has enough VRAM (12GB minimum recommended)
- Check pod logs: `docker logs <container_id>`

### Terminal looks broken
- Resize browser window to refresh layout
- Try a different browser

---

## Cost Optimization

- **Pods**: Hourly rate (best for extended sessions)
- **Community Cloud**: Cheaper but less reliable
- **Secure Cloud**: More expensive but enterprise-grade

**Stop pods when not in use to avoid charges.**

---

## Updating Your Deployment

The `start.sh` script automatically pulls the latest code from GitHub on each restart.

To force an update:
1. Restart your pod
2. Or SSH in and run: `cd /workspace/AO && git pull`

---

## Local Development

Run locally with the same web terminal experience:

```bash
# Install ttyd (macOS)
brew install ttyd

# Run
./run_tui_web.sh
```

Opens browser to `http://localhost:7681`

---

## Need Help?

- RunPod Docs: https://docs.runpod.io
- RunPod Discord: https://discord.gg/runpod
- Issues: https://github.com/ChuloIva/AO/issues
