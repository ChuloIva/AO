# RunPod Deployment Guide

This guide shows you how to deploy the Activation Oracles Kit on RunPod with automatic setup and GPU support.

## Deployment Options

### Option 1: Docker Hub (Recommended - One-Click Deploy)

This is the easiest method. Build once, deploy anywhere.

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
8. Set **"Expose HTTP Ports"** to: `8501`
9. (Optional) Add environment variable:
   - Key: `OPENROUTER_API_KEY`
   - Value: `your_openrouter_api_key_here`
10. Click **"Deploy"**

#### Step 3: Access Your App

1. Wait for the pod to start (1-2 minutes)
2. Click **"Connect"** → **"HTTP Service [Port 8501]"**
3. Streamlit app will open automatically
4. If you didn't set the API key, enter it in the Setup tab

---

### Option 2: GitHub + Manual Build (For Development)

RunPod can build directly from GitHub if you include the Dockerfile in your repo.

#### Step 1: Push Files to GitHub

```bash
cd activation-oracles-kit
git add Dockerfile start.sh .dockerignore RUNPOD_DEPLOYMENT.md
git commit -m "Add RunPod deployment files"
git push
```

#### Step 2: Deploy from GitHub

1. In RunPod, go to **Serverless** → **"Create Endpoint"**
2. Click **"Import from GitHub"**
3. Connect your GitHub account
4. Select repository: `ChuloIva/AO`
5. Set **Branch**: `main`
6. Set **Dockerfile Path**: `activation-oracles-kit/Dockerfile`
7. Configure GPU and workers
8. Click **"Deploy"**

---

### Option 3: Manual Setup (For Testing)

Quick and dirty for testing, but not persistent.

```bash
# SSH into your RunPod pod terminal

# Clone the repository
git clone https://github.com/ChuloIva/AO.git
cd AO/activation_oracles
pip install -e .

# Install kit requirements
cd ../activation-oracles-kit
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py --server.port 8501
```

Then connect via HTTP Service [Port 8501].

---

## Environment Variables

Set these in RunPod's environment variables section:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Optional | Your OpenRouter API key (can also enter in UI) |
| `STREAMLIT_SERVER_PORT` | No | Default: 8501 |

---

## Recommended GPU Specs

| Model | VRAM Required | Recommended GPUs |
|-------|---------------|------------------|
| Qwen3-1.7B | 4-6 GB | RTX 4090, A4000, A5000 |
| Qwen3-4B | 6-8 GB | RTX 4090, A5000, A6000 |
| Llama-3.2-1B | 3-4 GB | RTX 4090, A4000 |

All models use 8-bit quantization on RunPod for efficiency.

---

## Troubleshooting

### Port 8501 not accessible
- Check that you exposed port 8501 in pod settings
- Wait 30 seconds after pod starts for Streamlit to initialize
- Check pod logs for errors

### Model fails to load
- Ensure GPU has enough VRAM (12GB minimum recommended)
- Check pod logs: `docker logs <container_id>`
- Try a smaller model (Llama-3.2-1B or Qwen3-1.7B)

### Can't pull from GitHub
- Ensure repository is public or add GitHub credentials
- Check branch name is correct (`main`)

### Out of memory errors
- Use smaller batch sizes in config.yaml
- Switch to a GPU with more VRAM
- Use a smaller model

---

## Cost Optimization

- **Serverless**: Pay per second of inference (best for intermittent use)
- **Pods**: Hourly rate (best for extended sessions)
- **Community Cloud**: Cheaper but less reliable
- **Secure Cloud**: More expensive but enterprise-grade

Stop pods when not in use to avoid charges.

---

## Updating Your Deployment

The `start.sh` script automatically pulls the latest code from GitHub on each restart.

To force an update:
1. Restart your pod
2. Or SSH in and run: `cd /workspace/AO && git pull && cd activation-oracles-kit`

---

## Need Help?

- RunPod Docs: https://docs.runpod.io
- RunPod Discord: https://discord.gg/runpod
- Issues: https://github.com/ChuloIva/AO/issues
