# RunPod Deployment Dockerfile for Activation Oracles Kit
# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone the repository
RUN git clone https://github.com/ChuloIva/AO.git && \
    cd AO

# Install activation_oracles as editable package
WORKDIR /workspace/AO/activation_oracles
RUN pip install --no-cache-dir -e .

# Install activation-oracles-kit requirements
WORKDIR /workspace/AO/activation-oracles-kit
RUN pip install --no-cache-dir -r requirements.txt

# Copy startup script
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Expose Streamlit port
EXPOSE 8501

# Set the working directory to the kit
WORKDIR /workspace/AO/activation-oracles-kit

# Run the startup script
CMD ["/workspace/start.sh"]
