# RunPod Deployment Dockerfile for Activation Oracles Kit
# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TERM=xterm-256color

# Install system dependencies including ttyd for web terminal
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ttyd \
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

# Expose ttyd web terminal port
EXPOSE 7681

# Set the working directory to the kit
WORKDIR /workspace/AO/activation-oracles-kit

# Run the startup script
CMD ["/workspace/start.sh"]
