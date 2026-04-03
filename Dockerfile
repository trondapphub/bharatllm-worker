# BharatLLM RunPod Serverless Worker
# ============================================================
# Base: NVIDIA PyTorch with CUDA support
# Includes: Unsloth, PEFT, RunPod SDK
# ============================================================

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    runpod>=1.6.0 \
    huggingface_hub>=0.20.0 \
    peft>=0.8.0 \
    bitsandbytes>=0.43.0 \
    accelerate>=0.27.0 \
    transformers>=4.38.0 \
    torch>=2.2.0 \
    sentencepiece \
    protobuf

# Install Unsloth for fast inference
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Create cache directory for adapters
RUN mkdir -p /workspace/model_cache

# Copy handler
COPY handler.py /app/handler.py

# Environment
ENV HF_HOME=/workspace/model_cache
ENV TRANSFORMERS_CACHE=/workspace/model_cache
ENV HF_ORG=FoundryAILabs
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import runpod; print('ok')" || exit 1

CMD ["python", "-u", "/app/handler.py"]
