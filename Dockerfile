# BharatLLM RunPod Serverless Worker
# ============================================================
# Base: NVIDIA PyTorch with CUDA support
# Includes: Unsloth, PEFT, RunPod SDK
# ============================================================
# BharatLLM RunPod Serverless Worker — Simplified
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    runpod>=1.6.0 \
    huggingface_hub>=0.20.0 \
    peft>=0.8.0 \
    accelerate>=0.27.0 \
    transformers>=4.38.0 \
    sentencepiece \
    protobuf

COPY handler.py /app/handler.py

ENV HF_HOME=/runpod-volume/model_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/model_cache
ENV HF_ORG=FoundryAILabs
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/app/handler.py"]

