"""
BharatLLM — RunPod Serverless Handler
Dynamic LoRA adapter loading for 13 models on a single endpoint.

Base model: Mistral-7B-Instruct-v0.3 (loaded once at cold start)
LoRA adapters: Downloaded from HuggingFace on first request, cached.

Request format:
{
    "input": {
        "prompt": "What is photosynthesis?",
        "model": "hindi",
        "max_tokens": 512,
        "temperature": 0.7
    }
}

Response format:
{
    "output": {
        "text": "Photosynthesis is...",
        "model": "bharat-hindi-7b-lora",
        "tokens": 142,
        "latency_ms": 1230
    }
}
"""

import os
import time
import torch
import runpod
from threading import Lock

# ============================================================
# Configuration
# ============================================================

HF_ORG = os.environ.get("HF_ORG", "FoundryAILabs")
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_SEQ_LENGTH = 2048
CACHE_DIR = "/workspace/model_cache"

# Model registry: request name -> HuggingFace repo
MODEL_REGISTRY = {
    "english":   f"{HF_ORG}/bharat-english-7b-lora",
    "hindi":     f"{HF_ORG}/bharat-hindi-7b-lora",
    "bengali":   f"{HF_ORG}/bharat-bengali-7b-lora",
    "telugu":    f"{HF_ORG}/bharat-telugu-7b-lora",
    "tamil":     f"{HF_ORG}/bharat-tamil-7b-lora",
    "kannada":   f"{HF_ORG}/bharat-kannada-7b-lora",
    "malayalam": f"{HF_ORG}/bharat-malayalam-7b-lora",
    "marathi":   f"{HF_ORG}/bharat-marathi-7b-lora",
    "gujarati":  f"{HF_ORG}/bharat-gujarati-7b-lora",
    "odia":      f"{HF_ORG}/bharat-odia-7b-lora",
    "punjabi":   f"{HF_ORG}/bharat-punjabi-7b-lora",
    "urdu":      f"{HF_ORG}/bharat-urdu-7b-lora",
    "btech":     f"{HF_ORG}/bharat-btech-7b-lora",
}

# ============================================================
# Global State
# ============================================================

base_model = None
base_tokenizer = None
loaded_adapters = {}
adapter_lock = Lock()


def load_base_model():
    """Load Mistral-7B base model at cold start."""
    global base_model, base_tokenizer

    print(f"[INIT] Loading base model: {BASE_MODEL}")
    start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    elapsed = time.time() - start
    print(f"[INIT] Base model loaded in {elapsed:.1f}s")
    print(f"[INIT] GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")


def get_adapter(model_name):
    """Load or retrieve cached LoRA adapter."""
    global base_model

    if model_name in loaded_adapters:
        return loaded_adapters[model_name]

    with adapter_lock:
        if model_name in loaded_adapters:
            return loaded_adapters[model_name]

        repo_id = MODEL_REGISTRY.get(model_name)
        if not repo_id:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

        print(f"[ADAPTER] Loading {model_name} from {repo_id}...")
        start = time.time()

        from peft import PeftModel

        adapter_model = PeftModel.from_pretrained(
            base_model,
            repo_id,
            cache_dir=CACHE_DIR,
        )
        adapter_model.eval()

        elapsed = time.time() - start
        print(f"[ADAPTER] {model_name} loaded in {elapsed:.1f}s")

        # Keep max 3 adapters in memory
        if len(loaded_adapters) >= 3:
            oldest_key = next(iter(loaded_adapters))
            del loaded_adapters[oldest_key]
            torch.cuda.empty_cache()
            print(f"[ADAPTER] Evicted {oldest_key} from cache")

        loaded_adapters[model_name] = adapter_model
        return adapter_model


# ============================================================
# Handler
# ============================================================

def handler(job):
    """Process a single inference request."""
    job_input = job["input"]

    prompt = job_input.get("prompt", "")
    model_name = job_input.get("model", "english")
    max_tokens = min(job_input.get("max_tokens", 512), 2048)
    temperature = job_input.get("temperature", 0.7)
    mode = job_input.get("mode", "k12")

    if mode == "btech":
        model_name = "btech"

    if not prompt:
        return {"error": "No prompt provided"}

    start_time = time.time()

    try:
        model = get_adapter(model_name)

        if model_name == "btech":
            system = job_input.get("system", "You are BharatLLM, an expert engineering tutor.")
            formatted = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            formatted = f"[INST] {prompt} [/INST]"

        inputs = base_tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                repetition_penalty=1.1,
                pad_token_id=base_tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        text = base_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        token_count = len(generated_tokens)

        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "text": text.strip(),
            "model": f"bharat-{model_name}-7b-lora",
            "tokens": token_count,
            "latency_ms": latency_ms,
            "cached": model_name in loaded_adapters,
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "error": str(e),
            "model": model_name,
            "latency_ms": latency_ms,
        }


# ============================================================
# Initialize and Start
# ============================================================

print("=" * 55)
print("  BharatLLM RunPod Serverless Worker")
print(f"  Models: {len(MODEL_REGISTRY)} (12 K-12 + 1 BTech)")
print(f"  Base: {BASE_MODEL}")
print("=" * 55)

load_base_model()

runpod.serverless.start({"handler": handler})
