"""
BharatLLM RunPod Serverless Worker — v4 (lazy loading, robust)
"""
import os
import time
import runpod

HF_ORG = os.environ.get("HF_ORG", "FoundryAILabs")
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Try multiple cache locations
for cache_dir in ["/runpod-volume/model_cache", "/workspace/model_cache", "/tmp/model_cache"]:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # Test write
        test_file = os.path.join(cache_dir, ".test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        CACHE_DIR = cache_dir
        break
    except Exception:
        continue
else:
    CACHE_DIR = "/tmp/model_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

print(f"[INIT] Cache dir: {CACHE_DIR}")

MODEL_REGISTRY = {
    "english": f"{HF_ORG}/bharat-english-7b-lora",
    "hindi": f"{HF_ORG}/bharat-hindi-7b-lora",
    "bengali": f"{HF_ORG}/bharat-bengali-7b-lora",
    "telugu": f"{HF_ORG}/bharat-telugu-7b-lora",
    "tamil": f"{HF_ORG}/bharat-tamil-7b-lora",
    "kannada": f"{HF_ORG}/bharat-kannada-7b-lora",
    "malayalam": f"{HF_ORG}/bharat-malayalam-7b-lora",
    "marathi": f"{HF_ORG}/bharat-marathi-7b-lora",
    "gujarati": f"{HF_ORG}/bharat-gujarati-7b-lora",
    "odia": f"{HF_ORG}/bharat-odia-7b-lora",
    "punjabi": f"{HF_ORG}/bharat-punjabi-7b-lora",
    "urdu": f"{HF_ORG}/bharat-urdu-7b-lora",
    "btech": f"{HF_ORG}/bharat-btech-7b-lora",
}

# Lazy-loaded globals
_model = None
_tokenizer = None
_adapter_name = None
_adapter_model = None


def get_model_and_tokenizer():
    """Lazy-load base model on first request (not at startup)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[MODEL] Loading {BASE_MODEL} (float16)...")
    start = time.time()

    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    elapsed = time.time() - start
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"[MODEL] Loaded in {elapsed:.1f}s, GPU: {mem:.1f}GB")
    return _model, _tokenizer


def get_adapter(model_name):
    """Load LoRA adapter (cached)."""
    global _adapter_name, _adapter_model
    if model_name == _adapter_name and _adapter_model is not None:
        return _adapter_model

    from peft import PeftModel

    base, _ = get_model_and_tokenizer()
    repo = MODEL_REGISTRY.get(model_name)
    if not repo:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"[ADAPTER] Loading {model_name} from {repo}")
    start = time.time()
    _adapter_model = PeftModel.from_pretrained(base, repo, cache_dir=CACHE_DIR)
    _adapter_model.eval()
    _adapter_name = model_name
    print(f"[ADAPTER] {model_name} loaded in {time.time()-start:.1f}s")
    return _adapter_model


def handler(job):
    """Process inference request."""
    try:
        inp = job["input"]
        prompt = inp.get("prompt", "")
        model_name = inp.get("model", "english")
        max_tokens = min(inp.get("max_tokens", 512), 2048)
        temperature = inp.get("temperature", 0.7)

        if inp.get("mode") == "btech":
            model_name = "btech"

        if not prompt:
            return {"error": "No prompt provided"}

        import torch

        start = time.time()
        _, tokenizer = get_model_and_tokenizer()
        model = get_adapter(model_name)

        if model_name == "btech":
            sys_msg = inp.get("system", "You are BharatLLM, an expert engineering tutor.")
            formatted = f"[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            formatted = f"[INST] {prompt} [/INST]"

        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        return {
            "text": text.strip(),
            "model": f"bharat-{model_name}-7b-lora",
            "tokens": len(out[0]) - input_len,
            "latency_ms": int((time.time() - start) * 1000),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


print("=" * 50)
print("  BharatLLM Worker Ready (lazy loading)")
print(f"  Models: {len(MODEL_REGISTRY)}")
print(f"  Cache: {CACHE_DIR}")
print("=" * 50)

runpod.serverless.start({"handler": handler})
