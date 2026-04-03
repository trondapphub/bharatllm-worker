"""
BharatLLM RunPod Serverless Worker — Simplified (float16, no bitsandbytes)
"""
import os
import time
import runpod

HF_ORG = os.environ.get("HF_ORG", "FoundryAILabs")
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR = os.environ.get("HF_HOME", "/runpod-volume/model_cache")

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

base_model = None
base_tokenizer = None
current_adapter_name = None
current_model = None

print("=" * 50)
print("  BharatLLM RunPod Worker Starting...")
print(f"  Models: {len(MODEL_REGISTRY)}")
print(f"  Cache: {CACHE_DIR}")
print("=" * 50)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"[INIT] Loading tokenizer: {BASE_MODEL}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    print(f"[INIT] Loading base model (float16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    print(f"[INIT] Base model loaded! GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

except Exception as e:
    print(f"[INIT ERROR] {e}")
    import traceback
    traceback.print_exc()


def handler(job):
    global current_adapter_name, current_model

    try:
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

        # Load adapter if different from current
        if model_name != current_adapter_name:
            repo_id = MODEL_REGISTRY.get(model_name)
            if not repo_id:
                return {"error": f"Unknown model: {model_name}"}

            print(f"[ADAPTER] Loading {model_name} from {repo_id}")
            current_model = PeftModel.from_pretrained(
                base_model, repo_id, cache_dir=CACHE_DIR
            )
            current_model.eval()
            current_adapter_name = model_name
            print(f"[ADAPTER] {model_name} loaded!")

        # Format prompt
        if model_name == "btech":
            system = job_input.get("system", "You are BharatLLM, an expert engineering tutor.")
            formatted = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            formatted = f"[INST] {prompt} [/INST]"

        inputs = base_tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=2048
        ).to(current_model.device)

        with torch.no_grad():
            outputs = current_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                repetition_penalty=1.1,
                pad_token_id=base_tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        text = base_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        latency = int((time.time() - start_time) * 1000)

        return {
            "text": text.strip(),
            "model": f"bharat-{model_name}-7b-lora",
            "tokens": len(outputs[0]) - input_len,
            "latency_ms": latency,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
