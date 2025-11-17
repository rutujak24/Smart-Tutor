#!/usr/bin/env python3
# ============================================================
# Mixture-of-Experts (MoE) Inference — Improved Natural Replies
# Author: Ramanathan Swaminathan (Georgia Tech)
# ============================================================

import os, json, torch
# --- Offline-safe env ---
os.environ["HF_HOME"] = "/home/hice1/rswaminathan38/scratch/CS-6200-Project_ram/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["UNSLOTH_OFFLINE"] = "1"
os.environ["UNSLOTH_SILENT"] = "1"

# --- Imports (Unsloth first) ---
from unsloth import FastLanguageModel
from transformers import DistilBertTokenizer, DistilBertConfig, AutoTokenizer
from distilbert_router import DistilBertRouter, RouterInference

# ============================================================
# Configuration
# ============================================================
BASE = "/home/hice1/rswaminathan38/scratch/CS-6200-Project_ram"
CACHE = os.path.join(BASE, "hf_cache")
ROUTER_PATH = os.path.join(BASE, "router_model_moe_3class")
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = os.path.join(
    CACHE,
    "models--unsloth--meta-llama-3.1-8b-instruct-bnb-4bit",
    "snapshots",
    "f15c379fb32bb402fa06a7ae9aecb1febf4b79ec",
)

CONFIDENCE_THRESHOLD = 0.70

# ============================================================
# Experts — 3 domains + combined fallback
# ============================================================
EXPERTS = {
    "math": {
        "model": os.path.join(CACHE, "math_model"),
        "tokenizer": os.path.join(CACHE, "math_tokenizer"),
    },
    "science": {
        "model": os.path.join(CACHE, "science_model"),
        "tokenizer": os.path.join(CACHE, "science_tokenizer"),
    },
    "general": {
        "model": os.path.join(CACHE, "general_model"),
        "tokenizer": os.path.join(CACHE, "general_tokenizer"),
    },
    "combined": {
        "model": os.path.join(CACHE, "combined_model"),
        "tokenizer": os.path.join(CACHE, "combined_tokenizer"),
    },
}

# Map router → expert
label_to_expert = {"math": "math", "science": "science", "general": "general"}

# ============================================================
# Load router
# ============================================================
print("🧭 Loading trained DistilBERT router model...")
cfg = DistilBertConfig.from_pretrained("distilbert-base-uncased")
router_model = DistilBertRouter.load_pretrained(ROUTER_PATH, cfg, device=device)
router_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

with open(os.path.join(ROUTER_PATH, "router_config.json")) as f:
    meta = json.load(f)
router_label_map = {str(i): name for i, name in enumerate(meta["category_names"])}

print("✅ Router model loaded.")
print("🧩 Router label mapping:", router_label_map)
print("✅ Experts mapped and ready.\n")

router = RouterInference(router_model, router_tokenizer, device=device)

# ============================================================
# Prompt style templates
# ============================================================
PROMPT_STYLES = {
    "math": "Show reasoning step by step and end with the final numeric answer.",
    "science": "Explain clearly in natural scientific language using 2–4 sentences.",
    "general": "Answer concisely and naturally for a general audience.",
    "combined": "Provide a factual, balanced explanation suitable for any topic.",
}

# ============================================================
# Expert loader (cached) + fallback
# ============================================================
_loaded_experts = {}

def load_expert(model_dir, tokenizer_dir, expert_key):
    if expert_key in _loaded_experts:
        print(f"⚡ Using cached expert → {expert_key}")
        return _loaded_experts[expert_key]

    try:
        print(f"📦 Loading expert adapter from {model_dir} ...")
        base_model, _ = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            dtype=torch.float16,
            load_in_4bit=False,
            trust_remote_code=True,
            local_files_only=True,
            resume_download=False,
        )
        base_model.load_adapter(model_dir)
        base_model = FastLanguageModel.for_inference(base_model)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        _loaded_experts[expert_key] = (base_model, tokenizer)
        print(f"✅ Expert ready: {expert_key}")
        return base_model, tokenizer
    except Exception as e:
        print(f"❌ Error loading expert {expert_key}: {e}")
        try:
            print("📂 Folder contents:", os.listdir(model_dir))
        except Exception:
            pass
        return None, None

# ============================================================
# Logging
# ============================================================
LOG_PATH = os.path.join(BASE, "results", "moe_inference_log.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
def log_decision(query, category, conf, expert_key):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({
            "query": query,
            "router_category": category,
            "router_confidence": conf,
            "expert_used": expert_key,
        }) + "\n")

# ============================================================
# Interactive inference
# ============================================================
print("💡 Router Inference Active. Type a query or 'exit' to quit.\n")

while True:
    try:
        query = input(">> ").strip()
        if not query or query.lower() == "exit":
            print("🛑 Exiting interactive session.")
            break

        # 1️⃣ Route
        route = router.route_query(query)
        category, conf = route["category"], route["confidence"]
        print(f"🔍 Router prediction → {category} (conf={conf:.3f})")

        # 2️⃣ Choose expert
        expert_key = (
            label_to_expert.get(category)
            if conf >= CONFIDENCE_THRESHOLD
            else "combined"
        )
        expert = EXPERTS.get(expert_key, EXPERTS["combined"])

        # 3️⃣ Load expert
        model, tokenizer = load_expert(expert["model"], expert["tokenizer"], expert_key)
        if model is None:
            print("⚠️ Expert load failed — using combined fallback.\n")
            model, tokenizer = load_expert(
                EXPERTS["combined"]["model"], EXPERTS["combined"]["tokenizer"], "combined"
            )
            if model is None:
                continue

        # 4️⃣ Instruction-style prompt (per expert)
        style = PROMPT_STYLES.get(expert_key, "")
        prompt = f"<|begin_of_text|><|user|>\n{style}\n\n{query.strip()}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.75,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = text.split("<|assistant|>")[-1].strip()
        print(f"\n🤖 Expert ({expert_key}) reply:\n{response}\n")
        log_decision(query, category, conf, expert_key)

    except KeyboardInterrupt:
        print("\n🛑 Session terminated.")
        break
