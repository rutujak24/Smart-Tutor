#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from peft import PeftModel

# ======================================================
# PATHS
# ======================================================
BASE = "/home/hice1/rswaminathan38/scratch/CS-6220-Project_ram_router_v2/new_hf_cache"

ROUTER_DIR = f"{BASE}/router_v2_models/deberta_base_clean"
BASE_MODEL_DIR = f"{BASE}/base_model_extracted"

EXPERT_DIRS = {
    0: f"{BASE}/general_expert",
    1: f"{BASE}/math_expert",
    2: f"{BASE}/science_expert",
}

# EXACT training label order:
# LABEL_MAP = {"MATH": 0, "SCIENCE": 1, "GENERAL": 2}
LABELS = ["MATH", "SCIENCE", "GENERAL"]

MAX_LEN = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# FASTAPI SETUP
# ======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    prompt: str

# ======================================================
# LOAD ROUTER (DeBERTa)
# ======================================================
print("\n=== Loading DeBERTa Router ===")
router_tok = AutoTokenizer.from_pretrained(
    ROUTER_DIR,
    local_files_only=True
)

router_model = AutoModelForSequenceClassification.from_pretrained(
    ROUTER_DIR,
    local_files_only=True
).to(device)
router_model.eval()

# ======================================================
# LOAD BASE LLAMA MODEL
# ======================================================
print("=== Loading Base Llama Model ===")

base_tok = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR,
    local_files_only=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
base_model.eval()

# ======================================================
# LOAD LoRA EXPERT MODELS (once)
# ======================================================
print("\n=== Loading LoRA Experts ===")
expert_models = {}

for idx, path in EXPERT_DIRS.items():
    print(f"Loading expert {idx} → {path}")
    expert_models[idx] = PeftModel.from_pretrained(
        base_model,
        path,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    expert_models[idx].eval()

print("\n=== MoE API Server Ready ===\n")

# ======================================================
# HEALTH CHECK
# ======================================================
@app.get("/health")
def health():
    return {"status": "ok"}

# ======================================================
# CHAT ENDPOINT
# ======================================================
@app.post("/chat")
def chat(query: Query):
    prompt = query.prompt

    # -------- ROUTER INFERENCE (MATCH TRAINING EXACTLY) --------
    inputs = router_tok(
        prompt,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = router_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    expert_name = LABELS[pred_idx]
    expert_model = expert_models[pred_idx]

    # -------- GENERATE WITH EXPERT --------
    encoded = base_tok(prompt, return_tensors="pt").to(expert_model.device)

    with torch.no_grad():
        output_ids = expert_model.generate(
            **encoded,
            max_new_tokens=200,
            temperature=0.7
        )

    response = base_tok.decode(output_ids[0], skip_special_tokens=True)

    return {
        "expert": expert_name,
        "confidence": round(confidence, 4),
        "response": response
    }

# ======================================================
# START SERVER (NO DOUBLE LOADING)
# ======================================================
if __name__ == "__main__":
    import uvicorn
    print("=== Starting MoE API Server ===")
    uvicorn.run(
        app,               # <-- FIXED: PREVENT DOUBLE-LOADING
        host="0.0.0.0",
        port=8510,
        reload=False,
        workers=1
    )
