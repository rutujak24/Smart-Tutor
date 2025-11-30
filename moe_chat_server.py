#!/usr/bin/env python3
import torch
import os
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from peft import PeftModel

# ============================================================
# HARD-WIRED ABSOLUTE PATHS (MATCH YOUR PACE DIRECTORY)
# ============================================================
BASE = "/home/hice1/rswaminathan38/scratch/CS-6220-Project_ram_router_v2/new_hf_cache"

ROUTER_DIR = f"{BASE}/router_v2_models/deberta_base_clean"   # <-- THE NEW CLEAN ROUTER
BASE_MODEL_DIR = f"{BASE}/base_model_extracted"

EXPERT_DIRS = {
    0: f"{BASE}/general_expert",
    1: f"{BASE}/math_expert",
    2: f"{BASE}/science_expert",
}

LABELS = ["MATH", "SCIENCE", "GENERAL"]   # MUST MATCH TRAINING LABEL_MAP order:
# LABEL_MAP = {"MATH": 0, "SCIENCE": 1, "GENERAL": 2}

MAX_LEN = 128   # EXACT MATCH TO TRAINING + EVAL

# ============================================================
# LOAD ROUTER
# ============================================================
print("\n=== Loading Router (DeBERTa-v3-base CLEAN) ===")

router_tok = AutoTokenizer.from_pretrained(
    ROUTER_DIR,
    local_files_only=True
)

router_model = AutoModelForSequenceClassification.from_pretrained(
    ROUTER_DIR,
    local_files_only=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
router_model.to(device)
router_model.eval()

# ============================================================
# LOAD BASE LLAMA MODEL (ONCE)
# ============================================================
print("\n=== Loading Base Llama Model ===")

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

# ============================================================
# LOAD ALL EXPERTS (LoRA)
# ============================================================
print("\n=== Loading All LoRA Experts ===")
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

print("\n=== MoE Chat Server READY ===")
print("Type your question (or 'exit' to quit)\n")

# ============================================================
# CHAT LOOP
# ============================================================
while True:
    user = input("You: ").strip()
    if user.lower() in ["exit", "quit"]:
        print("Exiting chat server.")
        break

    # ================================
    # ROUTER INFERENCE (MATCH TRAINING EXACTLY)
    # ================================
    encoded = router_tok(
        user,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = router_model(**encoded).logits
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    expert_name = LABELS[pred]

    print(f"\n[Router → {expert_name}]  (confidence={confidence:.3f})\n")

    # LOW CONFIDENCE FAIL-SAFE
    if confidence < 0.40:
        print("[Low-Confidence Router → DEFAULT: GENERAL]")
        pred = 2    # GENERAL = 2
        expert_name = LABELS[pred]

    # ================================
    # GENERATE WITH EXPERT
    # ================================
    expert_model = expert_models[pred]

    inp = base_tok(
        user,
        return_tensors="pt"
    ).to(expert_model.device)

    with torch.no_grad():
        output_ids = expert_model.generate(
            **inp,
            max_new_tokens=200,
            temperature=0.7
        )

    response = base_tok.decode(output_ids[0], skip_special_tokens=True)

    print(f"Assistant ({expert_name}): {response}\n")
