#!/usr/bin/env python3
import sys
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from peft import PeftModel

# ============================
# PATHS
# ============================
BASE = "/home/hice1/rswaminathan38/scratch/CS-6220-Project_ram/new_hf_cache"

ROUTER_DIR = f"{BASE}/router_deberta"          # NEW ROUTER
BASE_MODEL_DIR = f"{BASE}/base_model_extracted"  # CORRECT LLAMA BASE

EXPERT_DIRS = {
    0: f"{BASE}/general_expert",
    1: f"{BASE}/math_expert",
    2: f"{BASE}/science_expert",
}

# ============================
# HELPER — Resolve adapter file
# ============================
def resolve_adapter_model(path):
    safetensors = os.path.join(path, "adapter_model.safetensors")
    binfile = os.path.join(path, "adapter_model.bin")
    if os.path.exists(safetensors):
        return safetensors
    if os.path.exists(binfile):
        return binfile
    raise FileNotFoundError(
        f"No adapter_model.safetensors or adapter_model.bin in {path}"
    )

# ============================
# LOAD ROUTER
# ============================
print("\n=== Loading DeBERTa Router Tokenizer ===")
router_tok = AutoTokenizer.from_pretrained(ROUTER_DIR)

print("=== Loading DeBERTa Router Model ===")
router_model = AutoModelForSequenceClassification.from_pretrained(ROUTER_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
router_model.to(device)
router_model.eval()

# ============================
# GET USER PROMPT
# ============================
if len(sys.argv) < 2:
    print("Usage:\n  python3 moe_router_inference_final.py \"your prompt here\"")
    sys.exit(1)

USER_PROMPT = sys.argv[1]
print(f"\n=== USER PROMPT: {USER_PROMPT} ===\n")

# ============================
# ROUTE PROMPT → EXPERT
# ============================
inputs = router_tok(USER_PROMPT, return_tensors="pt").to(device)

with torch.no_grad():
    logits = router_model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()

expert_name = ["GENERAL", "MATH", "SCIENCE"][pred]
expert_path = EXPERT_DIRS[pred]

print(f"=== ROUTER DECISION: {expert_name} ({pred}) ===")
print(f"→ Using Expert Adapter: {expert_path}\n")

# ============================
# LOAD BASE LLAMA MODEL
# ============================
print("=== Loading Base Llama-3.1-8B-Instruct ===")
base_tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ============================
# APPLY LORA EXPERT
# ============================
print("=== Applying LoRA Expert Adapter ===")
adapter_model_path = resolve_adapter_model(expert_path)

model = PeftModel.from_pretrained(
    base_model,
    expert_path,
    torch_dtype=torch.float16,
)
model.eval()

# ============================
# GENERATE OUTPUT
# ============================
print("=== Generating Answer ===")
inputs = base_tok(USER_PROMPT, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

output_text = base_tok.decode(output_ids[0], skip_special_tokens=True)

print("\n======= MODEL OUTPUT =======")
print(output_text)
print("============================\n")
