import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from peft import PeftModel

# =================================
# PATHS
# =================================

BASE = "/home/hice1/rswaminathan38/scratch/CS-6220-Project_ram_router_v2/new_hf_cache"

ROUTER_DIR = os.path.join(BASE, "router_v2_models/deberta_large_router")

BASE_MODEL_DIR = os.path.join(BASE, "models--meta-llama--Llama-3.1-8B-Instruct")

EXPERT_DIRS = {
    0: os.path.join(BASE, "general_expert"),
    1: os.path.join(BASE, "math_expert"),
    2: os.path.join(BASE, "science_expert")
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# =================================
# LOAD ROUTER (DeBERTa)
# =================================
print("\n=== Loading Router (DeBERTa-v3-large) ===")

router_tok = AutoTokenizer.from_pretrained(ROUTER_DIR)

# IMPORTANT: force safe tensors to be used
router_model = AutoModelForSequenceClassification.from_pretrained(
    ROUTER_DIR,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

router_model.to(device)
router_model.eval()

print("Router loaded successfully!")

# =================================
# LOAD BASE LLAMA MODEL ONCE
# =================================
print("\n=== Loading Base Llama Model (ONE TIME) ===")

base_tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)

base_model.eval()
print("Base model loaded.")

# =================================
# LOAD ALL EXPERTS ONE TIME
# =================================
print("\n=== Loading LoRA Experts ===")

expert_models = {}

for idx, path in EXPERT_DIRS.items():
    print(f"Loading expert [{idx}] → {path}")
    expert_models[idx] = PeftModel.from_pretrained(
        base_model,
        path,
        torch_dtype=torch.float16
    )
    expert_models[idx].eval()

print("\n=== MoE Chat Server Ready ===")
print("Type 'exit' to quit.\n")

# =================================
# CHAT LOOP
# =================================

while True:
    user = input("You: ").strip()
    if user.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    # ----- Router prediction -----
    inputs = router_tok(user, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = router_model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    expert_name = ["GENERAL", "MATH", "SCIENCE"][pred]

    print(f"\n[Router → {expert_name} Expert]\n")

    # ----- Generate with expert -----
    expert_model = expert_models[pred]

    encoded = base_tok(user, return_tensors="pt").to(expert_model.device)

    with torch.no_grad():
        output = expert_model.generate(
            **encoded,
            max_new_tokens=180,
            temperature=0.7
        )

    response = base_tok.decode(output[0], skip_special_tokens=True)

    print(f"Assistant ({expert_name}): {response}\n")
