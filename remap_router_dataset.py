#!/usr/bin/env python3
# ============================================================
# Remap Original Router Dataset → MoE-Compatible Labels
# ============================================================
# Converts: code → general, commonsense → general
# Keeps: math, science
# Adds: optional combined fallback for balance
# ============================================================

import json
import random
import os

BASE = "/home/hice1/rswaminathan38/scratch/CS-6200-Project_ram"
INPUT_PATH = os.path.join(BASE, "router_combined.jsonl")
OUTPUT_PATH = os.path.join(BASE, "router_moe_ready.jsonl")

print(f"📂 Loading dataset from {INPUT_PATH}")
data = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        data.append(row)

print(f"✅ Loaded {len(data)} records.")

# --- Map labels ---
remap = {
    "code": "general",
    "commonsense": "general",
    "math": "math",
    "science": "science",
}

remapped = []
for d in data:
    old = d["category"].strip().lower()
    if old not in remap:
        continue
    new = remap[old]
    remapped.append({"instruction": d["instruction"], "category": new})

# --- Optional fallback balancing ---
# Add a small subset labeled as 'combined' for fallback routing (5–10%)
combined_ratio = 0.08
n_combined = int(len(remapped) * combined_ratio)
sampled = random.sample(remapped, n_combined)
for s in sampled:
    remapped.append({"instruction": s["instruction"], "category": "combined"})

print(f"✅ Remapped to categories: math, science, general, combined")
print(f"   → {len(remapped)} total examples ({n_combined} fallback)")

# --- Save output ---
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for row in remapped:
        f.write(json.dumps(row) + "\n")

print(f"💾 Saved remapped dataset to {OUTPUT_PATH}")
