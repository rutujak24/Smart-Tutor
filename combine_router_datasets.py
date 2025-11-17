"""
combine_router_datasets.py
--------------------------
Builds a balanced router training dataset from multiple Hugging Face datasets.
Categories come from router_config.yaml:
    - code
    - commonsense
    - general
    - math
    - science
The output is router_combined.jsonl where each line:
    {"instruction": "...", "category": "math"}
"""

import yaml, json, random
from datasets import load_dataset

# ============================================================
# 1️⃣ Load category configuration
# ============================================================
with open("router_config.yaml") as f:
    cfg = yaml.safe_load(f)

categories = cfg["categories"]
mapping = cfg["dataset_category_mapping"]

# ============================================================
# 2️⃣ Helper functions
# ============================================================
def get_question(ex):
    """Extract a meaningful text field from a dataset example."""
    for k in ["question", "prompt", "instruction", "text", "query", "problem", "title"]:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def load_split(name):
    """Unified wrapper to load each dataset split safely."""
    if name == "gsm8k":
        return load_dataset("gsm8k", "main", split="train")
    if name == "openbookqa":
        return load_dataset("openbookqa", "main", split="train")
    if name == "arc_easy":
        return load_dataset("ai2_arc", "ARC-Easy", split="train")
    if name == "arc_challenge":
        return load_dataset("ai2_arc", "ARC-Challenge", split="train")
    if name == "trivia_qa":
        return load_dataset("trivia_qa", "rc", split="train")
    if name == "commonsense_qa":
        return load_dataset("commonsense_qa", split="train")
    if name == "humaneval":
        return load_dataset("openai_humaneval", split="test")
    if name == "mbpp":
        try:
            return load_dataset("mbpp", "sanitized", split="train")
        except Exception:
            return load_dataset("mbpp", split="train")
    # fallback: try directly
    return load_dataset(name, split="train")

# ============================================================
# 3️⃣ Build per-label buckets
# ============================================================
bucket = {lab: [] for lab in categories}
print("🔧 Building router dataset from configured sources...\n")

for ds_key, label in mapping.items():
    try:
        ds = load_split(ds_key)
        total = len(ds)

        # define sampling limits
        take = 1000
        if label in ["science", "code"]:
            take = min(1500, total)
        else:
            take = min(1000, total)

        idxs = random.sample(range(total), take)
        kept = 0
        for i in idxs:
            q = get_question(ds[i])
            if q:
                bucket[label].append({"instruction": q, "category": label})
                kept += 1
        print(f"{ds_key:<15} → {label:<12} kept {kept}")
    except Exception as e:
        print(f"⚠️ Skipping {ds_key}: {e}")

# ============================================================
# 4️⃣ Safe balancing
# ============================================================
print("\n🔧 Balancing categories safely...")
valid_labels = [lab for lab, samples in bucket.items() if len(samples) > 0]

if not valid_labels:
    raise RuntimeError("No datasets loaded successfully — check your YAML mapping and HF connectivity.")

min_count = min(len(bucket[lab]) for lab in valid_labels)
print(f"→ Minimum available samples per valid category: {min_count}")

balanced = []
for lab in valid_labels:
    if len(bucket[lab]) < min_count:
        print(f"⚠️ Skipping undersized category '{lab}' ({len(bucket[lab])} samples)")
        continue
    balanced.extend(random.sample(bucket[lab], min_count))

random.shuffle(balanced)

# ============================================================
# 5️⃣ Write output
# ============================================================
out_path = "router_combined.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for row in balanced:
        f.write(json.dumps(row) + "\n")

print(f"\n✅ Wrote {len(balanced)} balanced samples ({min_count} per class) to {out_path}")
print(f"Included labels: {valid_labels}")
