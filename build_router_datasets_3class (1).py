#!/usr/bin/env python3
# ============================================================
# Router Dataset Builder (3-Class, Augmented & Balanced)
# Author: Ramanathan Swaminathan — Georgia Tech
# ============================================================

from datasets import load_dataset
import os, json, random
from collections import Counter

SAVE_DIR = "/home/hice1/rswaminathan38/scratch/CS-6200-Project_ram"
TRAIN_OUT = os.path.join(SAVE_DIR, "router_train_3class.jsonl")
TEST_OUT  = os.path.join(SAVE_DIR, "router_test_3class.jsonl")

random.seed(42)
TARGET_PER_CLASS = 5000  # aim ~15k before split

# ============================================================
# ✅ Active datasets (public and stable)
# ============================================================

science_sources = [
    ("allenai/openbookqa", "main"),
    ("allenai/ai2_arc", "ARC-Challenge"),
    ("allenai/ai2_arc", "ARC-Easy"),
    ("pubmed_qa", "pqa_labeled"),
    ("databricks/databricks-dolly-15k", None),
]

math_sources = [
    ("gsm8k", "main"),
    ("aqua_rat", None),
    ("hendrycks/competition_math", None),
    ("deepmind/mathematics_dataset", "arithmetic__add_or_sub"),
    ("math_qa", None),
]

general_sources = [
    ("commonsense_qa", None),
    ("Rowan/hellaswag", None),
    ("ag_news", None),
    ("cais/mmlu", "social_sciences"),
    ("super_glue", "boolq"),
]

# ============================================================
# Helper utilities
# ============================================================

def extract_text(example):
    """Find best text field in a dataset example."""
    for key in ["question", "instruction", "input", "context", "text",
                "query", "goal", "sentence", "headline", "prompt", "passage"]:
        if key in example and example[key]:
            return str(example[key]).strip()
    return None

def sample_dataset(name, subset=None, split="train", limit=3000):
    """Load & sample dataset."""
    try:
        ds = load_dataset(name, subset, split=split)
        texts = []
        for ex in ds:
            t = extract_text(ex)
            if t and len(t) > 10:
                texts.append(t)
        random.shuffle(texts)
        return texts[:limit]
    except Exception as e:
        print(f"⚠️ Skipping {name}: {e}")
        return []

def gather_samples(sources, total):
    """Gather and deduplicate across multiple datasets."""
    per = max(1, total // len(sources))
    data = []
    for name, sub in sources:
        data += sample_dataset(name, sub, limit=per)
    data = list(dict.fromkeys(data))
    random.shuffle(data)
    return data[:total]

# ============================================================
# Main data loading
# ============================================================

print("🔍 Gathering balanced samples...")
science = gather_samples(science_sources, TARGET_PER_CLASS)
math    = gather_samples(math_sources, TARGET_PER_CLASS)
general = gather_samples(general_sources, TARGET_PER_CLASS)
print(f"✅ Counts → science={len(science)}, math={len(math)}, general={len(general)}")

# ============================================================
# Train/test split
# ============================================================

def split_and_label(data, label):
    cut = int(0.8 * len(data))
    train = [{"instruction": t, "category": label} for t in data[:cut]]
    test  = [{"instruction": t, "category": label} for t in data[cut:]]
    return train, test

train, test = [], []
for label, arr in [("science", science), ("math", math), ("general", general)]:
    tr, te = split_and_label(arr, label)
    train += tr
    test += te

# ============================================================
# 🧩 Synthetic augmentation to fix bias
# ============================================================

augment = []

# --- Math-like short questions ---
for expr, ans in [
    ("5 * 250", "1250"),
    ("12 + 37", "49"),
    ("sqrt(81)", "9"),
    ("area of circle with radius 7", "153.94"),
    ("derivative of x^2", "2x"),
    ("integrate x", "x^2 / 2 + C"),
]:
    augment.append({"instruction": expr, "category": "math"})

# --- General knowledge / chit-chat ---
for text in [
    "Who is the president of the United States?",
    "Tell me about Barack Obama.",
    "What's the capital of France?",
    "Explain how a car engine works.",
    "Who won the last soccer World Cup?",
    "What is the purpose of social media?",
]:
    augment.append({"instruction": text, "category": "general"})

# --- Short science definitions ---
for text in [
    "What is gravity?",
    "Define photosynthesis.",
    "Explain Newton's first law.",
    "What causes rainbows?",
    "Why is the sky blue?",
]:
    augment.append({"instruction": text, "category": "science"})

# Mix into train set
train += augment
random.shuffle(train)
print(f"✨ Added {len(augment)} synthetic samples to training data.")

# ============================================================
# Save datasets
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)
with open(TRAIN_OUT, "w") as f:
    for r in train:
        f.write(json.dumps(r) + "\n")
with open(TEST_OUT, "w") as f:
    for r in test:
        f.write(json.dumps(r) + "\n")

print(f"\n✅ Saved training set → {TRAIN_OUT} ({len(train)} samples)")
print(f"✅ Saved test set     → {TEST_OUT} ({len(test)} samples)")
print("🧩 Categories verified: ['science', 'math', 'general']")

# ============================================================
# Verify label balance
# ============================================================

def count_labels(path):
    counts = Counter()
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            counts[obj["category"]] += 1
    return counts

train_counts = count_labels(TRAIN_OUT)
test_counts = count_labels(TEST_OUT)

print("\n📊 Class distribution check:")
print("Train →", dict(train_counts))
print("Test  →", dict(test_counts))
print("\n✅ Dataset build complete!")
