import json

IN_PATH = "router_combined.jsonl"
OUT_PATH = "router_moe_ready_3class.jsonl"

data = [json.loads(x) for x in open(IN_PATH)]

mapped = []
for d in data:
    cat = d["category"]
    if cat in ["commonsense", "code"]:
        new_cat = "general"
    elif cat in ["math", "science"]:
        new_cat = cat
    else:
        continue
    mapped.append({"instruction": d["instruction"], "category": new_cat})

print(f"✅ Remapped {len(mapped)} entries into 3 categories (math, science, general).")

with open(OUT_PATH, "w") as f:
    for r in mapped:
        f.write(json.dumps(r) + "\n")

print(f"💾 Saved to {OUT_PATH}")
