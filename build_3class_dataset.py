"""
build_3class_dataset.py
-----------------------
Creates a 3-class router dataset: math, science, general
Remaps the existing 4-class dataset to 3 classes:
- math → math
- science → science  
- code → general
- commonsense → general
"""

import json
from collections import Counter

# Read the existing 4-class dataset
input_file = "router_combined.jsonl"
output_file = "router_3class.jsonl"

# Define the mapping from 4 classes to 3 classes
class_mapping = {
    "math": "math",
    "science": "science",
    "code": "general",
    "commonsense": "general"
}

data = []
with open(input_file, "r") as f:
    for line in f:
        item = json.loads(line)
        old_category = item["category"]
        new_category = class_mapping.get(old_category, "general")
        data.append({
            "instruction": item["instruction"],
            "category": new_category
        })

# Count categories
category_counts = Counter(item["category"] for item in data)
print("Category distribution:")
for cat, count in sorted(category_counts.items()):
    print(f"  {cat}: {count}")

# Write the remapped dataset
with open(output_file, "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

print(f"\n✅ Created 3-class dataset: {output_file}")
print(f"Total samples: {len(data)}")
print(f"Classes: {sorted(category_counts.keys())}")
