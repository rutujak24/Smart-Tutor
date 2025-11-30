import json
import random
from pathlib import Path

random.seed(42)

# ============================================================
# SEED TOPICS
# ============================================================

MATH_TOPICS = [
    "linear algebra", "calculus", "probability", "statistics", "real analysis",
    "complex analysis", "geometry", "number theory", "optimization",
    "graph theory", "discrete math", "differential equations",
    "matrix factorization", "eigenvalues", "gradients", "tensors",
    "Fourier transforms", "Bayesian inference"
]

SCIENCE_TOPICS = [
    "quantum physics", "classical mechanics", "thermodynamics",
    "electromagnetism", "organic chemistry", "inorganic chemistry",
    "cell biology", "genetics", "evolution", "astronomy",
    "cosmology", "neuroscience", "geology", "meteorology",
    "ecology", "biophysics", "materials science", "medicine"
]

GENERAL_TOPICS = [
    "cooking", "travel", "movies", "music", "relationships",
    "career advice", "fitness", "history", "politics",
    "writing", "productivity", "mental health", "gaming",
    "programming", "daily life", "sports", "shopping",
    "technology news"
]

# ============================================================
# TEMPLATE GENERATORS
# ============================================================

def create_short_math_prompt(topic):
    return f"solve a {topic} problem"

def create_long_math_prompt(topic):
    return (
        f"Explain in detail how to approach a challenging {topic} question, "
        f"including definitions, step-by-step reasoning, and the final solution."
    )

def create_short_science_prompt(topic):
    return f"explain basic {topic}"

def create_long_science_prompt(topic):
    return (
        f"Describe the fundamental principles of {topic} and how they are applied "
        f"in real-world scientific or engineering contexts."
    )

def create_short_general_prompt(topic):
    return f"tell me something about {topic}"

def create_long_general_prompt(topic):
    return (
        f"Write a detailed explanation or piece of advice about {topic}, "
        f"including examples and practical insights."
    )

# ============================================================
# ASSEMBLE DATASET
# ============================================================

def generate_class_samples(
    topics, short_fn, long_fn, n_samples, label
):
    samples = []
    for _ in range(n_samples):
        t = random.choice(topics)
        if random.random() < 0.5:
            prompt = short_fn(t)
        else:
            prompt = long_fn(t)
        samples.append({"text": prompt, "label": label})
    return samples


def main():
    output_file = Path("router_dataset.jsonl")
    total_per_class = 1200  # math:1200, sci:1200, general:1200

    # Generate samples
    math_samples = generate_class_samples(
        MATH_TOPICS,
        create_short_math_prompt,
        create_long_math_prompt,
        total_per_class,
        label=1
    )

    science_samples = generate_class_samples(
        SCIENCE_TOPICS,
        create_short_science_prompt,
        create_long_science_prompt,
        total_per_class,
        label=2
    )

    general_samples = generate_class_samples(
        GENERAL_TOPICS,
        create_short_general_prompt,
        create_long_general_prompt,
        total_per_class,
        label=0
    )

    # Combine and shuffle
    full_dataset = math_samples + science_samples + general_samples
    random.shuffle(full_dataset)

    # Write JSONL
    with open(output_file, "w") as f:
        for item in full_dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Dataset saved to {output_file} with {len(full_dataset)} samples.")


if __name__ == "__main__":
    main()
