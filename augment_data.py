"""
Simple data augmentation for router training dataset.
Generates paraphrases and variations of existing samples.
"""
import json
import random

# Augmentation templates for each category
MATH_VARIANTS = [
    "Calculate {task}",
    "Solve {task}",
    "What is {task}?",
    "Find {task}",
    "Compute {task}",
]

SCIENCE_VARIANTS = [
    "Explain {task}",
    "Describe {task}",
    "What is {task}?",
    "Tell me about {task}",
    "Define {task}",
]

CODE_VARIANTS = [
    "Write a Python function to {task}",
    "Create a Python function for {task}",
    "Implement {task} in Python",
    "Code a function to {task}",
    "Write code to {task}",
]

GENERAL_VARIANTS = [
    "{task}",
    "Who {task}?",
    "What {task}?",
    "When {task}?",
    "Tell me {task}",
]

COMMONSENSE_VARIANTS = [
    "Why {task}?",
    "Explain why {task}",
    "What is the reason {task}?",
    "Why do {task}?",
    "What causes {task}?",
]

# Sample base content for augmentation
BASE_SAMPLES = {
    "math": [
        "2+2", "the square root of 144", "5 multiplied by 7", 
        "the derivative of x^2", "100 divided by 4", "factorial of 5",
        "sin(45 degrees)", "log base 10 of 100", "3^4"
    ],
    "science": [
        "photosynthesis", "gravity", "the water cycle", "cell division",
        "Newton's laws", "DNA", "evolution", "the solar system", "atoms"
    ],
    "code": [
        "reverse a string", "sort a list", "find duplicates in an array",
        "calculate fibonacci", "implement binary search", "parse JSON",
        "read a file", "merge two dictionaries", "remove whitespace"
    ],
    "general": [
        "won World Cup 2018", "is the capital of France", "invented the telephone",
        "won the Nobel Prize in 2020", "is the tallest mountain", "wrote Hamlet",
        "is the largest ocean", "founded Microsoft", "painted the Mona Lisa"
    ],
    "commonsense": [
        "do we sleep", "is the sky blue", "do birds fly south", "does ice float",
        "do we get thirsty", "does fire need oxygen", "do magnets attract",
        "does the sun rise in the east", "do we need food"
    ]
}

def augment_dataset(input_file, output_file, multiplier=10):
    """
    Augment the dataset by creating variations of existing samples.
    
    Args:
        input_file: Original JSONL file
        output_file: Augmented JSONL file
        multiplier: How many samples to generate per category
    """
    # Load original data
    original = []
    with open(input_file, 'r') as f:
        for line in f:
            original.append(json.loads(line))
    
    augmented = []
    
    # Generate augmented samples
    for category in ["math", "science", "code", "general", "commonsense"]:
        bases = BASE_SAMPLES[category]
        
        if category == "math":
            templates = MATH_VARIANTS
        elif category == "science":
            templates = SCIENCE_VARIANTS
        elif category == "code":
            templates = CODE_VARIANTS
        elif category == "general":
            templates = GENERAL_VARIANTS
        else:
            templates = COMMONSENSE_VARIANTS
        
        # Generate samples
        for _ in range(multiplier):
            template = random.choice(templates)
            base = random.choice(bases)
            
            if "{task}" in template:
                query = template.replace("{task}", base)
            else:
                query = template
            
            augmented.append({
                "query": query,
                "instruction": query,
                "category": category
            })
    
    # Add original samples
    augmented.extend(original)
    
    # Shuffle
    random.shuffle(augmented)
    
    # Save
    with open(output_file, 'w') as f:
        for item in augmented:
            f.write(json.dumps(item) + '\n')
    
    print(f"Generated {len(augmented)} samples (from {len(original)} original)")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Combine train and val, then augment
    combined = []
    for file in ["data/router_train.jsonl", "data/router_val.jsonl"]:
        with open(file, 'r') as f:
            for line in f:
                combined.append(json.loads(line))
    
    # Save combined
    with open("data/router_combined.jsonl", 'w') as f:
        for item in combined:
            f.write(json.dumps(item) + '\n')
    
    # Augment
    augment_dataset("data/router_combined.jsonl", "data/router_augmented.jsonl", multiplier=20)
    
    print("\nCategory distribution:")
    cats = {}
    with open("data/router_augmented.jsonl", 'r') as f:
        for line in f:
            cat = json.loads(line)["category"]
            cats[cat] = cats.get(cat, 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
