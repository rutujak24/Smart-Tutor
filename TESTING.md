# Router Testing Guide

Quick commands to test your DistilBERT router implementation.

## 1. Validate Setup

Check if everything is installed correctly:

```bash
# Check Python version (need 3.8+)
python --version

# Install dependencies
pip install -r requirements_router.txt

# Quick test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "from distilbert_router import DistilBertRouter; print('✓ Router imports OK')"
```

## 2. Create Test Data

Create some sample data to train on:

```bash
# Create data directory
mkdir -p data

# Create sample training data
cat > data/router_train.jsonl << 'EOF'
{"query": "What is 2+2?", "category": "math"}
{"query": "Calculate 15% of 200", "category": "math"}
{"query": "Solve x^2 = 16", "category": "math"}
{"query": "Explain photosynthesis", "category": "science"}
{"query": "What is gravity?", "category": "science"}
{"query": "How does DNA work?", "category": "science"}
{"query": "Write a hello world program", "category": "code"}
{"query": "Debug this Python code", "category": "code"}
{"query": "Create a function to sort array", "category": "code"}
{"query": "Who won World Cup 2018?", "category": "general"}
{"query": "Capital of France?", "category": "general"}
{"query": "When did WW2 end?", "category": "general"}
{"query": "Why do we need umbrellas?", "category": "commonsense"}
{"query": "What happens if you don't sleep?", "category": "commonsense"}
{"query": "Why do birds fly south?", "category": "commonsense"}
EOF

# Create validation data
cat > data/router_val.jsonl << 'EOF'
{"query": "What is 50 + 50?", "category": "math"}
{"query": "Explain mitosis", "category": "science"}
{"query": "Write a for loop", "category": "code"}
{"query": "Who is the president?", "category": "general"}
{"query": "Why do we eat food?", "category": "commonsense"}
EOF

echo "✓ Test data created in data/"
```

## 3. Test Model Creation

Test if the router model can be created:

```bash
python -c "
from distilbert_router import DistilBertRouter

categories = ['math', 'science', 'code', 'general', 'commonsense']
model = DistilBertRouter(num_categories=len(categories), category_names=categories)
print(f'✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters')
"
```

## 4. Train Router (Quick Test)

Train for just 1 epoch to test the pipeline:

```bash
python train_router.py \
  --train_data data/router_train.jsonl \
  --val_data data/router_val.jsonl \
  --output_dir router_model_test \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --device cpu
```

**Expected output:**
- Loading data...
- Training progress bar
- Validation metrics
- Model saved to `router_model_test/`

## 5. Test Inference - Single Query

Test routing a single query:

```bash
python router_inference.py \
  --model_path router_model_test \
  --mode single \
  --query "What is the square root of 144?"
```

**Expected output:**
```json
{
  "query": "What is the square root of 144?",
  "category": "math",
  "confidence": 0.XX,
  "all_scores": { ... }
}
```

## 6. Test Inference - Interactive Mode

Test interactive mode:

```bash
python router_inference.py \
  --model_path router_model_test \
  --mode interactive
```

**Try these queries:**
```
What is 2+2?                          → should be "math"
Explain how plants grow               → should be "science"  
Write a Python function               → should be "code"
Who won the Olympics?                 → should be "general"
Why do we need to sleep?              → should be "commonsense"
exit                                  → to quit
```

## 7. Test Inference - Batch Mode

Create test queries file:

```bash
cat > test_queries.jsonl << 'EOF'
{"query": "What is 10 * 5?"}
{"query": "How does the sun work?"}
{"query": "Debug my JavaScript code"}
{"query": "When was America founded?"}
{"query": "Why do people laugh?"}
EOF
```

Run batch inference:

```bash
python router_inference.py \
  --model_path router_model_test \
  --mode batch \
  --input_file test_queries.jsonl \
  --output_file test_results.jsonl
```

Check results:

```bash
cat test_results.jsonl
```

## 8. Test Inference - Evaluate Mode

Test evaluation with labeled data:

```bash
python router_inference.py \
  --model_path router_model_test \
  --mode evaluate \
  --test_file data/router_val.jsonl
```

**Expected output:**
- Accuracy score
- Classification report with precision/recall/F1
- Confusion matrix

## 9. Test Python API

Test using the router in Python:

```bash
python << 'EOF'
from distilbert_router import RouterInference

# Load router
print("Loading router...")
router = RouterInference.from_pretrained("router_model_test", device="cpu")

# Test single query
queries = [
    "What is the integral of x^2?",
    "Explain photosynthesis",
    "Write a sorting algorithm",
    "Who invented the lightbulb?",
    "Why do we dream?"
]

print("\nTesting queries:\n")
for query in queries:
    result = router.route_query(query)
    print(f"Query: {query[:40]}")
    print(f"  → {result['category']} (confidence: {result['confidence']:.3f})\n")

print("✓ Python API test complete!")
EOF
```

## 10. Performance Test

Test inference speed:

```bash
python << 'EOF'
from distilbert_router import RouterInference
import time

router = RouterInference.from_pretrained("router_model_test", device="cpu")

# Test single query speed
query = "What is 2+2?"
start = time.time()
for _ in range(10):
    router.route_query(query)
avg_time = (time.time() - start) / 10
print(f"Average inference time: {avg_time*1000:.2f}ms")

# Test batch speed
queries = ["Test query"] * 32
start = time.time()
router.route_batch(queries, batch_size=32)
batch_time = time.time() - start
print(f"Batch (32 queries) time: {batch_time*1000:.2f}ms")
print(f"Throughput: {32/batch_time:.1f} queries/sec")
EOF
```

## 11. Check Model Info

Get model details:

```bash
python << 'EOF'
from distilbert_router import DistilBertRouter
import os

model = DistilBertRouter.load_pretrained("router_model_test", device="cpu")

print(f"Categories: {model.category_names}")
print(f"Num categories: {model.num_categories}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check model file size
model_size = os.path.getsize("router_model_test/router_model.pt") / 1024 / 1024
print(f"Model size: {model_size:.1f} MB")
EOF
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Use CPU instead
python train_router.py --device cpu ...
```

### ImportError
```bash
# Reinstall dependencies
pip install -r requirements_router.txt
```

### Low Accuracy
- Need more training data (at least 100+ examples per category)
- Train for more epochs (3-5)
- Balance categories (equal samples per category)

### Slow Training
- Increase batch size: `--batch_size 16`
- Use GPU: `--device cuda:0`

## Quick Test Summary

```bash
# Full quick test
pip install -r requirements_router.txt && \
mkdir -p data && \
echo '{"query": "What is 2+2?", "category": "math"}' > data/router_train.jsonl && \
echo '{"query": "What is 5+5?", "category": "math"}' > data/router_val.jsonl && \
python train_router.py --train_data data/router_train.jsonl --val_data data/router_val.jsonl --output_dir router_test --epochs 1 --batch_size 2 --device cpu && \
python router_inference.py --model_path router_test --mode single --query "What is 10+10?"
```

If all tests pass, your router is working correctly! ✅
