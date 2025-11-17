import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizer, DistilBertConfig
from distilbert_router import DistilBertRouter

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def load_jsonl(path):
    """Load dataset from JSONL."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def plot_confusion_matrix(cm, labels, save_path):
    """Save confusion matrix heatmap."""
    try:
        import seaborn as sns
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Router Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"🖼️  Confusion matrix saved to {save_path}")
    except ImportError:
        print("(Plot skipped: seaborn not installed)")

# -------------------------------------------------------------
# Core evaluation
# -------------------------------------------------------------
def evaluate(model, tokenizer, data, device, label2id):
    model.eval()
    id2label = {v: k for k, v in label2id.items()}
    preds, trues = [], []

    with torch.no_grad():
        for sample in tqdm(data, desc="🔍 Evaluating"):
            text = sample["instruction"]
            true_label = sample["category"]
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt"
            ).to(device)
            out = model(**enc)
            pred_id = torch.argmax(out["logits"], dim=1).item()
            preds.append(pred_id)
            trues.append(label2id[true_label])

    cm = confusion_matrix(trues, preds)
    rep = classification_report(
        trues, preds,
        target_names=[id2label[i] for i in range(len(label2id))],
        output_dict=True,
        zero_division=0
    )
    acc = rep["accuracy"]
    macro_p = np.mean([rep[l]["precision"] for l in rep if l in label2id])
    macro_r = np.mean([rep[l]["recall"] for l in rep if l in label2id])
    macro_f1 = np.mean([rep[l]["f1-score"] for l in rep if l in label2id])
    return acc, macro_p, macro_r, macro_f1, rep, cm

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="router_test_3class.jsonl")
    parser.add_argument("--model_dir", default="router_model_moe_3class")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    print(f"📂 Loading dataset from {args.data_path}")
    data = load_jsonl(args.data_path)
    print(f"✅ Loaded {len(data)} records.")

    categories = sorted(list(set(x["category"] for x in data)))
    print(f"Detected categories → {categories}")
    label2id = {c: i for i, c in enumerate(categories)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device = {device}")

    # ---------------------------------------------------------
    # Load model safely even if file is router_model.pt
    # ---------------------------------------------------------
    cfg_path = os.path.join(args.model_dir, "router_config.json")
    if os.path.exists(cfg_path):
        cfg = DistilBertConfig.from_dict(json.load(open(cfg_path)))
    else:
        cfg = DistilBertConfig.from_pretrained("distilbert-base-uncased")
    cfg.num_labels = len(categories)

    model = DistilBertRouter(cfg, num_categories=len(categories), category_names=categories)
    weights_path_pt = os.path.join(args.model_dir, "router_model.pt")
    weights_path_bin = os.path.join(args.model_dir, "pytorch_model.bin")

    if os.path.exists(weights_path_pt):
        model.load_state_dict(torch.load(weights_path_pt, map_location=device))
        print(f"✅ Loaded weights from {weights_path_pt}")
    elif os.path.exists(weights_path_bin):
        model.load_state_dict(torch.load(weights_path_bin, map_location=device))
        print(f"✅ Loaded weights from {weights_path_bin}")
    else:
        raise FileNotFoundError("No weight file found (expected router_model.pt or pytorch_model.bin)")

    model.to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_dir)

    print("🧭 Evaluating router model...")
    acc, macro_p, macro_r, macro_f1, rep, cm = evaluate(model, tokenizer, data, device, label2id)

    print("\n=== 📊 ROUTER EVALUATION RESULTS ===")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall   : {macro_r:.4f}")
    print(f"Macro F1-score : {macro_f1:.4f}")

    metrics_path = os.path.join(args.results_dir, "router_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== ROUTER METRICS ===\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro Precision: {macro_p:.4f}\n")
        f.write(f"Macro Recall: {macro_r:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write("=== Detailed Classification Report ===\n")
        json.dump(rep, f, indent=2)
        f.write("\n\n=== Confusion Matrix ===\n")
        json.dump(cm.tolist(), f, indent=2)
    print(f"\n✅ Metrics saved to {metrics_path}")

    cm_path = os.path.join(args.results_dir, "router_confusion_matrix.png")
    plot_confusion_matrix(cm, categories, cm_path)


if __name__ == "__main__":
    main()
