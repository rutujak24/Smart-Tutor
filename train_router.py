import os, json, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import (
    DistilBertTokenizer,
    DistilBertConfig,
    get_cosine_schedule_with_warmup,
)
from distilbert_router import DistilBertRouter


class RouterDataset(Dataset):
    def __init__(self, data, tokenizer, category_to_id, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.category_to_id = category_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Support multiple possible text fields
        text = item.get("instruction") or item.get("query") or item.get("text") or ""
        # Optionally concatenate input field if present
        input_part = item.get("input")
        if input_part and isinstance(input_part, str) and input_part.strip():
            text = f"{text}\n\n{input_part}" if text else input_part
        label = self.category_to_id[item["category"]]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_jsonl(p): return [json.loads(x) for x in open(p, "r")]


def evaluate(model, dl, device, names):
    model.eval()
    preds, labs = [], []
    for b in tqdm(dl, desc="Evaluating"):
        b = {k: v.to(device) for k, v in b.items()}
        out = model(**b)
        preds += out["predictions"].cpu().tolist()
        labs += b["labels"].cpu().tolist()
    # Use explicit labels to allow reporting of all categories even if absent in tiny val set
    rep = classification_report(
        labs,
        preds,
        labels=list(range(len(names))),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labs, preds).tolist()
    return rep, cm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="router_model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum improvement to reset patience")
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--class_weights", action="store_true", help="Use class weights (inverse frequency)")
    parser.add_argument("--finetune_lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="Number of bottom transformer layers to freeze")
    args = parser.parse_args()

    # Load dataset
    data = load_jsonl(args.data_path)
    cats = sorted(list(set(x["category"] for x in data)))
    cat2id = {c: i for i, c in enumerate(cats)}

    # Robust split: handle tiny datasets where stratified split is impossible
    from collections import Counter
    label_counts = Counter([x["category"] for x in data])
    can_stratify = min(label_counts.values()) >= 2 and len(set(label_counts.values())) > 0
    try:
        if can_stratify:
            train, val = train_test_split(data, test_size=0.2, stratify=[x["category"] for x in data])
        else:
            # Fallback: simple shuffle split without stratification
            import random
            random.shuffle(data)
            split_idx = max(1, int(0.2 * len(data))) if len(data) > 5 else 1
            val = data[:split_idx]
            train = data[split_idx:]
            if len(train) == 0:  # ensure non-empty train
                train, val = data, data[:]
            print(f"Stratified split skipped (insufficient samples per class). Using {len(train)} train / {len(val)} val.")
    except Exception as e:
        print(f"Stratified split failed ({e}); using non-stratified fallback.")
        import random
        random.shuffle(data)
        split_idx = max(1, int(0.2 * len(data))) if len(data) > 5 else 1
        val = data[:split_idx]
        train = data[split_idx:]
        if len(train) == 0:
            train, val = data, data[:]
        print(f"Using {len(train)} train / {len(val)} val.")
    tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tr_dl = DataLoader(RouterDataset(train, tok, cat2id), batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(RouterDataset(val, tok, cat2id), batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DistilBertConfig.from_pretrained("distilbert-base-uncased")
    cfg.num_labels = len(cats)
    model = DistilBertRouter(cfg, num_categories=len(cats), category_names=cats).to(device)

    # Optionally freeze lower layers
    if args.freeze_layers > 0:
        for param in model.distilbert.transformer.layer[:args.freeze_layers].parameters():
            param.requires_grad = False
        print(f"Froze first {args.freeze_layers} transformer layers.")

    # Class weights (optional)
    class_weights = None
    if args.class_weights:
        counts = np.zeros(len(cats))
        for item in train:
            counts[cat2id[item["category"]]] += 1
        inv_freq = 1.0 / (counts + 1e-6)
        class_weights = torch.tensor(inv_freq / inv_freq.sum() * len(cats), dtype=torch.float, device=device)
        model.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        if class_weights is not None:
            print(f"Applied class weights: {class_weights.tolist()}\nCounts: {counts.tolist()}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup_steps, len(tr_dl)*args.epochs)

    print(f"Training router ({len(cats)} classes) on {len(train)} samples...")
    best_acc = 0
    saved_any = False
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.startswith("cuda"))
    patience_counter = 0

    for e in range(1, args.epochs + 1):
        model.train()
        tot = 0
        opt.zero_grad()
        for step, b in enumerate(tqdm(tr_dl, desc=f"Epoch {e}")):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.cuda.amp.autocast(enabled=args.use_amp and device.startswith("cuda")):
                out = model(**b)
                loss = out["loss"] / args.grad_accum_steps
            scaler.scale(loss).backward()
            tot += loss.item() * args.grad_accum_steps
            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
        rep, cm = evaluate(model, va_dl, device, cats)
        acc = rep["accuracy"]
        print(f"Epoch {e} | Loss {tot/len(tr_dl):.4f} | Acc {acc:.4f}")
        improved = acc - best_acc > args.min_delta
        if improved:
            best_acc = acc
            model.save_pretrained(args.output_dir)
            tok.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
                json.dump({"report": rep, "cm": cm}, f, indent=2)
            print(f"Best model saved (Acc={best_acc:.4f})")
            patience_counter = 0
            saved_any = True
        else:
            patience_counter += 1
            print(f"No improvement (patience {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # Save final model if none saved (tiny dataset safety net)
    if not saved_any:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tok.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
            json.dump({"report": rep, "cm": cm}, f, indent=2)
        print("Final model saved despite no accuracy improvement (tiny dataset).")

    # Optional fine-tuning phase
    if args.finetune_lr:
        print(f"\nStarting fine-tuning phase with lr={args.finetune_lr} ...")
        opt = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr)
        sched = get_cosine_schedule_with_warmup(opt, 100, len(tr_dl)*3)
        for e in range(1, 4):
            model.train()
            for b in tqdm(tr_dl, desc=f"Fine-tune Epoch {e}"):
                b = {k: v.to(device) for k, v in b.items()}
                out = model(**b)
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        rep, cm = evaluate(model, va_dl, device, cats)
        model.save_pretrained(args.output_dir)
        tok.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "eval_metrics_finetune.json"), "w") as f:
            json.dump({"report": rep, "cm": cm}, f, indent=2)
        print("Fine-tuning complete.")

    print("Training complete.")


if __name__ == "__main__":
    main()
