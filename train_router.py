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

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("instruction", "")
        label = self.category_to_id[item["category"]]
        enc = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt"
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
    rep = classification_report(labs, preds, target_names=names, output_dict=True)
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
    parser.add_argument("--finetune_lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="Number of bottom transformer layers to freeze")
    args = parser.parse_args()

    # Load dataset
    data = load_jsonl(args.data_path)
    cats = sorted(list(set(x["category"] for x in data)))
    cat2id = {c: i for i, c in enumerate(cats)}

    train, val = train_test_split(data, test_size=0.2, stratify=[x["category"] for x in data])
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
        print(f"🧊 Froze first {args.freeze_layers} transformer layers.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup_steps, len(tr_dl)*args.epochs)

    print(f"🚀 Training router ({len(cats)} classes) on {len(train)} samples...")
    best_acc = 0
    for e in range(1, args.epochs+1):
        model.train()
        tot = 0
        for b in tqdm(tr_dl, desc=f"Epoch {e}"):
            b = {k: v.to(device) for k, v in b.items()}
            out = model(**b)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
        rep, cm = evaluate(model, va_dl, device, cats)
        acc = rep["accuracy"]
        print(f"Epoch {e} | Loss {tot/len(tr_dl):.4f} | Acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(args.output_dir)
            tok.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
                json.dump({"report": rep, "cm": cm}, f, indent=2)
            print(f"💾 Best model saved (Acc={best_acc:.4f})")

    # Optional fine-tuning phase
    if args.finetune_lr:
        print(f"\n🎯 Starting fine-tuning phase with lr={args.finetune_lr} ...")
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
        print("✅ Fine-tuning complete.")

    print("✅ Training complete.")


if __name__ == "__main__":
    main()
