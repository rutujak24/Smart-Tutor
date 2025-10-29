"""
Training script for DistilBERT Router
Trains a router to classify queries into task categories
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from distilbert_router import DistilBertRouter
import json
import argparse
from tqdm import tqdm
import os
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Optional wandb import - only required if using --wandb_project
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class RouterDataset(Dataset):
    """Dataset for training the router."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: DistilBertTokenizer,
        category_to_id: Dict[str, int],
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            data: List of dictionaries with 'instruction', 'input', 'category'
            tokenizer: DistilBERT tokenizer
            category_to_id: Mapping from category names to IDs
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.category_to_id = category_to_id
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine instruction and input
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        
        if input_text and input_text.strip():
            text = f"{instruction}\n\n{input_text}"
        else:
            text = instruction
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get category label
        category = item.get("category", "general")
        label = self.category_to_id.get(category, 0)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_combined_dataset(data_path: str) -> List[Dict]:
    """Load the combined dataset from JSONL file."""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_metrics(predictions, labels, category_names):
    """Compute classification metrics."""
    # Classification report
    report = classification_report(
        labels, 
        predictions, 
        target_names=category_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    return report, cm


def train_epoch(
    model: DistilBertRouter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs["logits"], labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(
    model: DistilBertRouter,
    dataloader: DataLoader,
    device: str,
    category_names: List[str]
) -> Dict:
    """Evaluate the model."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs["logits"], labels)
            
            total_loss += loss.item()
            all_predictions.extend(outputs["predictions"].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    report, cm = compute_metrics(all_predictions, all_labels, category_names)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = report["accuracy"]
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT Router")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to combined dataset JSONL file")
    parser.add_argument("--output_dir", type=str, default="./router_model",
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use for validation")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Weights & Biases project name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize wandb if specified
    if args.wandb_project:
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is required when using --wandb_project. "
                "Install it with: pip install wandb"
            )
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print("Loading dataset...")
    data = load_combined_dataset(args.data_path)
    print(f"Loaded {len(data)} samples")
    
    # Extract unique categories
    categories = sorted(list(set(item.get("category", "general") for item in data)))
    print(f"Found {len(categories)} categories: {categories}")
    
    category_to_id = {cat: idx for idx, cat in enumerate(categories)}
    
    # Split data
    train_data, val_data = train_test_split(
        data, 
        test_size=args.test_size, 
        random_state=args.seed,
        stratify=[item.get("category", "general") for item in data]
    )
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create datasets
    train_dataset = RouterDataset(train_data, tokenizer, category_to_id, args.max_length)
    val_dataset = RouterDataset(val_data, tokenizer, category_to_id, args.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Initialize model
    print("Initializing model...")
    model = DistilBertRouter(
        num_categories=len(categories),
        category_names=categories
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Starting training...")
    best_accuracy = 0.0
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch + 1)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        eval_results = evaluate(model, val_loader, device, categories)
        print(f"Val Loss: {eval_results['loss']:.4f}")
        print(f"Val Accuracy: {eval_results['accuracy']:.4f}")
        
        # Log to wandb
        if args.wandb_project:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": eval_results['loss'],
                "val_accuracy": eval_results['accuracy']
            })
        
        # Print per-category metrics
        print("\nPer-category metrics:")
        for category in categories:
            if category in eval_results['classification_report']:
                metrics = eval_results['classification_report'][category]
                print(f"  {category}: Precision={metrics['precision']:.4f}, "
                      f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
        
        # Save best model
        if eval_results['accuracy'] > best_accuracy:
            best_accuracy = eval_results['accuracy']
            print(f"\nNew best accuracy: {best_accuracy:.4f}, saving model...")
            model.save_pretrained(args.output_dir)
            
            # Save tokenizer
            tokenizer.save_pretrained(args.output_dir)
            
            # Save metrics
            with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
                json.dump(eval_results, f, indent=2, default=str)
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved to {args.output_dir}")
    
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
