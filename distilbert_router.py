"""
distilbert_router.py
--------------------
DistilBERT-based router model for query classification.

Purpose:
    Acts as a lightweight classifier to route user queries
    to specialized LLM experts (math, science, code, etc.)

Compatible with:
    - train_router.py  (for supervised training)
    - moe_router_inference.py  (for live routing)
"""

import os
import json
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertPreTrainedModel
from typing import List, Optional, Dict


# ============================================================
#  Main Router Model
# ============================================================
class DistilBertRouter(DistilBertPreTrainedModel):
    """
    DistilBERT-based classification head for routing tasks.
    """

    def __init__(self, config, num_categories: int = 5, category_names: Optional[List[str]] = None):
        super().__init__(config)
        self.num_labels = num_categories or config.num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.category_names = category_names or ["code", "commonsense", "general", "math", "science"]
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass through DistilBERT encoder + classification head.
        Returns logits, predictions, and optional loss.
        """
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]          # [CLS] token embedding
        cls_token = self.dropout(cls_token)
        logits = self.classifier(cls_token)
        preds = torch.argmax(logits, dim=-1)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits, "predictions": preds}

    # ------------------------------------------------------------
    # Save / Load utilities
    # ------------------------------------------------------------
    def save_pretrained(self, save_dir: str):
        """
        Save model weights and router metadata for later inference.
        """
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "router_model.pt"))
        meta = {
            "num_labels": self.num_labels,
            "category_names": self.category_names,
        }
        with open(os.path.join(save_dir, "router_config.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Router model saved to {save_dir}")

    @classmethod
    def load_pretrained(cls, save_dir: str, config, device: str = None):
        """
        Load a trained router from disk.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(save_dir, "router_config.json"), "r") as f:
            meta = json.load(f)
        model = cls(config, num_categories=meta["num_labels"], category_names=meta["category_names"])
        model.load_state_dict(torch.load(os.path.join(save_dir, "router_model.pt"), map_location=device))
        return model.to(device)


# ============================================================
#  Optional Inference Wrapper
# ============================================================
class RouterInference:
    """
    Helper wrapper for single-query inference.
    """

    def __init__(self, model: DistilBertRouter, tokenizer, device: str = None, max_length: int = 256):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model.to(self.device)

    @torch.no_grad()
    def route_query(self, query: str, return_confidence: bool = True) -> Dict[str, float]:
        """
        Classify a query into one of the router’s categories.
        Returns the predicted label and confidence scores.
        """
        enc = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        category = self.model.category_names[pred_idx]
        confidence = float(probs[0, pred_idx])
        result = {"category": category, "confidence": confidence}
        if return_confidence:
            result["all_confidences"] = {
                self.model.category_names[i]: float(probs[0, i]) for i in range(len(self.model.category_names))
            }
        return result


# ============================================================
#  Manual Debug Test
# ============================================================
if __name__ == "__main__":
    from transformers import DistilBertTokenizer, DistilBertConfig

    print("Testing DistilBERT Router...")

    categories = ["code", "commonsense", "general", "math", "science"]
    cfg = DistilBertConfig.from_pretrained("distilbert-base-uncased")
    model = DistilBertRouter(cfg, num_categories=len(categories), category_names=categories)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    router = RouterInference(model, tokenizer)
    query = "Explain Newton's laws of motion."
    result = router.route_query(query)
    print(f"Query: {query}")
    print(f"Predicted Category: {result['category']}")
    print(f"Confidence: {result['confidence']:.4f}")
