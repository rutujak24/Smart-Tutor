"""
DistilBERT Router for Query Classification
Routes queries to specialized models based on task category
"""

import torch
import torch.nn as nn
from transformers import (
    DistilBertModel, 
    DistilBertTokenizer
)
from typing import Dict, List, Optional, Tuple
import json
import numpy as np


class DistilBertRouter(nn.Module):
    """
    DistilBERT-based router that classifies queries into categories
    to route them to appropriate specialized models.
    """
    
    def __init__(
        self, 
        num_categories: int,
        category_names: List[str],
        pretrained_model: str = "distilbert-base-uncased",
        dropout_rate: float = 0.1,
        hidden_size: int = 768
    ):
        """
        Initialize the DistilBERT Router.
        
        Args:
            num_categories: Number of task categories (e.g., math, science, general)
            category_names: List of category names
            pretrained_model: Name of pretrained DistilBERT model
            dropout_rate: Dropout rate for classification head
            hidden_size: Hidden size of DistilBERT (default 768)
        """
        super(DistilBertRouter, self).__init__()
        
        self.num_categories = num_categories
        self.category_names = category_names
        self.category_to_id = {name: idx for idx, name in enumerate(category_names)}
        self.id_to_category = {idx: name for idx, name in enumerate(category_names)}
        
        # Load pretrained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_categories)
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the router.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary with logits and predictions
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        
        return {
            "logits": logits,
            "predictions": predictions,
            "probabilities": probabilities
        }
    
    def predict_category(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_probabilities: bool = False
    ) -> Tuple[List[str], Optional[torch.Tensor]]:
        """
        Predict categories for input queries.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_probabilities: Whether to return confidence scores
            
        Returns:
            Tuple of (category names, probabilities if requested)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
        predictions = outputs["predictions"].cpu().numpy()
        category_names = [self.id_to_category[pred] for pred in predictions]
        
        if return_probabilities:
            return category_names, outputs["probabilities"]
        return category_names, None
    
    def save_pretrained(self, save_path: str):
        """Save the router model and configuration."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_path, "router_model.pt"))
        
        # Save configuration
        config = {
            "num_categories": self.num_categories,
            "category_names": self.category_names,
            "category_to_id": self.category_to_id,
            "id_to_category": self.id_to_category
        }
        with open(os.path.join(save_path, "router_config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load_pretrained(cls, load_path: str, device: str = "cuda"):
        """Load a pretrained router model."""
        import os
        
        # Load configuration
        config_path = os.path.join(load_path, "router_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(
            num_categories=config["num_categories"],
            category_names=config["category_names"]
        )
        
        # Load weights
        weights_path = os.path.join(load_path, "router_model.pt")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        
        return model.to(device)


class RouterInference:
    """
    Inference wrapper for the DistilBERT Router.
    Handles tokenization and model inference.
    """
    
    def __init__(
        self, 
        model: DistilBertRouter,
        tokenizer: DistilBertTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512
    ):
        """
        Initialize the router inference wrapper.
        
        Args:
            model: Trained DistilBertRouter model
            tokenizer: DistilBERT tokenizer
            device: Device to run inference on
            max_length: Maximum sequence length
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()
        
    def route_query(
        self, 
        query: str,
        return_confidence: bool = True
    ) -> Dict[str, any]:
        """
        Route a single query to the appropriate category.
        
        Args:
            query: Input query text
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with category and optional confidence scores
        """
        # Tokenize
        inputs = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        
        predicted_category = self.model.id_to_category[
            outputs["predictions"].item()
        ]
        
        result = {"category": predicted_category}
        
        if return_confidence:
            probabilities = outputs["probabilities"][0].cpu().numpy()
            confidence_scores = {
                self.model.id_to_category[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            result["confidence_scores"] = confidence_scores
            result["confidence"] = float(probabilities[outputs["predictions"].item()])
        
        return result
    
    def route_batch(
        self,
        queries: List[str],
        batch_size: int = 32,
        return_confidence: bool = True
    ) -> List[Dict[str, any]]:
        """
        Route multiple queries in batches.
        
        Args:
            queries: List of query texts
            batch_size: Batch size for processing
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of routing results
        """
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_queries,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model.forward(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
            
            # Process results
            predictions = outputs["predictions"].cpu().numpy()
            probabilities = outputs["probabilities"].cpu().numpy()
            
            for j, pred in enumerate(predictions):
                result = {
                    "query": batch_queries[j],
                    "category": self.model.id_to_category[pred]
                }
                
                if return_confidence:
                    confidence_scores = {
                        self.model.id_to_category[k]: float(prob)
                        for k, prob in enumerate(probabilities[j])
                    }
                    result["confidence_scores"] = confidence_scores
                    result["confidence"] = float(probabilities[j][pred])
                
                results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    # Define categories based on your dataset
    categories = ["math", "science", "general", "commonsense", "code"]
    
    # Initialize router
    router = DistilBertRouter(
        num_categories=len(categories),
        category_names=categories
    )
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create inference wrapper
    inference = RouterInference(router, tokenizer)
    
    # Example query
    query = "What is 2 + 2?"
    result = inference.route_query(query)
    print(f"Query: {query}")
    print(f"Predicted category: {result['category']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"All scores: {result['confidence_scores']}")
