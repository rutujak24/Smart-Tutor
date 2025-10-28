"""
Router Inference Script
Uses the trained DistilBERT router to classify queries and route them to appropriate models
"""

import torch
from transformers import DistilBertTokenizer
from distilbert_router import DistilBertRouter, RouterInference
import json
import argparse
from typing import List, Dict
import os


def load_router(model_path: str, device: str = "cuda") -> tuple:
    """
    Load the trained router model and tokenizer.
    
    Args:
        model_path: Path to the saved router model
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading router from {model_path}...")
    
    # Load model
    model = DistilBertRouter.load_pretrained(model_path, device)
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    
    print(f"Router loaded successfully!")
    print(f"Categories: {model.category_names}")
    
    return model, tokenizer


def route_single_query(
    inference: RouterInference,
    query: str,
    verbose: bool = True
) -> Dict:
    """
    Route a single query and optionally print results.
    
    Args:
        inference: RouterInference object
        query: Query text
        verbose: Whether to print results
        
    Returns:
        Routing result dictionary
    """
    result = inference.route_query(query, return_confidence=True)
    
    if verbose:
        print(f"\nQuery: {query}")
        print(f"Predicted Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nAll Category Scores:")
        for cat, score in sorted(result['confidence_scores'].items(), 
                                 key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {score:.4f}")
    
    return result


def route_batch_queries(
    inference: RouterInference,
    queries: List[str],
    output_file: str = None,
    batch_size: int = 32
) -> List[Dict]:
    """
    Route multiple queries in batch mode.
    
    Args:
        inference: RouterInference object
        queries: List of query texts
        output_file: Optional path to save results
        batch_size: Batch size for processing
        
    Returns:
        List of routing results
    """
    print(f"\nRouting {len(queries)} queries...")
    results = inference.route_batch(queries, batch_size=batch_size)
    
    # Print summary
    category_counts = {}
    for result in results:
        cat = result['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nRouting Summary:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} queries ({count/len(queries)*100:.1f}%)")
    
    # Save results if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", 
                    exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


def interactive_mode(inference: RouterInference):
    """
    Interactive mode for testing the router with manual queries.
    
    Args:
        inference: RouterInference object
    """
    print("\n" + "="*60)
    print("Interactive Router Mode")
    print("Type 'quit' or 'exit' to stop")
    print("="*60)
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode.")
                break
            
            if not query:
                continue
            
            route_single_query(inference, query, verbose=True)
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"Error: {e}")


def evaluate_on_test_set(
    inference: RouterInference,
    test_data_path: str,
    output_file: str = None
) -> Dict:
    """
    Evaluate router on a test dataset with known categories.
    
    Args:
        inference: RouterInference object
        test_data_path: Path to test JSONL file with 'instruction' and 'category'
        output_file: Optional path to save evaluation results
        
    Returns:
        Evaluation metrics
    """
    print(f"\nEvaluating on test set: {test_data_path}")
    
    # Load test data
    test_data = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Prepare queries and true labels
    queries = []
    true_labels = []
    for item in test_data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        if input_text and input_text.strip():
            query = f"{instruction}\n\n{input_text}"
        else:
            query = instruction
        queries.append(query)
        true_labels.append(item.get("category", "general"))
    
    # Get predictions
    results = inference.route_batch(queries, batch_size=32, return_confidence=True)
    predictions = [r['category'] for r in results]
    
    # Calculate accuracy
    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    accuracy = correct / len(predictions)
    
    # Per-category accuracy
    category_metrics = {}
    for category in set(true_labels):
        cat_indices = [i for i, label in enumerate(true_labels) if label == category]
        cat_correct = sum(1 for i in cat_indices if predictions[i] == true_labels[i])
        category_metrics[category] = {
            "total": len(cat_indices),
            "correct": cat_correct,
            "accuracy": cat_correct / len(cat_indices) if cat_indices else 0.0
        }
    
    # Summary
    evaluation = {
        "overall_accuracy": accuracy,
        "total_samples": len(predictions),
        "correct_predictions": correct,
        "category_metrics": category_metrics
    }
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
    print("\nPer-Category Accuracy:")
    for cat, metrics in sorted(category_metrics.items()):
        print(f"  {cat}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", 
                    exist_ok=True)
        
        detailed_results = {
            "evaluation_metrics": evaluation,
            "detailed_results": [
                {
                    "query": q,
                    "true_category": t,
                    "predicted_category": p,
                    "correct": p == t,
                    "confidence": r['confidence'],
                    "all_scores": r['confidence_scores']
                }
                for q, t, p, r in zip(queries, true_labels, predictions, results)
            ]
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")
    
    return evaluation


def main():
    parser = argparse.ArgumentParser(description="Router Inference and Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained router model")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "batch", "interactive", "evaluate"],
                        help="Inference mode")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query for 'single' mode")
    parser.add_argument("--queries_file", type=str, default=None,
                        help="File with queries (one per line) for 'batch' mode")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Test JSONL file for 'evaluate' mode")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for batch mode")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Load router
    model, tokenizer = load_router(args.model_path, args.device)
    inference = RouterInference(model, tokenizer, args.device)
    
    # Run appropriate mode
    if args.mode == "single":
        if not args.query:
            print("Error: --query is required for single mode")
            return
        route_single_query(inference, args.query)
    
    elif args.mode == "batch":
        if not args.queries_file:
            print("Error: --queries_file is required for batch mode")
            return
        
        # Load queries from file
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        
        route_batch_queries(inference, queries, args.output_file, args.batch_size)
    
    elif args.mode == "interactive":
        interactive_mode(inference)
    
    elif args.mode == "evaluate":
        if not args.test_data:
            print("Error: --test_data is required for evaluate mode")
            return
        
        evaluate_on_test_set(inference, args.test_data, args.output_file)


if __name__ == "__main__":
    main()
