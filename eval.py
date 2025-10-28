import lm_eval
import os
import sys
import argparse
import json
import torch
from datetime import datetime
import wandb
wandb.login(key=os.getenv('WANDB'), force=True)
from lm_eval import evaluator
from lm_eval.loggers.wandb_logger import WandbLogger


pretrained = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"



def run_eval_for_model(model_name: str, task: str, device: str, batch_size: int, save_json: str, limit: int | None , wandb_project: str | None, wandb_group: str | None):

    model_args = {"pretrained": pretrained, "peft": model_name, "parallelize": True}

    print(f"Evaluating model {pretrained} on task {task} (device={device}, limit={limit})")
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks = [task],
        device=device,
        batch_size=batch_size,
        limit=limit,
        log_samples=True,
        apply_chat_template=True,
    )

    # pretty print a short summary
    if results and "results" in results:
        print("Results summary:\n", json.dumps(results.get("results", {}), indent=2))

    

    # Save JSON
    if results is not None:
        try:
            out_path = save_json
            os.makedirs(out_path, exist_ok=True)
            model_name = model_name.replace("/", "_")
            final_path = os.path.join(out_path, f"{model_name}_{task}_results.json")
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(results, f, default=str, indent=2)
            print(f"Saved results to {final_path}")
        except Exception as e:
            print("Failed to save results:", repr(e))
    
    if wandb_project is not None:
        wandb_args = {"project": wandb_project, "group": wandb_group, "name": f"{model_name}_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
        wandb_logger = WandbLogger(init_args= wandb_args)
        wandb_logger.post_init(results)
        wandb_logger.log_eval_result()
        

    return results


def main():
    parse = argparse.ArgumentParser(description="Evaluate multiple models on multiple tasks with lm_eval.")
    parse.add_argument("--model_name", type=str, required=True, help="Model identifier for the evaluation")
    parse.add_argument("--tasks", type=str, required=True, help="Task name for evaluation")
    parse.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation on (default: cuda:0)")
    parse.add_argument("--limit", type=int, default=None,
                          help="Limit the number of samples to evaluate (default: None, meaning no limit)")
    parse.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation (default: 1)")
    parse.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results")
    parse.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
    parse.add_argument("--wandb_group", type=str, default=None, help="Weights & Biases group name")
                       
    args = parse.parse_args()
    run_eval_for_model(args.model_name, args.tasks, args.device, args.batch_size, args.output_dir, args.limit, args.wandb_project, args.wandb_group)


if __name__ == "__main__":
    main()

