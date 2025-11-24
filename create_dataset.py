import json
import os
import argparse
import yaml
from typing import List, Dict, Any, Optional
from datasets import load_dataset

def get_nested_field(item: Dict, field_path: str) -> Any:
    """
    Extract field from item using dot notation for nested fields.
    Example: "answer.value" will get item["answer"]["value"]
    """
    if not field_path:
        return None
    
    try:
        value = item
        for key in field_path.split('.'):
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value
    except:
        return None
    
def extract_text_value(value: Any) -> str:
    if value is None:
        return ""
    elif isinstance(value, str):
        return value
    elif isinstance(value, list):
        # Take first non-empty item
        for item in value:
            text = extract_text_value(item)
            if text:
                return text
        return ""
    elif isinstance(value, dict):
        # Try common keys
        for key in ["value", "text", "answer", "normalized_value"]:
            if key in value:
                return extract_text_value(value[key])
        # If no common key, return first value
        for v in value.values():
            text = extract_text_value(v)
            if text:
                return text
        return ""
    else:
        return str(value)

def format_choices(choices: Any, answer_key: str = None) -> tuple:
    """
    Format multiple choice options and get answer text.
    Returns: (formatted_choices_text, answer_text)
    """
    if not choices:
        return "", ""
    
    formatted_lines = []
    answer_text = ""
    
    try:
        # Handle different choice formats
        if isinstance(choices, dict):
            # Format: {"label": ["A", "B", ...], "text": ["option1", "option2", ...]}
            if "label" in choices and "text" in choices:
                labels = choices["label"]
                texts = choices["text"]
                for label, text in zip(labels, texts):
                    formatted_lines.append(f"{label}. {text}")
                    if answer_key and label == answer_key:
                        answer_text = text
            # Format: {"A": "text1", "B": "text2", ...}
            else:
                for label, text in choices.items():
                    formatted_lines.append(f"{label}. {text}")
                    if answer_key and label == answer_key:
                        answer_text = text
        
        elif isinstance(choices, list):
            # Format: ["option1", "option2", ...]
            for i, text in enumerate(choices):
                label = chr(65 + i)  # A, B, C, ...
                formatted_lines.append(f"{label}. {text}")
                if answer_key and (label == answer_key or str(i) == str(answer_key)):
                    answer_text = text
        
        formatted_text = "\n".join(formatted_lines) if formatted_lines else ""
        return formatted_text, answer_text
        
    except Exception as e:
        print(f"Error formatting choices: {e}")
        return "", ""

def process_item(item: Dict, config: Dict) -> Optional[Dict]:
    """
    Process a single item using the configuration.
    Returns None if item is invalid.
    """
    try:
        # Extract instruction
        instruction = extract_text_value(
            get_nested_field(item, config.get("instruction_field"))
        )
        
        if not instruction:
            return None
        
        # Extract input (optional)
        input_text = ""
        if config.get("input_field"):
            input_text = extract_text_value(
                get_nested_field(item, config.get("input_field"))
            )
        
        # Extract output
        output = extract_text_value(
            get_nested_field(item, config.get("output_field"))
        )
        
        # Handle multiple choice questions
        if config.get("choices_field"):
            choices = get_nested_field(item, config.get("choices_field"))
            answer_key = extract_text_value(
                get_nested_field(item, config.get("answer_key_field"))
            ) if config.get("answer_key_field") else None
            
            choices_text, answer_text = format_choices(choices, answer_key)
            
            # Add choices to instruction if requested
            if config.get("include_choices_in_instruction") and choices_text:
                instruction = f"{instruction}\n\nChoices:\n{choices_text}"
            
            # Enhance output with answer text
            if answer_text:
                if answer_key:
                    output = f"The answer is {answer_key}: {answer_text}"
                else:
                    output = answer_text
        
        if not output:
            return None
        
        return {
            "instruction": instruction.strip(),
            "input": input_text.strip(),
            "output": output.strip()
        }
        
    except Exception as e:
        return None

def load_and_process_dataset(config: Dict) -> List[Dict]:
    """Load and process a single dataset using its configuration"""
    name = config["name"]
    path = config["path"]
    config_name = config.get("config")
    split = config.get("split", "train")
    limit = config.get("limit")
    
    print(f"\nLoading {name}...")
    print(f"  Path: {path}")
    if config_name:
        print(f"  Config: {config_name}")
    print(f"  Split: {split}")
    
    try:
        # Load dataset
        if config_name:
            dataset = load_dataset(path, config_name, split=split)
        else:
            dataset = load_dataset(path, split=split)
        
        print(f"Loaded {len(dataset)} samples")
        
        # Apply limit
        if limit and len(dataset) > limit:
            dataset = dataset.select(range(limit))
            print(f"Limited to {len(dataset)} samples")
        
        # Process all items
        processed_data = []
        skipped = 0
        
        for item in dataset:
            processed = process_item(item, config)
            if processed:
                processed_data.append(processed)
            else:
                skipped += 1
        
        print(f"Successfully processed {len(processed_data)} samples")
        if skipped > 0:
            print(f"Skipped {skipped} invalid samples")
        
        return processed_data
        
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return []

def save_to_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL format"""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
    print(f"\nSaved {len(data):,} samples to: {filepath}")
    print(f"File size: {file_size:.2f} MB")

def load_config(config_path: str) -> Dict:
    """Load dataset configurations from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def show_preview(data: List[Dict], dataset_name: str, category: str, num_examples: int = 3):
    """Show preview of processed data"""
    print("DATA PREVIEW")
    
    for i, sample in enumerate(data[:num_examples], 1):
        print(f"\nExample {i} ({dataset_name} - {category}):")
        print(f"  Instruction: {sample['instruction'][:200]}{'...' if len(sample['instruction']) > 200 else ''}")
        if sample['input']:
            print(f"  Input: {sample['input'][:100]}{'...' if len(sample['input']) > 100 else ''}")
        print(f"  Output: {sample['output'][:200]}{'...' if len(sample['output']) > 200 else ''}")

def process_single_dataset(dataset_name: str, config_path: str, output_dir: str, preview: bool = True):
    """Process a single dataset"""
    # Load configuration
    all_configs = load_config(config_path)
    
    if dataset_name not in all_configs['datasets']:
        print(f"Error: Dataset '{dataset_name}' not found in config file.")
        print(f"Available datasets: {', '.join(all_configs['datasets'].keys())}")
        return
    
    config = all_configs['datasets'][dataset_name]
    config['name'] = dataset_name
    
    print(f"Processing dataset: {dataset_name}")
    print("="*60)
    
    # Process dataset
    data = load_and_process_dataset(config)
    
    if not data:
        print("\nWarning: No data to save!")
        return
    
    # Print summary
    print(f"Dataset: {dataset_name}")
    print(f"Category: {config.get('category', 'unknown')}")
    print(f"Total samples: {len(data):,}")
    
    # Save data
    output_file = os.path.join(output_dir, f"{dataset_name}_dataset.jsonl")
    save_to_jsonl(data, output_file)
    
    # Show preview
    if preview and data:
        show_preview(data, dataset_name, config.get('category', 'unknown'))

def process_all_datasets(config_path: str, output_dir: str, preview: bool = True):
    """Process all datasets from config file"""
    # Load configuration
    all_configs = load_config(config_path)
    
    all_data = []
    dataset_stats = {}
        
    for dataset_name, config in all_configs['datasets'].items():
        config['name'] = dataset_name
        data = load_and_process_dataset(config)
        all_data.extend(data)
        dataset_stats[dataset_name] = {
            "count": len(data),
            "category": config.get("category", "unknown")
        }
    
    print(f"Total samples: {len(all_data):,}\n")
    
    # Group by category
    categories = {}
    for name, stats in dataset_stats.items():
        cat = stats["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, stats["count"]))
    
    for category, datasets in sorted(categories.items()):
        total = sum(count for _, count in datasets)
        print(f"{category.upper()}:")
        for name, count in datasets:
            percentage = (count / len(all_data) * 100) if all_data else 0
            print(f"  • {name:20} {count:>6,} samples ({percentage:>5.1f}%)")
        print(f"  Subtotal: {total:,}\n")
    
    if all_data:
        # Save combined dataset
        output_file = os.path.join(output_dir, "combined_dataset.jsonl")
        save_to_jsonl(all_data, output_file)
        
        # Show preview
        if preview:
            shown_categories = set()
            examples_shown = 0            
            for dataset_name, config in all_configs['datasets'].items():
                cat = config.get("category", "unknown")
                if cat not in shown_categories and examples_shown < 5:
                    # Find first example from this dataset
                    for sample in all_data:
                        examples_shown += 1
                        shown_categories.add(cat)
                        
                        print(f"\nExample {examples_shown} ({dataset_name} - {cat}):")
                        print(f"  Instruction: {sample['instruction'][:200]}{'...' if len(sample['instruction']) > 200 else ''}")
                        if sample['input']:
                            print(f"  Input: {sample['input'][:100]}{'...' if len(sample['input']) > 100 else ''}")
                        print(f"  Output: {sample['output'][:200]}{'...' if len(sample['output']) > 200 else ''}")
                        break
                
                if examples_shown >= 5:
                    break
    else:
        print("\nWarning: No data to save!")

def main():
    parser = argparse.ArgumentParser(
        description="Process and create datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single dataset
  python create_dataset.py --dataset gsm8k
  
  # Process all datasets
  python create_dataset.py --all
  
  # Use custom config file
  python create_dataset.py --dataset trivia_qa --config my_config.yaml
  
  # Change output directory
  python create_dataset.py --all --output-dir ./my_data
  
  # List available datasets
  python create_dataset.py --list
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Name of the dataset to process (e.g., gsm8k, trivia_qa)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all datasets in the config file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='dataset_config.yaml',
        help='Path to the config YAML file (default: dataset_config.yaml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for the generated datasets (default: data/)'
    )
    
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Disable data preview output'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets in the config file'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return
    
    if args.list:
        all_configs = load_config(args.config)
        print("Available datasets:")
        for name, config in all_configs['datasets'].items():
            print(f"  • {name:20} (category: {config.get('category', 'unknown')})")
        return
    
    if not args.all and not args.dataset:
        parser.print_help()
        print("\nError: You must specify either --dataset or --all")
        return
    
    if args.all and args.dataset:
        print("Error: Cannot use both --dataset and --all together")
        return
    
    if args.dataset:
        process_single_dataset(
            args.dataset,
            args.config,
            args.output_dir,
            preview=not args.no_preview
        )
    else:
        process_all_datasets(
            args.config,
            args.output_dir,
            preview=not args.no_preview
        )

if __name__ == "__main__":
    main()

