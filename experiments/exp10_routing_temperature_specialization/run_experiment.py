"""
Main script to run temperature routing experiments
"""
import argparse
import json
import os
import sys
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(project_root))

from configs.moe_config import MoEModelConfig
from configs.dataset_config import DataConfig
from utils.helpers import set_seed
from utils.logger import setup_logging
from config import (
    get_experiment_config,
    list_experiments,
    TEMPERATURE_ABLATION,
    TEMPERATURE_SCHEDULES,
    ALL_EXPERIMENTS,
)
from tracking_trainer import train_with_temperature_tracking
from temperature_model import create_temperature_moe_model


def prepare_data(config: MoEModelConfig):
    """Prepare train and validation data loaders"""
    print("Loading dataset with Hugging Face Datasets API...")
    data_cfg = DataConfig(
        dataset_path="HuggingFaceTB/smollm-corpus",
        dataset_name="cosmopedia-v2",
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        seq_length=config.max_seq_len,
        num_samples=config.num_documents,
        cache_dir="./hf_cache",
    )
    
    # Split documents BEFORE tokenization to prevent data leakage
    from datasets import load_dataset, Dataset
    print("Loading raw dataset and splitting documents...")
    raw_dataset = load_dataset(
        data_cfg.dataset_path,
        data_cfg.dataset_name,
        split=data_cfg.split,
        cache_dir=data_cfg.cache_dir,
        streaming=True,
    )
    
    # Take samples and split into train/val
    raw_samples = list(raw_dataset.take(data_cfg.num_samples))
    random.shuffle(raw_samples)
    num_val = int(len(raw_samples) * 0.1)
    num_train = len(raw_samples) - num_val
    
    raw_train = Dataset.from_list(raw_samples[:num_train])
    raw_val = Dataset.from_list(raw_samples[num_train:])
    print(f"Split into {len(raw_train):,} train docs and {len(raw_val):,} val docs")
    
    # Now tokenize each split separately
    from data.loader import setup_tokenizer, tokenize_and_chunk, finalize_dataset
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size
    
    print("Tokenizing train set...")
    train_ds = tokenize_and_chunk(raw_train, tokenizer, data_cfg)
    train_ds = finalize_dataset(train_ds, data_cfg)
    
    print("Tokenizing validation set...")
    val_ds = tokenize_and_chunk(raw_val, tokenizer, data_cfg)
    val_ds = finalize_dataset(val_ds, data_cfg)
    
    print(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")
    
    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    
    return train_loader, val_loader


def run_single_experiment(exp_name: str, output_dir: str = "./results"):
    """Run a single temperature experiment"""
    logger = setup_logging(log_dir="./logs")
    logger.info(f"Running experiment: {exp_name}")
    
    set_seed(42)
    
    # Get experiment configuration
    temp_config = get_experiment_config(exp_name)
    
    # Create model config
    model_config = MoEModelConfig()
    model_config.max_steps = temp_config.max_steps
    
    # Prepare data
    train_loader, val_loader = prepare_data(model_config)
    
    # Create model with temperature-aware routing
    model = create_temperature_moe_model(model_config)
    
    # Create experiment output directory
    exp_output_dir = Path(output_dir) / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"Description: {temp_config.description}")
    print(f"Temperature: {temp_config.temperature}")
    print(f"Schedule: {temp_config.temperature_schedule or 'constant'}")
    print(f"Steps: {temp_config.max_steps}")
    print(f"Output: {exp_output_dir}")
    print(f"{'='*80}\n")
    
    # Train model
    model, metrics, history, routing_stats = train_with_temperature_tracking(
        model=model,
        config=model_config,
        temp_config=temp_config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(exp_output_dir),
    )
    
    # Save results
    results = {
        'experiment_name': exp_name,
        'description': temp_config.description,
        'temperature': temp_config.temperature,
        'temperature_schedule': temp_config.temperature_schedule,
        'final_metrics': metrics,
        'history': history,
        'routing_stats': routing_stats,
        'config': {
            'max_steps': temp_config.max_steps,
            'batch_size': model_config.batch_size,
            'num_experts': model_config.num_experts,
            'expert_top_k': model_config.expert_top_k,
        }
    }
    
    # Save metrics
    metrics_file = exp_output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Results saved to {metrics_file}")
    
    # Save model checkpoint
    model_file = exp_output_dir / "model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'metrics': metrics,
    }, model_file)
    print(f"✅ Model saved to {model_file}")
    
    return results


def run_multiple_experiments(exp_names: list, output_dir: str = "./results"):
    """Run multiple experiments sequentially"""
    logger = setup_logging(log_dir="./logs")
    logger.info(f"Running {len(exp_names)} experiments")
    
    results = {}
    
    for i, exp_name in enumerate(exp_names):
        print(f"\n{'='*80}")
        print(f"Running experiment {i+1}/{len(exp_names)}: {exp_name}")
        print(f"{'='*80}\n")
        
        try:
            result = run_single_experiment(exp_name, output_dir)
            results[exp_name] = result
            
            print(f"\n✅ Experiment '{exp_name}' completed successfully")
            print(f"   Final loss: {result['final_metrics']['val_loss']:.4f}")
            print(f"   Final accuracy: {result['final_metrics']['val_accuracy']:.4f}")
            
        except Exception as e:
            import traceback
            print(f"\n❌ Experiment '{exp_name}' failed with error: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            logger.error(f"Experiment '{exp_name}' failed: {e}")
            logger.error(traceback.format_exc())
            continue
    
    # Save summary
    summary_file = Path(output_dir) / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'experiments': list(results.keys()),
            'num_completed': len(results),
            'num_requested': len(exp_names),
        }, f, indent=2)
    print(f"\n📁 Experiment summary saved to {summary_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run routing temperature experiments for MoE training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available experiments
  python run_experiment.py --list
  
  # Run single temperature
  python run_experiment.py --experiment temp_1.0
  
  # Run temperature ablation
  python run_experiment.py --ablation
  
  # Run temperature schedules
  python run_experiment.py --schedules
  
  # Run all experiments
  python run_experiment.py --all
        """
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        help='Single experiment to run'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        help='Multiple experiments to run (space-separated)'
    )
    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Run all temperature ablation experiments'
    )
    parser.add_argument(
        '--schedules',
        action='store_true',
        help='Run all temperature schedule experiments'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available experiments and exit'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        help='Run with specific temperature (creates temp_X.X experiment)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    # Determine which experiments to run
    exp_names = []
    
    if args.all:
        exp_names = list(ALL_EXPERIMENTS.keys())
        print(f"Running all {len(exp_names)} experiments...")
    elif args.ablation:
        exp_names = list(TEMPERATURE_ABLATION.keys())
        print(f"Running temperature ablation ({len(exp_names)} experiments)...")
    elif args.schedules:
        exp_names = list(TEMPERATURE_SCHEDULES.keys())
        print(f"Running temperature schedules ({len(exp_names)} experiments)...")
    elif args.experiments:
        exp_names = args.experiments
    elif args.experiment:
        exp_names = [args.experiment]
    elif args.temperature:
        # Create custom temperature experiment
        exp_name = f"temp_{args.temperature}"
        print(f"Running custom temperature experiment: {exp_name}")
        run_single_experiment(exp_name, args.output_dir)
        return
    else:
        parser.print_help()
        print("\n❌ No experiments specified. Use --list to see available experiments.")
        return
    
    # Run experiments
    if len(exp_names) == 1:
        run_single_experiment(exp_names[0], args.output_dir)
    else:
        run_multiple_experiments(exp_names, args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ All experiments completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

