"""
Run Muon vs Adam optimizer comparison experiments for MoE training
"""
import argparse
import json
import time
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root and exp9 to path (project_root last so it's checked first)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir))  # For local configs/training
sys.path.insert(0, str(project_root))  # For project modules (checked first)

import torch
from torch.utils.data import DataLoader

from configs.moe_config import MoEModelConfig
from configs.dataset_config import DataConfig
from exp_configs.experiment_configs import ExperimentConfig, get_experiment, list_experiments, EXPERIMENTS
from exp_training.experiment_trainer import train_experiment
from utils.helpers import set_seed
from utils.logger import setup_logging


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


def run_single_experiment(exp_name: str, output_dir: str = "."):
    """Run a single experiment by name"""
    logger = setup_logging(log_dir="./logs")
    logger.info(f"Running experiment: {exp_name}")
    
    set_seed(42)
    
    # Get experiment configuration
    exp_config = get_experiment(exp_name)
    
    # Create base model config
    base_config = MoEModelConfig()
    
    # Convert to model config with experiment overrides
    config = exp_config.to_moe_config(base_config)
    
    # Prepare data (shared across experiments)
    train_loader, val_loader = prepare_data(config)
    
    # Run experiment
    start_time = time.time()
    model, metrics, history = train_experiment(
        config=config,
        exp_config=exp_config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir
    )
    elapsed = (time.time() - start_time) / 60
    
    print(f"\n✅ Experiment '{exp_name}' completed in {elapsed:.2f} minutes")
    logger.info(f"Experiment '{exp_name}' completed. Final val loss: {metrics['val_loss']:.4f}")
    
    return metrics, history


def run_multiple_experiments(exp_names: list, output_dir: str = "."):
    """Run multiple experiments sequentially"""
    logger = setup_logging(log_dir="./logs")
    logger.info(f"Running {len(exp_names)} experiments: {exp_names}")
    
    set_seed(42)
    
    # Create base model config and prepare data once
    base_config = MoEModelConfig()
    
    # Use first experiment config to set up data (they should all use same data settings)
    exp_config = get_experiment(exp_names[0])
    config = exp_config.to_moe_config(base_config)
    train_loader, val_loader = prepare_data(config)
    
    # Update base_config with vocab_size for all subsequent experiments
    base_config.vocab_size = config.vocab_size
    
    results = {}
    
    for exp_name in exp_names:
        print(f"\n{'='*80}")
        print(f"Starting experiment {len(results)+1}/{len(exp_names)}: {exp_name}")
        print(f"{'='*80}\n")
        
        try:
            exp_config = get_experiment(exp_name)
            config = exp_config.to_moe_config(base_config)
            
            start_time = time.time()
            model, metrics, history = train_experiment(
                config=config,
                exp_config=exp_config,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=output_dir
            )
            elapsed = (time.time() - start_time) / 60
            
            results[exp_name] = {
                'metrics': metrics,
                'history': history,
                'time_minutes': elapsed,
                'optimizer_type': exp_config.optimizer_type
            }
            
            print(f"\n✅ Experiment '{exp_name}' completed in {elapsed:.2f} minutes")
            logger.info(f"Experiment '{exp_name}' completed. Final val loss: {metrics['val_loss']:.4f}")
            
        except Exception as e:
            import traceback
            print(f"\n❌ Experiment '{exp_name}' failed with error: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            logger.error(f"Experiment '{exp_name}' failed: {e}")
            logger.error(traceback.format_exc())
            continue
    
    # Generate comparison report
    if len(results) > 1:
        compare_experiments(results, output_dir)
    
    return results


def compare_experiments(results: dict, output_dir: str = "."):
    """Generate comparison plots and report for multiple experiments"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("📊 EXPERIMENT COMPARISON: MUON VS ADAM")
    print(f"{'='*80}\n")
    
    # Print summary table
    print(f"{'Experiment':<25} {'Optimizer':<15} {'Final Loss':>12} {'Best Loss':>12} {'Final Acc':>12} {'Time (min)':>12}")
    print("-" * 100)
    
    comparison_data = []
    for exp_name, data in results.items():
        metrics = data['metrics']
        history = data['history']
        best_loss = min(history['val_losses'])
        optimizer_type = data.get('optimizer_type', 'unknown')
        
        print(f"{exp_name:<25} {optimizer_type:<15} {metrics['val_loss']:>12.4f} {best_loss:>12.4f} "
              f"{metrics['val_accuracy']:>12.4f} {data['time_minutes']:>12.2f}")
        
        comparison_data.append({
            'name': exp_name,
            'optimizer_type': optimizer_type,
            'final_loss': metrics['val_loss'],
            'best_loss': best_loss,
            'final_accuracy': metrics['val_accuracy'],
            'history': history
        })
    
    # Determine best optimizer
    muon_exps = [d for d in comparison_data if 'muon' in d['optimizer_type']]
    adam_exps = [d for d in comparison_data if d['optimizer_type'] == 'adam']
    
    if muon_exps and adam_exps:
        best_muon = min(muon_exps, key=lambda x: x['best_loss'])
        best_adam = min(adam_exps, key=lambda x: x['best_loss'])
        
        print(f"\n🏆 Best Muon: {best_muon['name']} (loss: {best_muon['best_loss']:.4f})")
        print(f"🏆 Best Adam: {best_adam['name']} (loss: {best_adam['best_loss']:.4f})")
        
        improvement = ((best_adam['best_loss'] - best_muon['best_loss']) / best_adam['best_loss']) * 100
        if improvement > 0:
            print(f"\n✨ Muon is better by {improvement:.2f}%")
        else:
            print(f"\n✨ Adam is better by {-improvement:.2f}%")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Muon vs Adam: Optimizer Comparison', fontsize=16, fontweight='bold')
    
    # Use different colors for Muon vs Adam
    muon_color = 'blue'
    adam_color = 'red'
    
    # Plot 1: Validation loss over steps
    ax = axes[0, 0]
    for exp_data in comparison_data:
        history = exp_data['history']
        color = muon_color if 'muon' in exp_data['optimizer_type'] else adam_color
        linestyle = '-' if exp_data['optimizer_type'] in ['muon_hybrid', 'adam'] else '--'
        ax.plot(history['steps'], history['val_losses'], 
                label=exp_data['name'], color=color, linewidth=2, 
                marker='o', markersize=3, linestyle=linestyle, alpha=0.7)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation loss over time
    ax = axes[0, 1]
    for exp_data in comparison_data:
        history = exp_data['history']
        color = muon_color if 'muon' in exp_data['optimizer_type'] else adam_color
        linestyle = '-' if exp_data['optimizer_type'] in ['muon_hybrid', 'adam'] else '--'
        ax.plot(history['elapsed_times'], history['val_losses'], 
                label=exp_data['name'], color=color, linewidth=2, 
                marker='o', markersize=3, linestyle=linestyle, alpha=0.7)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Time')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final metrics comparison
    ax = axes[1, 0]
    exp_names = [d['name'] for d in comparison_data]
    final_losses = [d['final_loss'] for d in comparison_data]
    best_losses = [d['best_loss'] for d in comparison_data]
    colors = [muon_color if 'muon' in d['optimizer_type'] else adam_color for d in comparison_data]
    
    x = np.arange(len(exp_names))
    width = 0.35
    ax.bar(x - width/2, final_losses, width, label='Final Loss', alpha=0.8, color=colors)
    ax.bar(x + width/2, best_losses, width, label='Best Loss', alpha=0.6, color=colors)
    ax.set_ylabel('Validation Loss')
    ax.set_title('Final vs Best Loss')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Accuracy comparison
    ax = axes[1, 1]
    for exp_data in comparison_data:
        history = exp_data['history']
        color = muon_color if 'muon' in exp_data['optimizer_type'] else adam_color
        linestyle = '-' if exp_data['optimizer_type'] in ['muon_hybrid', 'adam'] else '--'
        ax.plot(history['steps'], history['val_accuracies'], 
                label=exp_data['name'], color=color, linewidth=2, 
                marker='o', markersize=3, linestyle=linestyle, alpha=0.7)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = output_path / "comparison_plot.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to {comparison_plot_path}")
    
    # Save comparison data
    comparison_file = output_path / "comparison_summary.json"
    with open(comparison_file, 'w') as f:
        json.dump({
            'experiments': comparison_data,
            'best_experiment': min(comparison_data, key=lambda x: x['best_loss'])['name'],
            'best_muon': best_muon['name'] if muon_exps else None,
            'best_adam': best_adam['name'] if adam_exps else None,
        }, f, indent=2, default=str)
    print(f"📁 Comparison summary saved to {comparison_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Muon vs Adam optimizer comparison experiments for MoE training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline comparison
  python run_experiments.py -e muon_baseline adam_baseline
  
  # Run all experiments
  python run_experiments.py --all
  
  # List available experiments
  python run_experiments.py --list
  
  # Quick comparison
  python run_experiments.py --quick
        """
    )
    
    parser.add_argument(
        '--experiments', '-e',
        nargs='+',
        help='Experiment names to run (space-separated). Use --list to see available experiments.'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available experiments and exit'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all available experiments'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='.',
        help='Output directory for experiment results (default: current directory)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick comparison (muon_baseline vs adam_baseline)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    if args.all:
        exp_names = list(EXPERIMENTS.keys())
        print(f"Running all {len(exp_names)} experiments...")
    elif args.quick:
        exp_names = ['muon_baseline', 'adam_baseline']
        print(f"Running quick comparison: {exp_names}")
    elif args.experiments:
        exp_names = args.experiments
    else:
        parser.print_help()
        print("\nNo experiments specified. Use --list to see available experiments.")
        return
    
    # Run experiments
    results = run_multiple_experiments(exp_names, args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ Completed {len(results)}/{len(exp_names)} experiments")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

