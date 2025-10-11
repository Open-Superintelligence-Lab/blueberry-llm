"""
Learning Rate Ablation Study for H100 Configuration
Tests multiple learning rates with short training runs to find optimal LR
before committing to full 5000-step training.

Usage:
    python run_lr_ablation.py
"""

import torch
import sys
import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Fix tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp6_gated_deltanet_training.config import get_h100_optimized_config
from experiments.exp6_gated_deltanet_training.models import GatedDeltaNetWrapper
from experiments.exp6_gated_deltanet_training.run_experiment import Trainer
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from utils.helpers import set_seed
from torch.utils.data import DataLoader


def run_lr_experiment(lr, config, train_loader, val_loader, device, run_name):
    """Run a single LR experiment"""
    print(f"\n{'='*70}")
    print(f"Testing Learning Rate: {lr:.2e}")
    print(f"{'='*70}")
    
    # Update config with this LR
    config.learning_rate = lr
    
    # Create model
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = GatedDeltaNetWrapper(config)
    model = model.to(device=device, dtype=dtype)
    
    # Create trainer with specific save directory
    save_dir = Path(__file__).parent / "lr_ablation" / run_name
    trainer = Trainer(model, config, train_loader, val_loader, device, save_dir=save_dir)
    
    # Train
    start_time = time.time()
    results = trainer.train()
    elapsed = time.time() - start_time
    
    print(f"\n✓ LR {lr:.2e} completed in {elapsed:.1f}s")
    print(f"  Best Val Loss: {results['best_val_loss']:.4f}")
    if results['val_history']:
        final_val = results['val_history'][-1]
        print(f"  Final Val Loss: {final_val['loss']:.4f}")
        print(f"  Final Val Accuracy: {final_val['accuracy']*100:.2f}%")
        print(f"  Final Val Perplexity: {final_val['perplexity']:.2f}")
    
    return {
        'lr': lr,
        'best_val_loss': results['best_val_loss'],
        'train_history': results['train_history'],
        'val_history': results['val_history'],
        'elapsed_time': elapsed,
    }


def plot_lr_comparison(all_results, save_dir, is_partial=False):
    """Plot comparison of all learning rates"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    suffix = "_partial" if is_partial else ""
    
    colors = plt.cm.viridis([i / len(all_results) for i in range(len(all_results))])
    
    # Training loss curves
    for i, result in enumerate(all_results):
        if result['train_history']:
            steps = [h['step'] for h in result['train_history']]
            losses = [h['loss'] for h in result['train_history']]
            axes[0, 0].plot(steps, losses, label=f"LR {result['lr']:.2e}", 
                          color=colors[i], linewidth=2)
    
    axes[0, 0].set_xlabel('Step', fontweight='bold')
    axes[0, 0].set_ylabel('Train Loss', fontweight='bold')
    axes[0, 0].set_title('Training Loss by Learning Rate', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss curves
    for i, result in enumerate(all_results):
        if result['val_history']:
            steps = [h['step'] for h in result['val_history']]
            losses = [h['loss'] for h in result['val_history']]
            axes[0, 1].plot(steps, losses, label=f"LR {result['lr']:.2e}",
                          color=colors[i], linewidth=2, marker='o')
    
    axes[0, 1].set_xlabel('Step', fontweight='bold')
    axes[0, 1].set_ylabel('Val Loss', fontweight='bold')
    axes[0, 1].set_title('Validation Loss by Learning Rate', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Validation accuracy curves
    for i, result in enumerate(all_results):
        if result['val_history']:
            steps = [h['step'] for h in result['val_history']]
            accs = [h['accuracy'] * 100 for h in result['val_history']]
            axes[1, 0].plot(steps, accs, label=f"LR {result['lr']:.2e}",
                          color=colors[i], linewidth=2, marker='s')
    
    axes[1, 0].set_xlabel('Step', fontweight='bold')
    axes[1, 0].set_ylabel('Val Accuracy (%)', fontweight='bold')
    axes[1, 0].set_title('Validation Accuracy by Learning Rate', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Best validation loss bar chart
    lrs = [result['lr'] for result in all_results]
    best_losses = [result['best_val_loss'] for result in all_results]
    
    bars = axes[1, 1].bar(range(len(lrs)), best_losses, color=colors)
    axes[1, 1].set_xticks(range(len(lrs)))
    axes[1, 1].set_xticklabels([f"{lr:.2e}" for lr in lrs], rotation=45)
    axes[1, 1].set_xlabel('Learning Rate', fontweight='bold')
    axes[1, 1].set_ylabel('Best Val Loss', fontweight='bold')
    axes[1, 1].set_title('Best Validation Loss by Learning Rate', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Highlight best LR
    best_idx = best_losses.index(min(best_losses))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    filename = f'lr_ablation_comparison{suffix}.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    if not is_partial:
        print(f"\n📊 Final comparison plot saved to: {save_dir / filename}")
    else:
        print(f"📊 Progress plot saved to: {save_dir / filename} ({len(all_results)} LRs tested so far)")


def main():
    print("="*70)
    print("LEARNING RATE ABLATION STUDY - H100 Configuration")
    print("="*70)
    
    # Base config
    config = get_h100_optimized_config()
    
    # Shorten for ablation study
    config.max_steps = 1000  # Quick ablation runs
    config.warmup_steps = 100
    config.eval_interval = 100
    config.log_interval = 50
    config.save_interval = 500
    
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"Config: {config.hidden_size}d, {config.num_hidden_layers} layers, batch_size={config.batch_size}")
    print(f"Ablation: {config.max_steps} steps per LR\n")
    
    # Load data (once, reuse for all experiments)
    print("="*70)
    print("Loading Data")
    print("="*70)
    
    from dataclasses import dataclass
    @dataclass
    class DataConfig:
        num_documents: int = config.num_documents
        max_tokens: int = config.max_tokens
        vocab_size: int = config.vocab_size
    
    data_config = DataConfig()
    texts, tokenizer, tokens = load_and_cache_data(data_config)
    config.vocab_size = len(tokenizer)
    
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Total tokens: {len(tokens):,}")
    
    # Split data
    val_split_ratio = 0.1
    val_token_start = int(len(tokens) * (1 - val_split_ratio))
    train_tokens = tokens[:val_token_start]
    val_tokens = tokens[val_token_start:]
    
    train_dataset = TextTokenDataset(train_tokens, config.max_seq_len)
    val_dataset = TextTokenDataset(val_tokens, config.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    
    # Define learning rates to test
    # Test around our estimate (5.8e-4) and use sqrt scaling theory
    base_lr = 3e-4  # Base LR for batch_size=32
    
    learning_rates = [
        2e-4,    # Conservative (2/3 of base)
        3e-4,    # Base LR
        4e-4,    # 1.33x base
        5e-4,    # 1.67x base
        5.8e-4,  # Our sqrt-scaled estimate
        7e-4,    # More aggressive
        1e-3,    # 3.33x base (aggressive)
    ]
    
    print(f"\n{'='*70}")
    print(f"Testing {len(learning_rates)} learning rates:")
    for lr in learning_rates:
        print(f"  • {lr:.2e}")
    print(f"{'='*70}")
    
    # Run experiments
    all_results = []
    results_dir = Path(__file__).parent / "lr_ablation"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    for i, lr in enumerate(learning_rates, 1):
        print(f"\n{'#'*70}")
        print(f"Experiment {i}/{len(learning_rates)}")
        print(f"{'#'*70}")
        
        run_name = f"lr_{lr:.2e}".replace('.', '_').replace('-', '_')
        result = run_lr_experiment(lr, config, train_loader, val_loader, device, run_name)
        all_results.append(result)
        
        # Save intermediate results
        intermediate_summary = {
            'completed_experiments': len(all_results),
            'total_experiments': len(learning_rates),
            'results_so_far': [
                {
                    'lr': r['lr'],
                    'best_val_loss': r['best_val_loss'],
                    'final_val_loss': r['val_history'][-1]['loss'] if r['val_history'] else None,
                }
                for r in all_results
            ]
        }
        with open(results_dir / 'lr_ablation_progress.json', 'w') as f:
            json.dump(intermediate_summary, f, indent=2)
        
        # Generate progress plot after each experiment
        if len(all_results) > 1:  # Need at least 2 results to compare
            plot_lr_comparison(all_results, results_dir, is_partial=True)
        
        print(f"\n✅ Completed {len(all_results)}/{len(learning_rates)} experiments")
    
    # Analyze results
    print(f"\n\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    # Sort by best validation loss
    sorted_results = sorted(all_results, key=lambda x: x['best_val_loss'])
    
    print("Ranking by Best Validation Loss:")
    print(f"{'Rank':<6} {'LR':<12} {'Best Val Loss':<15} {'Final Val Loss':<15} {'Time (s)':<10}")
    print("-" * 70)
    
    for rank, result in enumerate(sorted_results, 1):
        final_val_loss = result['val_history'][-1]['loss'] if result['val_history'] else float('nan')
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank:<4} {result['lr']:.2e}    {result['best_val_loss']:<15.4f} "
              f"{final_val_loss:<15.4f} {result['elapsed_time']:<10.1f}")
    
    best_lr = sorted_results[0]['lr']
    print(f"\n🎯 BEST LEARNING RATE: {best_lr:.2e}")
    print(f"   Best Val Loss: {sorted_results[0]['best_val_loss']:.4f}")
    
    summary = {
        'config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'batch_size': config.batch_size,
            'max_seq_len': config.max_seq_len,
            'ablation_steps': config.max_steps,
        },
        'learning_rates_tested': learning_rates,
        'results': [
            {
                'lr': r['lr'],
                'best_val_loss': r['best_val_loss'],
                'final_val_loss': r['val_history'][-1]['loss'] if r['val_history'] else None,
                'final_val_accuracy': r['val_history'][-1]['accuracy'] if r['val_history'] else None,
                'elapsed_time': r['elapsed_time'],
            }
            for r in all_results
        ],
        'best_lr': best_lr,
        'best_val_loss': sorted_results[0]['best_val_loss'],
        'recommendation': f"Use learning_rate={best_lr:.2e} for full 5000-step training"
    }
    
    with open(results_dir / 'lr_ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_dir / 'lr_ablation_summary.json'}")
    
    # Plot comparison
    plot_lr_comparison(all_results, results_dir)
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION FOR FULL TRAINING")
    print(f"{'='*70}")
    print(f"\nUse the following command for full 5000-step training:")
    print(f"\n  python run_experiment.py --config h100 --extend-steps 5000")
    print(f"\nThen manually update config.py to set:")
    print(f"  learning_rate = {best_lr:.2e}")
    print(f"\nOr you can start from the best checkpoint:")
    best_checkpoint = results_dir / f"lr_{best_lr:.2e}".replace('.', '_').replace('-', '_') / "best_model.pt"
    if best_checkpoint.exists():
        print(f"  python run_experiment.py --resume {best_checkpoint} --extend-steps 5000")
    
    print(f"\n{'='*70}")
    print("Ablation study completed! ✨")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

