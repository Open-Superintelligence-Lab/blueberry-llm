"""
Train and Compare Best Configurations
- Best Baseline: 4_high_lr (LR=6e-4, no warmup)
- Best Reasoning: R02_minimal (H=1, L=1)

Both trained for 1000 steps
"""

import torch
import sys
import os
from pathlib import Path
from dataclasses import replace

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp8_reasoning_architecture.config import get_base_reasoning_config
from experiments.exp8_reasoning_architecture.models import ReasoningModelWrapper
from experiments.exp8_reasoning_architecture.run_experiment import Trainer, plot_training_curves
from data.loader import load_and_cache_data
from data.streaming_dataset import create_progressive_loaders
from utils.helpers import set_seed
import json


def train_model(name, config, use_recursive, recursive_config, train_loader, val_loader, device, save_dir):
    """Train a single model"""
    print("\n" + "="*70)
    print(f"Training: {name}")
    print("="*70)
    
    if use_recursive:
        config.recursive = recursive_config
    
    # Create model
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = ReasoningModelWrapper(config, use_recursive=use_recursive)
    model.print_info()
    model = model.to(device=device, dtype=dtype)
    
    # Train
    trainer = Trainer(model, config, train_loader, val_loader, device, 
                     save_dir=save_dir, use_recursive=use_recursive)
    results = trainer.train()
    
    # Save results
    results_summary = {
        'name': name,
        'use_recursive': use_recursive,
        'config': {
            'learning_rate': config.learning_rate,
            'warmup_steps': config.warmup_steps,
            'dropout': config.dropout,
            'gradient_clip': config.gradient_clip,
            'max_steps': config.max_steps,
        },
        'results': {
            'total_time': results['total_time'],
            'best_val_loss': results['best_val_loss'],
            'final_train_loss': results['train_history'][-1]['loss'] if results['train_history'] else None,
            'final_val_metrics': results['val_history'][-1] if results['val_history'] else None,
        },
        'train_history': results['train_history'],
        'val_history': results['val_history'],
    }
    
    if use_recursive:
        results_summary['recursive_config'] = recursive_config
    
    return results_summary


def main():
    print("="*70)
    print("COMPARISON: Best Baseline vs Best Reasoning")
    print("Training both for 1000 steps")
    print("="*70)
    
    # Load data
    base_config = get_base_reasoning_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(base_config.seed)
    
    from dataclasses import dataclass
    @dataclass
    class DataConfig:
        num_documents: int = base_config.num_documents
        max_tokens: int = base_config.max_tokens
        vocab_size: int = base_config.vocab_size
    
    data_config = DataConfig()
    texts, tokenizer, tokens = load_and_cache_data(data_config)
    base_config.vocab_size = len(tokenizer)
    
    print(f"\nVocabulary size: {base_config.vocab_size}")
    print(f"Total tokens: {len(tokens):,}")
    
    # Split tokens
    val_split_ratio = 0.1
    val_token_start = int(len(tokens) * (1 - val_split_ratio))
    train_tokens = tokens[:val_token_start]
    val_tokens = tokens[val_token_start:]
    
    # Create loaders
    train_loader, val_loader = create_progressive_loaders(
        train_tokens, val_tokens,
        base_config.max_seq_len, base_config.batch_size,
        None, None
    )
    
    exp_base_dir = Path(__file__).parent
    results_dir = exp_base_dir / "comparison_results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Train Best Baseline (4_high_lr)
    print("\n" + "="*70)
    print("TRAINING 1/2: Best Baseline")
    print("="*70)
    
    baseline_config = replace(base_config)
    baseline_config.learning_rate = 6e-4  # High LR wins!
    baseline_config.warmup_steps = 0      # No warmup
    baseline_config.dropout = 0.1         # Standard dropout
    baseline_config.max_steps = 1000
    baseline_config.eval_interval = 100
    baseline_config.save_interval = 500
    baseline_config.log_interval = 50
    
    set_seed(base_config.seed)
    baseline_results = train_model(
        "Best Baseline (4_high_lr)",
        baseline_config,
        use_recursive=False,
        recursive_config=None,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=exp_base_dir / "checkpoints_best_baseline"
    )
    
    # 2. Train Best Reasoning (R02_minimal)
    print("\n" + "="*70)
    print("TRAINING 2/2: Best Reasoning")
    print("="*70)
    
    reasoning_config = replace(base_config)
    reasoning_config.learning_rate = 3e-4  # Standard LR for recursive
    reasoning_config.warmup_steps = 100    # Some warmup for stability
    reasoning_config.dropout = 0.1
    reasoning_config.max_steps = 1000
    reasoning_config.eval_interval = 100
    reasoning_config.save_interval = 500
    reasoning_config.log_interval = 50
    
    minimal_recursive_config = {
        'H_cycles': 1,  # Minimal
        'L_cycles': 1,  # Minimal
        'halt_max_steps': 2,
        'halt_exploration_prob': 0.1,
        'use_act': True,
    }
    
    set_seed(base_config.seed)
    reasoning_results = train_model(
        "Best Reasoning (R02_minimal)",
        reasoning_config,
        use_recursive=True,
        recursive_config=minimal_recursive_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=exp_base_dir / "checkpoints_best_reasoning"
    )
    
    # Save comparison
    comparison = {
        'baseline': baseline_results,
        'reasoning': reasoning_results,
    }
    
    comparison_file = results_dir / 'best_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n📄 Comparison saved to: {comparison_file}")
    
    # Plot comparison
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    ax = axes[0, 0]
    baseline_steps = [h['step'] for h in baseline_results['train_history']]
    baseline_losses = [h['loss'] for h in baseline_results['train_history']]
    reasoning_steps = [h['step'] for h in reasoning_results['train_history']]
    reasoning_losses = [h['loss'] for h in reasoning_results['train_history']]
    
    ax.plot(baseline_steps, baseline_losses, 'b-', linewidth=2, label='Best Baseline', alpha=0.8)
    ax.plot(reasoning_steps, reasoning_losses, 'r-', linewidth=2, label='Best Reasoning', alpha=0.8)
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('Training Loss Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation loss
    ax = axes[0, 1]
    baseline_val_steps = [h['step'] for h in baseline_results['val_history']]
    baseline_val_losses = [h['loss'] for h in baseline_results['val_history']]
    reasoning_val_steps = [h['step'] for h in reasoning_results['val_history']]
    reasoning_val_losses = [h['loss'] for h in reasoning_results['val_history']]
    
    ax.plot(baseline_val_steps, baseline_val_losses, 'b-', linewidth=2, marker='o', label='Best Baseline', alpha=0.8)
    ax.plot(reasoning_val_steps, reasoning_val_losses, 'r-', linewidth=2, marker='s', label='Best Reasoning', alpha=0.8)
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('Validation Loss', fontweight='bold')
    ax.set_title('Validation Loss Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation accuracy
    ax = axes[1, 0]
    baseline_val_accs = [h['accuracy'] * 100 for h in baseline_results['val_history']]
    reasoning_val_accs = [h['accuracy'] * 100 for h in reasoning_results['val_history']]
    
    ax.plot(baseline_val_steps, baseline_val_accs, 'b-', linewidth=2, marker='o', label='Best Baseline', alpha=0.8)
    ax.plot(reasoning_val_steps, reasoning_val_accs, 'r-', linewidth=2, marker='s', label='Best Reasoning', alpha=0.8)
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Validation Accuracy Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation perplexity
    ax = axes[1, 1]
    baseline_val_ppls = [h['perplexity'] for h in baseline_results['val_history']]
    reasoning_val_ppls = [h['perplexity'] for h in reasoning_results['val_history']]
    
    ax.plot(baseline_val_steps, baseline_val_ppls, 'b-', linewidth=2, marker='o', label='Best Baseline', alpha=0.8)
    ax.plot(reasoning_val_steps, reasoning_val_ppls, 'r-', linewidth=2, marker='s', label='Best Reasoning', alpha=0.8)
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('Perplexity', fontweight='bold')
    ax.set_title('Validation Perplexity Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plot_file = results_dir / 'best_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plots saved to: {plot_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<30} {'Best Baseline':<20} {'Best Reasoning':<20}")
    print("-"*70)
    print(f"{'Final Val Loss':<30} {baseline_results['results']['best_val_loss']:<20.4f} {reasoning_results['results']['best_val_loss']:<20.4f}")
    
    baseline_final_ppl = baseline_results['results']['final_val_metrics']['perplexity']
    reasoning_final_ppl = reasoning_results['results']['final_val_metrics']['perplexity']
    print(f"{'Final Perplexity':<30} {baseline_final_ppl:<20.2f} {reasoning_final_ppl:<20.2f}")
    
    baseline_final_acc = baseline_results['results']['final_val_metrics']['accuracy'] * 100
    reasoning_final_acc = reasoning_results['results']['final_val_metrics']['accuracy'] * 100
    print(f"{'Final Accuracy (%)':<30} {baseline_final_acc:<20.2f} {reasoning_final_acc:<20.2f}")
    
    print(f"{'Training Time (s)':<30} {baseline_results['results']['total_time']:<20.1f} {reasoning_results['results']['total_time']:<20.1f}")
    print(f"{'Learning Rate':<30} {baseline_config.learning_rate:<20.6f} {reasoning_config.learning_rate:<20.6f}")
    print("-"*70)
    
    # Determine winner
    if baseline_results['results']['best_val_loss'] < reasoning_results['results']['best_val_loss']:
        improvement = (reasoning_results['results']['best_val_loss'] - baseline_results['results']['best_val_loss']) / reasoning_results['results']['best_val_loss'] * 100
        print(f"\n🏆 WINNER: Best Baseline")
        print(f"   {improvement:.2f}% better than Best Reasoning")
    else:
        improvement = (baseline_results['results']['best_val_loss'] - reasoning_results['results']['best_val_loss']) / baseline_results['results']['best_val_loss'] * 100
        print(f"\n🏆 WINNER: Best Reasoning")
        print(f"   {improvement:.2f}% better than Best Baseline")
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()

