"""
Experiment 3: Dynamic Sparsity and Adaptive Attention Patterns

This script runs a comprehensive experiment comparing adaptive sparsity patterns
with fixed sparsity baselines to optimize both pretraining speed and quality.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Warning: matplotlib/seaborn not available. Visualization will be skipped.")

# Add project root to path
sys.path.append('/root/deepseek-sparse-attention-research')

# Import our modules
from adaptive_models import AdaptiveMoELLM, FixedSparseMoELLM
from simple_dataset import TinyStoriesDataset
# from training.trainer import Trainer  # Not used in this experiment
from utils.helpers import set_seed, count_parameters


class ExperimentConfig:
    """Configuration for Experiment 3."""
    
    def __init__(self):
        # Model configuration
        self.d_model = 512
        self.n_layers = 6
        self.n_heads = 8
        self.d_ff = 2048
        self.vocab_size = 10000
        self.max_position_embeddings = 2048
        self.num_experts = 8
        self.top_k = 2
        self.dropout = 0.1
        
        # Training configuration
        self.learning_rate = 1e-3
        self.batch_size = 16
        self.steps = 2000
        self.eval_every = 200
        self.warmup_steps = 100
        
        # Experiment configuration
        self.sequence_lengths = [64, 128, 256, 512, 1024, 2048]
        self.random_seeds = [42, 123, 456]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model variants to test
        self.model_variants = [
            {'name': 'dense', 'type': 'dense', 'sparsity_ratio': 1.0},
            {'name': 'fixed_sparse_25', 'type': 'fixed_sparse', 'sparsity_ratio': 0.25},
            {'name': 'fixed_sparse_50', 'type': 'fixed_sparse', 'sparsity_ratio': 0.5},
            {'name': 'fixed_sparse_75', 'type': 'fixed_sparse', 'sparsity_ratio': 0.75},
            {'name': 'adaptive_sparse', 'type': 'adaptive_sparse', 'sparsity_ratio': None},
        ]
        
        # Results storage
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)


def create_model_config(config: ExperimentConfig) -> Dict:
    """Create model configuration dictionary."""
    return {
        'd_model': config.d_model,
        'n_layers': config.n_layers,
        'n_heads': config.n_heads,
        'd_ff': config.d_ff,
        'vocab_size': config.vocab_size,
        'max_position_embeddings': config.max_position_embeddings,
        'num_experts': config.num_experts,
        'top_k': config.top_k,
        'dropout': config.dropout
    }


def create_model(model_variant: Dict, config: ExperimentConfig):
    """Create model based on variant specification."""
    model_config = create_model_config(config)
    
    if model_variant['type'] == 'dense':
        # Dense model (no sparsity)
        return FixedSparseMoELLM(model_config, sparsity_ratio=1.0)
    elif model_variant['type'] == 'fixed_sparse':
        # Fixed sparse model
        return FixedSparseMoELLM(model_config, sparsity_ratio=model_variant['sparsity_ratio'])
    elif model_variant['type'] == 'adaptive_sparse':
        # Adaptive sparse model
        return AdaptiveMoELLM(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_variant['type']}")


def train_model(model, train_loader, val_loader, config: ExperimentConfig, 
                model_name: str, seq_len: int, seed: int) -> Dict:
    """Train a single model configuration."""
    print(f"\nTraining {model_name} (seq_len={seq_len}, seed={seed})")
    
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=config.warmup_steps
    )
    
    # Move model to device
    model = model.to(config.device)
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    step_times = []
    memory_usage = []
    adaptive_stats_history = []
    
    model.train()
    start_time = time.time()
    
    for step in range(config.steps):
        step_start_time = time.time()
        
        # Get batch
        try:
            batch = next(train_loader)
            input_ids = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)
        except StopIteration:
            # Restart data loader if exhausted
            train_loader = iter(train_loader)
            batch = next(train_loader)
            input_ids = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record metrics
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)  # GB
        
        # Validation
        if step % config.eval_every == 0:
            model.eval()
            val_loss, val_acc = evaluate_model(model, val_loader, config)
            val_losses.append((step, val_loss))
            val_accuracies.append((step, val_acc))
            model.train()
            
            # Adaptive stats (for adaptive models)
            if hasattr(model, 'get_adaptive_stats'):
                adaptive_stats = model.get_adaptive_stats()
                adaptive_stats_history.append((step, adaptive_stats))
        
        # Record training loss
        train_losses.append((step, loss.item()))
        
        if step % 100 == 0:
            print(f"  Step {step}/{config.steps}: loss={loss.item():.4f}, "
                  f"val_loss={val_losses[-1][1]:.4f}, val_acc={val_accuracies[-1][1]:.4f}")
    
    total_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    final_val_loss, final_val_acc = evaluate_model(model, val_loader, config)
    
    # Collect adaptive stats
    final_adaptive_stats = None
    if hasattr(model, 'get_adaptive_stats'):
        final_adaptive_stats = model.get_adaptive_stats()
    
    return {
        'model_name': model_name,
        'sequence_length': seq_len,
        'seed': seed,
        'total_time': total_time,
        'final_train_loss': train_losses[-1][1],
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'step_times': step_times,
        'memory_usage': memory_usage,
        'adaptive_stats_history': adaptive_stats_history,
        'final_adaptive_stats': final_adaptive_stats,
        'model_parameters': count_parameters(model),
        'tokens_per_second': (len(train_losses) * config.batch_size * seq_len) / total_time
    }


def evaluate_model(model, val_loader, config: ExperimentConfig) -> Tuple[float, float]:
    """Evaluate model on validation set."""
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            # Calculate accuracy (next token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            predictions = torch.argmax(shift_logits, dim=-1)
            correct = (predictions == shift_labels).float()
            
            # Mask out padding tokens
            mask = (shift_labels != -100).float()
            total_correct += (correct * mask).sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, accuracy


def run_experiment_for_sequence_length(config: ExperimentConfig, seq_len: int) -> Dict:
    """Run experiment for a single sequence length."""
    print(f"\n{'='*60}")
    print(f"Running Experiment for Sequence Length: {seq_len}")
    print(f"{'='*60}")
    
    # Create datasets
    train_dataset = TinyStoriesDataset(
        seq_len=seq_len, 
        split='train',
        vocab_size=config.vocab_size
    )
    val_dataset = TinyStoriesDataset(
        seq_len=seq_len, 
        split='val',
        vocab_size=config.vocab_size
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for debugging
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    results = {}
    
    # Test each model variant
    for model_variant in config.model_variants:
        model_name = model_variant['name']
        print(f"\nTesting {model_name}...")
        
        variant_results = []
        
        # Test with multiple seeds
        for seed in config.random_seeds:
            # Create model
            model = create_model(model_variant, config)
            
            # Train model
            result = train_model(
                model, train_loader, val_loader, config, 
                model_name, seq_len, seed
            )
            
            variant_results.append(result)
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        
        results[model_name] = variant_results
    
    return results


def analyze_results(all_results: Dict) -> Dict:
    """Analyze and aggregate results across all experiments."""
    print(f"\n{'='*60}")
    print("ANALYZING RESULTS")
    print(f"{'='*60}")
    
    analysis = {}
    
    # Aggregate by sequence length
    for seq_len in [64, 128, 256, 512, 1024, 2048]:
        if seq_len not in all_results:
            continue
            
        print(f"\nSequence Length {seq_len}:")
        seq_results = all_results[seq_len]
        
        # Calculate statistics for each model variant
        model_stats = {}
        for model_name, runs in seq_results.items():
            if not runs:
                continue
                
            # Aggregate across seeds
            final_losses = [run['final_val_loss'] for run in runs]
            final_accuracies = [run['final_val_accuracy'] for run in runs]
            training_times = [run['total_time'] for run in runs]
            tokens_per_second = [run['tokens_per_second'] for run in runs]
            
            model_stats[model_name] = {
                'val_loss_mean': np.mean(final_losses),
                'val_loss_std': np.std(final_losses),
                'val_accuracy_mean': np.mean(final_accuracies),
                'val_accuracy_std': np.std(final_accuracies),
                'training_time_mean': np.mean(training_times),
                'training_time_std': np.std(training_times),
                'tokens_per_second_mean': np.mean(tokens_per_second),
                'tokens_per_second_std': np.std(tokens_per_second),
                'n_runs': len(runs)
            }
            
            print(f"  {model_name}:")
            print(f"    Val Loss: {np.mean(final_losses):.4f} ± {np.std(final_losses):.4f}")
            print(f"    Val Acc:  {np.mean(final_accuracies):.4f} ± {np.std(final_accuracies):.4f}")
            print(f"    Speed:    {np.mean(tokens_per_second):.0f} ± {np.std(tokens_per_second):.0f} tok/s")
        
        analysis[seq_len] = model_stats
    
    return analysis


def create_visualizations(all_results: Dict, analysis: Dict, config: ExperimentConfig):
    """Create comprehensive visualizations of results."""
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    if not HAS_VISUALIZATION:
        print("Skipping visualizations - matplotlib/seaborn not available")
        return
    
    # Set up plotting style
    try:
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    except:
        # Fallback if seaborn style not available
        plt.style.use('default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Experiment 3: Dynamic Sparsity vs Fixed Sparsity', fontsize=16)
    
    # 1. Validation Loss vs Sequence Length
    ax1 = axes[0, 0]
    sequence_lengths = sorted(analysis.keys())
    
    for model_name in ['dense', 'fixed_sparse_25', 'fixed_sparse_50', 'fixed_sparse_75', 'adaptive_sparse']:
        losses = []
        stds = []
        
        for seq_len in sequence_lengths:
            if model_name in analysis[seq_len]:
                losses.append(analysis[seq_len][model_name]['val_loss_mean'])
                stds.append(analysis[seq_len][model_name]['val_loss_std'])
            else:
                losses.append(np.nan)
                stds.append(0)
        
        ax1.errorbar(sequence_lengths, losses, yerr=stds, 
                    label=model_name, marker='o', capsize=5)
    
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss vs Sequence Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Accuracy vs Sequence Length
    ax2 = axes[0, 1]
    
    for model_name in ['dense', 'fixed_sparse_25', 'fixed_sparse_50', 'fixed_sparse_75', 'adaptive_sparse']:
        accuracies = []
        stds = []
        
        for seq_len in sequence_lengths:
            if model_name in analysis[seq_len]:
                accuracies.append(analysis[seq_len][model_name]['val_accuracy_mean'])
                stds.append(analysis[seq_len][model_name]['val_accuracy_std'])
            else:
                accuracies.append(np.nan)
                stds.append(0)
        
        ax2.errorbar(sequence_lengths, accuracies, yerr=stds,
                    label=model_name, marker='s', capsize=5)
    
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy vs Sequence Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Training Speed vs Sequence Length
    ax3 = axes[0, 2]
    
    for model_name in ['dense', 'fixed_sparse_25', 'fixed_sparse_50', 'fixed_sparse_75', 'adaptive_sparse']:
        speeds = []
        stds = []
        
        for seq_len in sequence_lengths:
            if model_name in analysis[seq_len]:
                speeds.append(analysis[seq_len][model_name]['tokens_per_second_mean'])
                stds.append(analysis[seq_len][model_name]['tokens_per_second_std'])
            else:
                speeds.append(np.nan)
                stds.append(0)
        
        ax3.errorbar(sequence_lengths, speeds, yerr=stds,
                    label=model_name, marker='^', capsize=5)
    
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Training Speed (tokens/sec)')
    ax3.set_title('Training Speed vs Sequence Length')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Speed vs Quality Trade-off (for longest sequence)
    ax4 = axes[1, 0]
    
    if max(sequence_lengths) in analysis:
        max_seq_results = analysis[max(sequence_lengths)]
        
        for model_name in ['dense', 'fixed_sparse_25', 'fixed_sparse_50', 'fixed_sparse_75', 'adaptive_sparse']:
            if model_name in max_seq_results:
                speed = max_seq_results[model_name]['tokens_per_second_mean']
                quality = max_seq_results[model_name]['val_accuracy_mean']
                ax4.scatter(speed, quality, label=model_name, s=100)
        
        ax4.set_xlabel('Training Speed (tokens/sec)')
        ax4.set_ylabel('Validation Accuracy')
        ax4.set_title(f'Speed vs Quality Trade-off (Seq Len {max(sequence_lengths)})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Adaptive Behavior Analysis (if available)
    ax5 = axes[1, 1]
    
    # Plot adaptive sparsity ratios over sequence lengths
    adaptive_sparsity_ratios = []
    seq_lens_for_adaptive = []
    
    for seq_len in sequence_lengths:
        if seq_len in all_results and 'adaptive_sparse' in all_results[seq_len]:
            runs = all_results[seq_len]['adaptive_sparse']
            for run in runs:
                if run.get('final_adaptive_stats'):
                    for layer_stats in run['final_adaptive_stats'].values():
                        if 'mean_sparsity_ratio' in layer_stats:
                            adaptive_sparsity_ratios.append(layer_stats['mean_sparsity_ratio'])
                            seq_lens_for_adaptive.append(seq_len)
    
    if adaptive_sparsity_ratios:
        ax5.scatter(seq_lens_for_adaptive, adaptive_sparsity_ratios, alpha=0.6)
        ax5.set_xlabel('Sequence Length')
        ax5.set_ylabel('Average Sparsity Ratio')
        ax5.set_title('Adaptive Sparsity Behavior')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No adaptive data available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Adaptive Sparsity Behavior')
    
    # 6. Improvement over Baselines
    ax6 = axes[1, 2]
    
    # Calculate improvement of adaptive over fixed 50% sparse
    improvements = []
    seq_lens_for_improvement = []
    
    for seq_len in sequence_lengths:
        if (seq_len in analysis and 
            'adaptive_sparse' in analysis[seq_len] and 
            'fixed_sparse_50' in analysis[seq_len]):
            
            adaptive_loss = analysis[seq_len]['adaptive_sparse']['val_loss_mean']
            fixed_loss = analysis[seq_len]['fixed_sparse_50']['val_loss_mean']
            
            improvement = (fixed_loss - adaptive_loss) / fixed_loss * 100
            improvements.append(improvement)
            seq_lens_for_improvement.append(seq_len)
    
    if improvements:
        ax6.bar(seq_lens_for_improvement, improvements, alpha=0.7)
        ax6.set_xlabel('Sequence Length')
        ax6.set_ylabel('Improvement over Fixed 50% (%)')
        ax6.set_title('Adaptive vs Fixed 50% Sparse')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No comparison data available', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Adaptive vs Fixed 50% Sparse')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'experiment_3_comprehensive_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {os.path.join(config.results_dir, 'experiment_3_comprehensive_analysis.png')}")


def save_results(all_results: Dict, analysis: Dict, config: ExperimentConfig):
    """Save detailed results to files."""
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save detailed results
    results_file = os.path.join(config.results_dir, 'experiment_3_detailed_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Convert all results
    serializable_results = {}
    for seq_len, seq_results in all_results.items():
        serializable_results[str(seq_len)] = {}
        for model_name, runs in seq_results.items():
            serializable_results[str(seq_len)][model_name] = []
            for run in runs:
                serializable_run = {}
                for key, value in run.items():
                    if isinstance(value, (list, tuple)):
                        serializable_run[key] = [convert_numpy(item) for item in value]
                    else:
                        serializable_run[key] = convert_numpy(value)
                serializable_results[str(seq_len)][model_name].append(serializable_run)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save analysis summary
    analysis_file = os.path.join(config.results_dir, 'experiment_3_analysis_summary.json')
    
    serializable_analysis = {}
    for seq_len, seq_analysis in analysis.items():
        serializable_analysis[str(seq_len)] = {}
        for model_name, stats in seq_analysis.items():
            serializable_analysis[str(seq_len)][model_name] = {
                k: convert_numpy(v) for k, v in stats.items()
            }
    
    with open(analysis_file, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"Detailed results saved to: {results_file}")
    print(f"Analysis summary saved to: {analysis_file}")


def main():
    """Main experiment function."""
    print("="*80)
    print("EXPERIMENT 3: DYNAMIC SPARSITY AND ADAPTIVE ATTENTION PATTERNS")
    print("="*80)
    
    # Initialize configuration
    config = ExperimentConfig()
    
    print(f"Device: {config.device}")
    print(f"Sequence lengths: {config.sequence_lengths}")
    print(f"Model variants: {[v['name'] for v in config.model_variants]}")
    print(f"Random seeds: {config.random_seeds}")
    print(f"Training steps: {config.steps}")
    
    # Run experiments for each sequence length
    all_results = {}
    
    for seq_len in config.sequence_lengths:
        try:
            results = run_experiment_for_sequence_length(config, seq_len)
            all_results[seq_len] = results
        except Exception as e:
            print(f"Error running experiment for sequence length {seq_len}: {e}")
            continue
    
    # Analyze results
    analysis = analyze_results(all_results)
    
    # Create visualizations
    create_visualizations(all_results, analysis, config)
    
    # Save results
    save_results(all_results, analysis, config)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("EXPERIMENT 3 COMPLETED")
    print(f"{'='*80}")
    
    print("\nKEY FINDINGS:")
    print("-" * 40)
    
    # Find best performing adaptive vs fixed
    for seq_len in sorted(analysis.keys()):
        if 'adaptive_sparse' in analysis[seq_len] and 'fixed_sparse_50' in analysis[seq_len]:
            adaptive_loss = analysis[seq_len]['adaptive_sparse']['val_loss_mean']
            fixed_loss = analysis[seq_len]['fixed_sparse_50']['val_loss_mean']
            improvement = (fixed_loss - adaptive_loss) / fixed_loss * 100
            
            adaptive_speed = analysis[seq_len]['adaptive_sparse']['tokens_per_second_mean']
            fixed_speed = analysis[seq_len]['fixed_sparse_50']['tokens_per_second_mean']
            speed_change = (adaptive_speed - fixed_speed) / fixed_speed * 100
            
            print(f"Seq Len {seq_len}:")
            print(f"  Loss improvement: {improvement:+.1f}%")
            print(f"  Speed change: {speed_change:+.1f}%")
    
    print(f"\nResults saved in: {config.results_dir}/")
    print("Experiment 3 completed successfully!")


if __name__ == "__main__":
    main()
