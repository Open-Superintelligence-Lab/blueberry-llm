"""
Experiment 5: Optimal Sparsity Analysis

Systematic analysis of optimal sparsity ratios across sequence lengths
for DeepSeek sparse attention to improve pretraining speed and quality.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
from typing import Dict, List, Tuple

# Add parent directories to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

from data.dataset import TextTokenDataset
from data.loader import load_and_cache_data
from configs.moe_config import MoEModelConfig
from exp5_models import create_sparse_model, count_parameters, get_sparse_top_k


# Test configurations
SEQUENCE_LENGTHS = [64, 128, 256, 512, 1024]
SPARSITY_RATIOS = [0.25, 0.33, 0.5, 0.67, 0.75, 0.9]  # 0.25 = 75% sparse, 0.9 = 10% sparse
NUM_RUNS = 3  # Multiple runs for statistical significance

# Base config
BASE_CONFIG = {
    'd_model': 256,  # Reduced from 512
    'n_heads': 8,
    'n_layers': 4,   # Reduced from 6
    'd_ff': 1024,    # Reduced from 2048
    'num_experts': 4,
    'expert_top_k': 2,
    'indexer_heads': 4,
    'indexer_dim': 64,
    'batch_size': 16,  # Reduced from 32
    'steps': 500,  # Reduced for systematic analysis
    'learning_rate': 1e-4,
    'eval_every': 100,
    'max_tokens': 100000,
    'num_documents': 2000,
    'dropout': 0.1,
    'load_balancing_weight': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def load_data(seq_len: int) -> Tuple[TextTokenDataset, TextTokenDataset, int]:
    """Load data for a given sequence length"""
    data_config = MoEModelConfig(
        max_seq_len=seq_len,
        max_tokens=BASE_CONFIG['max_tokens'],
        num_documents=BASE_CONFIG['num_documents']
    )
    texts, tokenizer, tokens = load_and_cache_data(data_config)
    
    full_dataset = TextTokenDataset(tokens, seq_len)
    val_size = len(full_dataset) // 10
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset, data_config.vocab_size


def evaluate(model, val_loader, vocab_size, device):
    """Evaluate model"""
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            logits, _, _ = model(input_ids, return_index_scores=False)
            
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, config, vocab_size, run_id):
    """Train a model for one run"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    results = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'steps': [],
        'time_per_step': []
    }
    
    model.train()
    step = 0
    
    while step < config['steps']:
        for batch in train_loader:
            if step >= config['steps']:
                break
            
            step_start = time.time()
            
            input_ids, targets = batch
            input_ids = input_ids.to(config['device'])
            targets = targets.to(config['device'])
            
            logits, aux_loss, _ = model(input_ids, return_index_scores=False)
            
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )
            if aux_loss is not None:
                loss = loss + aux_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step_time = time.time() - step_start
            results['time_per_step'].append(step_time)
            
            # Evaluate
            if (step + 1) % config['eval_every'] == 0:
                model.eval()
                val_loss, val_acc = evaluate(model, val_loader, vocab_size, config['device'])
                model.train()
                
                results['train_loss'].append(loss.item())
                results['val_loss'].append(val_loss)
                results['val_accuracy'].append(val_acc)
                results['steps'].append(step + 1)
                
                print(f"    Run {run_id}: Step {step+1}/{config['steps']}: "
                      f"Loss={loss.item():.4f}, Val Loss={val_loss:.4f}, "
                      f"Val Acc={val_acc:.4f}, Time={step_time:.3f}s")
            
            step += 1
    
    results['avg_time_per_step'] = sum(results['time_per_step']) / len(results['time_per_step'])
    results['final_val_loss'] = results['val_loss'][-1] if results['val_loss'] else float('inf')
    results['final_val_accuracy'] = results['val_accuracy'][-1] if results['val_accuracy'] else 0.0
    
    return results


def run_sparsity_experiment(seq_len: int, sparsity_ratio: float):
    """Run experiment for one sparsity ratio at one sequence length"""
    print(f"\n{'='*60}")
    print(f"SEQUENCE LENGTH: {seq_len}, SPARSITY RATIO: {sparsity_ratio:.2f}")
    print(f"{'='*60}")
    
    # Calculate top_k from sparsity ratio
    top_k = get_sparse_top_k(seq_len, sparsity_ratio)
    actual_sparsity = 1.0 - (top_k / seq_len)
    
    print(f"Top-k: {top_k}/{seq_len} tokens (actual sparsity: {actual_sparsity:.2f})")
    
    # Load data
    print(f"📚 Loading data for seq_len={seq_len}...")
    train_dataset, val_dataset, vocab_size = load_data(seq_len)
    train_loader = DataLoader(train_dataset, batch_size=BASE_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BASE_CONFIG['batch_size'], shuffle=False)
    
    # Create config for this experiment
    config = BASE_CONFIG.copy()
    config['max_seq_len'] = seq_len
    config['vocab_size'] = vocab_size
    config['sparse_top_k'] = top_k
    
    # Results storage
    all_runs = []
    
    # Run multiple times for statistical significance
    for run_id in range(1, NUM_RUNS + 1):
        print(f"\n🔄 Run {run_id}/{NUM_RUNS}")
        
        # Set seed for reproducibility (different seed per run)
        torch.manual_seed(42 + run_id)
        torch.cuda.manual_seed(42 + run_id)
        
        # Create model
        model = create_sparse_model(config).to(config['device'])
        print(f"   Parameters: {count_parameters(model):,}")
        
        # Train model
        run_results = train_model(model, train_loader, val_loader, config, vocab_size, run_id)
        all_runs.append(run_results)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Aggregate results across runs
    final_val_losses = [run['final_val_loss'] for run in all_runs]
    final_val_accuracies = [run['final_val_accuracy'] for run in all_runs]
    avg_times = [run['avg_time_per_step'] for run in all_runs]
    
    aggregated_results = {
        'seq_len': seq_len,
        'sparsity_ratio': sparsity_ratio,
        'top_k': top_k,
        'actual_sparsity': actual_sparsity,
        'runs': all_runs,
        'final_val_loss_mean': np.mean(final_val_losses),
        'final_val_loss_std': np.std(final_val_losses),
        'final_val_accuracy_mean': np.mean(final_val_accuracies),
        'final_val_accuracy_std': np.std(final_val_accuracies),
        'avg_time_mean': np.mean(avg_times),
        'avg_time_std': np.std(avg_times)
    }
    
    print(f"\n📊 Results for seq_len={seq_len}, sparsity={sparsity_ratio:.2f}:")
    print(f"   Val Loss: {aggregated_results['final_val_loss_mean']:.4f} ± {aggregated_results['final_val_loss_std']:.4f}")
    print(f"   Val Acc:  {aggregated_results['final_val_accuracy_mean']:.4f} ± {aggregated_results['final_val_accuracy_std']:.4f}")
    print(f"   Time:     {aggregated_results['avg_time_mean']:.3f} ± {aggregated_results['avg_time_std']:.3f}s/step")
    
    return aggregated_results


def find_optimal_sparsity(all_results: Dict) -> Dict:
    """Find optimal sparsity ratio for each sequence length"""
    optimal_results = {}
    
    for seq_len in SEQUENCE_LENGTHS:
        seq_results = [r for r in all_results if r['seq_len'] == seq_len]
        
        if not seq_results:
            continue
            
        # Find minimum validation loss
        best_result = min(seq_results, key=lambda x: x['final_val_loss_mean'])
        
        optimal_results[seq_len] = {
            'optimal_sparsity_ratio': best_result['sparsity_ratio'],
            'optimal_top_k': best_result['top_k'],
            'optimal_val_loss': best_result['final_val_loss_mean'],
            'optimal_val_loss_std': best_result['final_val_loss_std'],
            'optimal_val_accuracy': best_result['final_val_accuracy_mean'],
            'optimal_val_accuracy_std': best_result['final_val_accuracy_std'],
            'all_sparsity_results': seq_results
        }
    
    return optimal_results


def statistical_analysis(all_results: Dict) -> Dict:
    """Perform statistical analysis of results"""
    analysis = {}
    
    for seq_len in SEQUENCE_LENGTHS:
        seq_results = [r for r in all_results if r['seq_len'] == seq_len]
        
        if len(seq_results) < 2:
            continue
            
        # Extract data for ANOVA
        sparsity_ratios = [r['sparsity_ratio'] for r in seq_results]
        val_losses = [r['final_val_loss_mean'] for r in seq_results]
        val_loss_stds = [r['final_val_loss_std'] for r in seq_results]
        
        # Simple statistical tests
        # Find best and worst performing sparsity ratios
        best_idx = np.argmin(val_losses)
        worst_idx = np.argmax(val_losses)
        
        # Effect size (Cohen's d)
        if val_loss_stds[best_idx] > 0 and val_loss_stds[worst_idx] > 0:
            pooled_std = np.sqrt((val_loss_stds[best_idx]**2 + val_loss_stds[worst_idx]**2) / 2)
            cohens_d = abs(val_losses[best_idx] - val_losses[worst_idx]) / pooled_std
        else:
            cohens_d = 0.0
        
        analysis[seq_len] = {
            'best_sparsity': sparsity_ratios[best_idx],
            'worst_sparsity': sparsity_ratios[worst_idx],
            'best_loss': val_losses[best_idx],
            'worst_loss': val_losses[worst_idx],
            'effect_size': cohens_d,
            'loss_range': max(val_losses) - min(val_losses),
            'relative_improvement': (val_losses[worst_idx] - val_losses[best_idx]) / val_losses[worst_idx] * 100
        }
    
    return analysis


def plot_results(all_results: List[Dict], optimal_results: Dict, analysis: Dict):
    """Plot comprehensive results"""
    print(f"\n📈 Creating comprehensive plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Validation Loss vs Sparsity Ratio (all sequence lengths)
    ax1 = plt.subplot(3, 3, 1)
    for seq_len in SEQUENCE_LENGTHS:
        seq_results = [r for r in all_results if r['seq_len'] == seq_len]
        if seq_results:
            sparsity_ratios = [r['sparsity_ratio'] for r in seq_results]
            val_losses = [r['final_val_loss_mean'] for r in seq_results]
            val_loss_stds = [r['final_val_loss_std'] for r in seq_results]
            
            plt.errorbar(sparsity_ratios, val_losses, yerr=val_loss_stds, 
                        marker='o', label=f'L={seq_len}', linewidth=2, markersize=6)
    
    plt.xlabel('Sparsity Ratio')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Sparsity Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Optimal Sparsity Ratio vs Sequence Length
    ax2 = plt.subplot(3, 3, 2)
    seq_lens = list(optimal_results.keys())
    optimal_sparsities = [optimal_results[seq_len]['optimal_sparsity_ratio'] for seq_len in seq_lens]
    
    plt.plot(seq_lens, optimal_sparsities, 'o-', linewidth=3, markersize=8, color='red')
    plt.xlabel('Sequence Length')
    plt.ylabel('Optimal Sparsity Ratio')
    plt.title('Optimal Sparsity vs Sequence Length')
    plt.grid(True, alpha=0.3)
    
    # 3. Performance Improvement vs Sequence Length
    ax3 = plt.subplot(3, 3, 3)
    improvements = [analysis[seq_len]['relative_improvement'] for seq_len in seq_lens if seq_len in analysis]
    seq_lens_with_analysis = [seq_len for seq_len in seq_lens if seq_len in analysis]
    
    plt.bar(seq_lens_with_analysis, improvements, alpha=0.7, color='green')
    plt.xlabel('Sequence Length')
    plt.ylabel('Relative Improvement (%)')
    plt.title('Performance Improvement from Optimal Sparsity')
    plt.grid(True, alpha=0.3)
    
    # 4. Effect Size vs Sequence Length
    ax4 = plt.subplot(3, 3, 4)
    effect_sizes = [analysis[seq_len]['effect_size'] for seq_len in seq_lens if seq_len in analysis]
    
    plt.bar(seq_lens_with_analysis, effect_sizes, alpha=0.7, color='purple')
    plt.xlabel('Sequence Length')
    plt.ylabel("Cohen's d (Effect Size)")
    plt.title('Effect Size of Sparsity Optimization')
    plt.grid(True, alpha=0.3)
    
    # 5. Training Time vs Sparsity Ratio
    ax5 = plt.subplot(3, 3, 5)
    for seq_len in SEQUENCE_LENGTHS:
        seq_results = [r for r in all_results if r['seq_len'] == seq_len]
        if seq_results:
            sparsity_ratios = [r['sparsity_ratio'] for r in seq_results]
            times = [r['avg_time_mean'] for r in seq_results]
            time_stds = [r['avg_time_std'] for r in seq_results]
            
            plt.errorbar(sparsity_ratios, times, yerr=time_stds, 
                        marker='s', label=f'L={seq_len}', linewidth=2, markersize=6)
    
    plt.xlabel('Sparsity Ratio')
    plt.ylabel('Training Time (s/step)')
    plt.title('Training Speed vs Sparsity Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Validation Accuracy vs Sparsity Ratio
    ax6 = plt.subplot(3, 3, 6)
    for seq_len in SEQUENCE_LENGTHS:
        seq_results = [r for r in all_results if r['seq_len'] == seq_len]
        if seq_results:
            sparsity_ratios = [r['sparsity_ratio'] for r in seq_results]
            accuracies = [r['final_val_accuracy_mean'] for r in seq_results]
            accuracy_stds = [r['final_val_accuracy_std'] for r in seq_results]
            
            plt.errorbar(sparsity_ratios, accuracies, yerr=accuracy_stds, 
                        marker='^', label=f'L={seq_len}', linewidth=2, markersize=6)
    
    plt.xlabel('Sparsity Ratio')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Sparsity Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Heatmap of validation loss
    ax7 = plt.subplot(3, 3, 7)
    loss_matrix = np.full((len(SEQUENCE_LENGTHS), len(SPARSITY_RATIOS)), np.nan)
    
    for i, seq_len in enumerate(SEQUENCE_LENGTHS):
        for j, sparsity_ratio in enumerate(SPARSITY_RATIOS):
            result = next((r for r in all_results if r['seq_len'] == seq_len and r['sparsity_ratio'] == sparsity_ratio), None)
            if result:
                loss_matrix[i, j] = result['final_val_loss_mean']
    
    im = plt.imshow(loss_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Validation Loss')
    plt.xlabel('Sparsity Ratio Index')
    plt.ylabel('Sequence Length Index')
    plt.title('Validation Loss Heatmap')
    plt.xticks(range(len(SPARSITY_RATIOS)), [f'{r:.2f}' for r in SPARSITY_RATIOS])
    plt.yticks(range(len(SEQUENCE_LENGTHS)), SEQUENCE_LENGTHS)
    
    # 8. Optimal Top-k vs Sequence Length
    ax8 = plt.subplot(3, 3, 8)
    optimal_top_ks = [optimal_results[seq_len]['optimal_top_k'] for seq_len in seq_lens]
    
    plt.plot(seq_lens, optimal_top_ks, 'o-', linewidth=3, markersize=8, color='orange')
    plt.plot(seq_lens, [s/2 for s in seq_lens], '--', label='L/2 (baseline)', alpha=0.7)
    plt.xlabel('Sequence Length')
    plt.ylabel('Optimal Top-k')
    plt.title('Optimal Top-k vs Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = "Summary Statistics\n\n"
    for seq_len in seq_lens:
        if seq_len in analysis:
            summary_text += f"L={seq_len}:\n"
            summary_text += f"  Optimal sparsity: {optimal_results[seq_len]['optimal_sparsity_ratio']:.2f}\n"
            summary_text += f"  Improvement: {analysis[seq_len]['relative_improvement']:.1f}%\n"
            summary_text += f"  Effect size: {analysis[seq_len]['effect_size']:.2f}\n\n"
    
    plt.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Experiment 5: Optimal Sparsity Analysis Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    save_path = Path('results') / 'sparsity_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comprehensive analysis plot to {save_path}")


def main():
    """Main experiment"""
    print("\n" + "="*80)
    print("EXPERIMENT 5: Optimal Sparsity Analysis")
    print("Systematic analysis of optimal sparsity ratios across sequence lengths")
    print("="*80)
    print(f"\nTesting sequence lengths: {SEQUENCE_LENGTHS}")
    print(f"Testing sparsity ratios: {SPARSITY_RATIOS}")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Device: {BASE_CONFIG['device']}")
    print(f"Steps per model: {BASE_CONFIG['steps']}")
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Run all experiments
    all_results = []
    total_experiments = len(SEQUENCE_LENGTHS) * len(SPARSITY_RATIOS)
    experiment_count = 0
    
    for seq_len in SEQUENCE_LENGTHS:
        for sparsity_ratio in SPARSITY_RATIOS:
            experiment_count += 1
            print(f"\n🔬 Experiment {experiment_count}/{total_experiments}")
            
            result = run_sparsity_experiment(seq_len, sparsity_ratio)
            all_results.append(result)
            
            # Save intermediate results
            with open(results_dir / 'intermediate_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Find optimal sparsity ratios
    print(f"\n🎯 Finding optimal sparsity ratios...")
    optimal_results = find_optimal_sparsity(all_results)
    
    # Statistical analysis
    print(f"📊 Performing statistical analysis...")
    analysis = statistical_analysis(all_results)
    
    # Create comprehensive plots
    plot_results(all_results, optimal_results, analysis)
    
    # Save final results
    final_results = {
        'experiment_config': {
            'sequence_lengths': SEQUENCE_LENGTHS,
            'sparsity_ratios': SPARSITY_RATIOS,
            'num_runs': NUM_RUNS,
            'base_config': BASE_CONFIG
        },
        'all_results': all_results,
        'optimal_sparsity_results': optimal_results,
        'statistical_analysis': analysis
    }
    
    with open(results_dir / 'optimal_sparsity_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("✅ EXPERIMENT 5 COMPLETED")
    print(f"{'='*80}")
    print(f"\n🎯 OPTIMAL SPARSITY RATIOS:")
    for seq_len in SEQUENCE_LENGTHS:
        if seq_len in optimal_results:
            opt = optimal_results[seq_len]
            print(f"   L={seq_len}: {opt['optimal_sparsity_ratio']:.2f} "
                  f"(top-k={opt['optimal_top_k']}, loss={opt['optimal_val_loss']:.4f})")
    
    print(f"\n📈 KEY INSIGHTS:")
    for seq_len in SEQUENCE_LENGTHS:
        if seq_len in analysis:
            a = analysis[seq_len]
            print(f"   L={seq_len}: {a['relative_improvement']:.1f}% improvement, "
                  f"effect size={a['effect_size']:.2f}")
    
    print(f"\n📁 Results saved to: results/")
    print(f"   - Comprehensive analysis: results/sparsity_analysis.png")
    print(f"   - Detailed results: results/optimal_sparsity_results.json")
    print(f"   - Intermediate results: results/intermediate_results.json")


if __name__ == '__main__':
    main()
