"""
Experiment 5: DeepSeek MHLA with/without Sparse Attention - Sequence Length Comparison

Compares DeepSeek Multi-Head Latent Attention:
- Baseline: MHLA (dense)  
- Experimental: MHLA + Sparse Attention (with Lightning Indexer)

This tests whether sparse token selection improves MHLA's already efficient attention.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add parent directories to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

from data.dataset import TextTokenDataset
from data.loader import load_and_cache_data
from configs.moe_config import MoEModelConfig
from exp5_models import create_sparse_model, create_baseline_model, count_parameters


# Test sequence lengths
SEQUENCE_LENGTHS = [64, 128, 256, 1024, 2048]

# Base config
BASE_CONFIG = {
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'd_ff': 2048,
    'num_experts': 4,
    'expert_top_k': 2,
    'indexer_heads': 4,
    'indexer_dim': 64,
    'batch_size': 8,  # Smaller for longer sequences
    'steps': 500,  # Fewer steps for longer sequences to manage time
    'learning_rate': 3e-3,
    'eval_every': 100,  # More frequent evaluation for shorter training
    'max_tokens': 50000,
    'num_documents': 1000,
    'dropout': 0.1,
    'load_balancing_weight': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # MHLA parameters
    'q_lora_rank': None,  # Set to value for compressed queries
    'kv_lora_rank': 64,   # Latent KV dimension
    'qk_rope_head_dim': None,  # Will be set to d_model // n_heads
    'qk_nope_head_dim': 0,  # 0 means all goes to RoPE
    'v_head_dim': None,  # Will be set to d_model // n_heads
}


def load_data(seq_len):
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


def get_dynamic_batch_size(seq_len):
    """Get appropriate batch size based on sequence length"""
    if seq_len <= 256:
        return 8
    elif seq_len <= 512:
        return 4
    elif seq_len <= 1024:
        return 2
    else:  # 2048
        return 1


def evaluate(model, val_loader, vocab_size, device, is_sparse=False):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            if is_sparse:
                logits, _, _ = model(input_ids, return_index_scores=False)
            else:
                logits, _ = model(input_ids)
            
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


def train_model(model, train_loader, val_loader, config, vocab_size, is_sparse=False):
    """Train a model"""
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
            
            if is_sparse:
                logits, aux_loss, _ = model(input_ids, return_index_scores=False)
            else:
                logits, aux_loss = model(input_ids)
            
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
            
            if step % config['eval_every'] == 0:
                model.eval()
                val_loss, val_acc = evaluate(model, val_loader, vocab_size, config['device'], is_sparse)
                model.train()
                
                results['steps'].append(step)
                results['train_loss'].append(loss.item())
                results['val_loss'].append(val_loss)
                results['val_accuracy'].append(val_acc)
                
                print(f"  Step {step:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            step += 1
    
    # Final evaluation
    model.eval()
    final_val_loss, final_val_acc = evaluate(model, val_loader, vocab_size, config['device'], is_sparse)
    
    avg_time = sum(results['time_per_step']) / len(results['time_per_step'])
    
    return {
        **results,
        'final_val_loss': final_val_loss,
        'final_val_accuracy': final_val_acc,
        'avg_time_per_step': avg_time,
        'total_training_time': sum(results['time_per_step'])
    }


def run_for_sequence_length(seq_len):
    """Run both baseline and sparse models for a given sequence length"""
    print(f"\n{'='*80}")
    print(f"Testing Sequence Length: {seq_len}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading data...")
    train_dataset, val_dataset, vocab_size = load_data(seq_len)
    print(f"Dataset size: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create data loaders with dynamic batch size
    batch_size = get_dynamic_batch_size(seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Using batch size: {batch_size} (for seq_len={seq_len})")
    
    # Prepare config
    config = {
        **BASE_CONFIG,
        'max_seq_len': seq_len,
        'vocab_size': vocab_size,
        'sparse_top_k': int(seq_len * 0.5),  # 50% sparsity
    }
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # ===== BASELINE: Dense MHLA =====
    print("\n" + "="*80)
    print("Training BASELINE: DeepSeek MHLA (Dense)")
    print("="*80)
    
    baseline_model = create_baseline_model(config).to(config['device'])
    baseline_params = count_parameters(baseline_model)
    print(f"Baseline Model Parameters: {baseline_params:,}")
    
    baseline_results = train_model(
        baseline_model, train_loader, val_loader, config, vocab_size, is_sparse=False
    )
    
    print(f"\nBaseline Final Results:")
    print(f"  Val Loss: {baseline_results['final_val_loss']:.4f}")
    print(f"  Val Accuracy: {baseline_results['final_val_accuracy']:.4f}")
    print(f"  Avg Time/Step: {baseline_results['avg_time_per_step']:.4f}s")
    
    # Clear memory
    del baseline_model
    torch.cuda.empty_cache()
    
    # Reset seed for fair comparison
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # ===== EXPERIMENTAL: Sparse MHLA =====
    print("\n" + "="*80)
    print("Training EXPERIMENTAL: DeepSeek MHLA + Sparse Attention")
    print("="*80)
    
    sparse_model = create_sparse_model(config).to(config['device'])
    sparse_params = count_parameters(sparse_model)
    print(f"Sparse Model Parameters: {sparse_params:,}")
    print(f"Additional Parameters: {sparse_params - baseline_params:,} ({100*(sparse_params - baseline_params)/baseline_params:.1f}%)")
    
    sparse_results = train_model(
        sparse_model, train_loader, val_loader, config, vocab_size, is_sparse=True
    )
    
    print(f"\nSparse Final Results:")
    print(f"  Val Loss: {sparse_results['final_val_loss']:.4f}")
    print(f"  Val Accuracy: {sparse_results['final_val_accuracy']:.4f}")
    print(f"  Avg Time/Step: {sparse_results['avg_time_per_step']:.4f}s")
    
    # Clear memory
    del sparse_model
    torch.cuda.empty_cache()
    
    # Save results
    results_dir = Path(f"results/seq_{seq_len}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "baseline_results.json", 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    with open(results_dir / "sparse_results.json", 'w') as f:
        json.dump(sparse_results, f, indent=2)
    
    return {
        'seq_len': seq_len,
        'baseline': baseline_results,
        'sparse': sparse_results,
        'baseline_params': baseline_params,
        'sparse_params': sparse_params
    }


def plot_comparison(all_results):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Experiment 5: DeepSeek MHLA Dense vs Sparse Comparison', fontsize=16, fontweight='bold')
    
    seq_lengths = [r['seq_len'] for r in all_results]
    baseline_losses = [r['baseline']['final_val_loss'] for r in all_results]
    sparse_losses = [r['sparse']['final_val_loss'] for r in all_results]
    baseline_accs = [r['baseline']['final_val_accuracy'] * 100 for r in all_results]
    sparse_accs = [r['sparse']['final_val_accuracy'] * 100 for r in all_results]
    baseline_times = [r['baseline']['avg_time_per_step'] for r in all_results]
    sparse_times = [r['sparse']['avg_time_per_step'] for r in all_results]
    
    # Plot 1: Validation Loss
    axes[0, 0].plot(seq_lengths, baseline_losses, 'o-', label='Baseline MHLA', linewidth=2, markersize=8)
    axes[0, 0].plot(seq_lengths, sparse_losses, 's-', label='Sparse MHLA', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Sequence Length', fontsize=12)
    axes[0, 0].set_ylabel('Validation Loss', fontsize=12)
    axes[0, 0].set_title('Final Validation Loss vs Sequence Length', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    axes[0, 1].plot(seq_lengths, baseline_accs, 'o-', label='Baseline MHLA', linewidth=2, markersize=8)
    axes[0, 1].plot(seq_lengths, sparse_accs, 's-', label='Sparse MHLA', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Sequence Length', fontsize=12)
    axes[0, 1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Final Validation Accuracy vs Sequence Length', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training Time
    axes[1, 0].plot(seq_lengths, baseline_times, 'o-', label='Baseline MHLA', linewidth=2, markersize=8)
    axes[1, 0].plot(seq_lengths, sparse_times, 's-', label='Sparse MHLA', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Sequence Length', fontsize=12)
    axes[1, 0].set_ylabel('Time per Step (s)', fontsize=12)
    axes[1, 0].set_title('Training Time vs Sequence Length', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training curves for longest sequence
    longest_result = all_results[-1]
    axes[1, 1].plot(longest_result['baseline']['steps'], 
                   longest_result['baseline']['val_loss'],
                   'o-', label='Baseline MHLA', linewidth=2, markersize=6)
    axes[1, 1].plot(longest_result['sparse']['steps'],
                   longest_result['sparse']['val_loss'],
                   's-', label='Sparse MHLA', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Training Step', fontsize=12)
    axes[1, 1].set_ylabel('Validation Loss', fontsize=12)
    axes[1, 1].set_title(f'Training Curves (Seq Len = {longest_result["seq_len"]})', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/sequence_length_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved comparison plot to results/sequence_length_comparison.png")


def main():
    """Run experiment across all sequence lengths"""
    print("="*80)
    print("EXPERIMENT 5: DeepSeek MHLA Dense vs Sparse Comparison")
    print("="*80)
    print(f"\nTesting sequence lengths: {SEQUENCE_LENGTHS}")
    print(f"Model: {BASE_CONFIG['d_model']}d, {BASE_CONFIG['n_layers']}L, {BASE_CONFIG['n_heads']}H")
    print(f"Training steps: {BASE_CONFIG['steps']}")
    print(f"Device: {BASE_CONFIG['device']}")
    
    all_results = []
    
    for seq_len in SEQUENCE_LENGTHS:
        result = run_for_sequence_length(seq_len)
        all_results.append(result)
    
    # Create comparison plot
    plot_comparison(all_results)
    
    # Save summary
    summary = {
        'config': BASE_CONFIG,
        'sequence_lengths': SEQUENCE_LENGTHS,
        'results': [
            {
                'seq_len': r['seq_len'],
                'baseline': {
                    'val_loss': r['baseline']['final_val_loss'],
                    'val_accuracy': r['baseline']['final_val_accuracy'],
                    'time_per_step': r['baseline']['avg_time_per_step'],
                    'parameters': r['baseline_params']
                },
                'sparse': {
                    'val_loss': r['sparse']['final_val_loss'],
                    'val_accuracy': r['sparse']['final_val_accuracy'],
                    'time_per_step': r['sparse']['avg_time_per_step'],
                    'parameters': r['sparse_params']
                }
            }
            for r in all_results
        ]
    }
    
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*80)
    print("EXPERIMENT 5 COMPLETE - SUMMARY")
    print("="*80)
    print(f"\n{'Seq Len':<10} {'Baseline Loss':<15} {'Sparse Loss':<15} {'Improvement':<15} {'Baseline Acc':<15} {'Sparse Acc':<15}")
    print("-" * 95)
    
    for r in all_results:
        baseline_loss = r['baseline']['final_val_loss']
        sparse_loss = r['sparse']['final_val_loss']
        improvement = ((baseline_loss - sparse_loss) / sparse_loss) * 100
        baseline_acc = r['baseline']['final_val_accuracy'] * 100
        sparse_acc = r['sparse']['final_val_accuracy'] * 100
        
        print(f"{r['seq_len']:<10} {baseline_loss:<15.2f} {sparse_loss:<15.2f} {improvement:<14.0f}% {baseline_acc:<14.1f}% {sparse_acc:<14.1f}%")
    
    print("\n✅ All results saved to results/")
    print("   - sequence_length_comparison.png (visualization)")
    print("   - summary.json (numerical results)")
    print("   - seq_*/baseline_results.json")
    print("   - seq_*/sparse_results.json")


if __name__ == "__main__":
    main()

