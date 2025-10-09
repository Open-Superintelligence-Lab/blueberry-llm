"""
Experiment 4: DeepSeek Sparse vs Classic Attention - Sequence Length Comparison

Compares DeepSeek sparse attention (from paper) vs classic dense attention
across different sequence lengths.
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
from exp4_models import create_sparse_model, create_classic_model, count_parameters


# Test sequence lengths
SEQUENCE_LENGTHS = [64, 128, 256]

# Base config
BASE_CONFIG = {
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 4,
    'd_ff': 512,
    'num_experts': 4,
    'expert_top_k': 2,
    'indexer_heads': 4,
    'indexer_dim': 64,
    'batch_size': 16,
    'steps': 1000,  # Keep short for comparison
    'learning_rate': 3e-3,
    'eval_every': 200,
    'max_tokens': 50000,
    'num_documents': 1000,
    'dropout': 0.1,
    'load_balancing_weight': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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


def evaluate(model, val_loader, vocab_size, device, is_sparse=False):
    """Evaluate model"""
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
            
            # Evaluate
            if (step + 1) % config['eval_every'] == 0:
                model.eval()
                val_loss, val_acc = evaluate(model, val_loader, vocab_size, 
                                            config['device'], is_sparse)
                model.train()
                
                results['train_loss'].append(loss.item())
                results['val_loss'].append(val_loss)
                results['val_accuracy'].append(val_acc)
                results['steps'].append(step + 1)
                
                print(f"  Step {step+1}/{config['steps']}: "
                      f"Loss={loss.item():.4f}, Val Loss={val_loss:.4f}, "
                      f"Val Acc={val_acc:.4f}, Time={step_time:.3f}s")
            
            step += 1
    
    results['avg_time_per_step'] = sum(results['time_per_step']) / len(results['time_per_step'])
    return results


def run_for_sequence_length(seq_len):
    """Run both sparse and classic for a given sequence length"""
    print(f"\n{'='*80}")
    print(f"SEQUENCE LENGTH: {seq_len}")
    print(f"{'='*80}\n")
    
    # Load data
    print(f"📚 Loading data for seq_len={seq_len}...")
    train_dataset, val_dataset, vocab_size = load_data(seq_len)
    train_loader = DataLoader(train_dataset, batch_size=BASE_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BASE_CONFIG['batch_size'], shuffle=False)
    
    # Create config for this sequence length
    config = BASE_CONFIG.copy()
    config['max_seq_len'] = seq_len
    config['vocab_size'] = vocab_size
    config['sparse_top_k'] = int(seq_len * 0.5)  # Select 50% of tokens
    
    results_dir = Path('results') / f'seq_{seq_len}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Train Classic Attention
    print(f"\n🔵 Training Classic Attention (Dense)...")
    torch.manual_seed(42)  # Set seed for fair comparison
    torch.cuda.manual_seed(42)
    classic_model = create_classic_model(config).to(config['device'])
    print(f"   Parameters: {count_parameters(classic_model):,}")
    classic_results = train_model(classic_model, train_loader, val_loader, 
                                  config, vocab_size, is_sparse=False)
    
    with open(results_dir / 'classic_results.json', 'w') as f:
        json.dump(classic_results, f, indent=2)
    
    # Train Sparse Attention
    print(f"\n🟠 Training Sparse Attention (DeepSeek DSA)...")
    torch.manual_seed(42)  # Same seed for fair comparison
    torch.cuda.manual_seed(42)
    sparse_model = create_sparse_model(config).to(config['device'])
    print(f"   Parameters: {count_parameters(sparse_model):,}")
    print(f"   Sparse top-k: {config['sparse_top_k']} / {seq_len} tokens")
    sparse_results = train_model(sparse_model, train_loader, val_loader, 
                                 config, vocab_size, is_sparse=True)
    
    with open(results_dir / 'sparse_results.json', 'w') as f:
        json.dump(sparse_results, f, indent=2)
    
    # Summary
    print(f"\n📊 Results for seq_len={seq_len}:")
    print(f"   Classic: Loss={classic_results['val_loss'][-1]:.4f}, "
          f"Acc={classic_results['val_accuracy'][-1]:.4f}, "
          f"Time={classic_results['avg_time_per_step']:.3f}s/step")
    print(f"   Sparse:  Loss={sparse_results['val_loss'][-1]:.4f}, "
          f"Acc={sparse_results['val_accuracy'][-1]:.4f}, "
          f"Time={sparse_results['avg_time_per_step']:.3f}s/step")
    
    return classic_results, sparse_results, config


def plot_all_results(all_results):
    """Plot comparison across all sequence lengths"""
    print(f"\n📈 Plotting comparison curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Validation Loss vs Sequence Length
    seq_lens = []
    classic_losses = []
    sparse_losses = []
    
    for seq_len, (classic, sparse, _) in all_results.items():
        seq_lens.append(seq_len)
        classic_losses.append(classic['val_loss'][-1])
        sparse_losses.append(sparse['val_loss'][-1])
    
    axes[0, 0].plot(seq_lens, classic_losses, 'o-', label='Classic (Dense)', linewidth=2, markersize=8)
    axes[0, 0].plot(seq_lens, sparse_losses, 's-', label='Sparse (DSA)', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].set_title('Final Validation Loss vs Sequence Length')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy vs Sequence Length
    classic_accs = []
    sparse_accs = []
    
    for seq_len in seq_lens:
        classic, sparse, _ = all_results[seq_len]
        classic_accs.append(classic['val_accuracy'][-1])
        sparse_accs.append(sparse['val_accuracy'][-1])
    
    axes[0, 1].plot(seq_lens, classic_accs, 'o-', label='Classic (Dense)', linewidth=2, markersize=8)
    axes[0, 1].plot(seq_lens, sparse_accs, 's-', label='Sparse (DSA)', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].set_ylabel('Validation Accuracy')
    axes[0, 1].set_title('Final Validation Accuracy vs Sequence Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training Time vs Sequence Length
    classic_times = []
    sparse_times = []
    
    for seq_len in seq_lens:
        classic, sparse, _ = all_results[seq_len]
        classic_times.append(classic['avg_time_per_step'])
        sparse_times.append(sparse['avg_time_per_step'])
    
    axes[1, 0].plot(seq_lens, classic_times, 'o-', label='Classic (Dense)', linewidth=2, markersize=8)
    axes[1, 0].plot(seq_lens, sparse_times, 's-', label='Sparse (DSA)', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Sequence Length')
    axes[1, 0].set_ylabel('Time per Step (s)')
    axes[1, 0].set_title('Training Speed vs Sequence Length')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training curves for largest sequence length
    max_seq_len = max(seq_lens)
    classic, sparse, _ = all_results[max_seq_len]
    
    axes[1, 1].plot(classic['steps'], classic['val_loss'], 
                   'o-', label=f'Classic (Dense, L={max_seq_len})', linewidth=2, alpha=0.7)
    axes[1, 1].plot(sparse['steps'], sparse['val_loss'], 
                   's-', label=f'Sparse (DSA, L={max_seq_len})', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Training Steps')
    axes[1, 1].set_ylabel('Validation Loss')
    axes[1, 1].set_title(f'Training Curves (Sequence Length = {max_seq_len})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('DeepSeek Sparse vs Classic Attention: Sequence Length Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path('results') / 'sequence_length_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison plot to {save_path}")


def main():
    """Main experiment"""
    print("\n" + "="*80)
    print("EXPERIMENT 4: DeepSeek Sparse vs Classic Attention")
    print("Sequence Length Comparison")
    print("="*80)
    print(f"\nTesting sequence lengths: {SEQUENCE_LENGTHS}")
    print(f"Device: {BASE_CONFIG['device']}")
    print(f"Steps per model: {BASE_CONFIG['steps']}")
    
    all_results = {}
    
    for seq_len in SEQUENCE_LENGTHS:
        classic_results, sparse_results, config = run_for_sequence_length(seq_len)
        all_results[seq_len] = (classic_results, sparse_results, config)
    
    # Plot comparison
    plot_all_results(all_results)
    
    # Save summary
    summary = {
        'sequence_lengths': SEQUENCE_LENGTHS,
        'results': {}
    }
    
    for seq_len in SEQUENCE_LENGTHS:
        classic, sparse, _ = all_results[seq_len]
        summary['results'][seq_len] = {
            'classic': {
                'final_val_loss': classic['val_loss'][-1],
                'final_val_accuracy': classic['val_accuracy'][-1],
                'avg_time_per_step': classic['avg_time_per_step']
            },
            'sparse': {
                'final_val_loss': sparse['val_loss'][-1],
                'final_val_accuracy': sparse['val_accuracy'][-1],
                'avg_time_per_step': sparse['avg_time_per_step']
            }
        }
    
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ EXPERIMENT 4 COMPLETED")
    print(f"{'='*80}")
    print(f"\nResults saved to: results/")
    print(f"  - Comparison plot: results/sequence_length_comparison.png")
    print(f"  - Summary: results/summary.json")
    print(f"  - Per-length results: results/seq_*/")


if __name__ == '__main__':
    main()
