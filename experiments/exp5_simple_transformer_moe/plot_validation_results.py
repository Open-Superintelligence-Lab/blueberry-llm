#!/usr/bin/env python3
"""
Plot validation results comparison
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_dir = Path("results/ablation_batch_seqlen")

# Collect results for the 9 main configs
configs = [
    'large_batch_lr001', 'long_seq_lr001', 'balanced_lr001',
    'large_batch_lr002', 'long_seq_lr002', 'balanced_lr002',
    'large_batch_lr0005', 'long_seq_lr0005', 'balanced_lr0005',
]

# Define colors and styles
strategy_colors = {
    'large_batch': '#1f77b4',  # blue
    'long_seq': '#2ca02c',      # green
    'balanced': '#ff7f0e',      # orange
}

# Collect all results
all_data = []
for config_name in configs:
    result_file = results_dir / f"{config_name}_result.json"
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
            all_data.append(data)

print(f"Found {len(all_data)} configurations")

# Create figure with 2 subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Experiment 5: Validation Loss Comparison - All Ablation Configurations', 
             fontsize=18, fontweight='bold', y=0.98)

# Plot 1: Validation Loss vs Time (using training time)
ax = axes[0]

# Group by strategy
strategies = ['Large Batch', 'Long Seq', 'Balanced']
lrs = sorted(set(d['lr'] for d in all_data))

for strategy_name in strategies:
    strategy_key = strategy_name.lower().replace(' ', '_')
    color = strategy_colors[strategy_key]
    
    strategy_data = [d for d in all_data if strategy_key in d['config_name']]
    strategy_data.sort(key=lambda x: x['lr'])
    
    times = [d['time'] for d in strategy_data]
    val_losses = [d['val_loss'] for d in strategy_data]
    lr_labels = [d['lr'] for d in strategy_data]
    
    # Plot with markers
    ax.plot(times, val_losses, 'o-', color=color, linewidth=3, 
            markersize=12, label=strategy_name, alpha=0.8)
    
    # Add LR annotations
    for t, vl, lr in zip(times, val_losses, lr_labels):
        ax.annotate(f'LR={lr}', (t, vl), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=8, alpha=0.7)

ax.set_xlabel('Training Time (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Final Validation Loss', fontsize=14, fontweight='bold')
ax.set_title('Validation Loss vs Training Time', fontsize=16, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, framealpha=0.95, loc='upper right')

# Plot 2: Validation Loss vs Total Tokens
ax = axes[1]

for strategy_name in strategies:
    strategy_key = strategy_name.lower().replace(' ', '_')
    color = strategy_colors[strategy_key]
    
    strategy_data = [d for d in all_data if strategy_key in d['config_name']]
    strategy_data.sort(key=lambda x: x['lr'])
    
    # Calculate total tokens from training
    total_tokens = []
    for d in strategy_data:
        if 'train_history' in d and d['train_history']:
            max_tokens = max(h['tokens'] for h in d['train_history'])
            total_tokens.append(max_tokens / 1e6)  # Convert to millions
        else:
            # Estimate from batch size, seq len, and steps
            total_tokens.append(d['batch_size'] * d['seq_len'] * 50 / 1e6)
    
    val_losses = [d['val_loss'] for d in strategy_data]
    lr_labels = [d['lr'] for d in strategy_data]
    
    # Plot with markers
    ax.plot(total_tokens, val_losses, 'o-', color=color, linewidth=3, 
            markersize=12, label=strategy_name, alpha=0.8)
    
    # Add LR annotations
    for tt, vl, lr in zip(total_tokens, val_losses, lr_labels):
        ax.annotate(f'LR={lr}', (tt, vl), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=8, alpha=0.7)

ax.set_xlabel('Total Tokens Processed (Millions)', fontsize=14, fontweight='bold')
ax.set_ylabel('Final Validation Loss', fontsize=14, fontweight='bold')
ax.set_title('Validation Loss vs Total Tokens', fontsize=16, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, framealpha=0.95, loc='upper right')

plt.tight_layout()

# Save the plot
output_file = results_dir / 'validation_loss_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✅ Validation loss comparison plot saved to: {output_file}")
plt.close()

# Create individual plots
# Plot 1: Validation Loss vs Time (standalone)
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.suptitle('Final Validation Loss vs Training Time - All Configurations', 
             fontsize=18, fontweight='bold', y=0.98)

for strategy_name in strategies:
    strategy_key = strategy_name.lower().replace(' ', '_')
    color = strategy_colors[strategy_key]
    
    strategy_data = [d for d in all_data if strategy_key in d['config_name']]
    strategy_data.sort(key=lambda x: x['lr'])
    
    times = [d['time'] for d in strategy_data]
    val_losses = [d['val_loss'] for d in strategy_data]
    lr_labels = [d['lr'] for d in strategy_data]
    
    ax.plot(times, val_losses, 'o-', color=color, linewidth=3.5, 
            markersize=14, label=strategy_name, alpha=0.8)
    
    for t, vl, lr in zip(times, val_losses, lr_labels):
        ax.annotate(f'LR={lr}', (t, vl), textcoords="offset points", 
                   xytext=(0,12), ha='center', fontsize=9, alpha=0.7)

ax.set_xlabel('Training Time (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Final Validation Loss', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=13, framealpha=0.95, loc='upper right')

# Highlight best
best = min(all_data, key=lambda x: x['val_loss'])
ax.scatter(best['time'], best['val_loss'], s=500, facecolors='none', 
          edgecolors='red', linewidths=4, label='Best Config', zorder=5)

plt.tight_layout()
output_file1 = results_dir / 'val_loss_vs_time.png'
plt.savefig(output_file1, dpi=150, bbox_inches='tight')
print(f"✅ Validation loss vs time plot saved to: {output_file1}")
plt.close()

# Plot 2: Validation Loss vs Tokens (standalone)
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.suptitle('Final Validation Loss vs Total Tokens Processed - All Configurations', 
             fontsize=18, fontweight='bold', y=0.98)

for strategy_name in strategies:
    strategy_key = strategy_name.lower().replace(' ', '_')
    color = strategy_colors[strategy_key]
    
    strategy_data = [d for d in all_data if strategy_key in d['config_name']]
    strategy_data.sort(key=lambda x: x['lr'])
    
    total_tokens = []
    for d in strategy_data:
        if 'train_history' in d and d['train_history']:
            max_tokens = max(h['tokens'] for h in d['train_history'])
            total_tokens.append(max_tokens / 1e6)
        else:
            total_tokens.append(d['batch_size'] * d['seq_len'] * 50 / 1e6)
    
    val_losses = [d['val_loss'] for d in strategy_data]
    lr_labels = [d['lr'] for d in strategy_data]
    
    ax.plot(total_tokens, val_losses, 'o-', color=color, linewidth=3.5, 
            markersize=14, label=strategy_name, alpha=0.8)
    
    for tt, vl, lr in zip(total_tokens, val_losses, lr_labels):
        ax.annotate(f'LR={lr}', (tt, vl), textcoords="offset points", 
                   xytext=(0,12), ha='center', fontsize=9, alpha=0.7)

ax.set_xlabel('Total Tokens Processed (Millions)', fontsize=14, fontweight='bold')
ax.set_ylabel('Final Validation Loss', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=13, framealpha=0.95, loc='upper right')

# Highlight best
best_tokens = max(h['tokens'] for h in best['train_history']) / 1e6 if 'train_history' in best else best['batch_size'] * best['seq_len'] * 50 / 1e6
ax.scatter(best_tokens, best['val_loss'], s=500, facecolors='none', 
          edgecolors='red', linewidths=4, label='Best Config', zorder=5)

plt.tight_layout()
output_file2 = results_dir / 'val_loss_vs_tokens.png'
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"✅ Validation loss vs tokens plot saved to: {output_file2}")
plt.close()

print("\n📊 Summary:")
print(f"  - Combined: validation_loss_comparison.png")
print(f"  - Val Loss vs Time: val_loss_vs_time.png")
print(f"  - Val Loss vs Tokens: val_loss_vs_tokens.png")
print(f"\nNote: Validation was performed only at the end of training (50 steps).")
print(f"Best config: {best['config_name']} with val_loss={best['val_loss']:.4f}")

