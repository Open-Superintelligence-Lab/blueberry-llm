#!/usr/bin/env python3
"""
Plot ablation results
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

results = []
for config_name in configs:
    result_file = results_dir / f"{config_name}_result.json"
    if result_file.exists():
        with open(result_file) as f:
            results.append(json.load(f))

# Group by learning rate and strategy
lr_groups = {0.01: [], 0.02: [], 0.005: []}
strategies = ['Large Batch', 'Long Seq', 'Balanced']

for r in results:
    lr = r['lr']
    if 'large_batch' in r['config_name']:
        strategy_idx = 0
    elif 'long_seq' in r['config_name']:
        strategy_idx = 1
    else:
        strategy_idx = 2
    
    if lr in lr_groups:
        lr_groups[lr].append((strategy_idx, r))

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Experiment 5: MoE Ablation Study - Batch Size vs Sequence Length', fontsize=16, fontweight='bold')

# Plot 1: Validation Loss by Strategy and LR
ax = axes[0, 0]
x = np.arange(len(strategies))
width = 0.25

for i, (lr, lr_results) in enumerate(sorted(lr_groups.items())):
    sorted_results = sorted(lr_results, key=lambda x: x[0])
    losses = [r[1]['val_loss'] for r in sorted_results]
    ax.bar(x + i*width - width, losses, width, label=f'LR={lr}')

ax.set_xlabel('Strategy', fontweight='bold')
ax.set_ylabel('Validation Loss', fontweight='bold')
ax.set_title('Validation Loss by Strategy and Learning Rate')
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Validation Accuracy by Strategy and LR
ax = axes[0, 1]
for i, (lr, lr_results) in enumerate(sorted(lr_groups.items())):
    sorted_results = sorted(lr_results, key=lambda x: x[0])
    accs = [r[1]['val_acc'] * 100 for r in sorted_results]
    ax.bar(x + i*width - width, accs, width, label=f'LR={lr}')

ax.set_xlabel('Strategy', fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax.set_title('Validation Accuracy by Strategy and Learning Rate')
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Throughput by Strategy and LR
ax = axes[1, 0]
for i, (lr, lr_results) in enumerate(sorted(lr_groups.items())):
    sorted_results = sorted(lr_results, key=lambda x: x[0])
    throughputs = [r[1]['throughput'] / 1000 for r in sorted_results]  # Convert to k tokens/s
    ax.bar(x + i*width - width, throughputs, width, label=f'LR={lr}')

ax.set_xlabel('Strategy', fontweight='bold')
ax.set_ylabel('Throughput (k tokens/sec)', fontweight='bold')
ax.set_title('Throughput by Strategy and Learning Rate')
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Performance Summary - Val Loss vs Throughput
ax = axes[1, 1]
colors = {'large_batch': 'blue', 'long_seq': 'green', 'balanced': 'orange'}
markers = {0.005: 'o', 0.01: 's', 0.02: '^'}

for r in results:
    color_key = 'large_batch' if 'large_batch' in r['config_name'] else ('long_seq' if 'long_seq' in r['config_name'] else 'balanced')
    marker = markers.get(r['lr'], 'o')
    
    ax.scatter(r['throughput']/1000, r['val_loss'], 
              c=colors[color_key], marker=marker, s=150, alpha=0.7,
              label=f"{r['config_name']}")

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Large Batch'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Long Seq'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Balanced'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='LR=0.005'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='LR=0.01'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='LR=0.02'),
]
ax.legend(handles=legend_elements, loc='upper right')

ax.set_xlabel('Throughput (k tokens/sec)', fontweight='bold')
ax.set_ylabel('Validation Loss', fontweight='bold')
ax.set_title('Performance Trade-off: Validation Loss vs Throughput')
ax.grid(True, alpha=0.3)

# Highlight best config
best = min(results, key=lambda x: x['val_loss'])
ax.scatter(best['throughput']/1000, best['val_loss'], 
          s=300, facecolors='none', edgecolors='red', linewidths=3,
          label='Best Config')

plt.tight_layout()

# Save the plot
output_file = results_dir / 'experiment5_ablation_results.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✅ Plot saved to: {output_file}")

plt.close()

# Also print the best config info
print(f"\n🏆 Best Configuration: {best['config_name']}")
print(f"   Val Loss: {best['val_loss']:.4f}")
print(f"   Val Acc: {best['val_acc']*100:.2f}%")
print(f"   Throughput: {best['throughput']:.0f} tokens/sec")
print(f"   Batch: {best['batch_size']}, SeqLen: {best['seq_len']}, LR: {best['lr']}")
