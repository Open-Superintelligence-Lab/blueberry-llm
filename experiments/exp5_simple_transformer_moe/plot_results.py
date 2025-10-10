#!/usr/bin/env python3
"""Quick plot of all ablation results"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results/ablation_batch_seqlen/results.json') as f:
    data = json.load(f)

# Parse results
strategies = {'large_batch': [], 'long_seq': [], 'balanced': []}
for r in data['results']:
    name = r['config_name']
    for strategy in strategies:
        if strategy in name:
            strategies[strategy].append(r)

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MoE Ablation: Batch Size vs Sequence Length', fontsize=16, fontweight='bold')

colors = {'large_batch': '#2ecc71', 'long_seq': '#e74c3c', 'balanced': '#3498db'}
labels = {'large_batch': 'Large Batch (64×256)', 'long_seq': 'Long Seq (8×1024)', 'balanced': 'Balanced (24×512)'}

# 1. Validation Loss by Strategy
for strategy, results in strategies.items():
    lrs = [r['config_name'].split('lr')[1] for r in results]
    losses = [r['final_val_loss'] for r in results]
    ax1.plot(lrs, losses, 'o-', label=labels[strategy], color=colors[strategy], linewidth=2, markersize=8)
ax1.set_xlabel('Learning Rate', fontsize=11)
ax1.set_ylabel('Validation Loss', fontsize=11)
ax1.set_title('Validation Loss vs Learning Rate', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Validation Accuracy by Strategy
for strategy, results in strategies.items():
    lrs = [r['config_name'].split('lr')[1] for r in results]
    accs = [r['final_val_acc'] * 100 for r in results]
    ax2.plot(lrs, accs, 'o-', label=labels[strategy], color=colors[strategy], linewidth=2, markersize=8)
ax2.set_xlabel('Learning Rate', fontsize=11)
ax2.set_ylabel('Validation Accuracy (%)', fontsize=11)
ax2.set_title('Validation Accuracy vs Learning Rate', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Throughput Comparison
for strategy, results in strategies.items():
    lrs = [r['config_name'].split('lr')[1] for r in results]
    throughputs = [r['avg_tokens_per_sec'] / 1000 for r in results]
    ax3.plot(lrs, throughputs, 'o-', label=labels[strategy], color=colors[strategy], linewidth=2, markersize=8)
ax3.set_xlabel('Learning Rate', fontsize=11)
ax3.set_ylabel('Throughput (K tokens/s)', fontsize=11)
ax3.set_title('Training Throughput', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Best configs bar chart
best_per_strategy = {s: min(r, key=lambda x: x['final_val_loss']) for s, r in strategies.items()}
x_pos = np.arange(len(strategies))
losses = [best_per_strategy[s]['final_val_loss'] for s in strategies.keys()]
ax4.bar(x_pos, losses, color=[colors[s] for s in strategies.keys()], alpha=0.8)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([labels[s] for s in strategies.keys()], rotation=15, ha='right')
ax4.set_ylabel('Best Validation Loss', fontsize=11)
ax4.set_title('Best Loss per Strategy', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(losses):
    ax4.text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/ablation_batch_seqlen/ablation_results.png', dpi=150, bbox_inches='tight')
print('✅ Saved to results/ablation_batch_seqlen/ablation_results.png')
plt.show()

