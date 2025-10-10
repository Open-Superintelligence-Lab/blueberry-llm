#!/usr/bin/env python3
"""
Plot training curves: Loss vs Time and Loss vs Tokens
Fair comparison across different batch/seqlen strategies
"""
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Load all result files
results_dir = 'results/ablation_batch_seqlen'
result_files = glob.glob(f'{results_dir}/*_result.json')

if not result_files:
    print("❌ No result files found! Run some experiments first:")
    print("   python run_ablation.py large_batch")
    print("   python run_ablation.py long_seq")
    print("   python run_ablation.py balanced")
    exit(1)

results = []
for f in result_files:
    with open(f) as fp:
        data = json.load(fp)
        if 'train_history' in data and data['train_history']:
            results.append(data)

if not results:
    print("❌ No training history found in results!")
    print("   Re-run experiments to generate training curves.")
    exit(1)

print(f"📊 Found {len(results)} experiments with training history")

# Color scheme
colors = {
    'large_batch': '#2ecc71',
    'long_seq': '#e74c3c', 
    'balanced': '#3498db',
    'custom': '#9b59b6',
    'quick': '#f39c12',
    'max_batch': '#1abc9c',
    'max_seq': '#e67e22'
}

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('MoE Training Curves - Fair Comparison', fontsize=16, fontweight='bold')

# Plot 1: Loss vs Time
for result in results:
    name = result['config_name']
    history = result['train_history']
    times = [h['time'] for h in history]
    losses = [h['loss'] for h in history]
    
    color = colors.get(name, '#34495e')
    label = f"{name} (b={result['batch_size']}, s={result['seq_len']})"
    
    ax1.plot(times, losses, '-', label=label, color=color, linewidth=2, alpha=0.8)

ax1.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
ax1.set_title('Loss vs Wall-Clock Time', fontweight='bold', fontsize=12)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Loss vs Tokens
for result in results:
    name = result['config_name']
    history = result['train_history']
    tokens = [h['tokens'] / 1e6 for h in history]  # Millions of tokens
    losses = [h['loss'] for h in history]
    
    color = colors.get(name, '#34495e')
    label = f"{name}"
    
    ax2.plot(tokens, losses, '-', label=label, color=color, linewidth=2, alpha=0.8)

ax2.set_xlabel('Tokens Processed (Millions)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
ax2.set_title('Loss vs Tokens (Sample Efficiency)', fontweight='bold', fontsize=12)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: Throughput Comparison (Tokens/sec over time)
for result in results:
    name = result['config_name']
    history = result['train_history']
    times = [h['time'] for h in history]
    
    # Calculate instantaneous throughput
    throughputs = []
    for i in range(1, len(history)):
        dt = history[i]['time'] - history[i-1]['time']
        dtokens = history[i]['tokens'] - history[i-1]['tokens']
        throughput = dtokens / dt if dt > 0 else 0
        throughputs.append(throughput / 1000)  # K tokens/sec
    
    if throughputs:
        color = colors.get(name, '#34495e')
        ax3.plot(times[1:], throughputs, '-', label=name, color=color, linewidth=2, alpha=0.8)

ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Throughput (K tokens/sec)', fontsize=11, fontweight='bold')
ax3.set_title('Training Throughput Over Time', fontweight='bold', fontsize=12)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Final Results Bar Chart
config_names = [r['config_name'] for r in results]
val_losses = [r['val_loss'] for r in results]
throughputs = [r['throughput'] / 1000 for r in results]

x_pos = np.arange(len(results))
width = 0.35

# Dual y-axis
ax4_twin = ax4.twinx()

bars1 = ax4.bar(x_pos - width/2, val_losses, width, 
                label='Val Loss', color='#3498db', alpha=0.8)
bars2 = ax4_twin.bar(x_pos + width/2, throughputs, width,
                     label='Throughput', color='#2ecc71', alpha=0.8)

ax4.set_xlabel('Configuration', fontsize=11, fontweight='bold')
ax4.set_ylabel('Validation Loss', fontsize=11, fontweight='bold', color='#3498db')
ax4_twin.set_ylabel('Throughput (K tok/s)', fontsize=11, fontweight='bold', color='#2ecc71')
ax4.set_title('Final Performance Comparison', fontweight='bold', fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(config_names, rotation=30, ha='right', fontsize=9)
ax4.tick_params(axis='y', labelcolor='#3498db')
ax4_twin.tick_params(axis='y', labelcolor='#2ecc71')
ax4.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars1, val_losses)):
    ax4.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

for i, (bar, val) in enumerate(zip(bars2, throughputs)):
    ax4_twin.text(bar.get_x() + bar.get_width()/2, val, f'{val:.0f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

# Legend for dual axis
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.tight_layout()

# Save
output_file = f'{results_dir}/training_curves.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f'✅ Saved to {output_file}')

# Print summary
print(f"\n{'='*80}")
print("📈 TRAINING CURVE SUMMARY")
print(f"{'='*80}")
print(f"{'Config':<15} {'Val Loss':<12} {'Throughput':<15} {'Time (s)':<12} {'Tokens (M)':<12}")
print('-'*80)

for r in sorted(results, key=lambda x: x['val_loss']):
    total_tokens = r['train_history'][-1]['tokens'] / 1e6 if r['train_history'] else 0
    print(f"{r['config_name']:<15} {r['val_loss']:<12.4f} {r['throughput']/1000:<15.1f} "
          f"{r['time']:<12.1f} {total_tokens:<12.1f}")

print(f"{'='*80}\n")

plt.show()

