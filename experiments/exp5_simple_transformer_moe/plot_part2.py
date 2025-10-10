#!/usr/bin/env python3
"""
PART 2: Plot validation loss for 1000-step runs (Extended training with LR=0.03)
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_dir = Path("results/ablation_batch_seqlen")

# Part 2: 3 configurations (LR=0.03 × 3 strategies) - 1000 steps
configs = [
    'large_batch_lr003_1000steps',
    'long_seq_lr003_1000steps',
    'balanced_lr003_1000steps',
]

# Strategy colors
strategy_colors = {
    'large_batch': '#1f77b4',  # blue
    'long_seq': '#2ca02c',      # green
    'balanced': '#ff7f0e',      # orange
}

# Load all results
all_data = {}
for config_name in configs:
    result_file = results_dir / f"{config_name}_result.json"
    if result_file.exists():
        with open(result_file) as f:
            all_data[config_name] = json.load(f)

print(f"✅ Part 2: Loaded {len(all_data)} configurations")

# ============================================================================
# PLOT 1: Validation Loss vs Time - Single plot with all 3 strategies
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.suptitle('PART 2: Extended Training - Validation Loss vs Training Time (1000 Steps, LR=0.03)', 
             fontsize=18, fontweight='bold', y=0.98)

for config_name in configs:
    if config_name not in all_data:
        continue
        
    data = all_data[config_name]
    
    # Extract validation history
    if 'val_history' in data and data['val_history']:
        times = [v['time'] for v in data['val_history']]
        val_losses = [v['val_loss'] for v in data['val_history']]
        
        # Determine strategy and color
        if 'large_batch' in config_name:
            strategy = 'Large Batch (104×256)'
            color = strategy_colors['large_batch']
            marker = 's'
        elif 'long_seq' in config_name:
            strategy = 'Long Seq (6×4096)'
            color = strategy_colors['long_seq']
            marker = '^'
        else:
            strategy = 'Balanced (26×1024)'
            color = strategy_colors['balanced']
            marker = 'D'
        
        ax.plot(times, val_losses, 'o-', color=color, linewidth=3, 
               markersize=7, marker=marker, label=strategy, alpha=0.8, markevery=2)

ax.set_xlabel('Training Time (seconds)', fontsize=15, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=15, fontweight='bold')
ax.set_title('Learning Rate = 0.03, 1000 Steps', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=13, framealpha=0.95, loc='upper right')

# Add text annotation showing final values
for config_name in configs:
    if config_name in all_data:
        data = all_data[config_name]
        if 'val_history' in data and data['val_history']:
            final_loss = data['val_history'][-1]['val_loss']
            final_acc = data['val_history'][-1]['val_acc']
            if 'large_batch' in config_name:
                strategy_short = 'LB'
            elif 'long_seq' in config_name:
                strategy_short = 'LS'
            else:
                strategy_short = 'Bal'

plt.tight_layout()

# Save plot
output_file1 = results_dir / 'part2_val_loss_vs_time.png'
plt.savefig(output_file1, dpi=150, bbox_inches='tight')
print(f"📊 Saved: {output_file1}")
plt.close()

# ============================================================================
# PLOT 2: Validation Loss vs Total Tokens
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.suptitle('PART 2: Extended Training - Validation Loss vs Total Tokens Processed (1000 Steps, LR=0.03)', 
             fontsize=18, fontweight='bold', y=0.98)

for config_name in configs:
    if config_name not in all_data:
        continue
        
    data = all_data[config_name]
    
    # Extract validation history
    if 'val_history' in data and data['val_history']:
        tokens_millions = [v['tokens'] / 1e6 for v in data['val_history']]
        val_losses = [v['val_loss'] for v in data['val_history']]
        
        # Determine strategy and color
        if 'large_batch' in config_name:
            strategy = 'Large Batch (104×256)'
            color = strategy_colors['large_batch']
            marker = 's'
        elif 'long_seq' in config_name:
            strategy = 'Long Seq (6×4096)'
            color = strategy_colors['long_seq']
            marker = '^'
        else:
            strategy = 'Balanced (26×1024)'
            color = strategy_colors['balanced']
            marker = 'D'
        
        ax.plot(tokens_millions, val_losses, 'o-', color=color, linewidth=3, 
               markersize=7, marker=marker, label=strategy, alpha=0.8, markevery=2)

ax.set_xlabel('Total Tokens Processed (Millions)', fontsize=15, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=15, fontweight='bold')
ax.set_title('Learning Rate = 0.03, 1000 Steps', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=13, framealpha=0.95, loc='upper right')

plt.tight_layout()

# Save plot
output_file2 = results_dir / 'part2_val_loss_vs_tokens.png'
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"📊 Saved: {output_file2}")
plt.close()

# ============================================================================
# PLOT 3: Combined Time & Tokens side-by-side
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(24, 8))
fig.suptitle('PART 2: Extended Training - Strategy Comparison (1000 Steps, LR=0.03)', 
             fontsize=20, fontweight='bold', y=0.98)

# Left plot: vs Time
ax = axes[0]
for config_name in configs:
    if config_name not in all_data:
        continue
        
    data = all_data[config_name]
    
    if 'val_history' in data and data['val_history']:
        times = [v['time'] for v in data['val_history']]
        val_losses = [v['val_loss'] for v in data['val_history']]
        
        # Determine strategy and color
        if 'large_batch' in config_name:
            strategy = 'Large Batch (104×256)'
            color = strategy_colors['large_batch']
            marker = 's'
        elif 'long_seq' in config_name:
            strategy = 'Long Seq (6×4096)'
            color = strategy_colors['long_seq']
            marker = '^'
        else:
            strategy = 'Balanced (26×1024)'
            color = strategy_colors['balanced']
            marker = 'D'
        
        ax.plot(times, val_losses, 'o-', color=color, linewidth=3, 
               markersize=7, marker=marker, label=strategy, alpha=0.8, markevery=2)

ax.set_xlabel('Training Time (seconds)', fontsize=15, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=15, fontweight='bold')
ax.set_title('Validation Loss vs Training Time', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=13, framealpha=0.95, loc='upper right')

# Right plot: vs Tokens
ax = axes[1]
for config_name in configs:
    if config_name not in all_data:
        continue
        
    data = all_data[config_name]
    
    if 'val_history' in data and data['val_history']:
        tokens_millions = [v['tokens'] / 1e6 for v in data['val_history']]
        val_losses = [v['val_loss'] for v in data['val_history']]
        
        # Determine strategy and color
        if 'large_batch' in config_name:
            strategy = 'Large Batch (104×256)'
            color = strategy_colors['large_batch']
            marker = 's'
        elif 'long_seq' in config_name:
            strategy = 'Long Seq (6×4096)'
            color = strategy_colors['long_seq']
            marker = '^'
        else:
            strategy = 'Balanced (26×1024)'
            color = strategy_colors['balanced']
            marker = 'D'
        
        ax.plot(tokens_millions, val_losses, 'o-', color=color, linewidth=3, 
               markersize=7, marker=marker, label=strategy, alpha=0.8, markevery=2)

ax.set_xlabel('Total Tokens Processed (Millions)', fontsize=15, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=15, fontweight='bold')
ax.set_title('Validation Loss vs Total Tokens', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=13, framealpha=0.95, loc='upper right')

plt.tight_layout()

# Save plot
output_file3 = results_dir / 'part2_combined.png'
plt.savefig(output_file3, dpi=150, bbox_inches='tight')
print(f"📊 Saved: {output_file3}")
plt.close()

# ============================================================================
# PLOT 4: Validation Accuracy over time (bonus!)
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.suptitle('PART 2: Extended Training - Validation Accuracy vs Training Time (1000 Steps, LR=0.03)', 
             fontsize=18, fontweight='bold', y=0.98)

for config_name in configs:
    if config_name not in all_data:
        continue
        
    data = all_data[config_name]
    
    # Extract validation history
    if 'val_history' in data and data['val_history']:
        times = [v['time'] for v in data['val_history']]
        val_accs = [v['val_acc'] * 100 for v in data['val_history']]  # Convert to percentage
        
        # Determine strategy and color
        if 'large_batch' in config_name:
            strategy = 'Large Batch (104×256)'
            color = strategy_colors['large_batch']
            marker = 's'
        elif 'long_seq' in config_name:
            strategy = 'Long Seq (6×4096)'
            color = strategy_colors['long_seq']
            marker = '^'
        else:
            strategy = 'Balanced (26×1024)'
            color = strategy_colors['balanced']
            marker = 'D'
        
        ax.plot(times, val_accs, 'o-', color=color, linewidth=3, 
               markersize=7, marker=marker, label=strategy, alpha=0.8, markevery=2)

ax.set_xlabel('Training Time (seconds)', fontsize=15, fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontsize=15, fontweight='bold')
ax.set_title('Learning Rate = 0.03, 1000 Steps', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=13, framealpha=0.95, loc='lower right')
ax.set_ylim([0, 105])  # Set y-axis from 0 to 100%

plt.tight_layout()

# Save plot
output_file4 = results_dir / 'part2_val_accuracy_vs_time.png'
plt.savefig(output_file4, dpi=150, bbox_inches='tight')
print(f"📊 Saved: {output_file4}")
plt.close()

# ============================================================================
# PRINT SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 PART 2 SUMMARY - Extended Training (1000 Steps, LR=0.03)")
print("="*80)

for config_name in sorted(configs):
    if config_name in all_data:
        data = all_data[config_name]
        print(f"\n{config_name}:")
        print(f"  Final Val Loss: {data['val_loss']:.4f}")
        print(f"  Final Val Acc: {data['val_acc']*100:.2f}%")
        print(f"  Training Time: {data['time']/60:.2f} minutes")
        print(f"  Throughput: {data['throughput']:,.0f} tokens/sec")
        
        if 'val_history' in data and data['val_history']:
            initial_loss = data['val_history'][0]['val_loss']
            final_loss = data['val_history'][-1]['val_loss']
            improvement = (initial_loss - final_loss) / initial_loss * 100
            print(f"  Improvement: {initial_loss:.3f} → {final_loss:.3f} ({improvement:.1f}%)")

print("\n" + "="*80)
print("✅ Part 2 plots generated successfully!")
print("="*80 + "\n")

