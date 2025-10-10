#!/usr/bin/env python3
"""
PART 1: Plot validation loss for 100-step runs (LR comparison)
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_dir = Path("results/ablation_batch_seqlen")

# Part 1: 12 configurations (4 LRs × 3 strategies) - 100 steps
configs = [
    'large_batch_lr001', 'large_batch_lr002', 'large_batch_lr003', 'large_batch_lr004',
    'long_seq_lr001', 'long_seq_lr002', 'long_seq_lr003', 'long_seq_lr004',
    'balanced_lr001', 'balanced_lr002', 'balanced_lr003', 'balanced_lr004',
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

print(f"✅ Part 1: Loaded {len(all_data)} configurations")

# ============================================================================
# PLOT 1: Validation Loss vs Time (4 subplots by LR)
# ============================================================================
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle('PART 1: Validation Loss vs Training Time (100 Steps)', 
             fontsize=18, fontweight='bold', y=0.98)

learning_rates = [0.01, 0.02, 0.03, 0.04]

for idx, lr in enumerate(learning_rates):
    ax = axes[idx]
    
    # Filter configs for this learning rate
    lr_str = str(lr).replace('.', '')
    lr_configs = [c for c in configs if f'lr{lr_str}' in c or f'lr00{lr*1000:.0f}' in c]
    
    for config_name in lr_configs:
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
            elif 'long_seq' in config_name:
                strategy = 'Long Seq (6×4096)'
                color = strategy_colors['long_seq']
            else:
                strategy = 'Balanced (26×1024)'
                color = strategy_colors['balanced']
            
            ax.plot(times, val_losses, 'o-', color=color, linewidth=2.5, 
                   markersize=8, label=strategy, alpha=0.8)
    
    ax.set_xlabel('Training Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    ax.set_title(f'Learning Rate = {lr}', fontsize=15, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, framealpha=0.95, loc='upper right')

plt.tight_layout()

# Save plot
output_file1 = results_dir / 'part1_val_loss_vs_time.png'
plt.savefig(output_file1, dpi=150, bbox_inches='tight')
print(f"📊 Saved: {output_file1}")
plt.close()

# ============================================================================
# PLOT 2: Validation Loss vs Total Tokens (4 subplots by LR)
# ============================================================================
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle('PART 1: Validation Loss vs Total Tokens Processed (100 Steps)', 
             fontsize=18, fontweight='bold', y=0.98)

for idx, lr in enumerate(learning_rates):
    ax = axes[idx]
    
    # Filter configs for this learning rate
    lr_str = str(lr).replace('.', '')
    lr_configs = [c for c in configs if f'lr{lr_str}' in c or f'lr00{lr*1000:.0f}' in c]
    
    for config_name in lr_configs:
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
            elif 'long_seq' in config_name:
                strategy = 'Long Seq (6×4096)'
                color = strategy_colors['long_seq']
            else:
                strategy = 'Balanced (26×1024)'
                color = strategy_colors['balanced']
            
            ax.plot(tokens_millions, val_losses, 'o-', color=color, linewidth=2.5, 
                   markersize=8, label=strategy, alpha=0.8)
    
    ax.set_xlabel('Total Tokens Processed (Millions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    ax.set_title(f'Learning Rate = {lr}', fontsize=15, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, framealpha=0.95, loc='upper right')

plt.tight_layout()

# Save plot
output_file2 = results_dir / 'part1_val_loss_vs_tokens.png'
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"📊 Saved: {output_file2}")
plt.close()

# ============================================================================
# PLOT 3: Combined overview - all LRs and strategies
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('PART 1: Learning Rate & Strategy Comparison (100 Steps)', 
             fontsize=18, fontweight='bold', y=0.98)

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
            strategy = 'Large Batch'
            color = strategy_colors['large_batch']
        elif 'long_seq' in config_name:
            strategy = 'Long Seq'
            color = strategy_colors['long_seq']
        else:
            strategy = 'Balanced'
            color = strategy_colors['balanced']
        
        lr = data['lr']
        if lr == 0.01:
            linestyle = '--'
            marker = 's'
        elif lr == 0.02:
            linestyle = '-.'
            marker = '^'
        elif lr == 0.03:
            linestyle = ':'
            marker = 'D'
        else:  # 0.04
            linestyle = (0, (3, 1, 1, 1))
            marker = 'v'
        
        label = f"{strategy}, LR={lr}"
        ax.plot(times, val_losses, linestyle=linestyle, color=color, 
               linewidth=2, markersize=6, marker=marker, label=label, alpha=0.7)

ax.set_xlabel('Training Time (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
ax.set_title('Validation Loss vs Training Time', fontsize=15, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=9, framealpha=0.95, loc='upper right', ncol=2)

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
            strategy = 'Large Batch'
            color = strategy_colors['large_batch']
        elif 'long_seq' in config_name:
            strategy = 'Long Seq'
            color = strategy_colors['long_seq']
        else:
            strategy = 'Balanced'
            color = strategy_colors['balanced']
        
        lr = data['lr']
        if lr == 0.01:
            linestyle = '--'
            marker = 's'
        elif lr == 0.02:
            linestyle = '-.'
            marker = '^'
        elif lr == 0.03:
            linestyle = ':'
            marker = 'D'
        else:  # 0.04
            linestyle = (0, (3, 1, 1, 1))
            marker = 'v'
        
        label = f"{strategy}, LR={lr}"
        ax.plot(tokens_millions, val_losses, linestyle=linestyle, color=color, 
               linewidth=2, markersize=6, marker=marker, label=label, alpha=0.7)

ax.set_xlabel('Total Tokens Processed (Millions)', fontsize=14, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
ax.set_title('Validation Loss vs Total Tokens', fontsize=15, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=9, framealpha=0.95, loc='upper right', ncol=2)

plt.tight_layout()

# Save plot
output_file3 = results_dir / 'part1_combined.png'
plt.savefig(output_file3, dpi=150, bbox_inches='tight')
print(f"📊 Saved: {output_file3}")
plt.close()

print("\n" + "="*80)
print("✅ Part 1 plots generated successfully!")
print("="*80 + "\n")

