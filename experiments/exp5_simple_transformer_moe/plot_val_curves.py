#!/usr/bin/env python3
"""
Plot validation loss curves throughout training for all configurations
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_dir = Path("results/ablation_batch_seqlen")

# All 9 configurations
configs = [
    'large_batch_lr0005', 'large_batch_lr001', 'large_batch_lr002',
    'long_seq_lr0005', 'long_seq_lr001', 'long_seq_lr002',
    'balanced_lr0005', 'balanced_lr001', 'balanced_lr002',
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

print(f"✅ Loaded {len(all_data)} configurations with validation curves")

# Create figure with 3 subplots (one per learning rate)
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Validation Loss Curves Throughout Training - All Configurations', 
             fontsize=18, fontweight='bold', y=0.98)

learning_rates = [0.005, 0.01, 0.02]

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
            steps = [v['step'] for v in data['val_history']]
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
            
            ax.plot(steps, val_losses, 'o-', color=color, linewidth=2.5, 
                   markersize=8, label=strategy, alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    ax.set_title(f'Learning Rate = {lr}', fontsize=15, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, framealpha=0.95, loc='upper right')

plt.tight_layout()

# Save plot
output_file = results_dir / 'validation_loss_curves_all.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"📊 Saved validation curves plot: {output_file}")
plt.close()

# Create individual strategy comparison plots
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Validation Loss Curves - Strategy Comparison Across Learning Rates', 
             fontsize=18, fontweight='bold', y=0.98)

strategies = ['large_batch', 'long_seq', 'balanced']
strategy_names = {'large_batch': 'Large Batch (104×256)', 
                  'long_seq': 'Long Sequence (6×4096)', 
                  'balanced': 'Balanced (26×1024)'}

for idx, strategy in enumerate(strategies):
    ax = axes[idx]
    
    # Get all configs for this strategy
    strategy_configs = [c for c in configs if strategy in c]
    
    for config_name in strategy_configs:
        if config_name not in all_data:
            continue
            
        data = all_data[config_name]
        lr = data['lr']
        
        # Extract validation history
        if 'val_history' in data and data['val_history']:
            steps = [v['step'] for v in data['val_history']]
            val_losses = [v['val_loss'] for v in data['val_history']]
            
            # Different line styles for different LRs
            if lr == 0.005:
                linestyle = '-'
                marker = 'o'
            elif lr == 0.01:
                linestyle = '--'
                marker = 's'
            else:
                linestyle = '-.'
                marker = '^'
            
            ax.plot(steps, val_losses, linestyle=linestyle, marker=marker, 
                   linewidth=2.5, markersize=8, label=f'LR={lr}', alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    ax.set_title(strategy_names[strategy], fontsize=15, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, framealpha=0.95, loc='upper right')

plt.tight_layout()

# Save plot
output_file2 = results_dir / 'validation_loss_curves_by_strategy.png'
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"📊 Saved strategy comparison plot: {output_file2}")
plt.close()

# Create single combined plot with all curves
fig, ax = plt.subplots(figsize=(16, 10))
fig.suptitle('All Validation Loss Curves - Complete Comparison', 
             fontsize=18, fontweight='bold', y=0.96)

for config_name in configs:
    if config_name not in all_data:
        continue
        
    data = all_data[config_name]
    
    # Extract validation history
    if 'val_history' in data and data['val_history']:
        steps = [v['step'] for v in data['val_history']]
        val_losses = [v['val_loss'] for v in data['val_history']]
        
        # Determine strategy, color, and line style
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
        if lr == 0.005:
            linestyle = '-'
        elif lr == 0.01:
            linestyle = '--'
        else:
            linestyle = '-.'
        
        label = f"{strategy}, LR={lr}"
        ax.plot(steps, val_losses, linestyle=linestyle, color=color, 
               linewidth=2, markersize=6, marker='o', label=label, alpha=0.7)

ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10, framealpha=0.95, loc='upper right', ncol=2)

plt.tight_layout()

# Save plot
output_file3 = results_dir / 'validation_loss_curves_combined.png'
plt.savefig(output_file3, dpi=150, bbox_inches='tight')
print(f"📊 Saved combined curves plot: {output_file3}")
plt.close()

# Print summary
print("\n" + "="*80)
print("📈 VALIDATION CURVE SUMMARY")
print("="*80)

for config_name in configs:
    if config_name not in all_data:
        continue
    
    data = all_data[config_name]
    if 'val_history' in data and data['val_history']:
        initial_val_loss = data['val_history'][0]['val_loss']
        final_val_loss = data['val_history'][-1]['val_loss']
        improvement = initial_val_loss - final_val_loss
        
        print(f"{config_name:25} | Initial: {initial_val_loss:.4f} | "
              f"Final: {final_val_loss:.4f} | Improvement: {improvement:.4f}")

# Find best config
best_config = min(all_data.items(), 
                 key=lambda x: x[1]['val_history'][-1]['val_loss'] if 'val_history' in x[1] and x[1]['val_history'] else float('inf'))

print(f"\n🏆 Best Final Validation Loss: {best_config[0]}")
print(f"   Final Val Loss: {best_config[1]['val_history'][-1]['val_loss']:.4f}")
print(f"   Learning Rate: {best_config[1]['lr']}")
print(f"   Batch: {best_config[1]['batch_size']}, SeqLen: {best_config[1]['seq_len']}")

print(f"\n{'='*80}")
print("✅ All validation curve plots generated successfully!")
print(f"{'='*80}\n")

