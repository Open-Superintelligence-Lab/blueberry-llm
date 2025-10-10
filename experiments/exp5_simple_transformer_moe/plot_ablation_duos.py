#!/usr/bin/env python3
"""Plot ablation study results grouped by learning rate"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_ablation_duos(results_path='results/ablation_batch_seqlen/results.json'):
    """Create per-LR comparison plots showing all strategies together"""
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Extract unique learning rates
    lrs = sorted(set([float(r['config_name'].split('lr')[1]) for r in results]))
    
    # Strategy colors (consistent across all plots)
    strategy_colors = {
        'large_batch': '#FF6B6B',
        'long_seq': '#4ECDC4',
        'balanced': '#45B7D1'
    }
    
    strategy_labels = {
        'large_batch': 'Large Batch (64×256)',
        'long_seq': 'Long Seq (8×1024)',
        'balanced': 'Balanced (24×512)'
    }
    
    # Create one plot per learning rate
    for lr in lrs:
        # Filter results for this learning rate
        lr_results = [r for r in results if f'lr{lr}' in r['config_name']]
        
        # Group by strategy
        large_batch = [r for r in lr_results if 'large_batch' in r['config_name']]
        long_seq = [r for r in lr_results if 'long_seq' in r['config_name']]
        balanced = [r for r in lr_results if 'balanced' in r['config_name']]
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Strategy Comparison at Learning Rate = {lr}', fontsize=16, fontweight='bold', y=1.02)
        
        # Subplot 1: Training Loss Curves
        ax1 = axes[0]
        for strategy_name, strategy_results, color, label in [
            ('large_batch', large_batch, strategy_colors['large_batch'], strategy_labels['large_batch']),
            ('long_seq', long_seq, strategy_colors['long_seq'], strategy_labels['long_seq']),
            ('balanced', balanced, strategy_colors['balanced'], strategy_labels['balanced'])
        ]:
            if strategy_results:
                for r in strategy_results:
                    if r['train_losses']:
                        losses = [t['loss'] for t in r['train_losses']]
                        steps = [t['step'] for t in r['train_losses']]
                        ax1.plot(steps, losses, color=color, alpha=0.6, linewidth=2, label=label)
                        break  # Only plot once per strategy
        
        ax1.set_xlabel('Training Step', fontsize=11)
        ax1.set_ylabel('Training Loss', fontsize=11)
        ax1.set_title('Training Loss Curves', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Validation Metrics
        ax2 = axes[1]
        strategies = []
        val_losses = []
        val_accs = []
        
        for strategy_name, strategy_results, color, label in [
            ('large_batch', large_batch, strategy_colors['large_batch'], strategy_labels['large_batch']),
            ('long_seq', long_seq, strategy_colors['long_seq'], strategy_labels['long_seq']),
            ('balanced', balanced, strategy_colors['balanced'], strategy_labels['balanced'])
        ]:
            if strategy_results:
                strategies.append(label)
                val_losses.append(strategy_results[0]['final_val_loss'])
                val_accs.append(strategy_results[0]['final_val_acc'])
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, val_losses, width, label='Val Loss', alpha=0.7, 
                        color=[strategy_colors['large_batch'], strategy_colors['long_seq'], strategy_colors['balanced']])
        
        # Create a twin axis for accuracy
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x + width/2, val_accs, width, label='Val Accuracy', alpha=0.7,
                             color=[c + '80' for c in [strategy_colors['large_batch'], strategy_colors['long_seq'], strategy_colors['balanced']]])
        
        ax2.set_xlabel('Strategy', fontsize=11)
        ax2.set_ylabel('Validation Loss', fontsize=11, color='black')
        ax2_twin.set_ylabel('Validation Accuracy', fontsize=11, color='gray')
        ax2.set_title('Validation Metrics', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.split('(')[0].strip() for s in strategies], fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        
        # Subplot 3: Throughput Comparison
        ax3 = axes[2]
        throughputs = []
        
        for strategy_name, strategy_results in [
            ('large_batch', large_batch),
            ('long_seq', long_seq),
            ('balanced', balanced)
        ]:
            if strategy_results:
                throughputs.append(strategy_results[0]['avg_tokens_per_sec'])
        
        bars = ax3.bar(strategies, throughputs, alpha=0.7,
                       color=[strategy_colors['large_batch'], strategy_colors['long_seq'], strategy_colors['balanced']])
        
        ax3.set_xlabel('Strategy', fontsize=11)
        ax3.set_ylabel('Tokens/sec', fontsize=11)
        ax3.set_title('Training Throughput', fontsize=12, fontweight='bold')
        ax3.set_xticklabels([s.split('(')[0].strip() for s in strategies], fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.ticklabel_format(style='plain', axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        output_dir = os.path.dirname(results_path)
        lr_str = str(lr).replace('.', '')
        output_path = os.path.join(output_dir, f'duo_lr{lr_str}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📈 Saved LR {lr} comparison to {output_path}")
        
        plt.close()
    
    print(f"\n✅ Created {len(lrs)} duo comparison plots (one per learning rate)")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 PER-LR SUMMARY")
    print("="*80)
    
    for lr in lrs:
        lr_results = [r for r in results if f'lr{lr}' in r['config_name']]
        
        print(f"\n🎯 Learning Rate = {lr}")
        print("-" * 80)
        
        for strategy_key, strategy_label in [
            ('large_batch', 'Large Batch (64×256)'),
            ('long_seq', 'Long Seq (8×1024)'),
            ('balanced', 'Balanced (24×512)')
        ]:
            strategy_results = [r for r in lr_results if strategy_key in r['config_name']]
            if strategy_results:
                r = strategy_results[0]
                print(f"{strategy_label:30} | Loss: {r['final_val_loss']:.4f} | "
                      f"Acc: {r['final_val_acc']:.4f} | "
                      f"Throughput: {r['avg_tokens_per_sec']:,.0f} tok/s")
        
        # Determine best for this LR
        best = min(lr_results, key=lambda x: x['final_val_loss'])
        print(f"\n   ✨ Best: {best['config_name']} (Loss: {best['final_val_loss']:.4f})")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        plot_ablation_duos(sys.argv[1])
    else:
        plot_ablation_duos()

