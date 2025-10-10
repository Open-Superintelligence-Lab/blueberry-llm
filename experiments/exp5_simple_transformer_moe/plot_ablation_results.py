#!/usr/bin/env python3
"""Plot ablation study results"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_ablation_results(results_path='results/ablation_batch_seqlen/results.json'):
    """Create comprehensive visualization of ablation results"""
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Group by strategy
    large_batch = [r for r in results if 'large_batch' in r['config_name']]
    long_seq = [r for r in results if 'long_seq' in r['config_name']]
    balanced = [r for r in results if 'balanced' in r['config_name']]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Final Val Loss Comparison
    ax1 = plt.subplot(2, 3, 1)
    strategies = ['Large Batch\n(64×256)', 'Long Seq\n(8×1024)', 'Balanced\n(24×512)']
    losses = [
        [r['final_val_loss'] for r in large_batch],
        [r['final_val_loss'] for r in long_seq],
        [r['final_val_loss'] for r in balanced]
    ]
    
    positions = np.arange(len(strategies))
    bp1 = ax1.boxplot(losses, positions=positions, widths=0.6, patch_artist=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(strategies)
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Final Validation Loss by Strategy', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Final Val Accuracy Comparison
    ax2 = plt.subplot(2, 3, 2)
    accs = [
        [r['final_val_acc'] for r in large_batch],
        [r['final_val_acc'] for r in long_seq],
        [r['final_val_acc'] for r in balanced]
    ]
    
    bp2 = ax2.boxplot(accs, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(strategies)
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Final Validation Accuracy by Strategy', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Throughput Comparison
    ax3 = plt.subplot(2, 3, 3)
    throughputs = [
        [r['avg_tokens_per_sec'] for r in large_batch],
        [r['avg_tokens_per_sec'] for r in long_seq],
        [r['avg_tokens_per_sec'] for r in balanced]
    ]
    
    bp3 = ax3.boxplot(throughputs, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_xticks(positions)
    ax3.set_xticklabels(strategies)
    ax3.set_ylabel('Tokens/sec')
    ax3.set_title('Training Throughput by Strategy', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='plain', axis='y')
    
    # 4. Training Loss Curves
    ax4 = plt.subplot(2, 3, 4)
    
    for result in large_batch:
        losses_over_time = [t['loss'] for t in result['train_losses']]
        steps = [t['step'] for t in result['train_losses']]
        ax4.plot(steps, losses_over_time, color='#FF6B6B', alpha=0.3, linewidth=1)
    
    for result in long_seq:
        losses_over_time = [t['loss'] for t in result['train_losses']]
        steps = [t['step'] for t in result['train_losses']]
        ax4.plot(steps, losses_over_time, color='#4ECDC4', alpha=0.3, linewidth=1)
    
    for result in balanced:
        losses_over_time = [t['loss'] for t in result['train_losses']]
        steps = [t['step'] for t in result['train_losses']]
        ax4.plot(steps, losses_over_time, color='#45B7D1', alpha=0.3, linewidth=1)
    
    # Add average lines
    if large_batch:
        avg_large = np.mean([[t['loss'] for t in r['train_losses']] for r in large_batch], axis=0)
        steps = [t['step'] for t in large_batch[0]['train_losses']]
        ax4.plot(steps, avg_large, color='#FF6B6B', linewidth=2, label='Large Batch (avg)')
    
    if long_seq:
        avg_long = np.mean([[t['loss'] for t in r['train_losses']] for r in long_seq], axis=0)
        steps = [t['step'] for t in long_seq[0]['train_losses']]
        ax4.plot(steps, avg_long, color='#4ECDC4', linewidth=2, label='Long Seq (avg)')
    
    if balanced:
        avg_bal = np.mean([[t['loss'] for t in r['train_losses']] for r in balanced], axis=0)
        steps = [t['step'] for t in balanced[0]['train_losses']]
        ax4.plot(steps, avg_bal, color='#45B7D1', linewidth=2, label='Balanced (avg)')
    
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Training Loss')
    ax4.set_title('Training Loss Curves', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Learning Rate Sensitivity
    ax5 = plt.subplot(2, 3, 5)
    
    lrs = sorted(set([float(r['config_name'].split('lr')[1]) for r in results]))
    
    for strategy_name, strategy_results, color in [
        ('Large Batch', large_batch, '#FF6B6B'),
        ('Long Seq', long_seq, '#4ECDC4'),
        ('Balanced', balanced, '#45B7D1')
    ]:
        lr_losses = []
        for lr in lrs:
            lr_results = [r for r in strategy_results if f'lr{lr}' in r['config_name']]
            if lr_results:
                lr_losses.append(lr_results[0]['final_val_loss'])
        
        ax5.plot(lrs, lr_losses, 'o-', color=color, linewidth=2, markersize=8, label=strategy_name)
    
    ax5.set_xlabel('Learning Rate')
    ax5.set_ylabel('Final Validation Loss')
    ax5.set_title('Learning Rate Sensitivity', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    
    # 6. Efficiency (Loss vs Throughput)
    ax6 = plt.subplot(2, 3, 6)
    
    for strategy_name, strategy_results, color in [
        ('Large Batch', large_batch, '#FF6B6B'),
        ('Long Seq', long_seq, '#4ECDC4'),
        ('Balanced', balanced, '#45B7D1')
    ]:
        losses = [r['final_val_loss'] for r in strategy_results]
        throughputs = [r['avg_tokens_per_sec'] for r in strategy_results]
        ax6.scatter(throughputs, losses, s=150, alpha=0.7, color=color, label=strategy_name)
    
    ax6.set_xlabel('Throughput (tokens/sec)')
    ax6.set_ylabel('Final Validation Loss')
    ax6.set_title('Efficiency: Loss vs Throughput', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.invert_yaxis()  # Lower loss is better
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(results_path)
    output_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Visualization saved to {output_path}")
    
    plt.close()
    
    # Create summary statistics
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    
    for strategy_name, strategy_results in [
        ('Large Batch (64×256)', large_batch),
        ('Long Sequence (8×1024)', long_seq),
        ('Balanced (24×512)', balanced)
    ]:
        if strategy_results:
            losses = [r['final_val_loss'] for r in strategy_results]
            accs = [r['final_val_acc'] for r in strategy_results]
            throughputs = [r['avg_tokens_per_sec'] for r in strategy_results]
            
            print(f"\n{strategy_name}:")
            print(f"   Val Loss:    {np.mean(losses):.4f} ± {np.std(losses):.4f}")
            print(f"   Val Acc:     {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(f"   Throughput:  {np.mean(throughputs):,.0f} ± {np.std(throughputs):,.0f} tok/s")
    
    # Statistical winner
    print(f"\n{'='*80}")
    print("🏆 STATISTICAL WINNER")
    print("="*80)
    
    all_strategy_results = [
        ('Large Batch (64×256)', large_batch),
        ('Long Sequence (8×1024)', long_seq),
        ('Balanced (24×512)', balanced)
    ]
    
    best_loss = min(all_strategy_results, key=lambda x: np.mean([r['final_val_loss'] for r in x[1]]))
    best_throughput = max(all_strategy_results, key=lambda x: np.mean([r['avg_tokens_per_sec'] for r in x[1]]))
    
    print(f"\n✅ Best Loss: {best_loss[0]}")
    print(f"   Avg Loss: {np.mean([r['final_val_loss'] for r in best_loss[1]]):.4f}")
    
    print(f"\n⚡ Best Throughput: {best_throughput[0]}")
    print(f"   Avg Throughput: {np.mean([r['avg_tokens_per_sec'] for r in best_throughput[1]]):,.0f} tok/s")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        plot_ablation_results(sys.argv[1])
    else:
        plot_ablation_results()

