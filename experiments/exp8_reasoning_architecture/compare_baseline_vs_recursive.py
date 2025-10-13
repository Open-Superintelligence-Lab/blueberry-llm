"""
Comparison Script for Baseline vs Recursive Reasoning Models
Loads results from both models and generates side-by-side comparisons
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import numpy as np


def load_results(results_dir, model_type):
    """Load training results for a specific model type"""
    results_file = results_dir / f'training_results_{model_type}.json'
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_comparison(baseline_results, recursive_results, save_path):
    """Generate comprehensive comparison plots"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract histories
    baseline_train = baseline_results['results'].get('final_train_loss')
    recursive_train = recursive_results['results'].get('final_train_loss')
    
    baseline_val = baseline_results['results'].get('final_val_metrics', {})
    recursive_val = recursive_results['results'].get('final_val_metrics', {})
    
    # 1. Validation Loss Comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Baseline', 'Recursive']
    val_losses = [
        baseline_results['results']['best_val_loss'],
        recursive_results['results']['best_val_loss']
    ]
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(models, val_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Validation Loss', fontweight='bold', fontsize=12)
    ax1.set_title('Best Validation Loss', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, val_losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Training Time Comparison (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    training_times = [
        baseline_results['results']['total_time'] / 60,  # Convert to minutes
        recursive_results['results']['total_time'] / 60
    ]
    bars = ax2.bar(models, training_times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Training Time (minutes)', fontweight='bold', fontsize=12)
    ax2.set_title('Training Time', fontweight='bold', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, training_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}m',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Accuracy Comparison (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    accuracies = [
        baseline_val.get('accuracy', 0) * 100,
        recursive_val.get('accuracy', 0) * 100
    ]
    bars = ax3.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax3.set_title('Final Validation Accuracy', fontweight='bold', fontsize=14)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # 4. Perplexity Comparison (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    perplexities = [
        baseline_val.get('perplexity', 0),
        recursive_val.get('perplexity', 0)
    ]
    bars = ax4.bar(models, perplexities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Perplexity', fontweight='bold', fontsize=12)
    ax4.set_title('Final Validation Perplexity', fontweight='bold', fontsize=14)
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, perplexities):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 5. Model Parameters Comparison (middle center)
    ax5 = fig.add_subplot(gs[1, 1])
    params = [
        baseline_results['model_info']['parameters']['total_millions'],
        recursive_results['model_info']['parameters']['total_millions']
    ]
    bars = ax5.bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Parameters (Millions)', fontweight='bold', fontsize=12)
    ax5.set_title('Model Size', fontweight='bold', fontsize=14)
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, params):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}M',
                ha='center', va='bottom', fontweight='bold')
    
    # 6. Improvement Summary (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Calculate improvements
    loss_improvement = (baseline_results['results']['best_val_loss'] - 
                       recursive_results['results']['best_val_loss']) / baseline_results['results']['best_val_loss'] * 100
    acc_improvement = ((recursive_val.get('accuracy', 0) - baseline_val.get('accuracy', 0)) / 
                       baseline_val.get('accuracy', 1e-9)) * 100
    
    # Add recursive-specific metrics if available
    recursive_config = recursive_results.get('recursive_config', {})
    
    summary_text = f"""
    COMPARISON SUMMARY
    {'='*40}
    
    Loss Improvement: {loss_improvement:+.2f}%
    Accuracy Improvement: {acc_improvement:+.2f}%
    
    Baseline Val Loss: {baseline_results['results']['best_val_loss']:.4f}
    Recursive Val Loss: {recursive_results['results']['best_val_loss']:.4f}
    
    """
    
    if recursive_config:
        summary_text += f"""
    Recursive Config:
    - H Cycles: {recursive_config.get('H_cycles', 'N/A')}
    - L Cycles: {recursive_config.get('L_cycles', 'N/A')}
    - Max Steps: {recursive_config.get('halt_max_steps', 'N/A')}
    - ACT Enabled: {recursive_config.get('use_act', 'N/A')}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 7-9. Configuration comparison tables (bottom row)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create comparison table
    table_data = [
        ['Metric', 'Baseline', 'Recursive', 'Δ'],
        ['Val Loss', f"{baseline_results['results']['best_val_loss']:.4f}",
         f"{recursive_results['results']['best_val_loss']:.4f}",
         f"{loss_improvement:+.2f}%"],
        ['Accuracy', f"{baseline_val.get('accuracy', 0)*100:.2f}%",
         f"{recursive_val.get('accuracy', 0)*100:.2f}%",
         f"{acc_improvement:+.2f}%"],
        ['Perplexity', f"{baseline_val.get('perplexity', 0):.2f}",
         f"{recursive_val.get('perplexity', 0):.2f}",
         f"{((baseline_val.get('perplexity', 0) - recursive_val.get('perplexity', 0)) / baseline_val.get('perplexity', 1))*100:+.2f}%"],
        ['Training Time', f"{baseline_results['results']['total_time']/60:.1f}m",
         f"{recursive_results['results']['total_time']/60:.1f}m",
         f"{((recursive_results['results']['total_time'] - baseline_results['results']['total_time']) / baseline_results['results']['total_time'])*100:+.2f}%"],
        ['Parameters', f"{params[0]:.1f}M", f"{params[1]:.1f}M",
         f"{((params[1] - params[0]) / params[0])*100:+.2f}%"],
    ]
    
    table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color improvement column
    for i in range(1, len(table_data)):
        delta_val = float(table_data[i][3].rstrip('%'))
        if delta_val > 0:
            table[(i, 3)].set_facecolor('#d4edda')
        elif delta_val < 0:
            table[(i, 3)].set_facecolor('#f8d7da')
    
    plt.suptitle('Baseline vs Recursive Reasoning - Comprehensive Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plots saved to: {save_path}")


def main():
    """Main comparison function"""
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    
    print("="*70)
    print("Baseline vs Recursive Reasoning - Comparison Analysis")
    print("="*70)
    
    # Load results
    print("\nLoading results...")
    baseline_results = load_results(results_dir, 'baseline')
    recursive_results = load_results(results_dir, 'recursive')
    
    if baseline_results is None or recursive_results is None:
        print("\n❌ Error: Could not load both result files")
        print("Make sure both baseline and recursive training completed successfully")
        return
    
    print("✓ Both result files loaded successfully")
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    comparison_plot_path = results_dir / 'comparison_plots.png'
    plot_comparison(baseline_results, recursive_results, comparison_plot_path)
    
    # Create summary JSON
    print("\nCreating comparison summary...")
    
    baseline_val_loss = baseline_results['results']['best_val_loss']
    recursive_val_loss = recursive_results['results']['best_val_loss']
    improvement = (baseline_val_loss - recursive_val_loss) / baseline_val_loss * 100
    
    summary = {
        'baseline': {
            'val_loss': baseline_val_loss,
            'training_time': baseline_results['results']['total_time'],
            'parameters': baseline_results['model_info']['parameters']['total_millions'],
        },
        'recursive': {
            'val_loss': recursive_val_loss,
            'training_time': recursive_results['results']['total_time'],
            'parameters': recursive_results['model_info']['parameters']['total_millions'],
            'config': recursive_results.get('recursive_config', {}),
        },
        'comparison': {
            'val_loss_improvement_percent': improvement,
            'winner': 'recursive' if recursive_val_loss < baseline_val_loss else 'baseline',
            'loss_difference': baseline_val_loss - recursive_val_loss,
        }
    }
    
    summary_file = results_dir / 'comparison_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Comparison summary saved to: {summary_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nBaseline Val Loss:   {baseline_val_loss:.4f}")
    print(f"Recursive Val Loss:  {recursive_val_loss:.4f}")
    print(f"Improvement:         {improvement:+.2f}%")
    print(f"\nWinner: {summary['comparison']['winner'].upper()}")
    print("="*70)
    
    print("\n✅ Comparison analysis complete!")
    print(f"\nOutputs:")
    print(f"  - Plots: {comparison_plot_path}")
    print(f"  - Summary: {summary_file}")


if __name__ == '__main__':
    main()

