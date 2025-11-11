"""
Comprehensive visualization of temperature experiment results
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List


def load_experiment_results(results_dir: Path) -> Dict:
    """Load all experiment results from results directory"""
    results = {}
    
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            metrics_file = exp_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    results[exp_dir.name] = data
    
    return results


def plot_temperature_comparison(results: Dict, output_dir: Path):
    """Plot comparison of different temperatures"""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Temperature Ablation: Comprehensive Comparison', fontsize=18, fontweight='bold')
    
    # Filter temperature ablation experiments
    temp_results = {k: v for k, v in results.items() if k.startswith('temp_') and not k.endswith('_long')}
    
    # Sort by temperature
    sorted_results = sorted(temp_results.items(), key=lambda x: x[1]['temperature'])
    
    # Extract data
    temps = [r[1]['temperature'] for r in sorted_results]
    names = [r[0] for r in sorted_results]
    
    # Colors based on temperature
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(temps)))
    
    # Plot 1: Validation Loss over Steps
    ax1 = fig.add_subplot(gs[0, :2])
    for i, (name, data) in enumerate(sorted_results):
        history = data['history']
        ax1.plot(history['steps'], history['val_losses'], 
                label=f"T={data['temperature']:.1f}", 
                color=colors[i], linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Validation Loss vs Training Steps', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss over Time
    ax2 = fig.add_subplot(gs[0, 2])
    for i, (name, data) in enumerate(sorted_results):
        history = data['history']
        ax2.plot(history['elapsed_times'], history['val_losses'], 
                color=colors[i], linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Loss vs Wall-Clock Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final Performance vs Temperature
    ax3 = fig.add_subplot(gs[1, 0])
    final_losses = [r[1]['final_metrics']['loss'] for r in sorted_results]
    best_losses = [min(r[1]['history']['val_losses']) for r in sorted_results]
    ax3.plot(temps, final_losses, 'o-', color='darkred', linewidth=2, markersize=8, label='Final Loss')
    ax3.plot(temps, best_losses, 's-', color='darkblue', linewidth=2, markersize=8, label='Best Loss')
    ax3.set_xlabel('Temperature', fontsize=12)
    ax3.set_ylabel('Validation Loss', fontsize=12)
    ax3.set_title('Performance vs Temperature', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Plot 4: Validation Accuracy vs Temperature
    ax4 = fig.add_subplot(gs[1, 1])
    final_accs = [r[1]['final_metrics']['accuracy'] for r in sorted_results]
    ax4.plot(temps, final_accs, 'o-', color='darkgreen', linewidth=2, markersize=8)
    ax4.set_xlabel('Temperature', fontsize=12)
    ax4.set_ylabel('Validation Accuracy', fontsize=12)
    ax4.set_title('Accuracy vs Temperature', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    # Plot 5: Routing Entropy vs Temperature
    ax5 = fig.add_subplot(gs[1, 2])
    final_entropies = []
    for name, data in sorted_results:
        if 'routing_entropies' in data['history'] and data['history']['routing_entropies']:
            final_entropies.append(data['history']['routing_entropies'][-1])
        else:
            final_entropies.append(0)
    ax5.plot(temps, final_entropies, 'o-', color='purple', linewidth=2, markersize=8)
    ax5.set_xlabel('Temperature', fontsize=12)
    ax5.set_ylabel('Routing Entropy', fontsize=12)
    ax5.set_title('Routing Entropy vs Temperature', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    
    # Plot 6: Accuracy over Steps
    ax6 = fig.add_subplot(gs[2, :2])
    for i, (name, data) in enumerate(sorted_results):
        history = data['history']
        ax6.plot(history['steps'], history['val_accuracies'], 
                label=f"T={data['temperature']:.1f}",
                color=colors[i], linewidth=2, marker='o', markersize=4)
    ax6.set_xlabel('Training Steps', fontsize=12)
    ax6.set_ylabel('Validation Accuracy', fontsize=12)
    ax6.set_title('Validation Accuracy vs Training Steps', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10, ncol=2)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Summary Statistics Table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('tight')
    ax7.axis('off')
    
    # Find best temperature
    best_idx = np.argmin(best_losses)
    best_temp = temps[best_idx]
    best_loss = best_losses[best_idx]
    
    table_data = [
        ['Metric', 'Value'],
        ['Best Temperature', f'{best_temp:.2f}'],
        ['Best Loss', f'{best_loss:.4f}'],
        ['Worst Loss', f'{max(best_losses):.4f}'],
        ['Improvement', f'{((max(best_losses) - best_loss) / max(best_losses) * 100):.1f}%'],
        ['Temps Tested', f'{len(temps)}'],
    ]
    
    table = ax7.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best result
    table[(1, 1)].set_facecolor('#FFD700')
    table[(2, 1)].set_facecolor('#FFD700')
    
    ax7.set_title('Summary Statistics', fontsize=14, fontweight='bold')
    
    plt.savefig(output_dir / 'temperature_ablation_comprehensive.png', 
                dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'temperature_ablation_comprehensive.png'}")
    plt.close()


def plot_routing_dynamics(results: Dict, output_dir: Path):
    """Plot routing dynamics over training"""
    temp_results = {k: v for k, v in results.items() if k.startswith('temp_') and not k.endswith('_long')}
    sorted_results = sorted(temp_results.items(), key=lambda x: x[1]['temperature'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Routing Dynamics Analysis', fontsize=18, fontweight='bold')
    
    temps = [r[1]['temperature'] for r in sorted_results]
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(temps)))
    
    # Plot 1: Routing Entropy Evolution
    ax = axes[0, 0]
    for i, (name, data) in enumerate(sorted_results):
        history = data['history']
        if 'routing_entropies' in history and history['routing_entropies']:
            ax.plot(history['steps'], history['routing_entropies'],
                   label=f"T={data['temperature']:.1f}",
                   color=colors[i], linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Routing Entropy', fontsize=12)
    ax.set_title('Routing Entropy Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Selection Confidence Evolution
    ax = axes[0, 1]
    for i, (name, data) in enumerate(sorted_results):
        history = data['history']
        if 'selection_confidences' in history and history['selection_confidences']:
            ax.plot(history['steps'], history['selection_confidences'],
                   label=f"T={data['temperature']:.1f}",
                   color=colors[i], linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Top-1 Selection Confidence', fontsize=12)
    ax.set_title('Selection Confidence Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Load Balancing Loss Evolution
    ax = axes[1, 0]
    for i, (name, data) in enumerate(sorted_results):
        history = data['history']
        if 'load_balancing_losses' in history and history['load_balancing_losses']:
            ax.plot(history['steps'], history['load_balancing_losses'],
                   label=f"T={data['temperature']:.1f}",
                   color=colors[i], linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Load Balancing Loss', fontsize=12)
    ax.set_title('Load Balancing Loss Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Temperature vs Final Entropy and Confidence
    ax = axes[1, 1]
    final_entropies = []
    final_confidences = []
    for name, data in sorted_results:
        history = data['history']
        if 'routing_entropies' in history and history['routing_entropies']:
            final_entropies.append(history['routing_entropies'][-1])
        else:
            final_entropies.append(0)
        if 'selection_confidences' in history and history['selection_confidences']:
            final_confidences.append(history['selection_confidences'][-1])
        else:
            final_confidences.append(0)
    
    ax2 = ax.twinx()
    line1 = ax.plot(temps, final_entropies, 'o-', color='purple', 
                    linewidth=2, markersize=8, label='Entropy')
    line2 = ax2.plot(temps, final_confidences, 's-', color='orange', 
                     linewidth=2, markersize=8, label='Confidence')
    
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Final Routing Entropy', fontsize=12, color='purple')
    ax2.set_ylabel('Final Selection Confidence', fontsize=12, color='orange')
    ax.set_title('Temperature vs Final Routing Metrics', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.tick_params(axis='y', labelcolor='purple')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'routing_dynamics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'routing_dynamics.png'}")
    plt.close()


def plot_expert_utilization(results: Dict, output_dir: Path):
    """Plot expert utilization patterns"""
    temp_results = {k: v for k, v in results.items() if k.startswith('temp_') and not k.endswith('_long')}
    sorted_results = sorted(temp_results.items(), key=lambda x: x[1]['temperature'])
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Expert Utilization Patterns', fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(sorted_results[:8]):  # Plot up to 8
        ax = axes[idx]
        
        # Get final expert utilization
        history = data['history']
        if 'expert_utilizations' in history and history['expert_utilizations']:
            final_util = history['expert_utilizations'][-1]
            
            experts = list(range(len(final_util)))
            bars = ax.bar(experts, final_util, color='steelblue', alpha=0.8)
            
            # Color bars by utilization
            max_util = max(final_util) if final_util else 1
            for bar, util in zip(bars, final_util):
                bar.set_color(plt.cm.RdYlGn(util / max_util))
            
            ax.axhline(y=1.0/len(final_util), color='red', linestyle='--', 
                      linewidth=2, label='Uniform')
            ax.set_xlabel('Expert Index', fontsize=10)
            ax.set_ylabel('Utilization', fontsize=10)
            ax.set_title(f'Temperature = {data["temperature"]:.1f}', 
                        fontsize=12, fontweight='bold')
            ax.set_ylim(0, max(final_util) * 1.2 if final_util else 1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(sorted_results), 8):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'expert_utilization.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'expert_utilization.png'}")
    plt.close()


def plot_schedule_comparison(results: Dict, output_dir: Path):
    """Plot temperature schedule comparisons"""
    schedule_results = {k: v for k, v in results.items() if k.startswith('schedule_')}
    
    if not schedule_results:
        print("⚠️ No schedule experiments found, skipping schedule comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temperature Schedule Comparison', fontsize=18, fontweight='bold')
    
    # Plot 1: Validation Loss
    ax = axes[0, 0]
    for name, data in schedule_results.items():
        history = data['history']
        ax.plot(history['steps'], history['val_losses'],
               label=name, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Loss: Schedule Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Temperature Evolution
    ax = axes[0, 1]
    for name, data in schedule_results.items():
        history = data['history']
        if 'temperatures' in history:
            ax.plot(history['steps'], history['temperatures'],
                   label=name, linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Temperature', fontsize=12)
    ax.set_title('Temperature Schedule Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy
    ax = axes[1, 0]
    for name, data in schedule_results.items():
        history = data['history']
        ax.plot(history['steps'], history['val_accuracies'],
               label=name, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Accuracy: Schedule Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final Metrics Comparison
    ax = axes[1, 1]
    schedule_names = list(schedule_results.keys())
    final_losses = [data['final_metrics']['loss'] for data in schedule_results.values()]
    final_accs = [data['final_metrics']['accuracy'] for data in schedule_results.values()]
    
    x = np.arange(len(schedule_names))
    width = 0.35
    
    ax.bar(x - width/2, final_losses, width, label='Loss', alpha=0.8, color='coral')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, final_accs, width, label='Accuracy', alpha=0.8, color='lightblue')
    
    ax.set_xlabel('Schedule', fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12, color='coral')
    ax2.set_ylabel('Final Accuracy', fontsize=12, color='lightblue')
    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(schedule_names, rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor='coral')
    ax2.tick_params(axis='y', labelcolor='lightblue')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'schedule_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'schedule_comparison.png'}")
    plt.close()


def generate_summary_report(results: Dict, output_dir: Path):
    """Generate comprehensive summary report"""
    report = {
        'experiment_overview': {
            'total_experiments': len(results),
            'experiment_names': list(results.keys()),
        },
        'best_results': {},
        'temperature_analysis': {},
        'schedule_analysis': {},
    }
    
    # Temperature ablation analysis
    temp_results = {k: v for k, v in results.items() if k.startswith('temp_') and not k.endswith('_long')}
    if temp_results:
        best_exp = min(temp_results.items(), key=lambda x: min(x[1]['history']['val_losses']))
        best_name, best_data = best_exp
        
        report['best_results']['temperature_ablation'] = {
            'experiment': best_name,
            'temperature': best_data['temperature'],
            'best_loss': min(best_data['history']['val_losses']),
            'final_loss': best_data['final_metrics']['loss'],
            'final_accuracy': best_data['final_metrics']['accuracy'],
        }
        
        # Temperature analysis
        temps = sorted([(v['temperature'], min(v['history']['val_losses'])) 
                       for v in temp_results.values()])
        report['temperature_analysis'] = {
            'tested_temperatures': [t[0] for t in temps],
            'losses': [t[1] for t in temps],
            'best_temperature': best_data['temperature'],
            'worst_temperature': max(temps, key=lambda x: x[1])[0],
            'improvement': ((max(t[1] for t in temps) - min(t[1] for t in temps)) / 
                          max(t[1] for t in temps) * 100),
        }
    
    # Schedule analysis
    schedule_results = {k: v for k, v in results.items() if k.startswith('schedule_')}
    if schedule_results:
        best_schedule = min(schedule_results.items(), 
                          key=lambda x: min(x[1]['history']['val_losses']))
        best_name, best_data = best_schedule
        
        report['best_results']['temperature_schedule'] = {
            'experiment': best_name,
            'schedule_type': best_data['temperature_schedule'],
            'best_loss': min(best_data['history']['val_losses']),
            'final_loss': best_data['final_metrics']['loss'],
            'final_accuracy': best_data['final_metrics']['accuracy'],
        }
        
        schedule_losses = {k: min(v['history']['val_losses']) 
                          for k, v in schedule_results.items()}
        report['schedule_analysis'] = {
            'schedules_tested': list(schedule_losses.keys()),
            'losses': schedule_losses,
            'best_schedule': best_name,
        }
    
    # Save report
    report_file = output_dir / 'summary_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved: {report_file}")
    
    # Print summary to console
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    
    if 'temperature_ablation' in report['best_results']:
        ta = report['best_results']['temperature_ablation']
        print(f"🏆 Best Temperature: {ta['temperature']:.2f}")
        print(f"   Loss: {ta['best_loss']:.4f}")
        print(f"   Accuracy: {ta['final_accuracy']:.4f}\n")
    
    if 'temperature_schedule' in report['best_results']:
        ts = report['best_results']['temperature_schedule']
        print(f"🏆 Best Schedule: {ts['experiment']}")
        print(f"   Loss: {ts['best_loss']:.4f}")
        print(f"   Accuracy: {ts['final_accuracy']:.4f}\n")
    
    if 'temperature_analysis' in report:
        ta = report['temperature_analysis']
        print(f"📊 Temperature Analysis:")
        print(f"   Improvement: {ta['improvement']:.2f}%")
        print(f"   Best: T={ta['best_temperature']:.2f}")
        print(f"   Worst: T={ta['worst_temperature']:.2f}\n")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot temperature experiment results")
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='./analysis',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {results_dir}...")
    results = load_experiment_results(results_dir)
    
    if not results:
        print(f"❌ No results found in {results_dir}")
        return
    
    print(f"Found {len(results)} experiments")
    print(f"Generating visualizations...\n")
    
    # Generate all plots
    plot_temperature_comparison(results, output_dir)
    plot_routing_dynamics(results, output_dir)
    plot_expert_utilization(results, output_dir)
    plot_schedule_comparison(results, output_dir)
    generate_summary_report(results, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ All visualizations saved to {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

