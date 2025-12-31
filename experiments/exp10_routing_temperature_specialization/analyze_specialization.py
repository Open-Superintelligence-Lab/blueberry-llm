"""
Analyze expert specialization patterns from routing statistics
"""
import argparse
import json
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_expert_specialization(results_dir: Path, output_dir: Path):
    """Analyze how experts specialize under different temperatures"""
    
    print("Loading experiment results...")
    results = {}
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            metrics_file = exp_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    results[exp_dir.name] = json.load(f)
    
    if not results:
        print(f"❌ No results found in {results_dir}")
        return
    
    print(f"Found {len(results)} experiments\n")
    
    # Analyze expert utilization distribution
    analyze_utilization_distribution(results, output_dir)
    
    # Analyze routing entropy trends
    analyze_entropy_trends(results, output_dir)
    
    # Generate specialization report
    generate_specialization_report(results, output_dir)


def analyze_utilization_distribution(results: Dict, output_dir: Path):
    """Analyze how evenly experts are utilized"""
    print("Analyzing expert utilization distribution...")
    
    temp_results = {k: v for k, v in results.items() if k.startswith('temp_')}
    sorted_results = sorted(temp_results.items(), key=lambda x: x[1]['temperature'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Expert Utilization Distribution Analysis', fontsize=18, fontweight='bold')
    
    # Plot 1: Gini coefficient vs Temperature
    ax = axes[0, 0]
    temps = []
    gini_coeffs = []
    
    for name, data in sorted_results:
        history = data['history']
        if 'expert_utilizations' in history and history['expert_utilizations']:
            final_util = np.array(history['expert_utilizations'][-1])
            # Compute Gini coefficient (measure of inequality)
            sorted_util = np.sort(final_util)
            n = len(sorted_util)
            gini = (2 * np.sum((np.arange(1, n+1)) * sorted_util)) / (n * np.sum(sorted_util)) - (n + 1) / n
            
            temps.append(data['temperature'])
            gini_coeffs.append(gini)
    
    ax.plot(temps, gini_coeffs, 'o-', linewidth=2, markersize=8, color='darkblue')
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Gini Coefficient', fontsize=12)
    ax.set_title('Utilization Inequality vs Temperature\n(Lower Gini = More Balanced)', 
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', label='Perfect Balance')
    ax.legend()
    
    # Plot 2: Utilization variance vs Temperature
    ax = axes[0, 1]
    variances = []
    
    for name, data in sorted_results:
        history = data['history']
        if 'expert_utilizations' in history and history['expert_utilizations']:
            final_util = np.array(history['expert_utilizations'][-1])
            variances.append(np.var(final_util))
    
    ax.plot(temps, variances, 'o-', linewidth=2, markersize=8, color='darkgreen')
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Utilization Variance', fontsize=12)
    ax.set_title('Utilization Variance vs Temperature\n(Lower = More Balanced)', 
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of expert utilization across temperatures
    ax = axes[1, 0]
    utilization_matrix = []
    temp_labels = []
    
    for name, data in sorted_results:
        history = data['history']
        if 'expert_utilizations' in history and history['expert_utilizations']:
            final_util = history['expert_utilizations'][-1]
            utilization_matrix.append(final_util)
            temp_labels.append(f"T={data['temperature']:.1f}")
    
    if utilization_matrix:
        utilization_matrix = np.array(utilization_matrix)
        sns.heatmap(utilization_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=[f'E{i}' for i in range(utilization_matrix.shape[1])],
                   yticklabels=temp_labels, ax=ax, cbar_kws={'label': 'Utilization'})
        ax.set_title('Expert Utilization Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Expert Index', fontsize=12)
        ax.set_ylabel('Temperature', fontsize=12)
    
    # Plot 4: Distribution statistics
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    table_data = [['Temperature', 'Gini', 'Variance', 'Min Util', 'Max Util']]
    for i, (name, data) in enumerate(sorted_results):
        if i < len(gini_coeffs):
            history = data['history']
            if 'expert_utilizations' in history and history['expert_utilizations']:
                final_util = np.array(history['expert_utilizations'][-1])
                table_data.append([
                    f"{data['temperature']:.1f}",
                    f"{gini_coeffs[i]:.3f}",
                    f"{variances[i]:.4f}",
                    f"{np.min(final_util):.3f}",
                    f"{np.max(final_util):.3f}",
                ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.2, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Utilization Statistics Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'expert_utilization_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'expert_utilization_analysis.png'}")
    plt.close()


def analyze_entropy_trends(results: Dict, output_dir: Path):
    """Analyze routing entropy evolution over training"""
    print("Analyzing routing entropy trends...")
    
    temp_results = {k: v for k, v in results.items() if k.startswith('temp_')}
    sorted_results = sorted(temp_results.items(), key=lambda x: x[1]['temperature'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Routing Entropy Analysis', fontsize=18, fontweight='bold')
    
    # Plot 1: Entropy evolution
    ax = axes[0]
    temps = [r[1]['temperature'] for r in sorted_results]
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(temps)))
    
    for i, (name, data) in enumerate(sorted_results):
        history = data['history']
        if 'routing_entropies' in history and history['routing_entropies']:
            ax.plot(history['steps'], history['routing_entropies'],
                   label=f"T={data['temperature']:.1f}",
                   color=colors[i], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Routing Entropy', fontsize=12)
    ax.set_title('Routing Entropy Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Entropy change rate
    ax = axes[1]
    
    for i, (name, data) in enumerate(sorted_results):
        history = data['history']
        if 'routing_entropies' in history and history['routing_entropies'] and len(history['routing_entropies']) > 1:
            entropies = np.array(history['routing_entropies'])
            steps = np.array(history['steps'])
            
            # Compute rate of change
            entropy_diff = np.diff(entropies)
            step_diff = np.diff(steps)
            entropy_rate = entropy_diff / step_diff
            
            ax.plot(steps[1:], entropy_rate,
                   label=f"T={data['temperature']:.1f}",
                   color=colors[i], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Entropy Change Rate', fontsize=12)
    ax.set_title('Routing Entropy Change Rate', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'entropy_analysis.png'}")
    plt.close()


def generate_specialization_report(results: Dict, output_dir: Path):
    """Generate detailed specialization report"""
    print("Generating specialization report...")
    
    report = {
        'summary': {},
        'per_experiment': {},
        'insights': []
    }
    
    temp_results = {k: v for k, v in results.items() if k.startswith('temp_')}
    
    for name, data in temp_results.items():
        history = data['history']
        
        analysis = {
            'temperature': data['temperature'],
            'final_loss': data['final_metrics']['val_loss'],
            'final_accuracy': data['final_metrics']['val_accuracy'],
        }
        
        # Analyze expert utilization
        if 'expert_utilizations' in history and history['expert_utilizations']:
            final_util = np.array(history['expert_utilizations'][-1])
            
            analysis['expert_utilization'] = {
                'distribution': final_util.tolist(),
                'mean': float(np.mean(final_util)),
                'std': float(np.std(final_util)),
                'min': float(np.min(final_util)),
                'max': float(np.max(final_util)),
                'gini': float((2 * np.sum((np.arange(1, len(final_util)+1)) * np.sort(final_util))) / 
                             (len(final_util) * np.sum(final_util)) - (len(final_util) + 1) / len(final_util)),
            }
        
        # Analyze routing entropy
        if 'routing_entropies' in history and history['routing_entropies']:
            entropies = history['routing_entropies']
            analysis['routing_entropy'] = {
                'initial': entropies[0] if entropies else None,
                'final': entropies[-1] if entropies else None,
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
            }
        
        report['per_experiment'][name] = analysis
    
    # Generate insights
    if temp_results:
        best_exp = min(temp_results.items(), key=lambda x: x[1]['final_metrics']['val_loss'])
        report['summary']['best_temperature'] = {
            'experiment': best_exp[0],
            'temperature': best_exp[1]['temperature'],
            'loss': best_exp[1]['final_metrics']['val_loss'],
        }
        
        # Find most balanced utilization
        gini_scores = {name: data['expert_utilization']['gini'] 
                      for name, data in report['per_experiment'].items() 
                      if 'expert_utilization' in data}
        if gini_scores:
            most_balanced = min(gini_scores.items(), key=lambda x: x[1])
            report['summary']['most_balanced_experts'] = {
                'experiment': most_balanced[0],
                'gini_coefficient': most_balanced[1],
            }
        
        # Insights
        report['insights'].append("Lower temperature (< 1.0) leads to sharper routing but may cause load imbalance")
        report['insights'].append("Higher temperature (> 1.0) improves load balancing but may reduce specialization")
        report['insights'].append("Optimal temperature balances exploration and exploitation")
    
    # Save report
    report_file = output_dir / 'specialization_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved: {report_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SPECIALIZATION ANALYSIS SUMMARY")
    print(f"{'='*80}\n")
    
    if 'best_temperature' in report['summary']:
        bt = report['summary']['best_temperature']
        print(f"🏆 Best Performance: {bt['experiment']}")
        print(f"   Temperature: {bt['temperature']:.2f}")
        print(f"   Loss: {bt['loss']:.4f}\n")
    
    if 'most_balanced_experts' in report['summary']:
        mb = report['summary']['most_balanced_experts']
        print(f"⚖️  Most Balanced Experts: {mb['experiment']}")
        print(f"   Gini Coefficient: {mb['gini_coefficient']:.3f}\n")
    
    print("💡 Key Insights:")
    for insight in report['insights']:
        print(f"   • {insight}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze expert specialization patterns")
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='./analysis',
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyze_expert_specialization(results_dir, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ Specialization analysis complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

