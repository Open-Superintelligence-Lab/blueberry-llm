"""
View and analyze experiment results
"""
import argparse
import json
import sys
from pathlib import Path

# Add exp8 directory to path for local imports
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from exp_configs.experiment_configs import list_experiments, EXPERIMENTS


def view_results(output_dir: str = "./experiments"):
    """View results from completed experiments"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Experiments directory not found: {output_path}")
        return
    
    print(f"\n{'='*80}")
    print("📊 EXPERIMENT RESULTS")
    print(f"{'='*80}\n")
    
    # Find all experiment directories with results
    experiment_dirs = [d for d in output_path.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    
    if not experiment_dirs:
        print(f"No completed experiments found in {output_path}")
        return
    
    print(f"Found {len(experiment_dirs)} completed experiments:\n")
    print(f"{'Experiment':<25} {'Final Loss':>12} {'Best Loss':>12} {'Final Acc':>12} {'Steps':>8} {'Early Stop':>12}")
    print("-" * 100)
    
    results = []
    for exp_dir in sorted(experiment_dirs):
        metrics_file = exp_dir / "metrics.json"
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        exp_name = data['experiment_config']['name']
        final_loss = data['final_metrics']['val_loss']
        best_loss = min(data['history']['val_losses'])
        final_acc = data['final_metrics']['val_accuracy']
        actual_steps = data['actual_steps']
        stopped_early = "Yes" if data['stopped_early'] else "No"
        
        print(f"{exp_name:<25} {final_loss:>12.4f} {best_loss:>12.4f} "
              f"{final_acc:>12.4f} {actual_steps:>8} {stopped_early:>12}")
        
        results.append({
            'name': exp_name,
            'final_loss': final_loss,
            'best_loss': best_loss,
            'dir': exp_dir
        })
    
    # Show best experiment
    if results:
        best = min(results, key=lambda x: x['best_loss'])
        print(f"\n🏆 Best experiment: {best['name']} (best loss: {best['best_loss']:.4f})")
        print(f"   Results: {best['dir']}")
    
    # Check for comparison files
    comparison_file = output_path / "comparison_summary.json"
    if comparison_file.exists():
        print(f"\n📊 Comparison plot available: {output_path / 'comparison_plot.png'}")
        print(f"📁 Comparison data: {comparison_file}")


def view_specific_experiment(exp_name: str, output_dir: str = "./experiments"):
    """View detailed results for a specific experiment"""
    exp_path = Path(output_dir) / exp_name
    metrics_file = exp_path / "metrics.json"
    
    if not metrics_file.exists():
        print(f"❌ Experiment '{exp_name}' not found or incomplete")
        print(f"   Looking for: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"📊 EXPERIMENT: {exp_name}")
    print(f"{'='*80}\n")
    
    config = data['experiment_config']
    print("Configuration:")
    print(f"  Description: {config['description']}")
    print(f"  Max steps: {config['max_steps']}")
    print(f"  LR schedule: {config['lr_schedule_type']}")
    print(f"  Early stopping: {config['use_early_stopping']}")
    print(f"  Load balancing weight: {config['load_balancing_weight']}")
    print(f"  Dropout: {config['dropout']}")
    
    print("\nResults:")
    metrics = data['final_metrics']
    print(f"  Validation loss: {metrics['val_loss']:.4f}")
    print(f"  Validation accuracy: {metrics['val_accuracy']:.4f}")
    print(f"  Validation perplexity: {metrics['val_perplexity']:.2f}")
    print(f"  Training time: {data['total_time_minutes']:.2f} minutes")
    print(f"  Actual steps: {data['actual_steps']}")
    if data['stopped_early']:
        print(f"  ⚠️  Training stopped early")
    
    history = data['history']
    best_loss = min(history['val_losses'])
    best_idx = history['val_losses'].index(best_loss)
    best_step = history['steps'][best_idx]
    
    print("\nBest validation loss:")
    print(f"  Loss: {best_loss:.4f}")
    print(f"  At step: {best_step}")
    print(f"  Accuracy: {history['val_accuracies'][best_idx]:.4f}")
    
    # Check for overfitting
    final_loss = history['val_losses'][-1]
    if final_loss > best_loss * 1.01:  # 1% worse than best
        print(f"\n⚠️  Possible overfitting detected:")
        print(f"     Best loss: {best_loss:.4f} at step {best_step}")
        print(f"     Final loss: {final_loss:.4f} at step {history['steps'][-1]}")
        print(f"     Degradation: {((final_loss - best_loss) / best_loss * 100):.1f}%")
    
    print(f"\nFiles:")
    print(f"  Metrics: {metrics_file}")
    print(f"  Plot: {exp_path / 'metrics_plot.png'}")
    print(f"  Model: {exp_path / 'model.pt'}")


def main():
    parser = argparse.ArgumentParser(description="View experiment results")
    parser.add_argument(
        '--experiment', '-e',
        help='View specific experiment by name'
    )
    parser.add_argument(
        '--list-available', '-l',
        action='store_true',
        help='List all available experiment configurations'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./experiments',
        help='Experiments output directory (default: ./experiments)'
    )
    
    args = parser.parse_args()
    
    if args.list_available:
        list_experiments()
    elif args.experiment:
        view_specific_experiment(args.experiment, args.output_dir)
    else:
        view_results(args.output_dir)


if __name__ == "__main__":
    main()

