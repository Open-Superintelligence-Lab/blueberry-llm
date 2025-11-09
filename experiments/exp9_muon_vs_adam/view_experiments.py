"""
View and analyze results from Muon vs Adam experiments
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))


def load_experiment_results(exp_dir: Path):
    """Load results from an experiment directory"""
    metrics_file = exp_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    return data


def print_experiment_summary(exp_name: str, data: dict):
    """Print a summary of experiment results"""
    exp_config = data.get('experiment_config', {})
    final_metrics = data.get('final_metrics', {})
    history = data.get('history', {})
    
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*80}")
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Optimizer: {exp_config.get('optimizer_type', 'unknown')}")
    print(f"  Steps: {exp_config.get('max_steps', 'unknown')}")
    print(f"  LR Schedule: {exp_config.get('lr_schedule_type', 'unknown')}")
    print(f"  Load Balancing: {exp_config.get('load_balancing_weight', 'unknown')}")
    print(f"  Early Stopping: {exp_config.get('use_early_stopping', False)}")
    
    # Results
    print(f"\nFinal Results:")
    print(f"  Validation Loss: {final_metrics.get('val_loss', 'N/A'):.4f}")
    print(f"  Validation Accuracy: {final_metrics.get('val_accuracy', 'N/A'):.4f}")
    print(f"  Validation Perplexity: {final_metrics.get('val_perplexity', 'N/A'):.2f}")
    print(f"  Training Time: {data.get('total_time_minutes', 'N/A'):.2f} minutes")
    
    # Best metrics
    if history and 'val_losses' in history:
        best_loss = min(history['val_losses'])
        best_idx = history['val_losses'].index(best_loss)
        best_step = history['steps'][best_idx] if 'steps' in history else 'unknown'
        
        print(f"\nBest Results:")
        print(f"  Best Loss: {best_loss:.4f}")
        print(f"  Best Step: {best_step}")
        
        final_loss = final_metrics.get('val_loss')
        if final_loss:
            degradation = ((final_loss - best_loss) / best_loss) * 100
            if degradation > 0.5:
                print(f"  ⚠️  Loss degradation: +{degradation:.2f}% from best")
            else:
                print(f"  ✓ Loss stable at end of training")
    
    if data.get('stopped_early'):
        print(f"\n  ⚠️  Training stopped early at step {data.get('actual_steps')}")


def compare_all_experiments(experiments_dir: Path):
    """Compare all experiments in the directory"""
    print(f"\n{'='*80}")
    print("MUON VS ADAM: EXPERIMENT COMPARISON")
    print(f"{'='*80}\n")
    
    # Find all experiment directories
    exp_dirs = [d for d in experiments_dir.iterdir() 
                if d.is_dir() and (d / "metrics.json").exists()]
    
    if not exp_dirs:
        print("No experiment results found.")
        return
    
    # Load all results
    results = {}
    for exp_dir in sorted(exp_dirs):
        exp_name = exp_dir.name
        data = load_experiment_results(exp_dir)
        if data:
            results[exp_name] = data
    
    # Print summary table
    print(f"{'Experiment':<25} {'Optimizer':<15} {'Final Loss':>12} {'Best Loss':>12} "
          f"{'Final Acc':>12} {'Time (min)':>12}")
    print("-" * 100)
    
    muon_results = []
    adam_results = []
    
    for exp_name, data in results.items():
        exp_config = data.get('experiment_config', {})
        final_metrics = data.get('final_metrics', {})
        history = data.get('history', {})
        
        optimizer_type = exp_config.get('optimizer_type', 'unknown')
        final_loss = final_metrics.get('val_loss', float('inf'))
        final_acc = final_metrics.get('val_accuracy', 0.0)
        time_min = data.get('total_time_minutes', 0.0)
        
        best_loss = min(history.get('val_losses', [float('inf')]))
        
        print(f"{exp_name:<25} {optimizer_type:<15} {final_loss:>12.4f} {best_loss:>12.4f} "
              f"{final_acc:>12.4f} {time_min:>12.2f}")
        
        # Categorize by optimizer
        if 'muon' in optimizer_type.lower():
            muon_results.append({
                'name': exp_name,
                'best_loss': best_loss,
                'final_loss': final_loss
            })
        elif 'adam' in optimizer_type.lower():
            adam_results.append({
                'name': exp_name,
                'best_loss': best_loss,
                'final_loss': final_loss
            })
    
    # Print winner analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    if muon_results:
        best_muon = min(muon_results, key=lambda x: x['best_loss'])
        print(f"\n🔵 Best Muon Configuration:")
        print(f"   Name: {best_muon['name']}")
        print(f"   Best Loss: {best_muon['best_loss']:.4f}")
        print(f"   Final Loss: {best_muon['final_loss']:.4f}")
    
    if adam_results:
        best_adam = min(adam_results, key=lambda x: x['best_loss'])
        print(f"\n🔴 Best Adam Configuration:")
        print(f"   Name: {best_adam['name']}")
        print(f"   Best Loss: {best_adam['best_loss']:.4f}")
        print(f"   Final Loss: {best_adam['final_loss']:.4f}")
    
    if muon_results and adam_results:
        print(f"\n{'='*80}")
        print("WINNER")
        print(f"{'='*80}")
        
        improvement = ((best_adam['best_loss'] - best_muon['best_loss']) / best_adam['best_loss']) * 100
        
        if abs(improvement) < 0.5:
            print(f"\n⚖️  Results are very close (difference: {abs(improvement):.2f}%)")
            print(f"   Both optimizers perform comparably on this task")
        elif improvement > 0:
            print(f"\n🏆 MUON WINS!")
            print(f"   Muon achieves {improvement:.2f}% better loss than Adam")
            print(f"   Recommendation: Use Muon for this model architecture")
        else:
            print(f"\n🏆 ADAM WINS!")
            print(f"   Adam achieves {-improvement:.2f}% better loss than Muon")
            print(f"   Recommendation: Stick with Adam for this model architecture")
    
    # Check for comparison plot
    comparison_plot = experiments_dir / "comparison_plot.png"
    if comparison_plot.exists():
        print(f"\n📊 Comparison plot available: {comparison_plot}")
    
    comparison_json = experiments_dir / "comparison_summary.json"
    if comparison_json.exists():
        print(f"📁 Comparison summary available: {comparison_json}")


def view_single_experiment(exp_name: str, experiments_dir: Path):
    """View a single experiment's results"""
    exp_dir = experiments_dir / exp_name
    
    if not exp_dir.exists():
        print(f"❌ Experiment '{exp_name}' not found in {experiments_dir}")
        return
    
    data = load_experiment_results(exp_dir)
    if not data:
        print(f"❌ No results found for experiment '{exp_name}'")
        return
    
    print_experiment_summary(exp_name, data)
    
    # Show plot if available
    plot_path = exp_dir / "metrics_plot.png"
    if plot_path.exists():
        print(f"\n📊 Plot available: {plot_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="View and analyze Muon vs Adam experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--experiment', '-e',
        help='View specific experiment by name'
    )
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Compare all experiments (default if no experiment specified)'
    )
    parser.add_argument(
        '--dir', '-d',
        default='.',
        help='Experiments directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    experiments_dir = Path(args.dir).resolve()
    
    if args.experiment:
        view_single_experiment(args.experiment, experiments_dir)
    else:
        compare_all_experiments(experiments_dir)
    
    print("\n")


if __name__ == "__main__":
    main()

