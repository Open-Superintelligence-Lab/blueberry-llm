#!/usr/bin/env python3
"""
Run Adam Optimization Suite with optimal LR (0.001)

Tests various Adam hyperparameters with the discovered optimal learning rate
to ensure fair comparison with Muon.

Total experiments: 7
Estimated time: ~14 minutes
"""
import sys
import os
from pathlib import Path

# Fix tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(project_root))

from run_experiments import run_multiple_experiments

def main():
    print("=" * 80)
    print("🔍 ADAM OPTIMIZATION SUITE")
    print("=" * 80)
    print("\nUsing discovered optimal LR: 0.001")
    print("\nThis suite will test:")
    print("  1. Weight decay variations")
    print("  2. Warmup sensitivity")
    print("  3. LR schedule types")
    print("\n" + "=" * 80)
    
    experiments = [
        # Core optimal
        "adam_optimal",
        
        # Weight decay
        "adam_optimal_wd_0.05",
        "adam_optimal_wd_0.2",
        
        # Warmup
        "adam_no_warmup",
        "adam_warmup_0.1",
        
        # Schedule types
        "adam_linear_decay",
        "adam_constant_lr_optimal",
    ]
    
    print(f"\n📊 Running {len(experiments)} experiments...")
    print(f"⏱️  Estimated time: ~{len(experiments) * 2} minutes\n")
    
    output_dir = "./adam_optimization_results"
    
    # Run all experiments
    results = run_multiple_experiments(experiments, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ ADAM OPTIMIZATION SUITE COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    
    # Print summary
    if results:
        print("\n📊 Quick Summary:")
        print("-" * 80)
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1]['metrics']['val_loss']
        )
        print(f"{'Experiment':<35} {'Val Loss':<12} {'Time (min)':<12}")
        print("-" * 80)
        for exp_name, data in sorted_results:
            status = "🏆" if exp_name == sorted_results[0][0] else ""
            print(f"{exp_name:<35} {data['metrics']['val_loss']:<12.4f} "
                  f"{data['time_minutes']:<12.2f} {status}")
        
        print("\n🎯 Best Adam configuration: " + sorted_results[0][0])
        print(f"   Loss: {sorted_results[0][1]['metrics']['val_loss']:.4f}")
        
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

