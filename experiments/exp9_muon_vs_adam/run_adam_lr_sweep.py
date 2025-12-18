#!/usr/bin/env python3
"""
Run Adam Learning Rate Sweep (200 steps each, fast iteration)

This finds the optimal learning rate for Adam optimizer to ensure
fair comparison with Muon.

Total experiments: 11
Estimated time: ~20-25 minutes
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
    print("🔍 ADAM LEARNING RATE SWEEP")
    print("=" * 80)
    print("\nSystematic LR search for Adam optimizer (200 steps each)")
    print("\nLearning rates to test:")
    print("  • 0.0001 (very conservative)")
    print("  • 0.0002")
    print("  • 0.0003")
    print("  • 0.0005")
    print("  • 0.0007")
    print("  • 0.001 (typical default)")
    print("  • 0.002")
    print("  • 0.003")
    print("  • 0.005")
    print("  • 0.007")
    print("  • 0.01 (aggressive)")
    print("\n" + "=" * 80)
    
    experiments = [
        "adam_lr_0.0001_fast",
        "adam_lr_0.0002_fast",
        "adam_lr_0.0003_fast",
        "adam_lr_0.0005_fast",
        "adam_lr_0.0007_fast",
        "adam_lr_0.001_fast",
        "adam_lr_0.002_fast",
        "adam_lr_0.003_fast",
        "adam_lr_0.005_fast",
        "adam_lr_0.007_fast",
        "adam_lr_0.01_fast",
    ]
    
    print(f"\n📊 Running {len(experiments)} experiments...")
    print(f"⏱️  Estimated time: ~{len(experiments) * 2} minutes\n")
    
    output_dir = "./adam_lr_sweep_results"
    
    # Run all experiments
    results = run_multiple_experiments(experiments, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ ADAM LR SWEEP COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    
    # Print summary
    if results:
        print("\n📊 Adam Learning Rate Results (sorted by best loss):")
        print("-" * 80)
        sorted_results = sorted(
            results.items(), 
            key=lambda x: min(x[1]['history']['val_losses'])
        )
        print(f"{'Experiment':<30} {'LR':<10} {'Best Loss':<12} {'Final Loss':<12}")
        print("-" * 80)
        
        for i, (exp_name, data) in enumerate(sorted_results):
            lr = data['history']['learning_rates'][0] if data['history']['learning_rates'] else 0
            best_loss = min(data['history']['val_losses'])
            final_loss = data['metrics']['val_loss']
            status = "🏆" if i == 0 else ""
            print(f"{exp_name:<30} {lr:<10.5f} {best_loss:<12.4f} {final_loss:<12.4f} {status}")
        
        # Get winner
        winner = sorted_results[0]
        winner_lr = winner[1]['history']['learning_rates'][0]
        winner_loss = min(winner[1]['history']['val_losses'])
        
        print("\n" + "=" * 80)
        print(f"🎯 Optimal Adam LR: {winner_lr:.5f}")
        print(f"   Best Loss: {winner_loss:.4f}")
        print(f"   Experiment: {winner[0]}")
        print("\n💡 Next step:")
        print(f"   Run full validation: python run_experiments.py -e adam_lr_{winner_lr:.4f}")
        print("=" * 80)
    
if __name__ == "__main__":
    main()

