#!/usr/bin/env python3
"""
Run complete Muon experiment suite with optimal settings (LR=0.07, momentum=0.9)

This script runs all experiments to explore different aspects of Muon optimizer
while keeping LR and momentum at their optimal values.

Total experiments: 15
Estimated time: ~30 minutes
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
    print("🚀 OPTIMAL MUON EXPERIMENT SUITE")
    print("=" * 80)
    print("\nUsing discovered optimal settings:")
    print("  ✓ Learning Rate: 0.07 (Muon), 0.007 (AdamW)")
    print("  ✓ Momentum: 0.9")
    print("\nThis suite will test:")
    print("  1. Comparison with Adam baseline")
    print("  2. Newton-Schulz iteration steps (speed vs accuracy)")
    print("  3. Pushing LR higher (0.09, 0.1)")
    print("  4. Fine-tuning momentum (0.85, 0.92)")
    print("  5. Nesterov momentum ablation")
    print("  6. Weight decay variations")
    print("  7. Warmup sensitivity")
    print("  8. LR schedule comparisons")
    print("\n" + "=" * 80)
    
    experiments = [
        # Core comparison
        "muon_optimal",           # Optimal settings baseline
        "adam_baseline",          # Compare against Adam
        
        # Newton-Schulz steps (computational trade-off)
        "muon_optimal_ns3",       # Fewer steps = faster
        "muon_optimal_ns10",      # More steps = more accurate
        
        # Push LR higher
        "muon_lr_0.09_momentum_0.9",  # Test 0.09
        "muon_lr_0.1_momentum_0.9",   # Test 0.1
        
        # Fine-tune momentum around 0.9
        "muon_momentum_0.85",     # Lower momentum
        "muon_momentum_0.92",     # Higher momentum
        
        # Nesterov momentum
        "muon_optimal_no_nesterov",  # Is Nesterov needed?
        
        # Weight decay
        "muon_optimal_wd_0.05",   # Lower weight decay
        "muon_optimal_wd_0.2",    # Higher weight decay
        
        # Warmup variations (with optimal settings)
        "muon_no_warmup",         # No warmup
        "muon_warmup_0.1",        # 10% warmup
        
        # LR schedule variations (with optimal settings)
        "muon_linear_decay",      # Linear decay
        "muon_step_decay",        # Step decay
    ]
    
    print(f"\n📊 Running {len(experiments)} experiments...")
    print(f"⏱️  Estimated time: ~{len(experiments) * 2} minutes\n")
    
    output_dir = "./optimal_muon_suite_results"
    
    # Run all experiments
    results = run_multiple_experiments(experiments, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ OPTIMAL MUON SUITE COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - comparison_plot.png")
    print(f"  - comparison_summary.json")
    print(f"  - Individual experiment directories")
    
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
        for exp_name, data in sorted_results[:5]:  # Top 5
            status = "🏆" if exp_name == sorted_results[0][0] else ""
            print(f"{exp_name:<35} {data['metrics']['val_loss']:<12.4f} "
                  f"{data['time_minutes']:<12.2f} {status}")
        
        print("\n🎯 Best performer: " + sorted_results[0][0])
        print(f"   Loss: {sorted_results[0][1]['metrics']['val_loss']:.4f}")
        
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

