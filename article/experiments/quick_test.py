#!/usr/bin/env python3
"""
Quick 10-step test to verify all experiments work before running full experiments
"""

import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from amp_experiment_runner import AMPExperimentRunner

def main():
    """Run quick test of all experiment configurations"""
    print("🧪 QUICK TEST: Running 10 steps for each experiment configuration")
    print("=" * 60)
    
    # Run in test mode
    runner = AMPExperimentRunner(test_mode=True)
    results = runner.run_all_experiments()
    
    # Save test results
    test_file = runner.save_results("quick_test_results.json")
    
    # Print detailed summary
    print("\n📊 DETAILED TEST RESULTS:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        status = "✅" if result.success else "❌"
        print(f"{status} Test {i:2d}: {result.config.d_model:3d}d × {result.config.n_layers}L, "
              f"batch={result.config.batch_size:2d}, AMP={str(result.config.use_amp):5s} "
              f"→ {result.tokens_per_second:6.0f} tok/s, {result.peak_memory_mb:6.0f} MB")
        
        if not result.success:
            print(f"    Error: {result.error_message}")
    
    # Check if all tests passed
    successful = sum(1 for r in results if r.success)
    total = len(results)
    
    print(f"\n🎯 TEST SUMMARY:")
    print(f"   Passed: {successful}/{total}")
    
    if successful == total:
        print("   🎉 All tests passed! Ready for full experiments.")
        print("   Run: python amp_experiment_runner.py --full")
    else:
        print("   ⚠️  Some tests failed. Check errors above before running full experiments.")
    
    return successful == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
