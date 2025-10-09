#!/usr/bin/env python3
"""
Test Implementation for Experiment 4

This script tests the various optimization components to ensure they work correctly
before running the full experiment.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directories to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

from optimized_indexers import (
    OptimizedLightningIndexer, MinimalLightningIndexer, UltraLightIndexer,
    create_optimized_indexer, benchmark_indexer
)
from efficient_patterns import (
    LocalGlobalPattern, SlidingWindowPattern, HierarchicalPattern,
    create_efficient_pattern, analyze_pattern_coverage
)
from adaptive_selection import (
    AdaptiveKSelector, FixedRatioSelector, ProgressiveSelector,
    create_adaptive_selector, analyze_k_distribution
)
from quantization_utils import (
    FP16IndexerWrapper, MixedPrecisionIndexer, DynamicQuantizedIndexer,
    QuantizationBenchmark, create_quantized_indexer
)
from exp4_models import create_optimized_model, compare_model_sizes


def test_optimized_indexers():
    """Test optimized indexer variants"""
    print("Testing Optimized Indexers...")
    
    d_model = 256
    seq_len = 128
    batch_size = 4
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test different indexer variants
    indexers = {
        'baseline': OptimizedLightningIndexer(d_model, indexer_heads=4, indexer_dim=64),
        'optimized': OptimizedLightningIndexer(d_model, indexer_heads=2, indexer_dim=32),
        'minimal': MinimalLightningIndexer(d_model, indexer_dim=16),
        'ultra_light': UltraLightIndexer(d_model, indexer_dim=8),
    }
    
    for name, indexer in indexers.items():
        print(f"  Testing {name}...")
        
        # Test forward pass
        output = indexer(x)
        assert output.shape == (batch_size, seq_len, seq_len), f"Wrong output shape for {name}"
        
        # Test parameter count
        params = sum(p.numel() for p in indexer.parameters())
        print(f"    Parameters: {params:,}")
        
        # Test benchmark
        benchmark_results = benchmark_indexer(indexer, x, num_runs=10)
        print(f"    Avg time: {benchmark_results['avg_time_ms']:.2f}ms")
        print(f"    Memory: {benchmark_results['memory_allocated_mb']:.1f}MB")
    
    print("  ✓ Optimized indexers working correctly\n")


def test_efficient_patterns():
    """Test efficient attention patterns"""
    print("Testing Efficient Patterns...")
    
    d_model = 256
    seq_len = 128
    batch_size = 4
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test different patterns
    patterns = {
        'local_global': LocalGlobalPattern(local_window=32, global_k=64, d_model=d_model),
        'sliding_window': SlidingWindowPattern(window_size=64, stride=32, d_model=d_model),
        'hierarchical': HierarchicalPattern(local_window=16, medium_window=64, global_k=32, d_model=d_model),
    }
    
    for name, pattern in patterns.items():
        print(f"  Testing {name}...")
        
        # Test forward pass
        mask = pattern(x)
        assert mask.shape == (batch_size, 1, seq_len, seq_len), f"Wrong mask shape for {name}"
        
        # Test coverage analysis
        coverage = analyze_pattern_coverage(pattern, seq_len)
        print(f"    Coverage: {coverage['total_coverage']:.1%}")
        print(f"    Avg per query: {coverage['avg_per_query']:.1f}")
    
    print("  ✓ Efficient patterns working correctly\n")


def test_adaptive_selection():
    """Test adaptive selection strategies"""
    print("Testing Adaptive Selection...")
    
    d_model = 256
    seq_len = 128
    batch_size = 4
    
    # Create test input and index scores
    x = torch.randn(batch_size, seq_len, d_model)
    index_scores = torch.randn(batch_size, seq_len, seq_len)
    
    # Test different selectors
    selectors = {
        'fixed_ratio_25': FixedRatioSelector(ratio=0.25),
        'fixed_ratio_50': FixedRatioSelector(ratio=0.5),
        'progressive': ProgressiveSelector(start_k=16, end_k=64, max_steps=100),
        'adaptive': AdaptiveKSelector(base_k=64, adaptation_strategy='entropy', d_model=d_model),
    }
    
    for name, selector in selectors.items():
        print(f"  Testing {name}...")
        
        # Test forward pass
        mask, k_values = selector(x, index_scores)
        assert mask.shape == (batch_size, seq_len, seq_len), f"Wrong mask shape for {name}"
        assert k_values.shape == (batch_size, seq_len), f"Wrong k_values shape for {name}"
        
        # Test k distribution analysis
        k_stats = analyze_k_distribution(selector, x, index_scores)
        print(f"    Mean k: {k_stats['mean_k']:.1f}")
        print(f"    K range: {k_stats['min_k']}-{k_stats['max_k']}")
    
    print("  ✓ Adaptive selection working correctly\n")


def test_quantization():
    """Test quantization utilities"""
    print("Testing Quantization...")
    
    d_model = 256
    seq_len = 128
    batch_size = 4
    
    # Create test input and original indexer
    x = torch.randn(batch_size, seq_len, d_model)
    original_indexer = OptimizedLightningIndexer(d_model, indexer_heads=2, indexer_dim=32)
    
    # Test different quantization strategies
    quantized_indexers = {
        'fp16': FP16IndexerWrapper(original_indexer),
        'mixed_precision': MixedPrecisionIndexer(original_indexer),
        'int8': DynamicQuantizedIndexer(original_indexer, quantization_bits=8, calibration_steps=10),
    }
    
    for name, quantized_indexer in quantized_indexers.items():
        print(f"  Testing {name}...")
        
        # Test forward pass
        original_output = original_indexer(x)
        quantized_output = quantized_indexer(x)
        
        assert quantized_output.shape == original_output.shape, f"Shape mismatch for {name}"
        
        # Test error analysis (for non-calibration cases)
        if name != 'int8':
            error = torch.mean(torch.abs(original_output - quantized_output)).item()
            print(f"    Mean absolute error: {error:.6f}")
    
    print("  ✓ Quantization working correctly\n")


def test_model_creation():
    """Test model creation and parameter counting"""
    print("Testing Model Creation...")
    
    # Test different model configurations
    strategies = ['baseline', 'optimized', 'minimal', 'ultra_light']
    
    for strategy in strategies:
        print(f"  Testing {strategy} model...")
        
        # Create model
        model = create_optimized_model(
            optimization_strategy=strategy,
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            max_seq_len=256
        )
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 64))
        logits = model(input_ids)
        assert logits.shape == (2, 64, 1000), f"Wrong logits shape for {strategy}"
        
        # Test parameter counting
        param_counts = model.get_parameter_counts()
        print(f"    Total params: {param_counts['total']:,}")
        print(f"    Indexer params: {param_counts['indexer']:,}")
        print(f"    Indexer ratio: {param_counts['indexer']/param_counts['total']*100:.1f}%")
        
        del model  # Clean up memory
    
    print("  ✓ Model creation working correctly\n")


def test_model_size_comparison():
    """Test model size comparison utility"""
    print("Testing Model Size Comparison...")
    
    strategies = ['baseline', 'optimized', 'minimal']
    
    size_comparison = compare_model_sizes(
        strategies=strategies,
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2
    )
    
    print("  Model size comparison:")
    for strategy, counts in size_comparison.items():
        print(f"    {strategy}:")
        print(f"      Total: {counts['total']:,}")
        print(f"      Indexer: {counts['indexer']:,}")
        print(f"      Indexer ratio: {counts['indexer']/counts['total']*100:.1f}%")
    
    print("  ✓ Model size comparison working correctly\n")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running Experiment 4 Implementation Tests")
    print("="*60)
    
    try:
        test_optimized_indexers()
        test_efficient_patterns()
        test_adaptive_selection()
        test_quantization()
        test_model_creation()
        test_model_size_comparison()
        
        print("="*60)
        print("✅ All tests passed! Implementation is ready for Experiment 4.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
