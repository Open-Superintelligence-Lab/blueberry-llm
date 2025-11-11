"""
Quick test script to verify all attention mechanisms work correctly
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp10_attention_mechanism_ablation.config import ALL_CONFIGS
from experiments.exp10_attention_mechanism_ablation.models import ModelWrapper


def test_mechanism(name: str, config_fn):
    """Test a single attention mechanism"""
    print(f"\nTesting {name}...")
    
    try:
        # Create config
        config = config_fn()
        config.max_steps = 10  # Quick test
        
        # Create model
        model = ModelWrapper(config)
        
        # Create dummy input
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        
        # Check outputs
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
        assert outputs["loss"] is not None
        assert not torch.isnan(outputs["loss"])
        
        # Count parameters
        params = model.model.count_parameters()
        
        print(f"  ✓ {name} works!")
        print(f"    Total params: {params['total']:,}")
        print(f"    Attention params: {params['attention']:,}")
        print(f"    Loss: {outputs['loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ {name} failed: {e}")
        return False


def main():
    """Test all attention mechanisms"""
    print("=" * 80)
    print("Testing All Attention Mechanisms")
    print("=" * 80)
    
    results = {}
    
    for name, config_fn in ALL_CONFIGS.items():
        # Skip linear attention if FLA not installed
        if name == "linear":
            try:
                import fla
                success = test_mechanism(name, config_fn)
            except ImportError:
                print(f"\n{name}: Skipped (FLA not installed)")
                success = None
        else:
            success = test_mechanism(name, config_fn)
        
        results[name] = success
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"\nTotal: {len(results)}")
    print(f"  ✓ Passed: {passed}")
    print(f"  ✗ Failed: {failed}")
    print(f"  - Skipped: {skipped}")
    
    if failed > 0:
        print("\nFailed mechanisms:")
        for name, success in results.items():
            if success is False:
                print(f"  • {name}")
        return 1
    else:
        print("\n✓ All mechanisms working!")
        return 0


if __name__ == "__main__":
    exit(main())

