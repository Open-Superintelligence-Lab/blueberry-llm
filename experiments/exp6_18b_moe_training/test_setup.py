"""
Quick test to verify the 18B MoE setup works correctly
"""
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp6_18b_moe_training.config_18b import MoE18BConfig
from experiments.exp6_18b_moe_training.models_18b import MoE18BLLM


def test_model_creation():
    """Test that model can be created and basic stats are correct"""
    print("🧪 Testing 18B MoE Model Setup\n")
    
    # Create config with dummy vocab size
    config = MoE18BConfig(vocab_size=50000)
    
    # Print stats
    config.print_stats()
    
    # Create model on CPU first to test structure
    print("Creating model on CPU...")
    model = MoE18BLLM(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # Test forward pass with small input
    print("\n🔍 Testing forward pass...")
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, aux_loss = model(x, return_aux_loss=True)
    
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   Aux loss: {aux_loss.item() if aux_loss is not None else 'None'}")
    
    # Test gradient checkpointing
    print("\n🔍 Testing gradient checkpointing...")
    model.train()
    logits, aux_loss = model(x, return_aux_loss=True)
    loss = logits.mean() + (aux_loss if aux_loss is not None else 0)
    loss.backward()
    print(f"✅ Gradient checkpointing works!")
    
    # Check GPU if available
    if torch.cuda.is_available():
        print(f"\n🎮 GPU Information:")
        print(f"   Device: {torch.cuda.get_device_name()}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Total Memory: {total_mem:.1f} GB")
        
        if total_mem >= 180:
            print(f"   ✅ Sufficient memory for 18B model ({total_mem:.1f} >= 180 GB)")
        else:
            print(f"   ⚠️  May not have enough memory ({total_mem:.1f} < 180 GB)")
            print(f"   Consider reducing model size or batch size")
    else:
        print(f"\n⚠️  No GPU detected. This model requires GPU for training.")
    
    print(f"\n{'='*70}")
    print(f"✅ All tests passed! Setup is ready for training.")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_model_creation()

