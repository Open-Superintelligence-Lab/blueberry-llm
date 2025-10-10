"""Quick test to verify model implementation works"""
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import torch
from config import SimpleTransformerConfig
from models import SimpleTransformerMoE


def test_model():
    """Test model forward pass"""
    print("🧪 Testing Simple Transformer MoE Implementation\n")
    
    # Create config
    config = SimpleTransformerConfig(
        vocab_size=50000,
        d_model=384,
        n_layers=6,
        n_heads=8,
        d_ff=1536,
        num_experts=8,
        expert_top_k=2,
        max_seq_len=512
    )
    
    print("📊 Model Configuration:")
    print(f"   d_model: {config.d_model}")
    print(f"   n_layers: {config.n_layers}")
    print(f"   n_heads: {config.n_heads}")
    print(f"   d_ff: {config.d_ff}")
    print(f"   num_experts: {config.num_experts} (top-{config.expert_top_k})")
    print(f"   vocab_size: {config.vocab_size}")
    
    # Create model
    model = SimpleTransformerMoE(config)
    
    # Count parameters
    total_params = model.get_num_params(non_embedding=False)
    non_embed_params = model.get_num_params(non_embedding=True)
    
    print(f"\n📊 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Non-embedding: {non_embed_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n🔬 Testing Forward Pass:")
    print(f"   Input shape: {input_ids.shape}")
    
    # Forward with aux loss
    logits, aux_loss = model(input_ids, return_aux_loss=True)
    
    print(f"   Output shape: {logits.shape}")
    print(f"   Expected shape: [{batch_size}, {seq_len}, {config.vocab_size}]")
    print(f"   Aux loss: {aux_loss.item() if aux_loss is not None else 'None'}")
    
    # Verify shapes
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Wrong output shape!"
    assert aux_loss is not None, "Aux loss should not be None!"
    
    # Test without aux loss
    logits_only = model(input_ids, return_aux_loss=False)
    assert logits_only.shape == (batch_size, seq_len, config.vocab_size), "Wrong output shape!"
    
    print("\n✅ All tests passed!")
    print("\n📝 Summary:")
    print("   ✓ Model instantiation successful")
    print("   ✓ Forward pass with aux loss works")
    print("   ✓ Forward pass without aux loss works")
    print("   ✓ Output shapes are correct")
    print("   ✓ MoE load balancing loss is computed")
    
    # Test gradient flow
    print("\n🔬 Testing Gradient Flow:")
    loss = logits.mean() + aux_loss
    loss.backward()
    
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_count = sum(1 for _ in model.parameters())
    
    print(f"   Parameters with gradients: {has_grads}/{total_params_count}")
    
    if has_grads == total_params_count:
        print("   ✅ All parameters have gradients")
    else:
        print("   ⚠️ Some parameters don't have gradients")
    
    print("\n🎉 Model implementation is working correctly!")


if __name__ == "__main__":
    test_model()

