"""
Gated DeltaNet Model for Training using FLA
Using FLA's optimized DeltaNet implementation with Triton kernels
"""

import torch
import torch.nn as nn
from typing import Optional

# Use FLA's DeltaNet implementation
from fla.models import DeltaNetConfig, DeltaNetForCausalLM


def create_gated_deltanet_model(config):
    """
    Create a DeltaNet model using FLA's optimized implementation
    
    Args:
        config: ExperimentConfig with model parameters
    
    Returns:
        DeltaNetForCausalLM model instance
    """
    # Convert ExperimentConfig to DeltaNetConfig
    deltanet_config = DeltaNetConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        
        # DeltaNet specific parameters
        expand_k=getattr(config, 'expand_k', 1.0),  # Key expansion ratio
        expand_v=getattr(config, 'expand_v', 1.0),  # Value expansion ratio
        
        # MLP configuration
        hidden_ratio=getattr(config, 'hidden_ratio', 4),  # MLP expansion ratio
        intermediate_size=config.intermediate_size if hasattr(config, 'intermediate_size') else None,
        
        # Regularization
        norm_eps=config.rms_norm_eps,
        
        # Optimization flags
        fuse_norm=True,  # Use fused normalization
        fuse_cross_entropy=True,  # Use fused cross entropy
        
        # Standard configs
        max_position_embeddings=config.max_position_embeddings,
        initializer_range=0.02,
        use_cache=True,
        
        # Tokenizer configs
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    
    # Create model using FLA's implementation
    model = DeltaNetForCausalLM(deltanet_config)
    
    return model


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'total_millions': total_params / 1_000_000,
        'trainable_millions': trainable_params / 1_000_000,
    }


def verify_model_architecture(model, config):
    """
    Verify that the model has the expected architecture
    Returns dict with architecture info and verification status
    """
    # Detect actual model type
    model_class_name = model.__class__.__name__
    is_deltanet = 'DeltaNet' in model_class_name
    
    # Count layers
    actual_num_layers = len(model.model.layers)
    expected_num_layers = config.num_hidden_layers
    
    # Verify layer structure
    layer_types = []
    all_layers_valid = True
    
    for i, layer in enumerate(model.model.layers):
        has_attn = hasattr(layer, 'attn')
        has_mlp = hasattr(layer, 'mlp')
        layer_type = layer.__class__.__name__
        
        layer_info = {
            'idx': i,
            'type': layer_type,
            'has_attn': has_attn,
            'has_mlp': has_mlp,
            'valid': has_attn and has_mlp,
        }
        layer_types.append(layer_info)
        
        if not (has_attn and has_mlp):
            all_layers_valid = False
    
    # Overall verification status
    verification_passed = (
        is_deltanet and
        actual_num_layers == expected_num_layers and
        all_layers_valid
    )
    
    info = {
        'verification_passed': verification_passed,
        'model_type': model_class_name,
        'is_deltanet': is_deltanet,
        'num_layers': actual_num_layers,
        'expected_num_layers': expected_num_layers,
        'layers_match': actual_num_layers == expected_num_layers,
        'all_layers_valid': all_layers_valid,
        'layer_types': layer_types,
    }
    
    return info


class GatedDeltaNetWrapper(nn.Module):
    """
    Wrapper for FLA's DeltaNet model with convenience methods
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = create_gated_deltanet_model(config)
        self.param_info = count_parameters(self.model)
        self.arch_info = verify_model_architecture(self.model, config)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def get_info(self):
        """Get model information"""
        return {
            'parameters': self.param_info,
            'architecture': self.arch_info,
            'config': {
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_hidden_layers,
                'num_heads': self.config.num_attention_heads,
                'max_seq_len': self.config.max_position_embeddings,
                'vocab_size': self.config.vocab_size,
            }
        }
    
    def print_info(self):
        """Print model information"""
        info = self.get_info()
        
        print("="*70)
        print("Gated DeltaNet Model Information (FLA Implementation)")
        print("="*70)
        
        print("\nParameters:")
        print(f"  Total: {info['parameters']['total']:,} ({info['parameters']['total_millions']:.2f}M)")
        print(f"  Trainable: {info['parameters']['trainable']:,} ({info['parameters']['trainable_millions']:.2f}M)")
        
        print("\nConfiguration:")
        for key, value in info['config'].items():
            print(f"  {key}: {value}")
        
        print("\nArchitecture:")
        arch = info['architecture']
        verification_status = "✓ PASSED" if arch['verification_passed'] else "✗ FAILED"
        print(f"  Verification: {verification_status}")
        print(f"  Model Type: {arch['model_type']}")
        print(f"  Is DeltaNet: {'✓' if arch['is_deltanet'] else '✗'}")
        print(f"  Number of layers: {arch['num_layers']} (expected: {arch['expected_num_layers']})")
        
        print("\nLayer Types:")
        for layer_info in arch['layer_types']:
            status = "✓" if layer_info['valid'] else "✗"
            attn_status = "attn" if layer_info['has_attn'] else "no-attn"
            mlp_status = "mlp" if layer_info['has_mlp'] else "no-mlp"
            print(f"  Layer {layer_info['idx']}: {layer_info['type']} ({attn_status}, {mlp_status}) {status}")
        
        print("\nFLA Optimizations:")
        print("  ✓ Fused normalization (RMSNorm)")
        print("  ✓ Fused cross entropy loss")
        print("  ✓ Triton-optimized kernels")
        print("  ✓ Chunk-based DeltaNet computation")
        
        print("="*70)
