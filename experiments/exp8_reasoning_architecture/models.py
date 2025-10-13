"""
Reasoning Model Architecture
Based on FLA's Gated DeltaNet with exp7's winning hybrid configuration

Uses the winning Hybrid Sparse 17% architecture:
- Attention at layers [5, 11]
- DeltaNet for remaining layers

Optional: Wrap with recursive reasoning for iterative refinement
"""

import torch
import torch.nn as nn
from typing import Optional

# Use FLA's Gated DeltaNet implementation (supports hybrid with attention)
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

# Import recursive reasoning wrapper
from experiments.exp8_reasoning_architecture.recursive_reasoning import (
    create_recursive_reasoning_model,
    RecursiveCarryState
)


def create_reasoning_model(config):
    """
    Create a reasoning model using FLA's Gated DeltaNet with hybrid attention
    Based on exp7's winning Hybrid Sparse 17% architecture
    
    Args:
        config: ExperimentConfig with model parameters
    
    Returns:
        GatedDeltaNetForCausalLM model instance configured for reasoning
    """
    # Convert ExperimentConfig to GatedDeltaNetConfig
    deltanet_config = GatedDeltaNetConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        
        # DeltaNet specific parameters
        expand_k=getattr(config, 'expand_k', 1.0),
        expand_v=getattr(config, 'expand_v', 1.0),
        
        # MLP configuration
        hidden_ratio=getattr(config, 'hidden_ratio', 4),
        intermediate_size=config.intermediate_size if hasattr(config, 'intermediate_size') else None,
        
        # Regularization
        norm_eps=config.rms_norm_eps,
        
        # Optimization flags
        fuse_norm=True,
        fuse_cross_entropy=True,
        
        # Standard configs
        max_position_embeddings=config.max_position_embeddings,
        initializer_range=0.02,
        use_cache=True,
        
        # Tokenizer configs
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    
    # Add hybrid attention configuration (WINNER from exp7)
    if hasattr(config, 'attn_config') and config.attn_config is not None:
        deltanet_config.attn = {
            'layers': config.attn_config.get('layers', []),
            'num_heads': config.num_attention_heads,
            'num_kv_heads': config.attn_config.get('num_kv_heads', config.num_attention_heads),
            'window_size': config.attn_config.get('window_size', 2048),
            'qkv_bias': config.attn_config.get('qkv_bias', False),
            'rope_theta': config.attn_config.get('rope_theta', 10000.0),
        }
    
    # Create model
    model = GatedDeltaNetForCausalLM(deltanet_config)
    
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
    Verify that the model has the expected reasoning architecture
    Returns dict with architecture info and verification status
    """
    model_class_name = model.__class__.__name__
    is_deltanet = 'DeltaNet' in model_class_name
    
    # Count layers
    actual_num_layers = len(model.model.layers)
    expected_num_layers = config.num_hidden_layers
    
    # Verify layer structure and identify layer types
    layer_types = []
    all_layers_valid = True
    deltanet_layers = []
    attention_layers = []
    
    for i, layer in enumerate(model.model.layers):
        has_mlp = hasattr(layer, 'mlp')
        layer_type = layer.__class__.__name__
        
        mixer_type = None
        if hasattr(layer, 'attn'):
            mixer = layer.attn
            mixer_class = mixer.__class__.__name__
            if 'DeltaNet' in mixer_class:
                mixer_type = 'GatedDeltaNet'
                deltanet_layers.append(i)
            elif 'Attention' in mixer_class:
                mixer_type = 'Attention'
                attention_layers.append(i)
            else:
                mixer_type = mixer_class
        elif hasattr(layer, 'mixer'):
            mixer = layer.mixer
            mixer_class = mixer.__class__.__name__
            if 'DeltaNet' in mixer_class:
                mixer_type = 'DeltaNet'
                deltanet_layers.append(i)
            elif 'Attention' in mixer_class:
                mixer_type = 'Attention'
                attention_layers.append(i)
            else:
                mixer_type = mixer_class
        
        layer_info = {
            'idx': i,
            'type': layer_type,
            'mixer_type': mixer_type,
            'has_mlp': has_mlp,
            'valid': mixer_type is not None and has_mlp,
        }
        layer_types.append(layer_info)
        
        if not (mixer_type is not None and has_mlp):
            all_layers_valid = False
    
    # Check if model is hybrid
    is_hybrid = len(attention_layers) > 0 and len(deltanet_layers) > 0
    
    # Verify it matches exp7 winner configuration
    expected_attention_layers = config.attn_config.get('layers', []) if config.attn_config else []
    is_winner_config = set(attention_layers) == set(expected_attention_layers)
    
    verification_passed = (
        is_deltanet and
        actual_num_layers == expected_num_layers and
        all_layers_valid and
        is_hybrid and
        is_winner_config
    )
    
    info = {
        'verification_passed': verification_passed,
        'model_type': model_class_name,
        'is_deltanet': is_deltanet,
        'is_hybrid': is_hybrid,
        'is_winner_config': is_winner_config,
        'num_layers': actual_num_layers,
        'expected_num_layers': expected_num_layers,
        'layers_match': actual_num_layers == expected_num_layers,
        'all_layers_valid': all_layers_valid,
        'deltanet_layers': deltanet_layers,
        'attention_layers': attention_layers,
        'expected_attention_layers': expected_attention_layers,
        'layer_types': layer_types,
    }
    
    return info


class ReasoningModelWrapper(nn.Module):
    """
    Wrapper for reasoning model with convenience methods
    Built on exp7's winning Hybrid Sparse 17% architecture
    
    Can optionally wrap with recursive reasoning for iterative refinement
    """
    def __init__(self, config, use_recursive=False):
        super().__init__()
        self.config = config
        self.use_recursive = use_recursive
        
        # Create base model (exp7 winner architecture)
        self.base_model = create_reasoning_model(config)
        
        # Optionally wrap with recursive reasoning
        if use_recursive:
            recursive_config = getattr(config, 'recursive', {
                'H_cycles': 3,
                'L_cycles': 3,
                'halt_max_steps': 5,
                'halt_exploration_prob': 0.1,
                'use_act': True
            })
            self.model = create_recursive_reasoning_model(self.base_model, recursive_config)
        else:
            self.model = self.base_model
        
        self.param_info = count_parameters(self.model)
        
        # Architecture verification (only for base model)
        if not use_recursive:
            self.arch_info = verify_model_architecture(self.model, config)
        else:
            # For recursive, verify the base model
            self.arch_info = verify_model_architecture(self.base_model, config)
            self.arch_info['is_recursive'] = True
    
    def forward(self, input_ids, attention_mask=None, labels=None, carry=None, **kwargs):
        """
        Forward pass through the model
        
        For recursive models, accepts optional carry state
        """
        if self.use_recursive:
            # Recursive forward with carry state
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                carry=carry,
                **kwargs
            )
        else:
            # Standard forward
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
                'attention_layers': self.config.attn_config.get('layers', []) if self.config.attn_config else [],
            }
        }
    
    def print_info(self):
        """Print model information"""
        info = self.get_info()
        
        print("="*70)
        print("Reasoning Architecture Model (Exp8)")
        print("Based on Exp7 Winner: Hybrid Sparse 17%")
        if self.use_recursive:
            print("Mode: RECURSIVE REASONING")
        else:
            print("Mode: BASELINE")
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
        print(f"  Is Hybrid: {'✓' if arch['is_hybrid'] else '✗'}")
        print(f"  Winner Config: {'✓' if arch.get('is_winner_config', False) else '✗'}")
        
        if arch.get('is_hybrid', False):
            print(f"\n  🏆 HYBRID MODE (Exp7 Winner Configuration)")
            print(f"  DeltaNet Layers: {len(arch['deltanet_layers'])} layers -> {arch['deltanet_layers']}")
            print(f"  Attention Layers: {len(arch['attention_layers'])} layers -> {arch['attention_layers']}")
            print(f"  Expected Attention: {arch.get('expected_attention_layers', [])}")
            print(f"  Attention %: {len(arch['attention_layers']) / arch['num_layers'] * 100:.1f}%")
        
        print(f"  Number of layers: {arch['num_layers']} (expected: {arch['expected_num_layers']})")
        
        print("\nLayer Details:")
        for layer_info in arch['layer_types']:
            status = "✓" if layer_info['valid'] else "✗"
            mixer = layer_info.get('mixer_type', 'unknown')
            mlp_status = "mlp" if layer_info['has_mlp'] else "no-mlp"
            layer_marker = "🎯" if layer_info['idx'] in arch.get('attention_layers', []) else "  "
            print(f"  {layer_marker} Layer {layer_info['idx']:2d}: {mixer:15s} + {mlp_status:6s} {status}")
        
        print("\nOptimizations:")
        print("  ✓ FLA Triton kernels")
        print("  ✓ Fused RMSNorm")
        print("  ✓ Fused cross entropy")
        print("  ✓ Hybrid DeltaNet + Attention")
        print("  ✓ Strategic attention placement (layers 5, 11)")
        
        print("="*70)

