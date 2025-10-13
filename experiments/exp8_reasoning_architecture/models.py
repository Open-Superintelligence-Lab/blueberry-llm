"""
Reasoning Model Architecture
Based on MoE (Mixture of Experts) LLM

Uses MoE architecture with multiple expert networks for improved capacity
and specialized processing.

Optional: Wrap with recursive reasoning for iterative refinement
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
import os

# Add project root to path for imports
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Use MoE implementation
from models.moe_llm import MoEMinimalLLM
from configs.moe_config import MoEModelConfig

# Import recursive reasoning wrapper
from experiments.exp8_reasoning_architecture.recursive_reasoning import (
    create_recursive_reasoning_model,
    RecursiveCarryState
)


def create_reasoning_model(config):
    """
    Create a reasoning model using MoE (Mixture of Experts)
    
    Args:
        config: ExperimentConfig with model parameters
    
    Returns:
        MoEMinimalLLM model instance configured for reasoning
    """
    # Convert ExperimentConfig to MoEModelConfig
    moe_config = MoEModelConfig(
        d_model=config.hidden_size,
        n_heads=config.num_attention_heads,
        n_layers=config.num_hidden_layers,
        d_ff=config.intermediate_size if hasattr(config, 'intermediate_size') else config.hidden_size * 4,
        max_seq_len=config.max_position_embeddings,
        vocab_size=config.vocab_size,
        
        # MoE specific parameters
        num_experts=getattr(config, 'num_experts', 8),
        expert_top_k=getattr(config, 'expert_top_k', 2),
        
        # Regularization
        dropout=getattr(config, 'dropout', 0.1),
        
        # Training params (for reference, not used by model)
        batch_size=config.batch_size,
        max_steps=config.max_steps,
    )
    
    # Create MoE model
    model = MoEMinimalLLM(moe_config)
    
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
    Verify that the model has the expected MoE architecture
    Returns dict with architecture info and verification status
    """
    model_class_name = model.__class__.__name__
    is_moe = 'MoE' in model_class_name
    
    # Count layers
    actual_num_layers = len(model.transformer_blocks)
    expected_num_layers = config.num_hidden_layers
    
    # Verify layer structure and identify layer types
    layer_types = []
    all_layers_valid = True
    moe_layers = []
    
    for i, layer in enumerate(model.transformer_blocks):
        layer_type = layer.__class__.__name__
        has_attention = hasattr(layer, 'attention') or hasattr(layer, 'attn')
        has_moe = hasattr(layer, 'moe') or 'MoE' in layer_type
        
        layer_info = {
            'idx': i,
            'type': layer_type,
            'has_attention': has_attention,
            'has_moe': has_moe,
            'valid': has_attention and has_moe,
        }
        layer_types.append(layer_info)
        
        if has_moe:
            moe_layers.append(i)
        
        if not (has_attention and has_moe):
            all_layers_valid = False
    
    # Get MoE config
    num_experts = config.num_experts if hasattr(config, 'num_experts') else 8
    expert_top_k = config.expert_top_k if hasattr(config, 'expert_top_k') else 2
    
    verification_passed = (
        is_moe and
        actual_num_layers == expected_num_layers and
        all_layers_valid and
        len(moe_layers) == expected_num_layers
    )
    
    info = {
        'verification_passed': verification_passed,
        'model_type': model_class_name,
        'is_moe': is_moe,
        'num_layers': actual_num_layers,
        'expected_num_layers': expected_num_layers,
        'layers_match': actual_num_layers == expected_num_layers,
        'all_layers_valid': all_layers_valid,
        'moe_layers': moe_layers,
        'num_experts': num_experts,
        'expert_top_k': expert_top_k,
        'layer_types': layer_types,
    }
    
    return info


class ReasoningModelWrapper(nn.Module):
    """
    Wrapper for reasoning model with convenience methods
    Built on MoE (Mixture of Experts) architecture
    
    Can optionally wrap with recursive reasoning for iterative refinement
    """
    def __init__(self, config, use_recursive=False):
        super().__init__()
        self.config = config
        self.use_recursive = use_recursive
        
        # Create base model (MoE architecture)
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
        Returns dict with 'loss', 'logits', and optionally 'aux_loss'
        """
        if self.use_recursive:
            # Recursive forward with carry state
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                carry=carry,
                **kwargs
            )
        else:
            # Standard forward through MoE
            # MoE returns (logits, aux_loss) or just logits
            model_output = self.model(input_ids, return_aux_loss=True)
            
            # Handle tuple output
            if isinstance(model_output, tuple):
                logits, aux_loss = model_output
            else:
                logits = model_output
                aux_loss = None
            
            # Compute loss if labels provided
            loss = None
            if labels is not None:
                # Shift logits and labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Calculate cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Add auxiliary loss if present
                if aux_loss is not None:
                    loss = loss + aux_loss
            
            # Return dict format expected by trainer
            outputs = {
                'loss': loss,
                'logits': logits,
            }
            
            if aux_loss is not None:
                outputs['aux_loss'] = aux_loss
        
        return outputs
    
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
                'num_experts': getattr(self.config, 'num_experts', 8),
                'expert_top_k': getattr(self.config, 'expert_top_k', 2),
            }
        }
    
    def print_info(self):
        """Print model information"""
        info = self.get_info()
        
        print("="*70)
        print("Reasoning Architecture Model (Exp8)")
        print("Based on MoE (Mixture of Experts)")
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
        print(f"  Is MoE: {'✓' if arch.get('is_moe', False) else '✗'}")
        
        print(f"  Number of layers: {arch['num_layers']} (expected: {arch['expected_num_layers']})")
        print(f"  MoE Layers: {len(arch.get('moe_layers', []))} layers")
        print(f"  Experts per layer: {arch.get('num_experts', 'N/A')}")
        print(f"  Active experts (top-k): {arch.get('expert_top_k', 'N/A')}")
        
        print("\nLayer Details:")
        for layer_info in arch['layer_types']:
            status = "✓" if layer_info['valid'] else "✗"
            layer_type = layer_info.get('type', 'unknown')
            attn_status = "attn" if layer_info.get('has_attention', False) else "no-attn"
            moe_status = "moe" if layer_info.get('has_moe', False) else "no-moe"
            print(f"  Layer {layer_info['idx']:2d}: {layer_type:20s} [{attn_status:7s} + {moe_status:6s}] {status}")
        
        print("\nOptimizations:")
        print("  ✓ Mixture of Experts (MoE)")
        print("  ✓ Sparse expert activation")
        print("  ✓ RMSNorm layers")
        print("  ✓ Efficient transformer blocks")
        
        print("="*70)

