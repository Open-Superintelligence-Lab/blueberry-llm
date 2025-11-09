"""
Integration example for Blueberry LLM with Manifold Muon.

This shows how to tag parameters and create the composed optimizer
for the MoE LLM architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from models.moe_llm import MoEMinimalLLM
from configs.moe_config import MoEModelConfig
from .composed_optimizer import ComposedOptimizer, compute_lr_scale


def tag_model_parameters(model: MoEMinimalLLM, config: MoEModelConfig):
    """
    Tag each parameter in the model with its optimizer group and learning rate scale.
    
    This function modifies parameters in-place by adding attributes:
    - optimizer_group: 'stiefel', 'hypersphere', or 'adamw'
    - lr_scale: learning rate multiplier based on layer depth and dimensions
    
    Args:
        model: MoE LLM model to tag
        config: Model configuration
    """
    n_layers = config.n_layers
    
    # 1. Token embeddings - use hypersphere (per embedding vector)
    model.token_embedding.weight.optimizer_group = 'hypersphere'
    model.token_embedding.weight.lr_scale = 1.0
    model.token_embedding.weight.hypersphere_mode = 'per_vector'
    
    # 2. Transformer blocks
    for layer_idx, block in enumerate(model.transformer_blocks):
        
        # 2a. Attention QKV projection - Stiefel manifold
        qkv_weight = block.attention.qkv.weight
        qkv_weight.optimizer_group = 'stiefel'
        qkv_weight.layer_idx = layer_idx
        
        # QKV has shape (d_model * 3, d_model)
        fan_in = config.d_model
        fan_out = config.d_model * 3
        qkv_weight.lr_scale = compute_lr_scale(layer_idx, n_layers, fan_in, fan_out)
        
        # 2b. Attention output projection - Stiefel manifold
        w_o_weight = block.attention.w_o.weight
        w_o_weight.optimizer_group = 'stiefel'
        w_o_weight.layer_idx = layer_idx
        
        # Output projection has shape (d_model, d_model)
        fan_in = config.d_model
        fan_out = config.d_model
        w_o_weight.lr_scale = compute_lr_scale(layer_idx, n_layers, fan_in, fan_out)
        
        # 2c. RMS Norm parameters - AdamW (unconstrained)
        block.norm1.weight.optimizer_group = 'adamw'
        block.norm1.weight.lr_scale = 10.0  # Higher LR for norms
        
        block.norm2.weight.optimizer_group = 'adamw'
        block.norm2.weight.lr_scale = 10.0
        
        # 2d. MoE Router gate - per-row hypersphere
        router_weight = block.feed_forward.router.gate.weight
        router_weight.optimizer_group = 'hypersphere'
        router_weight.lr_scale = 1.0
        router_weight.hypersphere_mode = 'per_row'
        
        # 2e. Expert networks
        for expert_idx, expert in enumerate(block.feed_forward.experts):
            
            # Expert up-projection (d_model -> d_ff)
            linear1_weight = expert.linear1.weight
            linear1_weight.optimizer_group = 'stiefel'
            linear1_weight.layer_idx = layer_idx
            
            fan_in = config.d_model
            fan_out = config.d_ff
            linear1_weight.lr_scale = compute_lr_scale(layer_idx, n_layers, fan_in, fan_out)
            
            # Expert down-projection (d_ff -> d_model)
            linear2_weight = expert.linear2.weight
            linear2_weight.optimizer_group = 'stiefel'
            linear2_weight.layer_idx = layer_idx
            
            fan_in = config.d_ff
            fan_out = config.d_model
            linear2_weight.lr_scale = compute_lr_scale(layer_idx, n_layers, fan_in, fan_out)
    
    # 3. Final RMS Norm
    model.norm.weight.optimizer_group = 'adamw'
    model.norm.weight.lr_scale = 10.0
    
    # 4. LM head (tied with embeddings, so skip)
    # The LM head shares weights with token_embedding, so it's already tagged


def create_manifold_optimizer(
    model: MoEMinimalLLM,
    config: MoEModelConfig,
    base_lr: float = 0.02,
    dual_steps: int = 10,
    dual_lr: float = 0.1,
) -> ComposedOptimizer:
    """
    Create a composed manifold optimizer for the MoE LLM.
    
    Args:
        model: MoE LLM model
        config: Model configuration
        base_lr: Base learning rate
        dual_steps: Number of dual ascent steps for Stiefel Muon
        dual_lr: Learning rate for dual ascent
        
    Returns:
        ComposedOptimizer ready for training
    """
    # First, tag all parameters
    tag_model_parameters(model, config)
    
    # Collect parameters by optimizer group
    stiefel_params = []
    hypersphere_per_vector_params = []
    hypersphere_per_row_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        opt_group = getattr(param, 'optimizer_group', 'adamw')
        lr_scale = getattr(param, 'lr_scale', 1.0)
        
        if opt_group == 'stiefel':
            stiefel_params.append({
                'param': param,
                'name': name,
                'lr_scale': lr_scale,
            })
        
        elif opt_group == 'hypersphere':
            mode = getattr(param, 'hypersphere_mode', 'per_vector')
            if mode == 'per_vector':
                hypersphere_per_vector_params.append({
                    'param': param,
                    'name': name,
                    'lr_scale': lr_scale,
                })
            elif mode == 'per_row':
                hypersphere_per_row_params.append({
                    'param': param,
                    'name': name,
                    'lr_scale': lr_scale,
                })
        
        elif opt_group == 'adamw':
            adamw_params.append({
                'param': param,
                'name': name,
                'lr_scale': lr_scale,
            })
    
    # Create parameter groups for ComposedOptimizer
    param_groups = []
    
    if stiefel_params:
        print(f"Stiefel Muon: {len(stiefel_params)} parameter groups")
        # Group by learning rate scale for efficiency
        lr_scales = set(p['lr_scale'] for p in stiefel_params)
        for lr_scale in sorted(lr_scales):
            params_at_scale = [p['param'] for p in stiefel_params if p['lr_scale'] == lr_scale]
            param_groups.append({
                'params': params_at_scale,
                'optimizer': 'stiefel_muon',
                'lr_scale': lr_scale,
                'dual_steps': dual_steps,
                'dual_lr': dual_lr,
                'ns_steps': 5,
                'momentum': 0.95,
                'nesterov': True,
            })
    
    if hypersphere_per_vector_params:
        print(f"Hypersphere Muon (per-vector): {len(hypersphere_per_vector_params)} parameters")
        param_groups.append({
            'params': [p['param'] for p in hypersphere_per_vector_params],
            'optimizer': 'hypersphere_muon',
            'lr_scale': 1.0,
            'mode': 'per_vector',
            'momentum': 0.95,
        })
    
    if hypersphere_per_row_params:
        print(f"Hypersphere Muon (per-row): {len(hypersphere_per_row_params)} parameters")
        param_groups.append({
            'params': [p['param'] for p in hypersphere_per_row_params],
            'optimizer': 'hypersphere_muon',
            'lr_scale': 1.0,
            'mode': 'per_row',
            'momentum': 0.95,
        })
    
    if adamw_params:
        print(f"AdamW: {len(adamw_params)} parameters")
        param_groups.append({
            'params': [p['param'] for p in adamw_params],
            'optimizer': 'adamw',
            'lr_scale': 10.0,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999),
        })
    
    return ComposedOptimizer(param_groups, base_lr=base_lr)


def print_optimizer_summary(optimizer: ComposedOptimizer):
    """Print a summary of the optimizer configuration."""
    print("\n" + "=" * 60)
    print("OPTIMIZER SUMMARY")
    print("=" * 60)
    
    total_params = 0
    for i, opt_dict in enumerate(optimizer.optimizers):
        opt_type = opt_dict['type']
        lr_scale = opt_dict['lr_scale']
        opt = opt_dict['optimizer']
        
        # Count parameters
        n_params = sum(p.numel() for group in opt.param_groups for p in group['params'])
        total_params += n_params
        
        effective_lr = optimizer.base_lr * lr_scale
        
        print(f"\nOptimizer {i+1}: {opt_type}")
        print(f"  Parameters: {n_params:,}")
        print(f"  LR Scale: {lr_scale:.4f}")
        print(f"  Effective LR: {effective_lr:.6f}")
        
        if opt_type == 'stiefel_muon' and hasattr(opt, 'param_groups'):
            print(f"  Dual Steps: {opt.param_groups[0].get('dual_steps', 'N/A')}")
            print(f"  Newton-Schulz Steps: {opt.param_groups[0].get('ns_steps', 'N/A')}")
        
        elif opt_type == 'hypersphere_muon' and hasattr(opt, 'param_groups'):
            print(f"  Mode: {opt.param_groups[0].get('mode', 'N/A')}")
    
    print(f"\nTotal parameters: {total_params:,}")
    print("=" * 60 + "\n")


# Example usage
if __name__ == "__main__":
    print("Manifold Muon Integration Example")
    print("=" * 60)
    
    # Create a small model for demonstration
    config = MoEModelConfig(
        vocab_size=10000,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4,
        max_seq_len=512,
        num_experts=4,
        expert_top_k=2,
        dropout=0.1,
    )
    
    print("\nCreating model...")
    model = MoEMinimalLLM(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nCreating manifold optimizer...")
    optimizer = create_manifold_optimizer(
        model,
        config,
        base_lr=0.02,
        dual_steps=10,
        dual_lr=0.1,
    )
    
    print_optimizer_summary(optimizer)
    
    # Demonstrate a training step
    print("Running sample training step...")
    
    # Dummy data
    batch_size = 8
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    optimizer.zero_grad()
    logits, aux_loss = model(input_ids)
    
    # Dummy loss
    target = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        target.view(-1)
    )
    
    if aux_loss is not None:
        loss = loss + aux_loss
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    # Check constraints
    print("\nChecking manifold constraints...")
    constraint_stats = optimizer.check_constraints()
    print(f"Total violations: {constraint_stats['total_violations']}")
    
    for opt_type, stats in constraint_stats['by_type'].items():
        print(f"\n{opt_type}:")
        print(f"  Max violation: {stats.get('max_violation', 0):.6f}")
        print(f"  Mean violation: {stats.get('mean_violation', 0):.6f}")
        print(f"  Num violations: {stats.get('num_violations', 0)}")
    
    print("\n✓ Integration example completed successfully!")




