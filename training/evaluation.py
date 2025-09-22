"""
Model evaluation functions.

This module provides utilities for evaluating model performance
including loss computation, perplexity, and accuracy metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import math
from typing import Dict, Any, Optional
from configs import AdaptiveMoEModelConfig


def evaluate_model(
    model: nn.Module, 
    val_loader: DataLoader, 
    config: AdaptiveMoEModelConfig,
    max_eval_steps: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model performance on validation data.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        config: Model configuration
        max_eval_steps: Maximum number of evaluation steps (None for full evaluation)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    num_steps = 0
    
    device = next(model.parameters()).device
    eval_steps = max_eval_steps or config.eval_steps

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= eval_steps:
                break
                
            x, y = x.to(device), y.to(device)
            num_steps += 1

            with autocast('cuda', enabled=config.use_amp):
                # Handle MoE models that return aux loss
                if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
                    logits = model(x, return_aux_loss=False)
                else:
                    logits = model(x)
                
                # Compute cross-entropy loss
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size), 
                    y.view(-1),
                    reduction='sum'
                )

            # Accumulate metrics
            batch_tokens = y.numel()
            total_loss += loss.item()
            total_tokens += batch_tokens

            # Compute accuracy
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    # Compute final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    perplexity = compute_perplexity(avg_loss)

    model.train()
    
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        'val_perplexity': perplexity,
        'total_tokens': total_tokens,
        'num_steps': num_steps
    }


def compute_perplexity(loss: float, max_perplexity: float = 1000.0) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        max_perplexity: Maximum perplexity to return (for numerical stability)
        
    Returns:
        Perplexity value
    """
    try:
        perplexity = math.exp(min(loss, math.log(max_perplexity)))
        return perplexity
    except (OverflowError, ValueError):
        return max_perplexity


def evaluate_generation_quality(
    model: nn.Module,
    tokenizer,
    prompts: list,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate generation quality with sample prompts.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        
    Returns:
        Dictionary with generation results
    """
    model.eval()
    device = next(model.parameters()).device
    
    results = {
        'prompts': prompts,
        'generations': [],
        'avg_length': 0.0
    }
    
    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            input_ids = torch.tensor(
                tokenizer.encode(prompt), 
                dtype=torch.long, 
                device=device
            ).unsqueeze(0)
            
            # Generate
            if hasattr(model, 'generate'):
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
            else:
                # Simple greedy generation fallback
                generated = simple_generate(
                    model, 
                    input_ids, 
                    max_new_tokens,
                    temperature
                )
            
            # Decode
            generated_text = tokenizer.decode(
                generated[0].cpu().tolist(), 
                skip_special_tokens=True
            )
            
            results['generations'].append({
                'prompt': prompt,
                'generated': generated_text,
                'length': len(generated[0]) - len(input_ids[0])
            })
    
    # Compute average generation length
    if results['generations']:
        results['avg_length'] = sum(
            gen['length'] for gen in results['generations']
        ) / len(results['generations'])
    
    model.train()
    return results


def simple_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Simple greedy/sampling generation for models without generate method.
    
    Args:
        model: Model to use for generation
        input_ids: Input token IDs
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated token sequence
    """
    for _ in range(max_new_tokens):
        # Get logits
        if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
            logits = model(input_ids, return_aux_loss=False)
        else:
            logits = model(input_ids)
        
        # Get next token logits
        next_token_logits = logits[:, -1, :] / temperature
        
        # Sample next token
        if temperature > 0:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Stop if sequence gets too long (prevent memory issues)
        if input_ids.size(1) > 2048:
            break
    
    return input_ids


def compute_model_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    config: AdaptiveMoEModelConfig,
    compute_mfu: bool = False,
    dt: float = 1.0
) -> Dict[str, Any]:
    """
    Compute comprehensive model metrics.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        config: Model configuration
        compute_mfu: Whether to compute model FLOPs utilization
        dt: Time per iteration (for MFU computation)
        
    Returns:
        Dictionary with all metrics
    """
    # Basic evaluation metrics
    eval_results = evaluate_model(model, val_loader, config)
    
    # Model size metrics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics = {
        **eval_results,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
    }
    
    # MoE-specific metrics
    if hasattr(model, 'config') and hasattr(model.config, 'num_experts'):
        active_params = 0
        expert_params = 0
        
        for name, param in model.named_parameters():
            if 'expert' in name:
                expert_params += param.numel()
            else:
                active_params += param.numel()
        
        metrics.update({
            'active_params': active_params,
            'expert_params': expert_params,
            'active_params_M': active_params / 1e6,
            'expert_params_M': expert_params / 1e6,
            'parameter_efficiency': active_params / total_params if total_params > 0 else 0.0,
        })
    
    # Model FLOPs utilization
    if compute_mfu and hasattr(model, 'estimate_mfu'):
        mfu = model.estimate_mfu(fwdbwd_per_iter=1, dt=dt)
        metrics['mfu'] = mfu
        metrics['mfu_percent'] = mfu * 100
    
    return metrics


def benchmark_model_speed(
    model: nn.Module,
    data_loader: DataLoader,
    config: AdaptiveMoEModelConfig,
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        data_loader: Data loader for benchmark data
        config: Model configuration
        num_iterations: Number of iterations to benchmark
        
    Returns:
        Dictionary with timing metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Warmup
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= 3:  # Warmup iterations
                break
            x = x.to(device)
            _ = model(x, return_aux_loss=False) if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames else model(x)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        import time
        start_time = time.time()
    
    total_tokens = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if i >= num_iterations:
                break
            
            x = x.to(device)
            _ = model(x, return_aux_loss=False) if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames else model(x)
            total_tokens += x.numel()
    
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000
    else:
        elapsed_s = time.time() - start_time
        elapsed_ms = elapsed_s * 1000
    
    model.train()
    
    return {
        'total_time_s': elapsed_s,
        'total_time_ms': elapsed_ms,
        'avg_time_per_iteration_ms': elapsed_ms / num_iterations,
        'tokens_per_second': total_tokens / elapsed_s if elapsed_s > 0 else 0,
        'total_tokens': total_tokens,
        'num_iterations': num_iterations
    }
