#!/usr/bin/env python3
"""
Simple profiler hooks for easy integration
"""

import torch
import functools
from .simple_gpu_profiler import SimpleGPUProfiler

def profile_forward_pass(profiler: SimpleGPUProfiler):
    """Decorator to profile forward pass of any model"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if profiler.is_profiling:
                # Profile input
                if args and isinstance(args[0], torch.Tensor):
                    profiler.profile_memory_allocation(
                        args[0].numel() * args[0].element_size(),
                        operation="model_input"
                    )
                
                # Execute original function
                result = func(*args, **kwargs)
                
                # Profile output
                if isinstance(result, torch.Tensor):
                    profiler.profile_memory_allocation(
                        result.numel() * result.element_size(),
                        operation="model_output"
                    )
                elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], torch.Tensor):
                    profiler.profile_memory_allocation(
                        result[0].numel() * result[0].element_size(),
                        operation="model_output"
                    )
                
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def profile_expert_forward(profiler: SimpleGPUProfiler):
    """Decorator to profile expert forward pass"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if profiler.is_profiling:
                # Get self and input tensor
                self = args[0]
                x = args[1] if len(args) > 1 else None
                
                if x is not None and isinstance(x, torch.Tensor):
                    # Profile expert input
                    profiler.profile_memory_allocation(
                        x.numel() * x.element_size(),
                        expert_id=getattr(self, 'expert_id', None),
                        operation="expert_input"
                    )
                
                # Start kernel profiling
                kernel_context = profiler.start_kernel_profiling(
                    f"expert_{getattr(self, 'expert_id', 'unknown')}",
                    expert_id=getattr(self, 'expert_id', None),
                    operation_type="expert_forward"
                )
                
                # Execute original function
                result = func(*args, **kwargs)
                
                # End kernel profiling
                kernel_time = profiler.end_kernel_profiling(kernel_context)
                
                # Profile expert output
                if isinstance(result, torch.Tensor):
                    profiler.profile_memory_allocation(
                        result.numel() * result.element_size(),
                        expert_id=getattr(self, 'expert_id', None),
                        operation="expert_output"
                    )
                
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def profile_router_forward(profiler: SimpleGPUProfiler):
    """Decorator to profile router forward pass"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if profiler.is_profiling:
                # Get the input tensor (first argument after self)
                x = args[1] if len(args) > 1 else None
                
                if x is not None and isinstance(x, torch.Tensor):
                    # Profile router input
                    profiler.profile_memory_allocation(
                        x.numel() * x.element_size(),
                        operation="router_input"
                    )
                
                # Start kernel profiling
                kernel_context = profiler.start_kernel_profiling(
                    "router_forward",
                    operation_type="routing"
                )
                
                # Execute original function
                result = func(*args, **kwargs)
                
                # End kernel profiling
                kernel_time = profiler.end_kernel_profiling(kernel_context)
                
                # Profile expert routing
                if isinstance(result, tuple) and len(result) >= 2:
                    expert_indices, token_count = result[0], result[1]
                    if isinstance(expert_indices, torch.Tensor):
                        expert_indices = expert_indices.tolist()
                    if isinstance(token_count, torch.Tensor):
                        token_count = token_count.item()
                    
                    profiler.profile_expert_routing(expert_indices, token_count)
                
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def apply_simple_profiling(model, profiler: SimpleGPUProfiler):
    """Apply simple profiling to a model"""
    # Always apply hooks - they will only execute when profiler.is_profiling is True
    
    # Profile main model forward pass
    if hasattr(model, 'forward'):
        model.forward = profile_forward_pass(profiler)(model.forward)
    
    # Profile expert modules
    for name, module in model.named_modules():
        if 'expert' in name.lower() and hasattr(module, 'forward'):
            module.forward = profile_expert_forward(profiler)(module.forward)
        
        # Profile router modules
        if 'router' in name.lower() and hasattr(module, 'forward'):
            module.forward = profile_router_forward(profiler)(module.forward)
    
    return model
