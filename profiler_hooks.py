#!/usr/bin/env python3
"""
Profiler Hooks - Clean integration for Advanced GPU Profiler
This module provides decorators and context managers for profiling without modifying core model code.
"""

import torch
import torch.nn as nn
import functools
import time
from typing import Optional, List, Any, Callable
from advanced_gpu_profiler import AdvancedGPUProfiler

class ProfilerHooks:
    """Clean profiler integration using hooks and decorators"""
    
    def __init__(self, profiler: Optional[AdvancedGPUProfiler] = None):
        self.profiler = profiler
        self.is_active = profiler is not None and profiler.is_profiling
    
    def profile_forward_pass(self, operation_name: str, expert_id: Optional[int] = None):
        """Decorator for profiling forward passes"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_active:
                    return func(*args, **kwargs)
                
                # Profile memory allocation for input
                if args and isinstance(args[0], torch.Tensor):
                    input_size_bytes = args[0].numel() * args[0].element_size()
                    self.profiler.profile_memory_allocation(input_size_bytes, expert_id, f"{operation_name}_input")
                
                # Profile kernel execution
                start_time = time.time()
                self.profiler.profile_kernel_execution(operation_name, expert_id, operation_name)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Profile output memory
                if isinstance(result, torch.Tensor):
                    output_size_bytes = result.numel() * result.element_size()
                    self.profiler.profile_memory_allocation(output_size_bytes, expert_id, f"{operation_name}_output")
                elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], torch.Tensor):
                    # Handle tuple returns (like MoE forward)
                    output_size_bytes = result[0].numel() * result[0].element_size()
                    self.profiler.profile_memory_allocation(output_size_bytes, expert_id, f"{operation_name}_output")
                
                return result
            return wrapper
        return decorator
    
    def profile_expert_routing(self, expert_indices: List[int], token_count: int):
        """Profile expert routing decisions"""
        if self.is_active:
            self.profiler.profile_expert_routing(expert_indices, token_count)
    
    def profile_data_transfer(self, size_bytes: int, transfer_type: str):
        """Profile data transfer"""
        if self.is_active:
            self.profiler.profile_data_transfer(size_bytes, transfer_type)

# Global profiler hooks instance
_profiler_hooks: Optional[ProfilerHooks] = None

def set_profiler_hooks(profiler: Optional[AdvancedGPUProfiler]):
    """Set global profiler hooks"""
    global _profiler_hooks
    _profiler_hooks = ProfilerHooks(profiler)

def get_profiler_hooks() -> Optional[ProfilerHooks]:
    """Get global profiler hooks"""
    return _profiler_hooks

# Convenience decorators
def profile_operation(operation_name: str, expert_id: Optional[int] = None):
    """Decorator for profiling any operation"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            hooks = get_profiler_hooks()
            if hooks and hooks.is_active:
                return hooks.profile_forward_pass(operation_name, expert_id)(func)(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def profile_memory_allocation(size_bytes: int, expert_id: Optional[int] = None, operation: str = "unknown"):
    """Profile memory allocation"""
    hooks = get_profiler_hooks()
    if hooks:
        hooks.profiler.profile_memory_allocation(size_bytes, expert_id, operation)

def profile_kernel_execution(kernel_name: str, expert_id: Optional[int] = None, operation_type: str = "unknown"):
    """Profile kernel execution"""
    hooks = get_profiler_hooks()
    if hooks:
        hooks.profiler.profile_kernel_execution(kernel_name, expert_id, operation_type)

def profile_expert_routing(expert_indices: List[int], token_count: int):
    """Profile expert routing"""
    hooks = get_profiler_hooks()
    if hooks:
        hooks.profile_expert_routing(expert_indices, token_count)

def profile_data_transfer(size_bytes: int, transfer_type: str):
    """Profile data transfer"""
    hooks = get_profiler_hooks()
    if hooks:
        hooks.profile_data_transfer(size_bytes, transfer_type)

# Context manager for automatic profiling setup
class ProfilerContext:
    """Context manager for automatic profiler setup"""
    
    def __init__(self, profiler: AdvancedGPUProfiler):
        self.profiler = profiler
        self.original_hooks = None
    
    def __enter__(self):
        global _profiler_hooks
        self.original_hooks = _profiler_hooks
        _profiler_hooks = ProfilerHooks(self.profiler)
        self.profiler.start_profiling()
        return self.profiler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _profiler_hooks
        self.profiler.stop_profiling()
        _profiler_hooks = self.original_hooks

# Monkey patching approach for automatic profiling (optional)
def patch_model_for_profiling(model: nn.Module, profiler: AdvancedGPUProfiler):
    """Automatically patch model methods for profiling"""
    set_profiler_hooks(profiler)
    
    # Patch forward methods
    original_forward = model.forward
    
    @functools.wraps(original_forward)
    def profiled_forward(*args, **kwargs):
        if profiler.is_profiling:
            # Profile model input
            if args and isinstance(args[0], torch.Tensor):
                input_size_bytes = args[0].numel() * args[0].element_size()
                profiler.profile_memory_allocation(input_size_bytes, operation="model_input")
            
            # Execute original forward
            result = original_forward(*args, **kwargs)
            
            # Profile model output
            if isinstance(result, torch.Tensor):
                output_size_bytes = result.numel() * result.element_size()
                profiler.profile_memory_allocation(output_size_bytes, operation="model_output")
            elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], torch.Tensor):
                output_size_bytes = result[0].numel() * result[0].element_size()
                profiler.profile_memory_allocation(output_size_bytes, operation="model_output")
            
            return result
        else:
            return original_forward(*args, **kwargs)
    
    model.forward = profiled_forward
    return model

# Clean training wrapper
def train_with_profiling(train_function: Callable, profiler: AdvancedGPUProfiler, *args, **kwargs):
    """Wrapper for training functions with automatic profiling"""
    with ProfilerContext(profiler) as p:
        # Patch the model if it's in the arguments
        if 'model' in kwargs:
            patch_model_for_profiling(kwargs['model'], p)
        
        # Run training
        result = train_function(*args, **kwargs)
        
        # Print final dashboard
        p.print_dashboard()
        
        return result

# Example usage functions
def example_usage():
    """Example of clean profiler usage"""
    from advanced_gpu_profiler import AdvancedGPUProfiler
    
    # Create profiler
    profiler = AdvancedGPUProfiler(num_experts=8, enable_profiling=True)
    
    # Method 1: Context manager (cleanest)
    with ProfilerContext(profiler) as p:
        # Your training code here - no modifications needed
        # model = MoEMinimalLLM(config)
        # train_model(model, ...)
        pass
    
    # Method 2: Manual hooks
    set_profiler_hooks(profiler)
    profiler.start_profiling()
    
    # Use decorators in your code
    @profile_operation("custom_operation")
    def my_function():
        pass
    
    profiler.stop_profiling()
    
    # Method 3: Monkey patching (automatic)
    # model = MoEMinimalLLM(config)
    # patched_model = patch_model_for_profiling(model, profiler)
    # Now all forward passes are automatically profiled

if __name__ == "__main__":
    example_usage()
