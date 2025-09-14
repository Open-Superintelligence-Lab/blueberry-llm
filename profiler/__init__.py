"""
Advanced GPU Profiler for Blueberry LLM
"""

from .advanced_gpu_profiler import AdvancedGPUProfiler, ProfilerContext
from .profiler_hooks import (
    ProfilerHooks, 
    ProfilerContext as HooksContext,
    profile_operation,
    set_profiler_hooks,
    get_profiler_hooks,
    patch_model_for_profiling,
    train_with_profiling
)

__all__ = [
    'AdvancedGPUProfiler',
    'ProfilerContext', 
    'HooksContext',
    'ProfilerHooks',
    'profile_operation',
    'set_profiler_hooks',
    'get_profiler_hooks',
    'patch_model_for_profiling',
    'train_with_profiling'
]
