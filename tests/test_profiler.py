#!/usr/bin/env python3
"""
Test script for Advanced GPU Profiler
Demonstrates profiling capabilities with a simple MoE model
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler import AdvancedGPUProfiler, ProfilerContext, profile_operation, set_profiler_hooks

# Simple test MoE components
class SimpleExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class SimpleMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        
        # Create experts
        self.experts = nn.ModuleList([
            SimpleExpert(d_model, d_ff) for _ in range(num_experts)
        ])
        
        # Simple router
        self.router = nn.Linear(d_model, num_experts)
    
    @profile_operation("simple_router")
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Simple routing (select top-1 expert)
        router_logits = self.router(x)
        expert_indices = torch.argmax(router_logits, dim=-1)  # [batch_size, seq_len]
        
        # Process through experts
        output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            mask = (expert_indices == expert_idx)
            if mask.any():
                expert_input = x[mask]
                expert_output = self.experts[expert_idx](expert_input)
                output[mask] = expert_output
        
        return output

def test_profiler():
    """Test the advanced GPU profiler with a simple MoE model"""
    print("🧪 Testing Advanced GPU Profiler")
    print("=" * 50)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Device: {device}")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Profiler will run in CPU mode.")
    
    # Create profiler
    profiler = AdvancedGPUProfiler(num_experts=4, enable_profiling=True)
    
    # Create simple MoE model
    d_model = 128
    d_ff = 512
    num_experts = 4
    
    model = SimpleMoE(d_model, d_ff, num_experts)
    model = model.to(device)
    
    # Set up profiler hooks
    set_profiler_hooks(profiler)
    
    print(f"📊 Model: {d_model}d → {d_ff}ff → {d_model}d, {num_experts} experts")
    
    # Test data
    batch_size = 8
    seq_len = 32
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"📝 Input: {x.shape}")
    
    # Run profiling session
    with ProfilerContext(profiler) as p:
        print("\n🚀 Starting profiling session...")
        
        # Run multiple forward passes
        for i in range(10):
            print(f"  Forward pass {i+1}/10...")
            
            # Profile data transfer
            if i == 0:  # Only profile first transfer
                input_size_bytes = x.numel() * x.element_size()
                p.profile_data_transfer(input_size_bytes, "host_to_device")
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
            
            # Profile output transfer
            if i == 0:  # Only profile first transfer
                output_size_bytes = output.numel() * output.element_size()
                p.profile_data_transfer(output_size_bytes, "device_to_host")
            
            # Small delay to simulate real training
            time.sleep(0.1)
        
        print("\n📊 Profiling Results:")
        p.print_dashboard()
    
    print("\n✅ Profiler test completed!")
    
    # Test individual profiler components
    test_individual_profilers()

def test_individual_profilers():
    """Test individual profiler components"""
    print("\n🔬 Testing Individual Profiler Components")
    print("=" * 50)
    
    profiler = AdvancedGPUProfiler(num_experts=4, enable_profiling=True)
    
    # Test memory profiler
    print("📊 Testing Memory Profiler...")
    profiler.start_profiling()
    
    # Simulate memory allocations
    for i in range(5):
        size_bytes = (i + 1) * 1024 * 1024  # 1MB, 2MB, 3MB, 4MB, 5MB
        profiler.profile_memory_allocation(size_bytes, expert_id=i % 4, operation=f"test_alloc_{i}")
        time.sleep(0.1)
    
    memory_stats = profiler.memory_profiler.get_memory_efficiency()
    print(f"  Memory efficiency: {memory_stats}")
    
    # Test kernel profiler
    print("⚡ Testing Kernel Profiler...")
    for i in range(5):
        execution_time = profiler.profile_kernel_execution(f"test_kernel_{i}", expert_id=i % 4, operation_type="test")
        print(f"  Kernel {i}: {execution_time:.2f}ms")
    
    kernel_stats = profiler.kernel_profiler.get_kernel_performance()
    print(f"  Kernel performance: {kernel_stats}")
    
    # Test expert routing profiler
    print("🧠 Testing Expert Routing Profiler...")
    for i in range(10):
        expert_indices = [i % 4, (i + 1) % 4]  # Simulate top-2 routing
        token_count = 16
        profiler.profile_expert_routing(expert_indices, token_count)
    
    routing_stats = profiler.expert_routing_profiler.get_routing_efficiency()
    print(f"  Routing efficiency: {routing_stats}")
    
    # Test data movement profiler
    print("🔄 Testing Data Movement Profiler...")
    for i in range(5):
        size_bytes = (i + 1) * 512 * 1024  # 512KB, 1MB, 1.5MB, 2MB, 2.5MB
        profiler.profile_data_transfer(size_bytes, f"test_transfer_{i}")
    
    movement_stats = profiler.data_movement_profiler.get_movement_efficiency()
    print(f"  Data movement: {movement_stats}")
    
    profiler.stop_profiling()
    print("✅ Individual profiler tests completed!")

def test_profiler_integration():
    """Test profiler integration with actual training"""
    print("\n🎯 Testing Profiler Integration with Training")
    print("=" * 50)
    
    # This would test integration with the actual MoE model
    # For now, just demonstrate the concept
    print("💡 To test full integration, run:")
    print("   python llm.py  # This will now include profiling")
    print("   python train_auto.py  # This will include profiling in auto-training")

if __name__ == "__main__":
    try:
        test_profiler()
        test_profiler_integration()
        print("\n🎉 All profiler tests passed!")
    except Exception as e:
        print(f"\n❌ Profiler test failed: {e}")
        import traceback
        traceback.print_exc()
