#!/usr/bin/env python3
"""
Test script to verify profiler fixes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from profiler import SimpleGPUProfiler, apply_simple_profiling

class TestMoE(nn.Module):
    """Simple test MoE model"""
    def __init__(self, d_model=128, num_experts=4):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Simple experts
        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_experts)
        ])
        
        # Simple router
        self.router = nn.Linear(d_model, num_experts)
        
    def forward(self, x):
        # Router
        router_logits = self.router(x)
        expert_weights = torch.softmax(router_logits, dim=-1)
        
        # Select top expert
        top_expert = torch.argmax(expert_weights, dim=-1)
        
        # Expert forward pass
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (top_expert == i)
            if mask.any():
                expert_output = self.experts[i](x[mask])
                output[mask] = expert_output
        
        return output

def test_profiler_fix():
    """Test that profiler now collects data"""
    print("🧪 Testing Profiler Fix")
    print("=" * 40)
    
    # Create profiler
    profiler = SimpleGPUProfiler(num_experts=4, enable_profiling=True)
    
    # Create test model
    model = TestMoE(d_model=128, num_experts=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Apply profiling (this should now work even when profiler.is_profiling = False)
    model = apply_simple_profiling(model, profiler)
    print("✅ Profiler hooks applied successfully")
    
    # Start profiling
    profiler.start_profiling()
    print("✅ Profiler started")
    
    # Run some forward passes
    print("🚀 Running test forward passes...")
    for i in range(3):
        x = torch.randn(16, 128).to(device)
        with torch.no_grad():
            output = model(x)
        print(f"   Forward pass {i+1} completed")
    
    # Stop profiling
    profiler.stop_profiling()
    print("✅ Profiler stopped")
    
    # Check if data was collected
    print("\n📊 Profiler Data Check:")
    print(f"   Memory history entries: {len(profiler.memory_history)}")
    print(f"   Kernel history entries: {len(profiler.kernel_history)}")
    print(f"   Expert history entries: {len(profiler.expert_history)}")
    
    if profiler.memory_history:
        latest_mem = profiler.memory_history[-1]
        print(f"   Latest memory: {latest_mem.allocated_mb:.1f}MB")
    
    if profiler.kernel_history:
        print(f"   Kernel times collected: {len(profiler.kernel_times)}")
        for kernel_name, times in profiler.kernel_times.items():
            if times:
                avg_time = sum(times) / len(times)
                print(f"     {kernel_name}: {avg_time:.2f}ms avg")
    
    if profiler.expert_history:
        print(f"   Total tokens processed: {profiler.total_tokens}")
    
    # Check if report was generated
    import glob
    report_files = glob.glob("profiler_output/run_*_profiler_report.json")
    if report_files:
        print(f"✅ Report generated: {report_files[-1]}")
    else:
        print("❌ No report file found")

if __name__ == "__main__":
    try:
        test_profiler_fix()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
