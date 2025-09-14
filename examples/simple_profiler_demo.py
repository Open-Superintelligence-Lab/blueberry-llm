#!/usr/bin/env python3
"""
Simple Profiler Demo - Shows core GPU profiling measurements
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from profiler import SimpleGPUProfiler, apply_simple_profiling

class SimpleMoE(nn.Module):
    """Simple MoE model for demo"""
    def __init__(self, d_model=512, num_experts=4):
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

def demo_simple_profiler():
    """Demo the simple profiler"""
    print("🫐 Simple GPU Profiler Demo")
    print("=" * 40)
    
    # Create profiler
    profiler = SimpleGPUProfiler(num_experts=4, enable_profiling=True)
    
    # Create simple model
    model = SimpleMoE(d_model=512, num_experts=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Apply profiling
    model = apply_simple_profiling(model, profiler)
    
    # Start profiling
    profiler.start_profiling()
    
    print("🚀 Running MoE forward passes...")
    
    # Run some forward passes
    for i in range(5):
        x = torch.randn(32, 512).to(device)  # Batch of 32, 512-dim vectors
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Show dashboard every 2 steps
        if i % 2 == 0 and i > 0:
            profiler.print_dashboard()
    
    # Stop profiling
    profiler.stop_profiling()
    
    print("\n✅ Demo complete!")
    print("📊 Check profiler_output/ for the generated report")

if __name__ == "__main__":
    try:
        demo_simple_profiler()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
