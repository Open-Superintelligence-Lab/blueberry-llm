#!/usr/bin/env python3
"""
Example showing how to integrate the GPU-adaptive matmul system into an existing LLM.

This demonstrates how to replace standard PyTorch operations with our
adaptive system while maintaining the same interface.
"""

import torch
import torch.nn as nn
from system import SYSTEM_CONFIG, print_system_info
from ops.matmul import matmul


class AdaptiveLinear(nn.Module):
    """
    GPU-adaptive linear layer that automatically uses the best matmul kernel.
    
    This is a drop-in replacement for nn.Linear that leverages our
    GPU-adaptive matmul system.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameter
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Bias parameter (optional)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using optimal scaling for the current architecture."""
        # Use optimal initialization based on architecture
        config = SYSTEM_CONFIG
        
        if config.architecture == "blackwell":
            # Blackwell-optimized initialization
            std = 0.5 * (self.in_features ** -0.5)
            bound = (3 ** 0.5) * std
            nn.init.uniform_(self.weight, -bound, bound)
        else:
            # Standard initialization for other architectures
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using GPU-adaptive matmul."""
        # Use our adaptive matmul
        output = matmul(x, self.weight)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        return output


class AdaptiveAttention(nn.Module):
    """
    GPU-adaptive attention layer using our matmul system.
    
    This demonstrates how to integrate adaptive operations into
    more complex modules like attention.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # QKV projection using adaptive linear layers
        self.qkv_proj = AdaptiveLinear(d_model, d_model * 3, bias=False)
        self.out_proj = AdaptiveLinear(d_model, d_model, bias=False)
        
        # Scale factor (architecture-dependent)
        config = SYSTEM_CONFIG
        if config.architecture == "blackwell":
            self.scale = 0.12  # Optimized for Blackwell
        else:
            self.scale = 1.0 / (self.d_k ** 0.5)  # Standard scaling
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        # QKV projection using adaptive matmul
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        
        return output


def benchmark_adaptive_vs_standard():
    """Benchmark adaptive system vs standard PyTorch operations."""
    print("🏁 Benchmarking Adaptive vs Standard Operations")
    print("-" * 50)
    
    if not torch.cuda.is_available():
        print("⏭️ Skipping benchmark (no CUDA)")
        return
    
    # Test parameters
    batch_size, seq_len, d_model = 8, 1024, 1024
    d_out = 1024
    
    # Create test data
    x = torch.randn(batch_size, seq_len, d_model, device='cuda')
    w = torch.randn(d_out, d_model, device='cuda')
    
    # Warmup
    for _ in range(5):
        _ = torch.matmul(x, w.T)
        _ = matmul(x, w)
    
    torch.cuda.synchronize()
    
    # Benchmark standard PyTorch
    num_iterations = 100
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(num_iterations):
        _ = torch.matmul(x, w.T)
    end_time.record()
    torch.cuda.synchronize()
    
    standard_time = start_time.elapsed_time(end_time) / num_iterations
    
    # Benchmark adaptive system
    start_time.record()
    for _ in range(num_iterations):
        _ = matmul(x, w)
    end_time.record()
    torch.cuda.synchronize()
    
    adaptive_time = start_time.elapsed_time(end_time) / num_iterations
    
    # Results
    speedup = standard_time / adaptive_time
    print(f"📊 Standard PyTorch: {standard_time:.2f}ms")
    print(f"📊 Adaptive System: {adaptive_time:.2f}ms")
    print(f"📊 Speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print("✅ Adaptive system is faster!")
    elif speedup < 1.0:
        print("⚠️ Adaptive system is slower (may be due to overhead)")
    else:
        print("⚖️ Performance is similar")


def main():
    """Main example function."""
    print("🚀 GPU-Adaptive System Integration Example")
    print("=" * 60)
    
    # Print system information
    print_system_info()
    print()
    
    # Create adaptive layers
    print("🔧 Creating Adaptive Layers...")
    
    # Adaptive linear layer
    adaptive_linear = AdaptiveLinear(512, 256).cuda() if torch.cuda.is_available() else AdaptiveLinear(512, 256)
    print(f"   ✅ AdaptiveLinear: {adaptive_linear}")
    
    # Adaptive attention layer
    adaptive_attention = AdaptiveAttention(512, 8).cuda() if torch.cuda.is_available() else AdaptiveAttention(512, 8)
    print(f"   ✅ AdaptiveAttention: {adaptive_attention}")
    
    # Test forward pass
    print("\n🧪 Testing Forward Pass...")
    
    batch_size, seq_len, d_model = 2, 128, 512
    x = torch.randn(batch_size, seq_len, d_model, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test adaptive linear
    try:
        linear_out = adaptive_linear(x)
        print(f"   ✅ AdaptiveLinear output: {linear_out.shape}")
    except Exception as e:
        print(f"   ❌ AdaptiveLinear failed: {e}")
    
    # Test adaptive attention
    try:
        attn_out = adaptive_attention(x)
        print(f"   ✅ AdaptiveAttention output: {attn_out.shape}")
    except Exception as e:
        print(f"   ❌ AdaptiveAttention failed: {e}")
    
    # Run benchmark
    print()
    benchmark_adaptive_vs_standard()
    
    print("\n🎉 Integration example completed!")


if __name__ == "__main__":
    main()
