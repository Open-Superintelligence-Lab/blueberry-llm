from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MoE4090Config:
    """Configuration for MoE model optimized for RTX 4090 24GB"""
    
    # Model architecture - ~600M total params, ~200M active per token
    # Heavily scaled down from 18B to fit 24GB VRAM
    d_model: int = 1024          # Hidden dimension (vs 4096 for B200)
    n_heads: int = 8             # Attention heads (vs 32)
    n_layers: int = 12           # Transformer layers (vs 40)
    d_ff: int = 2752             # FFN dimension (per expert, vs 11008)
    
    # MoE specific parameters - same architecture
    num_experts: int = 8         # Total experts
    expert_top_k: int = 2        # Active experts per token
    load_balancing_weight: float = 0.01
    
    # Training parameters - adjusted for memory constraints
    batch_size: int = 2          # Tokens per batch (vs 4)
    max_seq_len: int = 1024      # Context window (vs 4096)
    gradient_accumulation_steps: int = 16  # Effective batch size = 32 (same as B200)
    max_steps: int = 50000       # Training steps
    
    # Optimizer settings - same as B200
    muon_lr: float = 0.01        # Muon learning rate
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Memory optimizations - critical for 24GB
    use_amp: bool = True         # Mixed precision training
    use_gradient_checkpointing: bool = True  # Save ~40% activation memory
    
    # Data parameters - same as B200
    num_documents: int = 100000  # Number of training documents
    max_tokens: int = 100_000_000  # 100M tokens
    
    # Evaluation - same as B200
    eval_every: int = 500        # Evaluate every N steps
    eval_steps: int = 100        # Steps to run during evaluation
    save_every: int = 5000       # Save checkpoint every N steps
    
    # Regularization - same as B200
    dropout: float = 0.1
    
    # Technical - same as B200
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (1000, 5000, 10000, 25000, 50000)
    
    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Calculate approximate model size
        self._calculate_model_stats()
    
    def _calculate_model_stats(self):
        """Calculate expected model statistics"""
        # Embedding parameters
        if self.vocab_size:
            embedding_params = self.vocab_size * self.d_model
        else:
            embedding_params = 50000 * self.d_model  # Estimate
        
        # Per-layer parameters
        # Attention: Q, K, V projections + output projection
        attn_params = 4 * (self.d_model * self.d_model)
        
        # MoE FFN: router + (num_experts * 2 linear layers)
        router_params = self.d_model * self.num_experts
        expert_params = self.num_experts * (2 * self.d_model * self.d_ff)
        ffn_params = router_params + expert_params
        
        # Layer norms (2 per layer)
        norm_params = 2 * self.d_model
        
        # Total per layer
        params_per_layer = attn_params + ffn_params + norm_params
        
        # Total model params
        total_params = embedding_params + (self.n_layers * params_per_layer) + self.d_model  # final norm
        
        # Active params per forward pass (only top_k experts active)
        active_expert_params = self.expert_top_k * (2 * self.d_model * self.d_ff)
        active_ffn_params = router_params + active_expert_params
        active_params_per_layer = attn_params + active_ffn_params + norm_params
        active_params = embedding_params + (self.n_layers * active_params_per_layer) + self.d_model
        
        self.total_params = total_params
        self.active_params = active_params
        self.params_per_layer = params_per_layer
        
        # Memory estimates (in GB)
        bytes_per_param_training = 12  # 2 (weights) + 2 (grads) + 8 (Muon momentum)
        self.estimated_param_memory_gb = (total_params * bytes_per_param_training) / 1e9
        
        # Activation memory (rough estimate)
        activation_memory_per_token = (
            self.n_layers * self.d_model * 
            (8 if self.use_gradient_checkpointing else 16)  # checkpointing saves ~50%
        )
        tokens_per_batch = self.batch_size * self.max_seq_len
        self.estimated_activation_memory_gb = (
            tokens_per_batch * activation_memory_per_token * 2  # BF16
        ) / 1e9
        
        self.estimated_total_memory_gb = (
            self.estimated_param_memory_gb + 
            self.estimated_activation_memory_gb
        )
    
    def print_stats(self):
        """Print model statistics"""
        print(f"\n{'='*70}")
        print(f"🚀 MoE Model Configuration (RTX 4090 Optimized)")
        print(f"{'='*70}")
        print(f"\n📐 Architecture:")
        print(f"   Layers: {self.n_layers}")
        print(f"   Hidden Size: {self.d_model}")
        print(f"   Attention Heads: {self.n_heads}")
        print(f"   FFN Size: {self.d_ff}")
        print(f"   Sequence Length: {self.max_seq_len}")
        
        print(f"\n🔀 Mixture of Experts:")
        print(f"   Total Experts: {self.num_experts}")
        print(f"   Active per Token: {self.expert_top_k}")
        print(f"   Expert Utilization: {self.expert_top_k}/{self.num_experts} ({self.expert_top_k/self.num_experts:.1%})")
        
        print(f"\n📊 Model Size:")
        print(f"   Total Parameters: {self.total_params/1e9:.2f}B")
        print(f"   Active Parameters: {self.active_params/1e9:.2f}B ({self.active_params/self.total_params:.1%})")
        print(f"   Parameters per Layer: {self.params_per_layer/1e6:.1f}M")
        
        print(f"\n💾 Memory Estimates (BF16 + Muon):")
        print(f"   Parameter Memory: {self.estimated_param_memory_gb:.1f} GB")
        print(f"   Activation Memory: {self.estimated_activation_memory_gb:.1f} GB")
        print(f"   Total Estimated: {self.estimated_total_memory_gb:.1f} GB")
        print(f"   Gradient Checkpointing: {'✅ Enabled' if self.use_gradient_checkpointing else '❌ Disabled'}")
        
        print(f"\n🎯 Training Settings:")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"   Effective Batch Size: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"   Total Steps: {self.max_steps:,}")
        print(f"   Learning Rate: {self.muon_lr}")
        print(f"   Mixed Precision: {'✅ FP16' if self.use_amp else '❌ FP32'}")
        
        print(f"\n📚 Data:")
        print(f"   Documents: {self.num_documents:,}")
        print(f"   Max Tokens: {self.max_tokens:,}")
        print(f"   Tokens per Step: {self.batch_size * self.max_seq_len * self.gradient_accumulation_steps:,}")
        
        print(f"{'='*70}\n")
