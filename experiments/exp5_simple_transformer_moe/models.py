"""Simple Transformer Model with MoE - No fancy attention mechanisms"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config import SimpleTransformerConfig


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cached cos and sin values if needed"""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor
        Args:
            x: [batch, seq_len, num_heads, head_dim]
        Returns:
            Rotated tensor of same shape
        """
        seq_len = x.shape[1]
        self._update_cache(seq_len, x.device)
        
        # Split into two halves for rotation
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotation
        cos = self._cos_cached[:, :seq_len, :, :x.shape[-1]//2]
        sin = self._sin_cached[:, :seq_len, :, :x.shape[-1]//2]
        
        return torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)


class StandardMultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention with RoPE - No fancy mechanisms"""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # QKV projection
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE for positional encoding
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)  # [B, T, 3*d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, H, T, D]
        
        # Apply RoPE to queries and keys
        # Convert to [B, T, H, D] for RoPE, then back to [B, H, T, D]
        q = self.rope(q.transpose(1, 2)).transpose(1, 2)
        k = self.rope(k.transpose(1, 2)).transpose(1, 2)
        
        # Standard scaled dot-product attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.output_proj(attn_output)


class Expert(nn.Module):
    """Single expert network - simple MLP"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN with GELU activation"""
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class MixtureOfExperts(nn.Module):
    """Simple MoE layer with top-k routing"""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # Router network
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: Load balancing loss
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [B*T, d_model]
        
        # Route to experts
        router_logits = self.router(x_flat)  # [B*T, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            if not expert_mask.any():
                continue
            
            expert_input = x_flat[expert_mask]
            expert_output = self.experts[i](expert_input)
            
            # Weight by routing probabilities
            expert_weights = top_k_probs[expert_mask]
            expert_weights = expert_weights[top_k_indices[expert_mask] == i].unsqueeze(-1)
            
            output[expert_mask] += expert_output * expert_weights
        
        output = output.view(batch_size, seq_len, d_model)
        
        # Compute load balancing loss
        # Encourage uniform distribution of tokens across experts
        expert_usage = router_probs.mean(dim=0)  # [num_experts]
        aux_loss = self.num_experts * (expert_usage ** 2).sum()
        
        return output, aux_loss


class TransformerBlock(nn.Module):
    """Standard Transformer block with MoE"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int,
        top_k: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Standard attention
        self.attention = StandardMultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        
        # MoE feedforward
        self.moe = MixtureOfExperts(d_model, d_ff, num_experts, top_k, dropout)
        
        # Normalization
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: Load balancing loss from MoE
        """
        # Self-attention with pre-norm
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # MoE feedforward with pre-norm
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out)
        
        return x, aux_loss


class SimpleTransformerMoE(nn.Module):
    """Simple Transformer with MoE - No fancy attention"""
    
    def __init__(self, config: SimpleTransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.max_seq_len,
                config.num_experts,
                config.expert_top_k,
                config.dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [batch, seq_len]
            return_aux_loss: Whether to return auxiliary loss
        Returns:
            logits: [batch, seq_len, vocab_size]
            aux_loss: Load balancing loss (if return_aux_loss=True)
        """
        # Embed tokens
        x = self.token_embedding(input_ids)
        x = x * math.sqrt(self.config.d_model)  # Scale embeddings
        
        # Collect auxiliary losses
        aux_losses = []
        
        # Process through transformer blocks
        for block in self.blocks:
            x, aux_loss = block(x)
            if aux_loss is not None and return_aux_loss:
                aux_losses.append(aux_loss)
        
        # Output projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Combine auxiliary losses
        total_aux_loss = None
        if aux_losses:
            total_aux_loss = sum(aux_losses) * self.config.load_balancing_weight
        
        if return_aux_loss:
            return logits, total_aux_loss
        return logits
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count model parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

