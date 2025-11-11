"""
Model implementations for attention mechanism ablation study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention_mechanisms import (
    MultiHeadAttention,
    MultiHeadLatentAttention,
    GroupedQueryAttention,
    MultiQueryAttention,
    SlidingWindowAttention,
    DeepSeekSparseAttention,
    IdentityAttention,
)
from .config import AttentionAblationConfig


class FeedForward(nn.Module):
    """Simple feed-forward network with SiLU activation"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Transformer block with configurable attention mechanism
    """
    def __init__(self, config: AttentionAblationConfig):
        super().__init__()
        
        # Create attention layer based on config
        self.attention = self._create_attention(config)
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            config.d_model,
            config.d_ff,
            config.dropout
        )
        
        # Normalization
        self.norm1 = nn.RMSNorm(config.d_model)
        self.norm2 = nn.RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def _create_attention(self, config: AttentionAblationConfig):
        """Create attention mechanism based on config"""
        attn_type = config.attention_type.lower()
        
        if attn_type == "mha":
            return MultiHeadAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout
            )
        elif attn_type == "mla":
            return MultiHeadLatentAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                qk_rope_dim=config.qk_rope_dim,
                qk_nope_dim=config.qk_nope_dim,
                kv_lora_rank=config.kv_lora_rank,
                v_dim=config.v_dim,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout
            )
        elif attn_type == "gqa":
            return GroupedQueryAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout
            )
        elif attn_type == "mqa":
            return MultiQueryAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout
            )
        elif attn_type == "sliding_window":
            return SlidingWindowAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                max_seq_len=config.max_seq_len,
                window_size=config.window_size,
                dropout=config.dropout
            )
        elif attn_type == "sparse":
            return DeepSeekSparseAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                max_seq_len=config.max_seq_len,
                indexer_heads=config.indexer_heads,
                indexer_dim=config.indexer_dim,
                sparse_top_k=config.sparse_top_k,
                dropout=config.dropout
            )
        elif attn_type == "identity":
            return IdentityAttention(d_model=config.d_model)
        elif attn_type == "linear":
            # Linear attention will use FLA's DeltaNet
            # Import here to avoid circular dependency
            try:
                from fla.layers import DeltaNet
                return DeltaNet(
                    d_model=config.d_model,
                    expand_k=1.0,
                    expand_v=1.0,
                    num_heads=config.n_heads,
                )
            except ImportError:
                raise ImportError(
                    "FLA library required for linear attention. "
                    "Install with: pip install fla-flash-linear-attention"
                )
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")
    
    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class AttentionAblationModel(nn.Module):
    """
    Simple language model for attention mechanism ablation
    """
    def __init__(self, config: AttentionAblationConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def count_parameters(self) -> dict:
        """Count parameters by component"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count by component
        embedding = sum(p.numel() for p in self.token_embedding.parameters())
        attention = sum(
            sum(p.numel() for p in block.attention.parameters())
            for block in self.blocks
        )
        feedforward = sum(
            sum(p.numel() for p in block.feed_forward.parameters())
            for block in self.blocks
        )
        
        return {
            "total": total,
            "trainable": trainable,
            "embedding": embedding,
            "attention": attention,
            "feedforward": feedforward,
            "other": total - embedding - attention - feedforward
        }
    
    def get_attention_type(self) -> str:
        """Get the attention type being used"""
        return self.config.attention_type


class ModelWrapper(nn.Module):
    """
    Wrapper for the ablation model with convenience methods
    """
    def __init__(self, config: AttentionAblationConfig):
        super().__init__()
        self.config = config
        self.model = AttentionAblationModel(config)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass with optional loss computation
        """
        logits = self.model(input_ids)
        
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {"logits": logits, "loss": loss}
    
    def print_model_info(self):
        """Print model information"""
        params = self.model.count_parameters()
        
        print("=" * 80)
        print(f"Attention Mechanism Ablation Model")
        print("=" * 80)
        print(f"\nArchitecture:")
        print(f"  Attention Type: {self.config.attention_type.upper()}")
        print(f"  Layers: {self.config.n_layers}")
        print(f"  Model Dim: {self.config.d_model}")
        print(f"  Num Heads: {self.config.n_heads}")
        print(f"  FF Dim: {self.config.d_ff}")
        print(f"  Vocab Size: {self.config.vocab_size}")
        print(f"  Max Seq Len: {self.config.max_seq_len}")
        
        if self.config.attention_type == "gqa":
            print(f"  KV Heads: {self.config.n_kv_heads}")
        elif self.config.attention_type == "mla":
            print(f"  QK RoPE Dim: {self.config.qk_rope_dim}")
            print(f"  QK NoPE Dim: {self.config.qk_nope_dim}")
            print(f"  KV LoRA Rank: {self.config.kv_lora_rank}")
            print(f"  V Dim: {self.config.v_dim}")
        elif self.config.attention_type == "sliding_window":
            print(f"  Window Size: {self.config.window_size}")
        elif self.config.attention_type == "sparse":
            print(f"  Sparse Top-K: {self.config.sparse_top_k}")
            print(f"  Indexer Heads: {self.config.indexer_heads}")
            print(f"  Indexer Dim: {self.config.indexer_dim}")
        
        print(f"\nParameters:")
        print(f"  Total: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,}")
        print(f"  Embedding: {params['embedding']:,} ({100*params['embedding']/params['total']:.1f}%)")
        print(f"  Attention: {params['attention']:,} ({100*params['attention']/params['total']:.1f}%)")
        print(f"  FeedForward: {params['feedforward']:,} ({100*params['feedforward']/params['total']:.1f}%)")
        print(f"  Other: {params['other']:,} ({100*params['other']/params['total']:.1f}%)")
        print("=" * 80)

