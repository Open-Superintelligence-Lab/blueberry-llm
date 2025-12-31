"""
Model creation with temperature-aware MoE components
"""
import sys
import torch
import torch.nn as nn
import math
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.moe_config import MoEModelConfig
from models.layers import MultiHeadAttention, MultiHeadLatentAttention
from temperature_moe import TemperatureMoE


class TemperatureMoETransformerBlock(nn.Module):
    """Transformer block with temperature-aware MoE"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        use_mla: bool,
        qk_rope_dim: int | None,
        qk_nope_dim: int | None,
        kv_lora_rank: int | None,
        v_dim: int | None,
        max_seq_len: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Attention layer (reuse from main models)
        if use_mla:
            self.attention = MultiHeadLatentAttention(
                d_model,
                n_heads,
                qk_rope_dim,
                qk_nope_dim,
                kv_lora_rank,
                v_dim,
                max_seq_len,
                dropout,
            )
        else:
            self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        
        # Temperature-aware MoE layer
        self.feed_forward = TemperatureMoE(d_model, d_ff, num_experts, top_k, dropout)
        
        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def set_temperature(self, temperature: float):
        """Set temperature for MoE routing"""
        self.feed_forward.set_temperature(temperature)
    
    def forward(self, x, return_routing_stats=False):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # MoE feed-forward
        ff_out, aux_loss, routing_stats = self.feed_forward(
            self.norm2(x),
            return_routing_stats=return_routing_stats
        )
        x = x + self.dropout(ff_out)
        
        if return_routing_stats:
            return x, aux_loss, routing_stats
        return x, aux_loss


class TemperatureMoEModel(nn.Module):
    """Complete MoE LLM with temperature-aware routing"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks with temperature-aware MoE
        self.transformer_blocks = nn.ModuleList(
            [
                TemperatureMoETransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.use_mla,
                    config.qk_rope_dim,
                    config.qk_nope_dim,
                    config.kv_lora_rank,
                    config.v_dim,
                    config.max_seq_len,
                    config.num_experts,
                    config.expert_top_k,
                    config.dropout,
                )
                for i in range(config.n_layers)
            ]
        )
        
        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def set_temperature(self, temperature: float):
        """Set routing temperature for all MoE layers"""
        for block in self.transformer_blocks:
            block.set_temperature(temperature)
    
    def forward(self, x, return_aux_loss=True, return_routing_stats=False):
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        
        # Collect auxiliary losses and routing stats
        aux_losses = []
        routing_stats_list = []
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            if return_routing_stats:
                x, aux_loss, routing_stats = block(x, return_routing_stats=True)
                if routing_stats is not None:
                    routing_stats_list.append(routing_stats)
            else:
                x, aux_loss = block(x, return_routing_stats=False)
            
            if aux_loss is not None and return_aux_loss:
                aux_losses.append(aux_loss)
        
        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        
        # Combine auxiliary losses
        total_aux_loss = sum(aux_losses) if aux_losses else None
        
        if return_routing_stats:
            return logits, total_aux_loss, routing_stats_list
        
        if return_aux_loss:
            return logits, total_aux_loss
        return logits


def create_temperature_moe_model(config: MoEModelConfig) -> TemperatureMoEModel:
    """
    Create a temperature-aware MoE model.
    
    Args:
        config: Model configuration
    
    Returns:
        TemperatureMoEModel instance
    """
    model = TemperatureMoEModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*80}")
    print(f"Temperature-aware MoE Model Created")
    print(f"{'='*80}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Number of experts: {config.num_experts}")
    print(f"Top-k routing: {config.expert_top_k}")
    print(f"{'='*80}\n")
    
    return model

