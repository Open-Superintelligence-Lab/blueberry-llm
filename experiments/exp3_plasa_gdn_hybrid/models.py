"""
Three model variants for Qwen3-Next experiment with Adaptive Sparse Attention:
1. BaselineQwen3: Standard Qwen3-Next
2. PLASAQwen3: All attention replaced with Per-Layer Adaptive Sparse Attention
3. HybridQwen3: Adaptive Sparse Attention for full_attention, Gated DeltaNet for linear_attention

This experiment tests the PROGRESSIVE_SPARSE schedule:
- Early layers: Dense (k=L)
- Middle layers: Aggressive sparse (k=L/4)
- Late layers: Moderate sparse (k=L/2)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os

# Add root to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# Import Qwen3-Next components
from models.qwen3_next.modular_qwen3_next import (
    Qwen3NextModel,
    Qwen3NextForCausalLM,
    Qwen3NextDecoderLayer,
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextRMSNorm,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
    Qwen3NextDynamicCache,
    Qwen3NextRotaryEmbedding,
)
from models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

# Import Adaptive Sparse Attention
from experiments.exp3_plasa_gdn_hybrid.adaptive_sparse_attention import (
    AdaptiveSparseAttention,
    SparsitySchedule,
    create_sparsity_schedule,
)

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack


class PLASADecoderLayer(nn.Module):
    """
    Decoder layer that uses Per-Layer Adaptive Sparse Attention for ALL attention
    (replaces both full_attention and linear_attention)
    """
    def __init__(self, config: Qwen3NextConfig, layer_idx: int, layer_top_k: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Use Per-Layer Adaptive Sparse Attention with layer-specific k
        self.self_attn = AdaptiveSparseAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            layer_idx=layer_idx,
            layer_top_k=layer_top_k,
            indexer_heads=getattr(config, 'indexer_heads', 4),
            indexer_dim=getattr(config, 'indexer_dim', 64),
            dropout=config.attention_dropout,
        )
        
        # MLP (same as Qwen3-Next)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3NextMLP(config, intermediate_size=config.intermediate_size)
        
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # DeepSeek Sparse Attention
        hidden_states, _ = self.self_attn(hidden_states)
        
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        
        return hidden_states


class HybridDecoderLayer(nn.Module):
    """
    Hybrid decoder layer:
    - full_attention layers use Per-Layer Adaptive Sparse Attention
    - linear_attention layers use Gated DeltaNet (original)
    """
    def __init__(self, config: Qwen3NextConfig, layer_idx: int, layer_top_k: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        # Token mixer - choose based on layer type
        if self.layer_type == "linear_attention":
            # Keep original Gated DeltaNet
            self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            # Replace with Per-Layer Adaptive Sparse Attention
            self.self_attn = AdaptiveSparseAttention(
                d_model=config.hidden_size,
                n_heads=config.num_attention_heads,
                max_seq_len=config.max_position_embeddings,
                layer_idx=layer_idx,
                layer_top_k=layer_top_k,
                indexer_heads=getattr(config, 'indexer_heads', 4),
                indexer_dim=getattr(config, 'indexer_dim', 64),
                dropout=config.attention_dropout,
            )
        
        # MLP (same as Qwen3-Next)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3NextMLP(config, intermediate_size=config.intermediate_size)
        
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Token Mixer - based on layer type
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            # Use DeepSeek Sparse Attention
            hidden_states, _ = self.self_attn(hidden_states)
        
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        
        return hidden_states


# ==================== Model Variants ====================

class BaselineQwen3(Qwen3NextForCausalLM):
    """
    Variant 1: Standard Qwen3-Next (no changes)
    Uses original full_attention and linear_attention layers
    """
    pass  # No changes needed - use as-is


class PLASAQwen3Model(nn.Module):
    """
    Variant 2: All attention layers replaced with Per-Layer Adaptive Sparse Attention
    Uses PROGRESSIVE_SPARSE schedule: Early=Dense, Middle=L/4, Late=L/2
    """
    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # Create sparsity schedule for per-layer k values
        sparsity_config = create_sparsity_schedule(
            schedule=SparsitySchedule.PROGRESSIVE_SPARSE,
            n_layers=config.num_hidden_layers,
            seq_len=config.max_position_embeddings
        )

        # Replace all layers with PLASA decoder layers with layer-specific k values
        self.layers = nn.ModuleList([
            PLASADecoderLayer(
                config,
                layer_idx,
                layer_top_k=sparsity_config.get_k_for_layer(layer_idx, config.max_position_embeddings)
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3NextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if cache_position is None:
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        
        hidden_states = self.norm(hidden_states)
        
        return type('ModelOutput', (), {
            'last_hidden_state': hidden_states,
            'past_key_values': past_key_values,
        })()


class PLASAQwen3(nn.Module):
    """Variant 2: Per-Layer Adaptive Sparse Attention (PLASA)-Only Qwen3 (for CausalLM)"""
    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.config = config
        self.model = PLASAQwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return type('CausalLMOutput', (), {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.past_key_values,
        })()


class HybridQwen3Model(nn.Module):
    """
    Variant 3: Hybrid model with PROGRESSIVE_SPARSE schedule
    - full_attention → Per-Layer Adaptive Sparse Attention (Early=Dense, Middle=L/4, Late=L/2)
    - linear_attention → Gated DeltaNet (original)
    """
    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # Create sparsity schedule for per-layer k values
        sparsity_config = create_sparsity_schedule(
            schedule=SparsitySchedule.PROGRESSIVE_SPARSE,
            n_layers=config.num_hidden_layers,
            seq_len=config.max_position_embeddings
        )

        # Use hybrid decoder layers with layer-specific k values
        self.layers = nn.ModuleList([
            HybridDecoderLayer(
                config,
                layer_idx,
                layer_top_k=sparsity_config.get_k_for_layer(layer_idx, config.max_position_embeddings)
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3NextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            past_key_values = Qwen3NextDynamicCache(config=self.config)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        
        hidden_states = self.norm(hidden_states)
        
        return type('ModelOutput', (), {
            'last_hidden_state': hidden_states,
            'past_key_values': past_key_values,
        })()


class HybridQwen3(nn.Module):
    """Variant 3: Hybrid Qwen3 (for CausalLM)"""
    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.config = config
        self.model = HybridQwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return type('CausalLMOutput', (), {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.past_key_values,
        })()


class EnhancedDecoderLayer(nn.Module):
    """
    Enhanced decoder layer that supports all three layer types:
    - full_attention: Standard Qwen3 attention
    - linear_attention: Gated DeltaNet
    - plasa_attention: Per-Layer Adaptive Sparse Attention
    """
    def __init__(self, config: Qwen3NextConfig, layer_idx: int, layer_top_k: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        # Token mixer - choose based on layer type
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(config, layer_idx)
        elif self.layer_type == "plasa_attention":
            self.plasa_attn = AdaptiveSparseAttention(
                d_model=config.hidden_size,
                n_heads=config.num_attention_heads,
                max_seq_len=config.max_position_embeddings,
                layer_idx=layer_idx,
                layer_top_k=layer_top_k,
                indexer_heads=getattr(config, 'indexer_heads', 4),
                indexer_dim=getattr(config, 'indexer_dim', 64),
                dropout=config.attention_dropout,
            )
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")
        
        # MLP (same as Qwen3-Next)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3NextMLP(config, intermediate_size=config.intermediate_size)
        
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Token Mixer - based on layer type
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        elif self.layer_type == "plasa_attention":
            hidden_states, _ = self.plasa_attn(hidden_states)
        
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        
        return hidden_states


class EnhancedQwen3NextModel(nn.Module):
    """Enhanced Qwen3 model supporting full_attention, linear_attention, and plasa_attention"""
    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # Create sparsity schedule for per-layer k values
        sparsity_config = create_sparsity_schedule(
            schedule=SparsitySchedule.PROGRESSIVE_SPARSE,
            n_layers=config.num_hidden_layers,
            seq_len=config.max_position_embeddings
        )

        # Use enhanced decoder layers with layer-specific k values
        self.layers = nn.ModuleList([
            EnhancedDecoderLayer(
                config,
                layer_idx,
                layer_top_k=sparsity_config.get_k_for_layer(layer_idx, config.max_position_embeddings)
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3NextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            past_key_values = Qwen3NextDynamicCache(config=self.config)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        
        hidden_states = self.norm(hidden_states)
        
        return type('ModelOutput', (), {
            'last_hidden_state': hidden_states,
            'past_key_values': past_key_values,
        })()


class EnhancedQwen3NextForCausalLM(nn.Module):
    """Enhanced CausalLM supporting full_attention, linear_attention, and dsa_attention"""
    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.config = config
        self.model = EnhancedQwen3NextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return type('CausalLMOutput', (), {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.past_key_values,
        })()


def create_model(variant: str, config):
    """
    Factory function to create model variants

    Args:
        variant: One of ["baseline", "plasa", "hybrid"]
        config: ExperimentConfig

    Returns:
        Model instance
    """
    # Convert ExperimentConfig to Qwen3NextConfig
    qwen_config = Qwen3NextConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        attention_dropout=config.attention_dropout,
        hidden_dropout_prob=config.hidden_dropout,
        rms_norm_eps=config.rms_norm_eps,
        layer_types=config.layer_types,
        linear_num_value_heads=config.linear_num_value_heads,
        linear_num_key_heads=config.linear_num_key_heads,
        linear_key_head_dim=config.linear_key_head_dim,
        linear_value_head_dim=config.linear_value_head_dim,
        linear_conv_kernel_dim=config.linear_conv_kernel_dim,
        num_experts=config.num_experts,
        num_local_experts=config.num_experts,  # Add for Mixtral compatibility
        router_jitter_noise=0.0,  # Add for Mixtral compatibility
        decoder_sparse_step=config.decoder_sparse_step,
        mlp_only_layers=config.mlp_only_layers,
        # Adaptive Sparse Attention specific
        indexer_heads=config.indexer_heads,
        indexer_dim=config.indexer_dim,
        sparse_top_k=config.sparse_top_k,  # Note: This is overridden by per-layer schedule
    )

    if variant == "baseline":
        return BaselineQwen3(qwen_config)
    elif variant == "plasa":
        return PLASAQwen3(qwen_config)
    elif variant == "hybrid":
        return HybridQwen3(qwen_config)
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose from: baseline, plasa, hybrid")

