"""
Model Definitions for Experiment 5: MHLA with/without Sparse Attention

This module defines:
1. Baseline Model: DeepSeek MHLA (dense)
2. Sparse Model: DeepSeek MHLA + Sparse Attention

Both use DeepSeek's Multi-Head Latent Attention with MoE.
The only difference is sparse token selection via Lightning Indexer.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os

# Add parent directories to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

# Import from parent models package
from models.components import MixtureOfExperts
from models.deepseek_v3.deepseek_modeling import DeepseekV3Attention, DeepseekV3RMSNorm
from models.deepseek_v3.configuration_deepseek import DeepseekV3Config

# Import local sparse_mhla_attention module
from sparse_mhla_attention import DeepSeekSparseMLHA


class SparseMLHATransformerBlock(nn.Module):
    """
    Transformer block with DeepSeek Sparse MHLA and MoE
    """
    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: int,
        num_experts: int = 4,
        expert_top_k: int = 2,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        sparse_top_k: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # DeepSeek Sparse MHLA
        self.attention = DeepSeekSparseMLHA(
            config=config,
            layer_idx=layer_idx,
            indexer_heads=indexer_heads,
            indexer_dim=indexer_dim,
            sparse_top_k=sparse_top_k
        )
        
        # MoE layer (GLM4 style)
        self.feed_forward = MixtureOfExperts(
            d_model=config.hidden_size,
            d_ff=config.intermediate_size,
            num_experts=num_experts,
            top_k=expert_top_k,
            dropout=dropout
        )
        
        # Normalization layers
        self.norm1 = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_index_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass
        
        Returns:
            - output: Block output
            - aux_loss: MoE auxiliary loss
            - index_scores: Index scores if return_index_scores=True
        """
        # Self-attention with sparse MHLA
        attn_out, _, _, index_scores = self.attention(
            self.norm1(x),
            attention_mask=attention_mask,
            return_index_scores=return_index_scores
        )
        x = x + self.dropout(attn_out)
        
        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x, aux_loss, index_scores


class BaselineMLHATransformerBlock(nn.Module):
    """
    Transformer block with DeepSeek MHLA (dense, no sparse) and MoE
    """
    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: int,
        num_experts: int = 4,
        expert_top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # DeepSeek MHLA (dense)
        self.attention = DeepseekV3Attention(
            config=config,
            layer_idx=layer_idx
        )
        
        # MoE layer (GLM4 style)
        self.feed_forward = MixtureOfExperts(
            d_model=config.hidden_size,
            d_ff=config.intermediate_size,
            num_experts=num_experts,
            top_k=expert_top_k,
            dropout=dropout
        )
        
        # Normalization layers
        self.norm1 = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Returns:
            - output: Block output
            - aux_loss: MoE auxiliary loss
        """
        # Self-attention with dense MHLA
        attn_out, _, _ = self.attention(
            self.norm1(x),
            attention_mask=attention_mask
        )
        x = x + self.dropout(attn_out)
        
        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x, aux_loss


class SparseMLHAMoELLM(nn.Module):
    """
    Language Model with DeepSeek Sparse MHLA and MoE
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 4,
        expert_top_k: int = 2,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        sparse_top_k: int = 512,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01,
        # MHLA specific params
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: int = 64,
        qk_rope_head_dim: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.load_balancing_weight = load_balancing_weight
        
        # Create DeepSeek config for MHLA
        self.config = DeepseekV3Config(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            intermediate_size=d_ff,
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len,
            attention_dropout=dropout,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim if qk_rope_head_dim else d_model // n_heads,
            qk_nope_head_dim=qk_nope_head_dim if qk_nope_head_dim is not None else 0,
            v_head_dim=v_head_dim if v_head_dim else d_model // n_heads
        )
        
        # Token embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks with sparse MHLA
        self.blocks = nn.ModuleList([
            SparseMLHATransformerBlock(
                config=self.config,
                layer_idx=i,
                num_experts=num_experts,
                expert_top_k=expert_top_k,
                indexer_heads=indexer_heads,
                indexer_dim=indexer_dim,
                sparse_top_k=sparse_top_k,
                dropout=dropout
            )
            for i in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = DeepseekV3RMSNorm(d_model, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight
        
    def _create_causal_mask(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for MHLA"""
        # Create 4D mask: (batch_size, 1, seq_len, seq_len)
        mask = torch.triu(
            torch.ones(batch_size, 1, seq_len, seq_len, device=device),
            diagonal=1
        )
        # Convert to additive mask (0 for attend, -inf for mask)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(
        self, 
        x: torch.Tensor,
        return_index_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Forward pass
        
        Args:
            x: Input token indices [batch_size, seq_len]
            return_index_scores: Whether to return index scores
            
        Returns:
            - logits: Output logits [batch_size, seq_len, vocab_size]
            - total_aux_loss: Total auxiliary loss (MoE load balancing)
            - index_scores_list: List of index scores per layer (if requested)
        """
        # Embed tokens
        x = self.embed(x)
        batch_size, seq_len, _ = x.shape
        
        # Create causal attention mask
        attention_mask = self._create_causal_mask(batch_size, seq_len, x.device)
        
        # Pass through transformer blocks
        total_aux_loss = 0.0
        index_scores_list = [] if return_index_scores else None
        
        for block in self.blocks:
            x, aux_loss, index_scores = block(
                x,
                attention_mask=attention_mask,
                return_index_scores=return_index_scores
            )
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
            if return_index_scores and index_scores is not None:
                index_scores_list.append(index_scores)
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Scale auxiliary loss
        if total_aux_loss != 0.0:
            total_aux_loss = total_aux_loss * self.load_balancing_weight
        else:
            total_aux_loss = None
            
        return logits, total_aux_loss, index_scores_list
    
    def enable_sparse_attention(self):
        """Enable sparse attention in all layers"""
        for block in self.blocks:
            block.attention.enable_sparse()
            
    def disable_sparse_attention(self):
        """Disable sparse attention (use dense MHLA) in all layers"""
        for block in self.blocks:
            block.attention.disable_sparse()


class BaselineMLHAMoELLM(nn.Module):
    """
    Language Model with DeepSeek MHLA (dense) and MoE (baseline)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 4,
        expert_top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01,
        # MHLA specific params
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: int = 64,
        qk_rope_head_dim: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.load_balancing_weight = load_balancing_weight
        
        # Create DeepSeek config for MHLA
        self.config = DeepseekV3Config(
            hidden_size=d_model,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            intermediate_size=d_ff,
            vocab_size=vocab_size,
            max_position_embeddings=max_seq_len,
            attention_dropout=dropout,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim if qk_rope_head_dim else d_model // n_heads,
            qk_nope_head_dim=qk_nope_head_dim if qk_nope_head_dim is not None else 0,
            v_head_dim=v_head_dim if v_head_dim else d_model // n_heads
        )
        
        # Token embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks with dense MHLA
        self.blocks = nn.ModuleList([
            BaselineMLHATransformerBlock(
                config=self.config,
                layer_idx=i,
                num_experts=num_experts,
                expert_top_k=expert_top_k,
                dropout=dropout
            )
            for i in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = DeepseekV3RMSNorm(d_model, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight
        
    def _create_causal_mask(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for MHLA"""
        # Create 4D mask: (batch_size, 1, seq_len, seq_len)
        mask = torch.triu(
            torch.ones(batch_size, 1, seq_len, seq_len, device=device),
            diagonal=1
        )
        # Convert to additive mask (0 for attend, -inf for mask)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input token indices [batch_size, seq_len]
            
        Returns:
            - logits: Output logits [batch_size, seq_len, vocab_size]
            - total_aux_loss: Total auxiliary loss (MoE load balancing)
        """
        # Embed tokens
        x = self.embed(x)
        batch_size, seq_len, _ = x.shape
        
        # Create causal attention mask
        attention_mask = self._create_causal_mask(batch_size, seq_len, x.device)
        
        # Pass through transformer blocks
        total_aux_loss = 0.0
        
        for block in self.blocks:
            x, aux_loss = block(x, attention_mask=attention_mask)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Scale auxiliary loss
        if total_aux_loss != 0.0:
            total_aux_loss = total_aux_loss * self.load_balancing_weight
        else:
            total_aux_loss = None
            
        return logits, total_aux_loss


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_sparse_model(config: dict) -> SparseMLHAMoELLM:
    """
    Create sparse MHLA model from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: SparseMLHAMoELLM instance
    """
    model = SparseMLHAMoELLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        num_experts=config.get('num_experts', 4),
        expert_top_k=config.get('expert_top_k', 2),
        indexer_heads=config.get('indexer_heads', 4),
        indexer_dim=config.get('indexer_dim', 64),
        sparse_top_k=config.get('sparse_top_k', 512),
        dropout=config.get('dropout', 0.1),
        load_balancing_weight=config.get('load_balancing_weight', 0.01),
        q_lora_rank=config.get('q_lora_rank', None),
        kv_lora_rank=config.get('kv_lora_rank', 64),
        qk_rope_head_dim=config.get('qk_rope_head_dim', None),
        qk_nope_head_dim=config.get('qk_nope_head_dim', None),
        v_head_dim=config.get('v_head_dim', None)
    )
    return model


def create_baseline_model(config: dict) -> BaselineMLHAMoELLM:
    """
    Create baseline MHLA model from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model: BaselineMLHAMoELLM instance
    """
    model = BaselineMLHAMoELLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        num_experts=config.get('num_experts', 4),
        expert_top_k=config.get('expert_top_k', 2),
        dropout=config.get('dropout', 0.1),
        load_balancing_weight=config.get('load_balancing_weight', 0.01),
        q_lora_rank=config.get('q_lora_rank', None),
        kv_lora_rank=config.get('kv_lora_rank', 64),
        qk_rope_head_dim=config.get('qk_rope_head_dim', None),
        qk_nope_head_dim=config.get('qk_nope_head_dim', None),
        v_head_dim=config.get('v_head_dim', None)
    )
    return model

