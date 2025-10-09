"""
Adaptive Sparsity Models for Experiment 3

This module implements models that integrate adaptive sparsity patterns with
DeepSeek's Multi-Head Latent Attention for optimal pretraining efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List

# Import from existing codebase
import sys
import os
sys.path.append('/root/deepseek-sparse-attention-research')

from models.components import MixtureOfExperts
from adaptive_sparsity import (
    DynamicSparsityController, 
    TopKTokenSelector, 
    create_sparse_mask
)
from simple_config import SimpleDeepSeekConfig


class LightningIndexer(nn.Module):
    """
    Lightning Indexer from DeepSeek sparse attention paper.
    
    Computes relevance scores between query and key tokens using lightweight
    attention heads with ReLU activation.
    """
    
    def __init__(self, d_model: int, indexer_heads: int = 4, indexer_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim
        
        # Indexer projections
        self.q_indexer = nn.Linear(d_model, indexer_heads * indexer_dim)
        self.k_indexer = nn.Linear(d_model, indexer_heads * indexer_dim)
        
        # Indexer weights (w_t,j^I from the paper) - ensure positive weights
        self.indexer_weights = nn.Parameter(torch.abs(torch.randn(indexer_heads) * 0.1))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.q_indexer.weight)
        nn.init.xavier_uniform_(self.k_indexer.weight)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            
        Returns:
            index_scores: [batch_size, seq_len, seq_len] - relevance scores
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Project to indexer space
        q_indexer = self.q_indexer(hidden_states)  # [batch_size, seq_len, indexer_heads * indexer_dim]
        k_indexer = self.k_indexer(hidden_states)  # [batch_size, seq_len, indexer_heads * indexer_dim]
        
        # Reshape for multi-head computation
        q_indexer = q_indexer.view(batch_size, seq_len, self.indexer_heads, self.indexer_dim)
        k_indexer = k_indexer.view(batch_size, seq_len, self.indexer_heads, self.indexer_dim)
        
        # Transpose for batch matrix multiplication
        q_indexer = q_indexer.transpose(1, 2)  # [batch_size, indexer_heads, seq_len, indexer_dim]
        k_indexer = k_indexer.transpose(1, 2)  # [batch_size, indexer_heads, seq_len, indexer_dim]
        
        # Compute dot products
        # q_indexer: [batch_size, indexer_heads, seq_len, indexer_dim]
        # k_indexer: [batch_size, indexer_heads, seq_len, indexer_dim]
        # We want: [batch_size, indexer_heads, seq_len, seq_len]
        dot_products = torch.matmul(q_indexer, k_indexer.transpose(-2, -1))
        
        # Apply ReLU activation
        dot_products = F.relu(dot_products)
        
        # Weight by indexer weights and sum over heads
        # indexer_weights: [indexer_heads] -> [1, indexer_heads, 1, 1]
        weighted_scores = dot_products * self.indexer_weights.view(1, -1, 1, 1)
        index_scores = weighted_scores.sum(dim=1)  # [batch_size, seq_len, seq_len]
        
        # Ensure non-negative scores
        index_scores = F.relu(index_scores)
        
        return index_scores


class AdaptiveDeepSeekSparseAttention(nn.Module):
    """
    DeepSeek Multi-Head Latent Attention with adaptive sparsity patterns.
    
    Integrates:
    1. Lightning Indexer for token relevance scoring
    2. Dynamic Sparsity Controller for adaptive k calculation
    3. DeepSeek's Multi-Head Latent Attention for efficient computation
    """
    
    def __init__(self, config, layer_idx: int, 
                 indexer_heads: int = 4, indexer_dim: int = 64):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Handle both dict and object configs
        if isinstance(config, dict):
            self.d_model = config['d_model']
        else:
            self.d_model = config.d_model
        
        # Import DeepSeek attention from existing codebase
        from models.deepseek_v3.deepseek_modeling import DeepseekV3Attention
        
        # Convert dict config to SimpleDeepSeekConfig if needed
        if isinstance(config, dict):
            deepseek_config = SimpleDeepSeekConfig(config)
        else:
            deepseek_config = config
            
        self.deepseek_attention = DeepseekV3Attention(deepseek_config, layer_idx)
        
        # Lightning Indexer
        self.indexer = LightningIndexer(
            self.d_model, indexer_heads, indexer_dim
        )
        
        # Dynamic Sparsity Controller
        if isinstance(config, dict):
            max_position_embeddings = config['max_position_embeddings']
        else:
            max_position_embeddings = config.max_position_embeddings
            
        self.sparsity_controller = DynamicSparsityController(
            self.d_model, max_position_embeddings
        )
        
        # Top-k Token Selector
        self.selector = TopKTokenSelector()
        
        # Track adaptive statistics
        self.adaptive_k_history = []
        self.sparsity_ratios_history = []
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with adaptive sparsity.
        
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask
            position_ids: Optional position embeddings
            past_key_value: Optional cached key-value states
            output_attentions: Whether to return attention weights
            use_cache: Whether to cache key-value states
            
        Returns:
            output: [batch_size, seq_len, d_model] - attention output
            past_key_value: Optional cached key-value states
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # 1. Calculate adaptive k values
        adaptive_k = self.sparsity_controller(hidden_states)
        
        # 2. Compute Lightning Indexer scores
        index_scores = self.indexer(hidden_states)
        
        # 3. Select top-k tokens with adaptive k
        top_k_mask, selected_indices = self.selector(index_scores, adaptive_k)
        
        # 4. Create sparse attention mask
        sparse_attention_mask = create_sparse_mask(top_k_mask)
        
        # 5. Apply DeepSeek attention with sparse mask
        if attention_mask is not None:
            # Combine with existing attention mask
            combined_mask = torch.minimum(attention_mask, sparse_attention_mask)
        else:
            combined_mask = sparse_attention_mask
        
        # Call DeepSeek attention
        attention_output = self.deepseek_attention(
            hidden_states=hidden_states,
            attention_mask=combined_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        
        # Track adaptive statistics
        with torch.no_grad():
            sparsity_ratio = 1.0 - (adaptive_k.float() / seq_len)
            self.adaptive_k_history.append(adaptive_k.float().mean().item())
            self.sparsity_ratios_history.append(sparsity_ratio.mean().item())
            
            # Keep only recent history to avoid memory issues
            if len(self.adaptive_k_history) > 1000:
                self.adaptive_k_history = self.adaptive_k_history[-500:]
                self.sparsity_ratios_history = self.sparsity_ratios_history[-500:]
        
        return attention_output
    
    def get_adaptive_stats(self) -> Dict[str, float]:
        """Get statistics about adaptive behavior."""
        if not self.adaptive_k_history:
            return {}
            
        return {
            'mean_adaptive_k': sum(self.adaptive_k_history) / len(self.adaptive_k_history),
            'mean_sparsity_ratio': sum(self.sparsity_ratios_history) / len(self.sparsity_ratios_history),
            'adaptive_k_std': torch.tensor(self.adaptive_k_history).std().item(),
            'sparsity_ratio_std': torch.tensor(self.sparsity_ratios_history).std().item(),
        }
    
    def reset_stats(self):
        """Reset adaptive statistics."""
        self.adaptive_k_history = []
        self.sparsity_ratios_history = []


class AdaptiveMoELLM(nn.Module):
    """
    Complete MoE LLM with adaptive sparsity attention.
    
    Integrates adaptive sparsity with:
    1. Multi-Head Latent Attention (DeepSeek style)
    2. Mixture of Experts (MoE) layers
    3. Adaptive sparsity patterns
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Handle both dict and object configs
        if isinstance(config, dict):
            self.d_model = config['d_model']
            self.n_layers = config['n_layers']
            self.n_heads = config['n_heads']
            self.d_ff = config['d_ff']
            self.vocab_size = config['vocab_size']
            self.max_seq_len = config['max_position_embeddings']
        else:
            self.d_model = config.d_model
            self.n_layers = config.n_layers
            self.n_heads = config.n_heads
            self.d_ff = config.d_ff
            self.vocab_size = config.vocab_size
            self.max_seq_len = config.max_position_embeddings
        
        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional embedding
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        
        # Transformer layers with adaptive sparsity
        self.layers = nn.ModuleList([
            AdaptiveTransformerLayer(config, layer_idx)
            for layer_idx in range(self.n_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: bool = False,
                use_cache: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the adaptive MoE model.
        
        Args:
            input_ids: [batch_size, seq_len] - input token ids
            attention_mask: Optional attention mask
            position_ids: Optional position embeddings
            labels: Optional labels for loss calculation
            output_attentions: Whether to return attention weights
            use_cache: Whether to cache intermediate states
            
        Returns:
            outputs: Dict containing logits, loss, and optional attention weights
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Transformer layers
        all_attentions = [] if output_attentions else None
        past_key_values = [] if use_cache else None
        
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions.append(layer_outputs[1])
                
            if use_cache:
                past_key_values.append(layer_outputs[-1])
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        outputs = {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }
        
        if output_attentions:
            outputs['attentions'] = all_attentions
            
        if use_cache:
            outputs['past_key_values'] = past_key_values
            
        return outputs
    
    def get_adaptive_stats(self) -> Dict[str, Dict[str, float]]:
        """Get adaptive statistics from all layers."""
        stats = {}
        for i, layer in enumerate(self.layers):
            layer_stats = layer.get_adaptive_stats()
            if layer_stats:
                stats[f'layer_{i}'] = layer_stats
        return stats
    
    def reset_adaptive_stats(self):
        """Reset adaptive statistics for all layers."""
        for layer in self.layers:
            layer.reset_adaptive_stats()


class AdaptiveTransformerLayer(nn.Module):
    """
    Single transformer layer with adaptive sparsity attention and MoE.
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Handle both dict and object configs
        if isinstance(config, dict):
            d_model = config['d_model']
            d_ff = config['d_ff']
            num_experts = config['num_experts']
            top_k = config['top_k']
            dropout = config['dropout']
        else:
            d_model = config.d_model
            d_ff = config.d_ff
            num_experts = config.num_experts
            top_k = config.top_k
            dropout = config.dropout
        
        # Adaptive sparse attention
        self.attention = AdaptiveDeepSeekSparseAttention(
            config, layer_idx, indexer_heads=4, indexer_dim=64
        )
        
        # MoE layer
        self.moe = MixtureOfExperts(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout
        )
        
        # Layer norms
        self.attention_norm = nn.LayerNorm(d_model)
        self.moe_norm = nn.LayerNorm(d_model)
        
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                output_attentions: bool = False,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of transformer layer.
        
        Returns:
            hidden_states: Layer output
            attention_weights: Optional attention weights
            past_key_value: Optional cached key-value states
        """
        # Self-attention with residual connection
        attn_outputs = self.attention(
            hidden_states=self.attention_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            attention_weights = attn_outputs[1] if output_attentions else None
            past_key_value = attn_outputs[-1] if use_cache else None
        else:
            attn_output = attn_outputs
            attention_weights = None
            past_key_value = None
            
        hidden_states = hidden_states + attn_output
        
        # MoE with residual connection
        moe_output, _ = self.moe(self.moe_norm(hidden_states))  # Ignore aux_loss
        hidden_states = hidden_states + moe_output
        
        outputs = (hidden_states,)
        if attention_weights is not None:
            outputs = outputs + (attention_weights,)
        if past_key_value is not None:
            outputs = outputs + (past_key_value,)
            
        return outputs
    
    def get_adaptive_stats(self) -> Dict[str, float]:
        """Get adaptive statistics from attention layer."""
        return self.attention.get_adaptive_stats()
    
    def reset_adaptive_stats(self):
        """Reset adaptive statistics."""
        self.attention.reset_stats()


# Baseline models for comparison
class FixedSparseMoELLM(nn.Module):
    """
    Baseline MoE LLM with fixed sparsity patterns.
    
    Uses the same architecture as AdaptiveMoELLM but with fixed k values.
    """
    
    def __init__(self, config, sparsity_ratio: float = 0.5):
        super().__init__()
        self.config = config
        self.sparsity_ratio = sparsity_ratio
        
        # Handle both dict and object configs
        if isinstance(config, dict):
            self.d_model = config['d_model']
            self.n_layers = config['n_layers']
            self.n_heads = config['n_heads']
            self.d_ff = config['d_ff']
            self.vocab_size = config['vocab_size']
            self.max_seq_len = config['max_position_embeddings']
        else:
            self.d_model = config.d_model
            self.n_layers = config.n_layers
            self.n_heads = config.n_heads
            self.d_ff = config.d_ff
            self.vocab_size = config.vocab_size
            self.max_seq_len = config.max_position_embeddings
        
        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional embedding
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        
        # Transformer layers with fixed sparse attention
        self.layers = nn.ModuleList([
            FixedSparseTransformerLayer(config, layer_idx, sparsity_ratio)
            for layer_idx in range(self.n_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: bool = False,
                use_cache: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with fixed sparsity."""
        batch_size, seq_len = input_ids.shape
        
        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Transformer layers
        all_attentions = [] if output_attentions else None
        past_key_values = [] if use_cache else None
        
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions.append(layer_outputs[1])
                
            if use_cache:
                past_key_values.append(layer_outputs[-1])
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        outputs = {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }
        
        if output_attentions:
            outputs['attentions'] = all_attentions
            
        if use_cache:
            outputs['past_key_values'] = past_key_values
            
        return outputs


class FixedSparseTransformerLayer(nn.Module):
    """
    Transformer layer with fixed sparse attention.
    """
    
    def __init__(self, config, layer_idx: int, sparsity_ratio: float):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.sparsity_ratio = sparsity_ratio
        
        # Handle both dict and object configs
        if isinstance(config, dict):
            d_model = config['d_model']
            d_ff = config['d_ff']
            num_experts = config['num_experts']
            top_k = config['top_k']
            dropout = config['dropout']
        else:
            d_model = config.d_model
            d_ff = config.d_ff
            num_experts = config.num_experts
            top_k = config.top_k
            dropout = config.dropout
        
        # Lightning Indexer
        self.indexer = LightningIndexer(d_model, indexer_heads=4, indexer_dim=64)
        
        # DeepSeek attention
        from models.deepseek_v3.deepseek_modeling import DeepseekV3Attention
        
        # Convert dict config to SimpleDeepSeekConfig if needed
        if isinstance(config, dict):
            deepseek_config = SimpleDeepSeekConfig(config)
        else:
            deepseek_config = config
            
        self.attention = DeepseekV3Attention(deepseek_config, layer_idx)
        
        # Top-k selector
        self.selector = TopKTokenSelector()
        
        # MoE layer
        self.moe = MixtureOfExperts(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout
        )
        
        # Layer norms
        self.attention_norm = nn.LayerNorm(d_model)
        self.moe_norm = nn.LayerNorm(d_model)
        
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                output_attentions: bool = False,
                use_cache: bool = False):
        """Forward pass with fixed sparsity."""
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Calculate fixed k
        fixed_k = int(seq_len * self.sparsity_ratio)
        fixed_k = torch.tensor([fixed_k] * batch_size, device=hidden_states.device)
        
        # Compute indexer scores
        index_scores = self.indexer(hidden_states)
        
        # Select top-k tokens
        top_k_mask, _ = self.selector(index_scores, fixed_k)
        
        # Create sparse attention mask
        sparse_attention_mask = create_sparse_mask(top_k_mask)
        
        # Apply attention
        if attention_mask is not None:
            combined_mask = torch.minimum(attention_mask, sparse_attention_mask)
        else:
            combined_mask = sparse_attention_mask
            
        attn_outputs = self.attention(
            hidden_states=self.attention_norm(hidden_states),
            attention_mask=combined_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            attention_weights = attn_outputs[1] if output_attentions else None
            past_key_value = attn_outputs[-1] if use_cache else None
        else:
            attn_output = attn_outputs
            attention_weights = None
            past_key_value = None
            
        hidden_states = hidden_states + attn_output
        
        # MoE
        moe_output, _ = self.moe(self.moe_norm(hidden_states))  # Ignore aux_loss
        hidden_states = hidden_states + moe_output
        
        outputs = (hidden_states,)
        if attention_weights is not None:
            outputs = outputs + (attention_weights,)
        if past_key_value is not None:
            outputs = outputs + (past_key_value,)
            
        return outputs


# Unit tests
if __name__ == "__main__":
    # Test adaptive models
    from types import SimpleNamespace
    
    config = SimpleNamespace(
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        vocab_size=10000,
        max_position_embeddings=2048,
        num_experts=8,
        top_k=2,
        dropout=0.1
    )
    
    # Test adaptive model
    model = AdaptiveMoELLM(config)
    
    # Test input
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Test adaptive stats
    stats = model.get_adaptive_stats()
    print(f"Adaptive stats: {stats}")
    
    # Test fixed sparse model
    fixed_model = FixedSparseMoELLM(config, sparsity_ratio=0.5)
    fixed_outputs = fixed_model(input_ids)
    
    print(f"Fixed model parameters: {sum(p.numel() for p in fixed_model.parameters()):,}")
    print(f"Fixed model loss: {fixed_outputs['loss'].item():.4f}")
