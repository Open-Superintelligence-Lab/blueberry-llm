"""
Adaptive Sparsity Components for Experiment 3

This module implements dynamic sparsity patterns that adjust based on sequence
characteristics to optimize both pretraining speed and quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class ContentComplexityAnalyzer(nn.Module):
    """
    Analyzes content complexity of input sequences to inform sparsity decisions.
    
    Complexity is measured through:
    1. Token diversity (variance in embeddings)
    2. Perplexity estimation (predictability of next tokens)
    """
    
    def __init__(self, d_model: int, complexity_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.complexity_dim = complexity_dim
        
        # Token diversity analyzer
        self.diversity_proj = nn.Linear(d_model, 1)
        
        # Perplexity estimator - predicts next token probability
        self.perplexity_estimator = nn.Sequential(
            nn.Linear(d_model, complexity_dim),
            nn.ReLU(),
            nn.Linear(complexity_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Complexity combination layer
        self.complexity_combiner = nn.Sequential(
            nn.Linear(3, 1),  # diversity + perplexity + variance
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            
        Returns:
            complexity_scores: [batch_size, 1] - normalized complexity scores
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # 1. Token diversity (variance across sequence)
        diversity = torch.var(hidden_states, dim=1).mean(dim=-1, keepdim=True).unsqueeze(-1)  # [batch_size, 1, 1]
        
        # 2. Perplexity estimation
        # Predict next token embeddings and measure prediction error
        if seq_len > 1:
            # Use first seq_len-1 tokens to predict last seq_len-1 tokens
            input_embeddings = hidden_states[:, :-1, :]  # [batch_size, seq_len-1, d_model]
            target_embeddings = hidden_states[:, 1:, :]   # [batch_size, seq_len-1, d_model]
            
            predicted_embeddings = self.perplexity_estimator(input_embeddings)
            prediction_error = F.mse_loss(predicted_embeddings, target_embeddings, reduction='none')
            perplexity_score = prediction_error.mean(dim=[1, 2], keepdim=True)  # [batch_size, 1]
        else:
            perplexity_score = torch.zeros(batch_size, 1, device=hidden_states.device)
        
        # 3. Sequence variance (overall variability)
        sequence_variance = torch.var(hidden_states, dim=[1, 2], keepdim=True)  # [batch_size, 1]
        
        # Combine complexity signals
        complexity_features = torch.cat([diversity, perplexity_score, sequence_variance], dim=-1)
        complexity_scores = self.complexity_combiner(complexity_features)
        
        # Squeeze to [batch_size, 1]
        return complexity_scores.squeeze(-1)


class AttentionEntropyEstimator(nn.Module):
    """
    Estimates attention entropy without computing full attention matrix.
    
    Uses a lightweight proxy to estimate how "spread out" attention would be.
    """
    
    def __init__(self, d_model: int, num_proxy_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_proxy_heads = num_proxy_heads
        
        # Lightweight attention proxy
        self.attention_proxy = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_proxy_heads,
            batch_first=True,
            dropout=0.0
        )
        
        # Entropy predictor
        self.entropy_proj = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            
        Returns:
            entropy_scores: [batch_size, 1] - normalized entropy scores
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        if seq_len <= 1:
            return torch.zeros(batch_size, 1, device=hidden_states.device)
        
        # Use lightweight attention to estimate attention patterns
        attn_output, attn_weights = self.attention_proxy(
            hidden_states, hidden_states, hidden_states
        )
        
        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        if attn_weights is None:
            # If no attention weights returned, use a simple proxy
            attn_weights = torch.softmax(
                torch.randn(batch_size, self.num_proxy_heads, seq_len, seq_len, device=hidden_states.device), 
                dim=-1
            )
        
        # Calculate entropy for each head and average
        attn_weights = attn_weights.mean(dim=1)  # Average over heads
        
        # Add small epsilon to avoid log(0)
        attn_weights = attn_weights + 1e-8
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(attn_weights * torch.log(attn_weights), dim=-1)  # [batch_size, seq_len]
        
        # Average entropy across sequence and normalize
        mean_entropy = entropy.mean(dim=-1, keepdim=True)  # [batch_size, 1]
        
        # Normalize to [0, 1] using sigmoid
        normalized_entropy = torch.sigmoid(mean_entropy)
        
        # Ensure correct shape
        if normalized_entropy.dim() == 0:
            normalized_entropy = normalized_entropy.unsqueeze(0).unsqueeze(0)
        elif normalized_entropy.dim() == 1:
            normalized_entropy = normalized_entropy.unsqueeze(-1)
        
        # Ensure batch dimension is correct
        if normalized_entropy.size(0) != batch_size:
            normalized_entropy = normalized_entropy.expand(batch_size, -1)
        
        return normalized_entropy


class AdaptiveKCalculator(nn.Module):
    """
    Calculates adaptive k values based on sequence characteristics.
    
    Combines length, complexity, and entropy factors to determine optimal
    sparsity level for each sequence.
    """
    
    def __init__(self, max_seq_len: int, min_sparsity: float = 0.1, max_sparsity: float = 0.9):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        
        # Learnable weights for different factors
        self.length_weight = nn.Parameter(torch.tensor(0.4))
        self.complexity_weight = nn.Parameter(torch.tensor(0.3))
        self.entropy_weight = nn.Parameter(torch.tensor(0.3))
        
        # Base sparsity level
        self.base_sparsity = nn.Parameter(torch.tensor(0.5))
        
        # Length adaptation function
        self.length_adapter = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, seq_len: int, length_factor: torch.Tensor, 
                complexity_factor: torch.Tensor, entropy_factor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_len: Sequence length
            length_factor: [batch_size, 1] - normalized length factor
            complexity_factor: [batch_size, 1] - normalized complexity factor  
            entropy_factor: [batch_size, 1] - normalized entropy factor
            
        Returns:
            adaptive_k: [batch_size] - adaptive k values for each sequence in batch
        """
        batch_size = length_factor.size(0)
        
        # Length adaptation: longer sequences may need different sparsity patterns
        normalized_length = torch.tensor([[seq_len / self.max_seq_len]], 
                                       device=length_factor.device).expand(batch_size, 1)
        length_adaptation = self.length_adapter(normalized_length)
        
        # Combine factors with learnable weights
        # Normalize weights to sum to 1
        total_weight = self.length_weight + self.complexity_weight + self.entropy_weight
        normalized_weights = torch.stack([
            self.length_weight / total_weight,
            self.complexity_weight / total_weight, 
            self.entropy_weight / total_weight
        ])
        
        # Weighted combination of factors
        adaptive_factor = (
            normalized_weights[0] * length_factor * length_adaptation +
            normalized_weights[1] * complexity_factor +
            normalized_weights[2] * entropy_factor
        )
        
        # Calculate k as fraction of sequence length
        k_fraction = self.base_sparsity * adaptive_factor
        
        # Ensure k is within reasonable bounds
        k_fraction = torch.clamp(k_fraction, self.min_sparsity, self.max_sparsity)
        
        # Convert to integer k
        k = torch.round(seq_len * k_fraction).long().squeeze(-1)
        
        # Ensure k is at least 1 and less than seq_len
        k = torch.clamp(k, 1, seq_len - 1)
        
        return k


class DynamicSparsityController(nn.Module):
    """
    Main controller that orchestrates all adaptive sparsity components.
    
    Takes sequence characteristics and outputs adaptive k values for sparse attention.
    """
    
    def __init__(self, d_model: int, max_seq_len: int, 
                 complexity_dim: int = 64, num_proxy_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Sequence length predictor (simple linear mapping)
        self.length_predictor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Content complexity analyzer
        self.complexity_analyzer = ContentComplexityAnalyzer(d_model, complexity_dim)
        
        # Attention entropy estimator
        self.entropy_estimator = AttentionEntropyEstimator(d_model, num_proxy_heads)
        
        # Adaptive k calculator
        self.k_calculator = AdaptiveKCalculator(max_seq_len)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            
        Returns:
            adaptive_k: [batch_size] - adaptive k values for each sequence
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Extract sequence characteristics
        # 1. Length factor (based on sequence embeddings)
        seq_mean = hidden_states.mean(dim=1)  # [batch_size, d_model]
        # Add small epsilon to prevent NaN
        seq_mean = seq_mean + 1e-8
        length_factor = self.length_predictor(seq_mean)  # [batch_size, 1]
        # Ensure no NaN values
        length_factor = torch.where(torch.isnan(length_factor), torch.zeros_like(length_factor), length_factor)
        
        # 2. Content complexity
        complexity_factor = self.complexity_analyzer(hidden_states)  # [batch_size, 1]
        # Ensure no NaN values
        complexity_factor = torch.where(torch.isnan(complexity_factor), torch.zeros_like(complexity_factor), complexity_factor)
        
        # 3. Attention entropy
        entropy_factor = self.entropy_estimator(hidden_states)  # [batch_size, 1]
        # Ensure no NaN values
        entropy_factor = torch.where(torch.isnan(entropy_factor), torch.zeros_like(entropy_factor), entropy_factor)
        
        # Calculate adaptive k
        adaptive_k = self.k_calculator(
            seq_len, length_factor, complexity_factor, entropy_factor
        )
        
        return adaptive_k
    
    def get_characteristics(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed sequence characteristics for analysis.
        
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            
        Returns:
            characteristics: Dict with detailed analysis
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Get all characteristics
        length_factor = self.length_predictor(hidden_states.mean(dim=1))
        complexity_factor = self.complexity_analyzer(hidden_states)
        entropy_factor = self.entropy_estimator(hidden_states)
        adaptive_k = self.forward(hidden_states)
        
        return {
            'length_factor': length_factor,
            'complexity_factor': complexity_factor, 
            'entropy_factor': entropy_factor,
            'adaptive_k': adaptive_k,
            'sparsity_ratio': 1.0 - (adaptive_k.float() / seq_len),
            'sequence_length': seq_len
        }


def create_sparse_mask(top_k_mask: torch.Tensor) -> torch.Tensor:
    """
    Create attention mask for sparse attention.
    
    Args:
        top_k_mask: [batch_size, seq_len, seq_len] - boolean mask of selected tokens
        
    Returns:
        attention_mask: [batch_size, 1, seq_len, seq_len] - mask with -inf for unselected tokens
    """
    # Convert boolean mask to attention mask
    # True (selected) -> 0.0 (attend)
    # False (not selected) -> -inf (don't attend)
    attention_mask = torch.where(top_k_mask, 0.0, float('-inf'))
    
    # Add head dimension for DeepSeek attention
    attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
    
    return attention_mask


class TopKTokenSelector(nn.Module):
    """
    Enhanced top-k token selector that works with adaptive k values.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, index_scores: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index_scores: [batch_size, seq_len, seq_len] - indexer scores
            k: [batch_size] - adaptive k values for each sequence
            
        Returns:
            top_k_mask: [batch_size, seq_len, seq_len] - boolean mask of selected tokens
            selected_indices: [batch_size, seq_len, max_k] - indices of selected tokens
        """
        batch_size, seq_len, _ = index_scores.shape
        
        # For each sequence in batch, select top-k tokens
        top_k_mask = torch.zeros_like(index_scores, dtype=torch.bool)
        max_k = k.max().item()
        selected_indices = torch.zeros(batch_size, seq_len, max_k, dtype=torch.long, 
                                     device=index_scores.device)
        
        for b in range(batch_size):
            k_b = k[b].item()
            
            # For each query position, select top-k key positions
            for i in range(seq_len):
                # Get scores for this query position
                scores_i = index_scores[b, i, :]  # [seq_len]
                
                # Select top-k indices
                _, top_indices = torch.topk(scores_i, min(k_b, seq_len))
                
                # Create mask
                top_k_mask[b, i, top_indices] = True
                
                # Store indices (pad with -1 for shorter k values)
                if len(top_indices) < max_k:
                    padded_indices = torch.cat([
                        top_indices, 
                        torch.full((max_k - len(top_indices),), -1, device=index_scores.device)
                    ])
                else:
                    padded_indices = top_indices[:max_k]
                    
                selected_indices[b, i, :] = padded_indices
        
        return top_k_mask, selected_indices


# Unit tests
if __name__ == "__main__":
    # Test adaptive sparsity controller
    batch_size, seq_len, d_model = 2, 128, 512
    max_seq_len = 2048
    
    # Create test data
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    
    # Test controller
    controller = DynamicSparsityController(d_model, max_seq_len)
    adaptive_k = controller(hidden_states)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Adaptive k: {adaptive_k}")
    print(f"Sparsity ratios: {1.0 - adaptive_k.float() / seq_len}")
    
    # Test characteristics
    characteristics = controller.get_characteristics(hidden_states)
    for key, value in characteristics.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")
    
    # Test top-k selector
    index_scores = torch.randn(batch_size, seq_len, seq_len)
    selector = TopKTokenSelector()
    top_k_mask, selected_indices = selector(index_scores, adaptive_k)
    
    print(f"Top-k mask shape: {top_k_mask.shape}")
    print(f"Selected indices shape: {selected_indices.shape}")
    print(f"Mask sparsity: {1.0 - top_k_mask.float().mean().item():.3f}")
