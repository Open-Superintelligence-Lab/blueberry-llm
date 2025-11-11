"""
Temperature-aware Mixture of Experts implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from temperature_router import TemperatureRouter


class Expert(nn.Module):
    """Single expert network (essentially a FeedForward layer)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))


class TemperatureMoE(nn.Module):
    """
    Mixture of Experts layer with temperature-controlled routing.
    
    This version extends the standard MoE with:
    1. Temperature-scaled routing
    2. Detailed routing statistics tracking
    3. Expert specialization analysis
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_weight = load_balancing_weight
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        
        # Create temperature-aware router
        self.router = TemperatureRouter(d_model, num_experts, top_k)
        
        # Expert statistics (accumulated over training)
        self.expert_activation_counts = torch.zeros(num_experts)
        self.expert_activation_history = []
    
    def set_temperature(self, temperature: float):
        """Set the routing temperature"""
        self.router.set_temperature(temperature)
    
    def forward(
        self,
        x: torch.Tensor,
        return_routing_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            return_routing_stats: Whether to return routing statistics
        
        Returns:
            - output: MoE output [batch_size, seq_len, d_model]
            - aux_loss: Load balancing auxiliary loss (only during training)
            - routing_stats: Routing statistics (if return_routing_stats=True)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get routing decisions
        router_weights, expert_indices, router_probs, routing_stats = self.router(x)
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Track expert activations
        expert_hits = torch.zeros(self.num_experts, device=x.device)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]
            
            if expert_mask.any():
                # Count activations
                expert_hits[expert_idx] = expert_mask.sum().item()
                
                # Get tokens for this expert
                expert_input = x[expert_mask]  # [num_tokens, d_model]
                
                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # Get weights for this expert
                mask_for_expert = (expert_indices == expert_idx)  # [batch, seq, top_k]
                positions = mask_for_expert[expert_mask].float().argmax(dim=-1)
                expert_weights = router_weights[expert_mask].gather(
                    -1, positions.unsqueeze(-1)
                ).squeeze(-1)
                
                # Add weighted expert output to result
                output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output
        
        # Update activation counts
        with torch.no_grad():
            self.expert_activation_counts += expert_hits.cpu()
        
        # Compute load balancing loss during training
        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balancing_loss(router_probs, expert_indices)
        
        # Return routing stats if requested
        if return_routing_stats:
            routing_stats['expert_hits'] = expert_hits.cpu().numpy().tolist()
            return output, aux_loss, routing_stats
        
        return output, aux_loss, None
    
    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to ensure balanced expert usage.
        This encourages the router to distribute tokens evenly across experts.
        """
        # Compute the fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=[0, 1, 2]) / expert_mask.sum()
        
        # Compute the average probability of routing to each expert
        router_prob_mean = router_probs.mean(dim=[0, 1])
        
        # Load balancing loss encourages uniform distribution
        aux_loss = torch.sum(tokens_per_expert * router_prob_mean) * self.num_experts
        
        return aux_loss * self.load_balancing_weight
    
    def get_expert_stats(self) -> dict:
        """Get expert activation statistics"""
        total_activations = self.expert_activation_counts.sum()
        return {
            'total_activations': total_activations.item(),
            'expert_counts': self.expert_activation_counts.numpy().tolist(),
            'expert_distribution': (self.expert_activation_counts / total_activations).numpy().tolist() if total_activations > 0 else None,
        }
    
    def reset_stats(self):
        """Reset expert statistics"""
        self.expert_activation_counts = torch.zeros(self.num_experts)
        self.expert_activation_history = []
        self.router.reset_stats()

