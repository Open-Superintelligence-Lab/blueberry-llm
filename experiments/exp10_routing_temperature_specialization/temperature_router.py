"""
Temperature-aware router for MoE experiments
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemperatureRouter(nn.Module):
    """
    Router that selects top-k experts with temperature-scaled softmax.
    
    Temperature controls the sharpness of the routing distribution:
    - Low temperature (< 1.0): Sharp, confident routing (exploitation)
    - Temperature = 1.0: Standard softmax (baseline)
    - High temperature (> 1.0): Soft, exploratory routing (exploration)
    - Very high temperature (>> 1.0): Nearly uniform routing
    """
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise_std = 0.1  # Standard deviation for noise during training
        
        # Temperature is set dynamically during forward pass
        self.current_temperature = 1.0
        
        # Statistics tracking
        self.expert_counts = None
        self.routing_entropy_history = []
        self.selection_confidence_history = []
    
    def set_temperature(self, temperature: float):
        """Set the current routing temperature"""
        self.current_temperature = max(temperature, 0.01)  # Avoid division by zero
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            - router_weights: Softmax weights for selected experts [batch_size, seq_len, top_k]
            - expert_indices: Indices of selected experts [batch_size, seq_len, top_k]
            - router_probs: Full probability distribution over experts (for load balancing loss)
            - routing_stats: Dictionary with routing statistics
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute router logits
        router_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        # Apply temperature scaling
        scaled_logits = router_logits / self.current_temperature
        
        # Get full probability distribution (for load balancing loss and analysis)
        router_probs = F.softmax(scaled_logits, dim=-1)
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(scaled_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute routing statistics
        routing_stats = self._compute_routing_stats(
            router_probs, top_k_weights, top_k_indices
        )
        
        return top_k_weights, top_k_indices, router_probs, routing_stats
    
    def _compute_routing_stats(
        self,
        router_probs: torch.Tensor,
        top_k_weights: torch.Tensor,
        top_k_indices: torch.Tensor
    ) -> dict:
        """
        Compute routing statistics for analysis.
        
        Returns dictionary with:
        - routing_entropy: Average entropy of routing distribution
        - selection_confidence: Average confidence in top-1 expert
        - expert_utilization: Fraction of tokens routed to each expert
        """
        with torch.no_grad():
            # Routing entropy: measure of routing diversity
            # High entropy = more uniform routing, low entropy = sharp routing
            entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10), dim=-1)
            avg_entropy = entropy.mean().item()
            
            # Selection confidence: how strongly the top expert is preferred
            top1_confidence = top_k_weights[:, :, 0].mean().item()
            
            # Expert utilization: how many tokens each expert processes
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
            expert_usage = expert_mask.sum(dim=[0, 1, 2]) / (expert_mask.sum() + 1e-10)
            
            # Update running statistics
            if self.expert_counts is None:
                self.expert_counts = expert_usage.cpu()
            else:
                self.expert_counts = 0.9 * self.expert_counts + 0.1 * expert_usage.cpu()
            
            return {
                'routing_entropy': avg_entropy,
                'selection_confidence': top1_confidence,
                'expert_utilization': expert_usage.cpu().numpy().tolist(),
                'temperature': self.current_temperature,
            }
    
    def get_routing_summary(self) -> dict:
        """Get summary statistics for the entire training run"""
        return {
            'final_expert_counts': self.expert_counts.numpy().tolist() if self.expert_counts is not None else None,
            'routing_entropy_history': self.routing_entropy_history,
            'selection_confidence_history': self.selection_confidence_history,
        }
    
    def reset_stats(self):
        """Reset routing statistics"""
        self.expert_counts = None
        self.routing_entropy_history = []
        self.selection_confidence_history = []

