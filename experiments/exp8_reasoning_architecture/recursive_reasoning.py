"""
Recursive Reasoning Module for Experiment 8
Integrates TRM-style recursive reasoning with FLA GatedDeltaNet

Based on Tiny Recursive Models (TRM) with:
- Hierarchical reasoning (H and L levels)
- Adaptive Compute Time (ACT) with Q-learning
- Carry state management for iterative refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from fla.models import GatedDeltaNetForCausalLM


@dataclass
class RecursiveCarryState:
    """
    Carry state for recursive reasoning
    Maintains reasoning state across iterations
    """
    z_H: torch.Tensor  # High-level reasoning state (batch, seq_len, hidden)
    z_L: torch.Tensor  # Low-level reasoning state (batch, seq_len, hidden)
    steps: torch.Tensor  # Number of reasoning steps taken (batch,)
    halted: torch.Tensor  # Whether sequence has halted (batch,)


class AdaptiveComputeHead(nn.Module):
    """
    Q-learning based halting mechanism from TRM
    Learns when to stop reasoning based on Q-values
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        # Two-head Q network: Q(halt) vs Q(continue)
        self.q_head = nn.Linear(hidden_size, 2, bias=True)
        
        # Initialize to almost zero for bootstrapping
        # Bias toward NOT halting initially (-5.0 makes sigmoid very small)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        
        Returns:
            q_halt: Q-value for halting (batch,)
            q_continue: Q-value for continuing (batch,)
        """
        # Use first token position (like TRM uses puzzle_emb position)
        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)
        return q_logits[..., 0], q_logits[..., 1]


class RecursiveReasoningWrapper(nn.Module):
    """
    Wraps FLA GatedDeltaNet with recursive reasoning capabilities
    
    Architecture:
    - Base: Exp7 winner (Hybrid Sparse 17% - DeltaNet + Attention)
    - Added: Hierarchical recursive cycles (H and L levels)
    - Added: Adaptive Compute Time (ACT) for dynamic reasoning depth
    
    Key insight: Don't modify FLA internals, just wrap and iterate
    """
    
    def __init__(self, base_model: GatedDeltaNetForCausalLM, config: dict):
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        
        # Extract dimensions from base model
        # Handle both hidden_size (standard) and d_model (MoE) naming conventions
        self.hidden_size = getattr(base_model.config, 'hidden_size', None) or getattr(base_model.config, 'd_model', None)
        if self.hidden_size is None:
            raise ValueError("Could not determine hidden size from base model config")
        self.vocab_size = base_model.config.vocab_size
        
        # Recursive reasoning parameters
        self.H_cycles = config.get('H_cycles', 3)  # High-level cycles
        self.L_cycles = config.get('L_cycles', 3)  # Low-level cycles
        self.halt_max_steps = config.get('halt_max_steps', 5)
        self.halt_exploration_prob = config.get('halt_exploration_prob', 0.1)
        
        # ACT halting mechanism
        self.use_act = config.get('use_act', True)
        if self.use_act:
            self.halt_head = AdaptiveComputeHead(self.hidden_size)
        
        # Initial states for carry (learnable parameters)
        # Small random initialization
        self.H_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
        self.L_init = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
        
        # Projection layers for hierarchical interaction
        # These allow H and L levels to communicate
        self.H_to_L_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.L_to_H_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Layer normalization for stability
        self.z_H_norm = nn.LayerNorm(self.hidden_size)
        self.z_L_norm = nn.LayerNorm(self.hidden_size)
    
    def initial_carry(self, batch_size: int, seq_len: int, device: torch.device) -> RecursiveCarryState:
        """Initialize carry state for new sequences"""
        return RecursiveCarryState(
            z_H=self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).clone(),
            z_L=self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).clone(),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=device)  # Start halted
        )
    
    def reset_carry(self, reset_flag: torch.Tensor, carry: RecursiveCarryState) -> RecursiveCarryState:
        """Reset carry for sequences that have halted"""
        batch_size, seq_len, _ = carry.z_H.shape
        device = carry.z_H.device
        
        H_init_expanded = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        L_init_expanded = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        return RecursiveCarryState(
            z_H=torch.where(
                reset_flag.view(-1, 1, 1), 
                H_init_expanded,
                carry.z_H
            ),
            z_L=torch.where(
                reset_flag.view(-1, 1, 1),
                L_init_expanded,
                carry.z_L
            ),
            steps=torch.where(reset_flag, torch.zeros_like(carry.steps), carry.steps),
            halted=reset_flag | carry.halted
        )
    
    def recursive_forward(self, 
                         input_ids: torch.Tensor, 
                         attention_mask: Optional[torch.Tensor] = None,
                         carry: Optional[RecursiveCarryState] = None) -> Tuple[torch.Tensor, RecursiveCarryState, Dict]:
        """
        Forward pass with recursive reasoning
        
        Process:
        1. Get base model embeddings/representations
        2. Run H_cycles of high-level reasoning
        3. Each H cycle contains L_cycles of low-level reasoning
        4. Use ACT to decide when to halt
        
        Returns:
            logits: Language model logits
            new_carry: Updated carry state
            metrics: Dict with reasoning metrics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize carry if not provided
        if carry is None:
            carry = self.initial_carry(batch_size, seq_len, device)
        
        # Reset carry for halted sequences
        carry = self.reset_carry(carry.halted, carry)
        
        # Get base model's embedding layer output
        # Handle both MoE (token_embedding) and HF-style (model.embeddings) models
        if hasattr(self.base_model, 'token_embedding'):
            embeddings = self.base_model.token_embedding(input_ids)
            # MoE model scales embeddings - match that scaling
            if hasattr(self.base_model.config, 'd_model'):
                import math
                embeddings = embeddings * math.sqrt(self.base_model.config.d_model)
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embeddings'):
            embeddings = self.base_model.model.embeddings(input_ids)
        else:
            raise ValueError("Could not find embedding layer in base model")
        
        # Recursive reasoning cycles
        z_H, z_L = carry.z_H, carry.z_L
        
        # Run H_cycles - 1 without gradient (efficiency like TRM)
        with torch.no_grad():
            for h_step in range(self.H_cycles - 1):
                # L-level cycles (low-level reasoning)
                for l_step in range(self.L_cycles):
                    # Inject input at L level with H-level guidance
                    l_input_embeds = self.z_L_norm(z_L + self.H_to_L_proj(z_H) + embeddings)
                    
                    # Run through base model layers
                    l_output = self.base_model(
                        inputs_embeds=l_input_embeds,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                        return_aux_loss=True
                    )
                    
                    # Update L state with residual
                    z_L = z_L + l_output.last_hidden_state
                
                # H-level update (high-level reasoning)
                h_input_embeds = self.z_H_norm(z_H + self.L_to_H_proj(z_L))
                h_output = self.base_model(
                    inputs_embeds=h_input_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    return_aux_loss=True
                )
                z_H = z_H + h_output.last_hidden_state
        
        # Final H cycle WITH gradient for learning
        for l_step in range(self.L_cycles):
            l_input_embeds = self.z_L_norm(z_L + self.H_to_L_proj(z_H) + embeddings)
            l_output = self.base_model(
                inputs_embeds=l_input_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                return_aux_loss=True
            )
            z_L = z_L + l_output.last_hidden_state
        
        h_input_embeds = self.z_H_norm(z_H + self.L_to_H_proj(z_L))
        h_output = self.base_model(
            inputs_embeds=h_input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            return_aux_loss=True
        )
        z_H = z_H + h_output.last_hidden_state
        
        # Generate final logits through base model's LM head
        # Use the logits from the last output instead of calling lm_head again
        logits = h_output.logits
        
        # ACT halting decision
        halt_decision = torch.zeros(batch_size, dtype=torch.bool, device=device)
        q_halt_logits = None
        q_continue_logits = None
        
        if self.use_act:
            q_halt_logits, q_continue_logits = self.halt_head(z_H)
            
            with torch.no_grad():
                new_steps = carry.steps + 1
                is_last_step = new_steps >= self.halt_max_steps
                
                # Halt if: (1) max steps reached, OR (2) Q(halt) > Q(continue)
                if self.training:
                    halt_decision = is_last_step | (q_halt_logits > q_continue_logits)
                    
                    # Exploration: randomly force longer reasoning
                    if self.halt_exploration_prob > 0:
                        explore = torch.rand(batch_size, device=device) < self.halt_exploration_prob
                        min_steps = torch.randint(2, self.halt_max_steps + 1, (batch_size,), device=device)
                        halt_decision = halt_decision & ((new_steps >= min_steps) | ~explore)
                else:
                    # During inference, always use max steps for consistency
                    halt_decision = is_last_step
        else:
            new_steps = carry.steps + 1
            halt_decision = new_steps >= self.halt_max_steps
        
        # Create new carry (detach for next iteration)
        new_carry = RecursiveCarryState(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            steps=carry.steps + 1,
            halted=halt_decision
        )
        
        # Metrics for monitoring
        metrics = {
            'reasoning_steps': carry.steps.float().mean().item(),
            'halt_rate': halt_decision.float().mean().item(),
            'q_halt_mean': q_halt_logits.mean().item() if q_halt_logits is not None else 0.0,
            'q_continue_mean': q_continue_logits.mean().item() if q_continue_logits is not None else 0.0,
        }
        
        return logits, new_carry, metrics
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                carry: Optional[RecursiveCarryState] = None,
                **kwargs) -> Dict:
        """
        Standard forward compatible with training loop
        
        Returns dict compatible with HuggingFace-style outputs
        """
        logits, new_carry, metrics = self.recursive_forward(input_ids, attention_mask, carry)
        
        loss = None
        if labels is not None:
            # Standard cross entropy loss (shift for causal LM)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Optional: Add ACT regularization loss
            # Encourage efficient halting (fewer steps = lower cost)
            if self.use_act and self.training:
                act_penalty = 0.01 * new_carry.steps.float().mean()
                loss = loss + act_penalty
                metrics['act_penalty'] = act_penalty.item()
        
        # Return dict with HuggingFace-compatible structure
        return {
            'loss': loss,
            'logits': logits,
            'carry': new_carry,
            'metrics': metrics,
        }


def create_recursive_reasoning_model(
    base_model: GatedDeltaNetForCausalLM, 
    recursive_config: Optional[dict] = None
) -> RecursiveReasoningWrapper:
    """
    Factory function to create recursive reasoning model
    
    Args:
        base_model: Your exp7 winner GatedDeltaNet model (Hybrid Sparse 17%)
        recursive_config: Dict with:
            - H_cycles: Number of high-level reasoning cycles (default: 3)
            - L_cycles: Number of low-level reasoning cycles (default: 3)
            - halt_max_steps: Maximum reasoning steps (default: 5)
            - halt_exploration_prob: Exploration probability for ACT (default: 0.1)
            - use_act: Whether to use adaptive compute time (default: True)
    
    Returns:
        RecursiveReasoningWrapper with recursive reasoning capabilities
    """
    if recursive_config is None:
        recursive_config = {
            'H_cycles': 3,
            'L_cycles': 3,
            'halt_max_steps': 5,
            'halt_exploration_prob': 0.1,
            'use_act': True
        }
    
    return RecursiveReasoningWrapper(base_model, recursive_config)

