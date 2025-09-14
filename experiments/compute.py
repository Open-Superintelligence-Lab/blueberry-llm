from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ComputeBudget:
    """Simple compute proxies for MoE FFN.

    We treat attention cost as constant across the grid. Proxies are
    sufficient to compare relative compute under fixed d_model.
    """

    d_model: int
    top_k: int
    d_ff_expert: int
    num_experts: int

    @property
    def active_params_per_token(self) -> int:
        # 2 * d_model * (top_k * d_ff_e)
        return 2 * self.d_model * (self.top_k * self.d_ff_expert)

    @property
    def flops_per_token_forward(self) -> int:
        # Same proxy constant as active params/token
        return self.active_params_per_token

    @property
    def total_expert_params(self) -> int:
        # 2 * d_model * (E * d_ff_e)
        return 2 * self.d_model * (self.num_experts * self.d_ff_expert)

    def as_dict(self) -> Dict[str, int]:
        return {
            "active_params_per_token": self.active_params_per_token,
            "flops_per_token_forward": self.flops_per_token_forward,
            "total_expert_params": self.total_expert_params,
        }


def d_ff_expert_for_fixed_active(d_ff_active: int, top_k: int) -> int:
    """Return expert width such that top_k * d_ff_e == d_ff_active."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return max(1, int(d_ff_active // top_k))


def d_ff_expert_for_fixed_total_params(
    d_ff_ref: int, e_ref: int, e_target: int
) -> int:
    """Return expert width to keep total expert params constant.

    E * d_ff_e == E_ref * d_ff_ref -> d_ff_e = (E_ref * d_ff_ref) / E
    """
    if e_target <= 0:
        raise ValueError("e_target must be positive")
    val = (e_ref * d_ff_ref) / e_target
    return max(1, int(val))

