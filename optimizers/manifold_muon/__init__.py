"""
Manifold Muon Optimizers

This package implements manifold-constrained optimization for neural networks,
based on the modular manifolds framework.

Available optimizers:
- StiefelMuon: For matrices constrained to Stiefel manifold (W^T W = I)
- HypersphereMuon: For vectors constrained to unit hypersphere
- ComposedOptimizer: Combines multiple optimizers for different parameter groups
"""

from .stiefel_muon import StiefelMuon
from .hypersphere_muon import HypersphereMuon
from .composed_optimizer import ComposedOptimizer
from .core import (
    matrix_sign_newton_schulz,
    nuclear_norm,
    spectral_norm,
    project_to_tangent_stiefel,
)

__all__ = [
    'StiefelMuon',
    'HypersphereMuon',
    'ComposedOptimizer',
    'matrix_sign_newton_schulz',
    'nuclear_norm',
    'spectral_norm',
    'project_to_tangent_stiefel',
]




