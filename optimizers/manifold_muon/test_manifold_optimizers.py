"""
Test script for manifold optimizers.

This script demonstrates basic usage and validates the manifold constraints.
Run with: python -m optimizers.manifold_muon.test_manifold_optimizers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .stiefel_muon import StiefelMuon
from .hypersphere_muon import HypersphereMuon
from .composed_optimizer import ComposedOptimizer
from .core import check_stiefel_constraint


def test_stiefel_muon_basic():
    """Test basic Stiefel Muon optimization on a toy problem."""
    print("=" * 60)
    print("Testing Stiefel Muon - Basic")
    print("=" * 60)
    
    # Create a simple linear layer
    d_in, d_out = 64, 64
    W = nn.Parameter(torch.randn(d_out, d_in))
    
    # Target: identity matrix (should be on Stiefel manifold)
    target = torch.eye(d_out, d_in)
    
    # Create optimizer
    optimizer = StiefelMuon([W], lr=0.1, dual_steps=5)
    
    print(f"Initial constraint error: {check_stiefel_constraint(W.data)[1]:.6f}")
    
    # Optimization loop
    for step in range(100):
        optimizer.zero_grad()
        
        # Loss: Frobenius norm to target
        loss = F.mse_loss(W, target)
        loss.backward()
        
        optimizer.step()
        
        if step % 20 == 0:
            satisfied, error = check_stiefel_constraint(W.data)
            print(f"Step {step:3d}: Loss={loss.item():.6f}, "
                  f"Constraint Error={error:.6f}, Satisfied={satisfied}")
    
    # Final check
    satisfied, error = check_stiefel_constraint(W.data)
    print(f"\nFinal constraint error: {error:.6f}")
    print(f"Constraint satisfied: {satisfied}")
    
    # Check singular values (should all be 1)
    U, S, V = torch.svd(W.data)
    print(f"Singular values: min={S.min():.4f}, max={S.max():.4f}, "
          f"mean={S.mean():.4f}")
    
    assert satisfied, "Stiefel constraint not satisfied!"
    assert error < 1e-3, f"Constraint error too large: {error}"
    print("✓ Test passed!\n")


def test_hypersphere_muon_basic():
    """Test basic Hypersphere Muon optimization."""
    print("=" * 60)
    print("Testing Hypersphere Muon - Basic")
    print("=" * 60)
    
    # Create embedding vectors
    vocab_size, d_model = 100, 64
    embeddings = nn.Parameter(torch.randn(vocab_size, d_model))
    
    # Target: random unit vectors
    torch.manual_seed(42)
    target = torch.randn(vocab_size, d_model)
    target = target / target.norm(dim=1, keepdim=True)
    
    # Create optimizer
    optimizer = HypersphereMuon([embeddings], lr=0.1, mode='per_row')
    
    # Check initial norms
    initial_norms = embeddings.data.norm(dim=1)
    print(f"Initial norms: min={initial_norms.min():.4f}, "
          f"max={initial_norms.max():.4f}, mean={initial_norms.mean():.4f}")
    
    # Optimization loop
    for step in range(100):
        optimizer.zero_grad()
        
        # Loss: cosine distance to target
        cos_sim = F.cosine_similarity(embeddings, target, dim=1)
        loss = (1 - cos_sim).mean()
        loss.backward()
        
        optimizer.step()
        
        if step % 20 == 0:
            norms = embeddings.data.norm(dim=1)
            stats = optimizer.check_constraints()
            print(f"Step {step:3d}: Loss={loss.item():.6f}, "
                  f"Norm error={stats['max_violation']:.6f}")
    
    # Final check
    norms = embeddings.data.norm(dim=1)
    print(f"\nFinal norms: min={norms.min():.4f}, max={norms.max():.4f}, "
          f"mean={norms.mean():.4f}")
    
    norm_errors = (norms - 1.0).abs()
    print(f"Max norm error: {norm_errors.max():.6f}")
    
    assert norm_errors.max() < 1e-3, f"Norm error too large: {norm_errors.max()}"
    print("✓ Test passed!\n")


def test_composed_optimizer():
    """Test composed optimizer with multiple parameter types."""
    print("=" * 60)
    print("Testing Composed Optimizer")
    print("=" * 60)
    
    # Create a simple model with different parameter types
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Parameter(torch.randn(50, 32))
            self.W1 = nn.Parameter(torch.randn(32, 32))
            self.W2 = nn.Parameter(torch.randn(32, 32))
            self.scale = nn.Parameter(torch.ones(32))
        
        def forward(self, x):
            x = self.embedding[x]
            x = F.linear(x, self.W1)
            x = F.relu(x)
            x = F.linear(x, self.W2)
            x = x * self.scale
            return x
    
    model = SimpleModel()
    
    # Create composed optimizer
    param_groups = [
        {
            'params': [model.embedding],
            'optimizer': 'hypersphere_muon',
            'mode': 'per_row',
            'lr_scale': 1.0,
        },
        {
            'params': [model.W1, model.W2],
            'optimizer': 'stiefel_muon',
            'dual_steps': 5,
            'lr_scale': 1.0,
        },
        {
            'params': [model.scale],
            'optimizer': 'adamw',
            'lr_scale': 10.0,
        }
    ]
    
    optimizer = ComposedOptimizer(param_groups, base_lr=0.02)
    
    # Dummy training loop
    for step in range(50):
        optimizer.zero_grad()
        
        # Dummy forward pass
        indices = torch.randint(0, 50, (16,))
        output = model(indices)
        target = torch.randn_like(output)
        
        loss = F.mse_loss(output, target)
        loss.backward()
        
        optimizer.step()
        
        if step % 10 == 0:
            stats = optimizer.check_constraints()
            print(f"Step {step:3d}: Loss={loss.item():.6f}, "
                  f"Violations={stats['total_violations']}")
    
    # Check final constraints
    stats = optimizer.check_constraints()
    print(f"\nFinal constraint violations: {stats['total_violations']}")
    print(f"Constraint details: {stats['by_type']}")
    
    assert stats['total_violations'] == 0, "Constraints violated!"
    print("✓ Test passed!\n")


def test_rectangular_matrices():
    """Test Stiefel Muon on rectangular (non-square) matrices."""
    print("=" * 60)
    print("Testing Stiefel Muon - Rectangular Matrices")
    print("=" * 60)
    
    # Test different aspect ratios
    test_cases = [
        (128, 64, "tall"),   # More rows than columns
        (64, 128, "wide"),   # More columns than rows
        (64, 64, "square"),  # Square
    ]
    
    for m, n, shape_type in test_cases:
        print(f"\nTesting {shape_type} matrix: {m}×{n}")
        
        W = nn.Parameter(torch.randn(m, n))
        optimizer = StiefelMuon([W], lr=0.1, dual_steps=5)
        
        # Simple optimization: minimize Frobenius norm
        for step in range(20):
            optimizer.zero_grad()
            loss = W.norm() ** 2
            loss.backward()
            optimizer.step()
        
        # Check constraint
        W_check = W.data if m >= n else W.data.T
        satisfied, error = check_stiefel_constraint(W_check)
        
        print(f"  Constraint error: {error:.6f}, Satisfied: {satisfied}")
        assert satisfied, f"Constraint not satisfied for {shape_type} matrix!"
    
    print("✓ All rectangular matrix tests passed!\n")


def test_learning_rate_scaling():
    """Test learning rate scaling in composed optimizer."""
    print("=" * 60)
    print("Testing Learning Rate Scaling")
    print("=" * 60)
    
    # Create parameters with different scales
    W1 = nn.Parameter(torch.randn(32, 32))
    W2 = nn.Parameter(torch.randn(32, 32))
    W3 = nn.Parameter(torch.randn(32, 32))
    
    param_groups = [
        {'params': [W1], 'optimizer': 'stiefel_muon', 'lr_scale': 0.5},
        {'params': [W2], 'optimizer': 'stiefel_muon', 'lr_scale': 1.0},
        {'params': [W3], 'optimizer': 'stiefel_muon', 'lr_scale': 2.0},
    ]
    
    base_lr = 0.02
    optimizer = ComposedOptimizer(param_groups, base_lr=base_lr)
    
    # Check learning rates
    lrs = optimizer.get_lr()
    print(f"Learning rates: {lrs}")
    
    expected_lrs = [0.01, 0.02, 0.04]
    for lr, expected_lr in zip(lrs, expected_lrs):
        assert abs(lr - expected_lr) < 1e-6, f"LR mismatch: {lr} vs {expected_lr}"
    
    # Test changing base LR
    new_base_lr = 0.01
    optimizer.set_lr(new_base_lr)
    lrs = optimizer.get_lr()
    print(f"After setting base_lr={new_base_lr}: {lrs}")
    
    expected_lrs = [0.005, 0.01, 0.02]
    for lr, expected_lr in zip(lrs, expected_lrs):
        assert abs(lr - expected_lr) < 1e-6, f"LR mismatch: {lr} vs {expected_lr}"
    
    print("✓ Test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MANIFOLD OPTIMIZER TEST SUITE")
    print("=" * 60 + "\n")
    
    # Run all tests
    try:
        test_stiefel_muon_basic()
        test_hypersphere_muon_basic()
        test_composed_optimizer()
        test_rectangular_matrices()
        test_learning_rate_scaling()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise




