"""
Core manifold operations for manifold Muon optimizers.

This module provides the fundamental operations needed for optimization
on matrix manifolds, particularly the Stiefel manifold.
"""

import torch
from typing import Tuple


@torch.compile
def matrix_sign_newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute the matrix sign function via Newton-Schulz iteration.
    
    The matrix sign function snaps singular values to +1 or -1.
    For matrices with all positive singular values, this is equivalent
    to projecting onto the Stiefel manifold.
    
    This is a key operation in manifold Muon, used both for:
    1. Computing the optimal update direction
    2. Retracting weights back to the manifold
    
    Args:
        G: Input matrix of shape (m, n)
        steps: Number of Newton-Schulz iterations (default: 5)
        eps: Small constant for numerical stability
    
    Returns:
        Matrix sign of G, same shape as G
        
    Notes:
        - Uses the 5th order Padé approximant for fast convergence
        - Automatically handles rectangular matrices
        - Compiled for efficiency
    """
    assert G.ndim == 2, f"Expected 2D matrix, got shape {G.shape}"
    
    # Padé approximant coefficients (5th order)
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Work in fp32 for numerical stability
    X = G.to(torch.float32)
    
    # Handle rectangular matrices by transposing if tall
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    
    # Normalize to improve conditioning
    norm = X.norm()
    if norm < eps:
        # Handle zero matrix edge case
        return torch.zeros_like(G)
    X = X / (norm + eps)
    
    # Newton-Schulz iteration: X_{k+1} = a·X_k + (b·A + c·A²)·X_k
    # where A = X_k @ X_k^T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    
    # Transpose back if needed
    if transposed:
        X = X.T
    
    return X.to(G.dtype)


def nuclear_norm(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the nuclear norm (sum of singular values) of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
    
    Returns:
        Nuclear norm (scalar tensor)
        
    Notes:
        - Nuclear norm = trace norm = sum of singular values
        - Used in the dual objective function
        - Dual of spectral norm
    """
    return torch.linalg.svdvals(A).sum()


def spectral_norm(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the spectral norm (largest singular value) of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
    
    Returns:
        Spectral norm (scalar tensor)
        
    Notes:
        - Spectral norm = operator norm = largest singular value
        - Used as the constraint in manifold Muon
        - Dual of nuclear norm
    """
    return torch.linalg.svdvals(A).max()


def project_to_tangent_stiefel(A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Project a matrix A onto the tangent space of the Stiefel manifold at W.
    
    The tangent space at W ∈ Stiefel(m,n) consists of all matrices A such that:
        A^T W + W^T A = 0
    
    The projection formula is:
        A_tan = A - W · sym(W^T A)
    where sym(X) = (X + X^T) / 2
    
    Args:
        A: Matrix to project, shape (m, n)
        W: Point on Stiefel manifold, shape (m, n), satisfying W^T W = I
    
    Returns:
        Projected matrix A_tan in tangent space at W
        
    Notes:
        - Used to ensure updates respect the manifold constraint
        - The projection is orthogonal w.r.t. Frobenius inner product
    """
    assert A.shape == W.shape, f"Shape mismatch: A {A.shape} vs W {W.shape}"
    
    WtA = W.T @ A
    WtA_sym = (WtA + WtA.T) / 2
    A_tan = A - W @ WtA_sym
    
    return A_tan


def compute_dual_gradient(
    W: torch.Tensor, 
    G: torch.Tensor, 
    Lambda: torch.Tensor, 
    lr: float,
    ns_steps: int = 5
) -> torch.Tensor:
    """
    Compute the gradient of the dual function H(Λ) for Stiefel Muon.
    
    The dual function is:
        D(Λ) = -η · ||G + 2W(Λ + Λ^T)||_nuclear
    
    Its gradient is:
        H(Λ) = ∇_Λ D(Λ) 
             = -η · [W^T · msign(G + 2W(Λ+Λ^T)) + msign(...)^T · W]
    
    Args:
        W: Current weights on Stiefel manifold, shape (m, n)
        G: Gradient of loss w.r.t. W, shape (m, n)
        Lambda: Current dual variable, shape (n, n)
        lr: Learning rate (η)
        ns_steps: Newton-Schulz iterations for matrix sign
    
    Returns:
        Gradient H(Λ), shape (n, n)
        
    Notes:
        - This is used in the dual ascent algorithm
        - Lambda should be symmetric (we enforce this by using Λ + Λ^T)
    """
    assert W.shape[1] == Lambda.shape[0] == Lambda.shape[1], \
        f"Dimension mismatch: W {W.shape}, Lambda {Lambda.shape}"
    
    # Make Lambda symmetric
    Lambda_sym = Lambda + Lambda.T
    
    # Compute argument to matrix sign
    arg = G + 2 * (W @ Lambda_sym)
    
    # Compute matrix sign
    M = matrix_sign_newton_schulz(arg, steps=ns_steps)
    
    # Compute dual gradient
    H = -lr * (W.T @ M + M.T @ W)
    
    return H


def compute_optimal_update(
    W: torch.Tensor,
    G: torch.Tensor,
    Lambda: torch.Tensor,
    lr: float,
    ns_steps: int = 5
) -> torch.Tensor:
    """
    Compute the optimal update direction A_opt for Stiefel Muon.
    
    Given the optimal dual variable Λ*, the primal optimal update is:
        A_opt = -η · msign(G + 2W(Λ* + Λ*^T))
    
    Args:
        W: Current weights, shape (m, n)
        G: Gradient, shape (m, n)
        Lambda: Optimal dual variable, shape (n, n)
        lr: Learning rate
        ns_steps: Newton-Schulz iterations
    
    Returns:
        Optimal update A_opt, shape (m, n)
    """
    Lambda_sym = Lambda + Lambda.T
    arg = G + 2 * (W @ Lambda_sym)
    M = matrix_sign_newton_schulz(arg, steps=ns_steps)
    A_opt = -lr * M
    
    return A_opt


def retract_to_stiefel(W: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
    """
    Retract a matrix back to the Stiefel manifold.
    
    After taking a step in the tangent space, we need to project back
    to the manifold. For the Stiefel manifold with spectral norm,
    the retraction is simply the matrix sign function.
    
    Args:
        W: Matrix to retract, shape (m, n)
        ns_steps: Newton-Schulz iterations
    
    Returns:
        Retracted matrix on Stiefel manifold, satisfying W^T W = I
    """
    return matrix_sign_newton_schulz(W, steps=ns_steps)


def check_stiefel_constraint(W: torch.Tensor, tol: float = 1e-4) -> Tuple[bool, float]:
    """
    Check if a matrix satisfies the Stiefel constraint W^T W = I.
    
    Args:
        W: Matrix to check, shape (m, n) with m >= n
        tol: Tolerance for constraint violation
    
    Returns:
        (satisfied, error) where:
            satisfied: True if ||W^T W - I|| < tol
            error: Frobenius norm of constraint violation
    """
    n = W.size(1)
    WtW = W.T @ W
    I = torch.eye(n, device=W.device, dtype=W.dtype)
    error = torch.norm(WtW - I, p='fro').item()
    
    return error < tol, error


def initialize_on_stiefel(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """
    Initialize a matrix on the Stiefel manifold.
    
    Uses QR decomposition of a random matrix to get orthonormal columns.
    
    Args:
        shape: (m, n) with m >= n
        device: Device to create tensor on
    
    Returns:
        Matrix W with W^T W = I
    """
    m, n = shape
    assert m >= n, f"Expected m >= n for Stiefel manifold, got {m} < {n}"
    
    # Random initialization
    W = torch.randn(m, n, device=device)
    
    # QR decomposition gives orthonormal columns
    Q, R = torch.linalg.qr(W)
    
    # Handle sign ambiguity: make diagonal of R positive
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)
    
    return Q


# Utility functions for monitoring

def singular_values(W: torch.Tensor) -> torch.Tensor:
    """Compute singular values of a matrix."""
    return torch.linalg.svdvals(W)


def condition_number(W: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute condition number (ratio of max to min singular value)."""
    sv = singular_values(W)
    return sv.max() / (sv.min() + eps)


def is_well_conditioned(W: torch.Tensor, max_condition: float = 100.0) -> bool:
    """Check if matrix is well-conditioned."""
    return condition_number(W).item() < max_condition




