"""
Main training loop and utilities.

This module provides the main training function and related utilities
for training LLM models with GPU-adaptive optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import math
import time
import os
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from configs import AdaptiveMoEModelConfig
from optimizers import setup_optimizers, get_lr_scheduler
from .evaluation import evaluate_model, compute_model_metrics
from system import print_system_info


def train_with_megatron(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: AdaptiveMoEModelConfig,
    device: Optional[torch.device] = None,
    resume_from_checkpoint: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Training function optimized for Megatron-LM distributed training.
    
    Args:
        model: Megatron-wrapped model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Model configuration
        device: Device to train on (auto-detected if None)
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Tuple of (trained_model, final_metrics)
    """
    print("🚀 Starting Megatron-LM distributed training...")
    
    # Initialize distributed training if not already done
    if not torch.distributed.is_initialized():
        try:
            # Try to initialize with environment variables (for torchrun)
            torch.distributed.init_process_group(backend='nccl')
            print("✅ Distributed training initialized")
            
            # Move model to appropriate GPU rank
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device(f'cuda:{local_rank}')
            model = model.to(device)
            print(f"✅ Model moved to GPU {local_rank}")
            
        except Exception as e:
            print(f"⚠️ Distributed initialization failed: {e}")
            print("   Falling back to native training")
            return train_model_native(model, train_loader, val_loader, config, device, resume_from_checkpoint)
    
    # Use the existing training logic but with Megatron optimizations
    # For minimal implementation, we'll use the same training loop
    # Future enhancement: integrate Megatron's training utilities
    return train_model_native(model, train_loader, val_loader, config, device, resume_from_checkpoint)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: AdaptiveMoEModelConfig,
    device: Optional[torch.device] = None,
    resume_from_checkpoint: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Main training function dispatcher for adaptive LLM models.
    
    Automatically selects the appropriate training backend based on model type.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Model configuration
        device: Device to train on (auto-detected if None)
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Tuple of (trained_model, final_metrics)
    """
    # Check if this is a Megatron-wrapped model
    from models.megatron_wrapper import is_megatron_model
    
    if is_megatron_model(model):
        return train_with_megatron(model, train_loader, val_loader, config, device, resume_from_checkpoint)
    else:
        return train_model_native(model, train_loader, val_loader, config, device, resume_from_checkpoint)


def train_model_native(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: AdaptiveMoEModelConfig,
    device: Optional[torch.device] = None,
    resume_from_checkpoint: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Native training function for adaptive LLM models.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Model configuration
        device: Device to train on (auto-detected if None)
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Tuple of (trained_model, final_metrics)
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Print system and model information
    print(f"\n🚀 Training {model.__class__.__name__}")
    print("=" * 60)
    print_system_info()
    config.print_config()
    
    # Print model parameter info
    _print_model_parameter_info(model)
    
    # Setup optimizers and schedulers
    optimizers = setup_optimizers(model, config, use_warmup=True)
    schedulers = [get_lr_scheduler(opt, config, "cosine_warmup") for opt in optimizers]
    
    # Setup gradient scaler for mixed precision
    scaler = GradScaler('cuda') if config.use_amp else None
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from_checkpoint:
        start_step = _load_checkpoint(model, optimizers, schedulers, scaler, resume_from_checkpoint)
        print(f"📁 Resumed training from step {start_step}")
    
    # Training state
    training_state = TrainingState(
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        scaler=scaler,
        config=config,
        device=device,
        start_step=start_step
    )
    
    # Main training loop
    final_metrics = _training_loop(training_state, train_loader, val_loader)
    
    print(f"\n🎯 Training completed!")
    print(f"📊 Final metrics: {final_metrics}")
    
    return model, final_metrics


class TrainingState:
    """Helper class to hold training state."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        schedulers: List,
        scaler: Optional[GradScaler],
        config: AdaptiveMoEModelConfig,
        device: torch.device,
        start_step: int = 0
    ):
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.scaler = scaler
        self.config = config
        self.device = device
        self.step = start_step
        self.best_val_loss = float('inf')
        self.training_start_time = time.time()


def _training_loop(
    state: TrainingState,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> Dict[str, Any]:
    """
    Main training loop.
    
    Args:
        state: Training state
        train_loader: Training data loader
        val_loader: Validation data loader
        
    Returns:
        Final training metrics
    """
    state.model.train()
    pbar = tqdm(total=state.config.max_steps, initial=state.step, desc="Training")
    
    # Training metrics
    total_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0
    
    while state.step < state.config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if state.step >= state.config.max_steps:
                break
            
            x, y = x.to(state.device), y.to(state.device)
            
            # Forward pass and backward pass
            loss_dict = _training_step(state, x, y)
            
            # Accumulate metrics
            total_loss += loss_dict['ce_loss']
            total_aux_loss += loss_dict.get('aux_loss', 0.0)
            num_batches += 1
            
            # Increment step counter first
            state.step += 1

            # Optimizer step (only when accumulation is complete)
            if state.step % state.config.gradient_accumulation_steps == 0:
                _optimizer_step(state)

            # Logging
            if state.step % 100 == 0:
                _log_training_progress(state, loss_dict, pbar)

            # Evaluation
            if state.step % state.config.eval_every == 0 and state.step > 0:
                eval_metrics = _evaluate_and_log(state, val_loader)

                # Check for best model
                if eval_metrics['val_loss'] < state.best_val_loss:
                    state.best_val_loss = eval_metrics['val_loss']
                    print(f"🎉 New best validation loss: {state.best_val_loss:.4f}")

            # Milestone evaluations
            if state.step in state.config.log_milestones:
                eval_metrics = _evaluate_and_log(state, val_loader)
                print(f"\n🧪 Milestone {state.step}: Val Loss: {eval_metrics['val_loss']:.4f}")

            # Update progress bar
            if state.step % 20 == 0:
                pbar.update(20)
    
    pbar.close()
    
    # Final evaluation
    final_metrics = _evaluate_and_log(state, val_loader, final=True)
    
    # Add training summary
    training_time = time.time() - state.training_start_time
    final_metrics.update({
        'training_time_minutes': training_time / 60,
        'training_time_hours': training_time / 3600,
        'steps_per_second': state.step / training_time,
        'final_step': state.step,
        'best_val_loss': state.best_val_loss
    })
    
    return final_metrics


def _training_step(
    state: TrainingState,
    x: torch.Tensor,
    y: torch.Tensor
) -> Dict[str, float]:
    """
    Single training step.
    
    Args:
        state: Training state
        x: Input tensor
        y: Target tensor
        
    Returns:
        Dictionary with loss values
    """
    if state.config.use_amp:
        with autocast('cuda'):
            # Handle models with auxiliary loss (MoE)
            if hasattr(state.model, 'forward') and 'return_aux_loss' in state.model.forward.__code__.co_varnames:
                logits, aux_loss = state.model(x, return_aux_loss=True)
            else:
                logits = state.model(x)
                aux_loss = None
            
            # Compute main loss
            ce_loss = F.cross_entropy(
                logits.view(-1, state.config.vocab_size), 
                y.view(-1)
            )
            
            # Check for NaN/inf in main loss
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                print(f"⚠️  WARNING: Invalid CE loss detected: {ce_loss}")
                print(f"   Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"   Logits contains NaN: {torch.isnan(logits).any()}")
                print(f"   Logits contains Inf: {torch.isinf(logits).any()}")
                # Set loss to a safe value to continue training
                ce_loss = torch.tensor(10.0, device=ce_loss.device, dtype=ce_loss.dtype)
            
            # Combine losses
            total_loss = ce_loss
            if aux_loss is not None:
                if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                    print(f"⚠️  WARNING: Invalid aux loss detected: {aux_loss}")
                    aux_loss = torch.tensor(0.0, device=aux_loss.device, dtype=aux_loss.dtype)
                total_loss = total_loss + aux_loss
            
            # Scale for gradient accumulation
            loss = total_loss / state.config.gradient_accumulation_steps
        
        # Backward pass with scaling
        state.scaler.scale(loss).backward()
    else:
        # Forward pass without AMP
        if hasattr(state.model, 'forward') and 'return_aux_loss' in state.model.forward.__code__.co_varnames:
            logits, aux_loss = state.model(x, return_aux_loss=True)
        else:
            logits = state.model(x)
            aux_loss = None
        
        ce_loss = F.cross_entropy(
            logits.view(-1, state.config.vocab_size), 
            y.view(-1)
        )
        
        # Check for NaN/inf in main loss
        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            print(f"⚠️  WARNING: Invalid CE loss detected: {ce_loss}")
            print(f"   Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            print(f"   Logits contains NaN: {torch.isnan(logits).any()}")
            print(f"   Logits contains Inf: {torch.isinf(logits).any()}")
            # Set loss to a safe value to continue training
            ce_loss = torch.tensor(10.0, device=ce_loss.device, dtype=ce_loss.dtype)
        
        total_loss = ce_loss
        if aux_loss is not None:
            if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                print(f"⚠️  WARNING: Invalid aux loss detected: {aux_loss}")
                aux_loss = torch.tensor(0.0, device=aux_loss.device, dtype=aux_loss.dtype)
            total_loss = total_loss + aux_loss
        
        loss = total_loss / state.config.gradient_accumulation_steps
        loss.backward()
    
    return {
        'ce_loss': ce_loss.item(),
        'aux_loss': aux_loss.item() if aux_loss is not None else 0.0,
        'total_loss': total_loss.item()
    }


def _optimizer_step(state: TrainingState):
    """
    Perform optimizer step with gradient clipping.
    
    Args:
        state: Training state
    """
    if state.config.use_amp:
        # Unscale gradients for clipping
        for optimizer in state.optimizers:
            state.scaler.unscale_(optimizer)
        
        # Check for NaN/inf gradients before clipping
        total_norm = 0.0
        param_count = 0
        for param in state.model.parameters():
            if param.grad is not None:
                param_count += 1
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"⚠️  WARNING: Invalid gradient detected in parameter")
                    print(f"   Param shape: {param.shape}")
                    print(f"   Grad contains NaN: {torch.isnan(param.grad).any()}")
                    print(f"   Grad contains Inf: {torch.isinf(param.grad).any()}")
                    # Zero out invalid gradients
                    param.grad.data.zero_()
        
        total_norm = total_norm ** (1. / 2)
        if total_norm > 1000.0:  # Very large gradient norm
            print(f"⚠️  WARNING: Very large gradient norm: {total_norm:.2f}")
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), state.config.grad_clip)
        
        # Optimizer step
        for optimizer in state.optimizers:
            state.scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
        
        # Update scaler
        state.scaler.update()
        
        # Update learning rate (after optimizer step)
        for scheduler in state.schedulers:
            scheduler.step()
    else:
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), state.config.grad_clip)
        
        # Optimizer step
        for optimizer in state.optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Update learning rate
        for scheduler in state.schedulers:
            scheduler.step()


def _log_training_progress(
    state: TrainingState,
    loss_dict: Dict[str, float],
    pbar: tqdm
):
    """
    Log training progress.
    
    Args:
        state: Training state
        loss_dict: Dictionary with loss values
        pbar: Progress bar
    """
    # Compute additional metrics
    perplexity = math.exp(min(loss_dict['ce_loss'], 20))
    
    # Current learning rates
    current_lrs = [group['lr'] for opt in state.optimizers for group in opt.param_groups]
    avg_lr = sum(current_lrs) / len(current_lrs) if current_lrs else 0.0
    
    # Update progress bar
    pbar.set_postfix({
        'loss': f"{loss_dict['ce_loss']:.4f}",
        'aux': f"{loss_dict['aux_loss']:.4f}",
        'ppl': f"{perplexity:.1f}",
        'lr': f"{avg_lr:.2e}"
    })


def _evaluate_and_log(
    state: TrainingState,
    val_loader: DataLoader,
    final: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model and log results.
    
    Args:
        state: Training state
        val_loader: Validation data loader
        final: Whether this is the final evaluation
        
    Returns:
        Evaluation metrics
    """
    # Compute training time
    elapsed_time = time.time() - state.training_start_time
    
    # Evaluate model
    eval_metrics = evaluate_model(state.model, val_loader, state.config)
    
    # Add timing info
    eval_metrics.update({
        'step': state.step,
        'elapsed_time_minutes': elapsed_time / 60,
        'steps_per_second': state.step / elapsed_time if elapsed_time > 0 else 0
    })
    
    # Log results
    prefix = "📊 Final" if final else f"Step {state.step}"
    print(f"\n{prefix} Evaluation:")
    print(f"   Val Loss: {eval_metrics['val_loss']:.4f}")
    print(f"   Val Accuracy: {eval_metrics['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {eval_metrics['val_perplexity']:.2f}")
    
    if final:
        print(f"   Training Time: {elapsed_time/60:.1f} minutes")
        
        # Compute additional final metrics
        if hasattr(state.model, 'estimate_mfu'):
            mfu = state.model.estimate_mfu(
                fwdbwd_per_iter=1,
                dt=elapsed_time / state.step if state.step > 0 else 1.0
            )
            eval_metrics['mfu'] = mfu
            print(f"   Model FLOPs Utilization: {mfu*100:.1f}%")
    
    return eval_metrics


def _print_model_parameter_info(model: nn.Module):
    """Print detailed model parameter information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Parameter Information:")
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    
    # MoE-specific info
    if any('expert' in name for name, _ in model.named_parameters()):
        active_params = sum(
            p.numel() for name, p in model.named_parameters() 
            if 'expert' not in name
        )
        expert_params = total_params - active_params
        
        print(f"   Active parameters: {active_params:,} ({active_params/1e6:.1f}M)")
        print(f"   Expert parameters: {expert_params:,} ({expert_params/1e6:.1f}M)")
        print(f"   Parameter efficiency: {active_params/total_params:.1%} active per forward pass")


def _load_checkpoint(
    model: nn.Module,
    optimizers: List[torch.optim.Optimizer],
    schedulers: List,
    scaler: Optional[GradScaler],
    checkpoint_path: str
) -> int:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load state into
        optimizers: List of optimizers
        schedulers: List of schedulers
        scaler: Gradient scaler
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Step number to resume from
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer states
    if 'optimizer_states' in checkpoint:
        for opt, state in zip(optimizers, checkpoint['optimizer_states']):
            opt.load_state_dict(state)
    
    # Load scheduler states
    if 'scheduler_states' in checkpoint:
        for sched, state in zip(schedulers, checkpoint['scheduler_states']):
            sched.load_state_dict(state)
    
    # Load scaler state
    if scaler is not None and 'scaler_state' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state'])
    
    return checkpoint.get('step', 0)


def save_checkpoint(
    model: nn.Module,
    optimizers: List[torch.optim.Optimizer],
    schedulers: List,
    scaler: Optional[GradScaler],
    step: int,
    checkpoint_path: str,
    additional_info: Optional[Dict] = None
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizers: List of optimizers
        schedulers: List of schedulers
        scaler: Gradient scaler
        step: Current training step
        checkpoint_path: Path to save checkpoint
        additional_info: Additional information to save
    """
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'scheduler_states': [sched.state_dict() for sched in schedulers],
    }
    
    if scaler is not None:
        checkpoint['scaler_state'] = scaler.state_dict()
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Saved checkpoint to {checkpoint_path}")


def validate_training_setup(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: AdaptiveMoEModelConfig
) -> bool:
    """
    Validate training setup before starting training.
    
    Args:
        model: Model to validate
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Model configuration
        
    Returns:
        True if setup is valid
    """
    print("🔍 Validating training setup...")
    
    try:
        # Test forward pass
        device = next(model.parameters()).device
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
                logits, aux_loss = model(x, return_aux_loss=True)
            else:
                logits = model(x)
                aux_loss = None
            
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        
        print(f"   ✅ Forward pass successful")
        print(f"   ✅ Output shape: {logits.shape}")
        print(f"   ✅ Loss computation successful: {loss.item():.4f}")
        
        if aux_loss is not None:
            print(f"   ✅ Auxiliary loss: {aux_loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print(f"   ✅ Backward pass successful")
        
        # Clear gradients
        model.zero_grad()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        return False
