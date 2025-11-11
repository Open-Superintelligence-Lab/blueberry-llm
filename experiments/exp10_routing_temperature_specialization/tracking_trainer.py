"""
Custom trainer with routing statistics tracking
"""
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.moe_config import MoEModelConfig
from optimizers.muon import Muon
from training.evaluation import evaluate_model
from utils.logger import setup_logging
from config import TemperatureConfig


def setup_optimizer(model: nn.Module, config: MoEModelConfig):
    """Setup Muon optimizer with optimal settings from exp9"""
    # Separate parameters by dimensionality
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Use Muon for 2D parameters (weight matrices)
            if param.ndim >= 2 and 'embedding' not in name.lower() and 'norm' not in name.lower():
                muon_params.append(param)
            else:
                adamw_params.append(param)
    
    # Create Muon optimizer (handles both param groups internally)
    optimizer = Muon(
        muon_params=muon_params,
        lr=config.muon_lr,
        momentum=config.muon_momentum,
        nesterov=True,
        ns_steps=5,
        adamw_params=adamw_params,
        adamw_lr=config.adamw_lr,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_wd=config.weight_decay,
    )
    
    return optimizer


def get_lr_schedule(optimizer, config: MoEModelConfig, warmup_steps: int, total_steps: int):
    """Create cosine learning rate schedule with warmup"""
    import math
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_with_temperature_tracking(
    model: nn.Module,
    config: MoEModelConfig,
    temp_config: TemperatureConfig,
    train_loader,
    val_loader,
    output_dir: str = "."
):
    """
    Train model with temperature-controlled routing and comprehensive tracking.
    
    Args:
        model: MoE model with temperature-aware routing
        config: Model configuration
        temp_config: Temperature experiment configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Directory to save results
    
    Returns:
        model: Trained model
        metrics: Final evaluation metrics
        history: Training history with routing statistics
    """
    logger = setup_logging(log_dir=Path(output_dir) / "logs")
    logger.info(f"Training with temperature config: {temp_config.name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(model, config)
    
    warmup_steps = int(config.warmup_ratio * config.max_steps)
    scheduler = get_lr_schedule(optimizer, config, warmup_steps, config.max_steps)
    
    # Use AMP for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    
    # Training state
    global_step = 0
    start_time = time.time()
    
    # History tracking
    history = {
        'steps': [],
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': [],
        'val_perplexities': [],
        'elapsed_times': [],
        'learning_rates': [],
        'temperatures': [],
        'routing_entropies': [],
        'selection_confidences': [],
        'expert_utilizations': [],
        'load_balancing_losses': [],
    }
    
    logger.info(f"Starting training for {config.max_steps} steps")
    logger.info(f"Temperature: {temp_config.temperature}, Schedule: {temp_config.temperature_schedule}")
    
    model.train()
    accumulated_loss = 0.0
    accumulated_aux_loss = 0.0
    optimizer.zero_grad()
    
    train_iter = iter(train_loader)
    
    for step in range(config.max_steps):
        # Update temperature based on schedule
        current_temp = temp_config.get_temperature_at_step(step)
        
        # Set temperature for all MoE layers
        for module in model.modules():
            if hasattr(module, 'set_temperature'):
                module.set_temperature(current_temp)
        
        # Get next batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            logits, aux_loss = model(input_ids, return_aux_loss=True)
            
            # Compute language modeling loss
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Total loss = LM loss + load balancing loss
            loss = lm_loss
            if aux_loss is not None:
                loss = loss + aux_loss
        
        # Backward pass with gradient scaling
        scaled_loss = loss / config.gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()
        
        accumulated_loss += lm_loss.item()
        if aux_loss is not None:
            accumulated_aux_loss += aux_loss.item()
        
        # Update weights after accumulation steps
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        # Evaluation
        if (step + 1) % config.eval_every == 0:
            elapsed_time = (time.time() - start_time) / 60
            current_lr = scheduler.get_last_lr()[0]
            
            # Evaluate on validation set
            model.eval()
            val_metrics = evaluate_model(model, val_loader, device, config.eval_steps)
            model.train()
            
            # Collect routing statistics from MoE layers
            routing_stats = collect_routing_stats(model)
            
            # Log progress
            avg_train_loss = accumulated_loss / config.eval_every
            avg_aux_loss = accumulated_aux_loss / config.eval_every
            
            logger.info(
                f"Step {step+1}/{config.max_steps} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Temp: {current_temp:.2f} | "
                f"Entropy: {routing_stats['avg_entropy']:.3f} | "
                f"Time: {elapsed_time:.2f}m"
            )
            
            # Update history
            history['steps'].append(step + 1)
            history['train_losses'].append(avg_train_loss)
            history['val_losses'].append(val_metrics['loss'])
            history['val_accuracies'].append(val_metrics['accuracy'])
            history['val_perplexities'].append(val_metrics['perplexity'])
            history['elapsed_times'].append(elapsed_time)
            history['learning_rates'].append(current_lr)
            history['temperatures'].append(current_temp)
            history['routing_entropies'].append(routing_stats['avg_entropy'])
            history['selection_confidences'].append(routing_stats['avg_confidence'])
            history['expert_utilizations'].append(routing_stats['expert_utilization'])
            history['load_balancing_losses'].append(avg_aux_loss)
            
            # Reset accumulators
            accumulated_loss = 0.0
            accumulated_aux_loss = 0.0
    
    # Final evaluation
    model.eval()
    final_metrics = evaluate_model(model, val_loader, device, config.eval_steps)
    final_routing_stats = collect_routing_stats(model)
    
    logger.info(f"Training complete!")
    logger.info(f"Final validation loss: {final_metrics['loss']:.4f}")
    logger.info(f"Final validation accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"Final routing entropy: {final_routing_stats['avg_entropy']:.3f}")
    
    return model, final_metrics, history, final_routing_stats


def collect_routing_stats(model: nn.Module) -> dict:
    """Collect routing statistics from all MoE layers"""
    entropies = []
    confidences = []
    utilizations = []
    
    for module in model.modules():
        if hasattr(module, 'router') and hasattr(module.router, 'routing_entropy_history'):
            router = module.router
            
            # Get latest stats
            if hasattr(router, 'routing_entropy_history') and router.routing_entropy_history:
                entropies.append(router.routing_entropy_history[-1])
            if hasattr(router, 'selection_confidence_history') and router.selection_confidence_history:
                confidences.append(router.selection_confidence_history[-1])
            if hasattr(router, 'expert_counts') and router.expert_counts is not None:
                utilizations.append(router.expert_counts.numpy().tolist())
    
    # Aggregate statistics across layers
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Average expert utilization across layers
    if utilizations:
        import numpy as np
        avg_utilization = np.mean(utilizations, axis=0).tolist()
    else:
        avg_utilization = []
    
    return {
        'avg_entropy': avg_entropy,
        'avg_confidence': avg_confidence,
        'expert_utilization': avg_utilization,
        'num_layers': len(entropies),
    }

