import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path


def setup_muon_optimizer(model: nn.Module, config):
    """Setup Muon optimizer with hybrid AdamW approach"""
    from optimizers.muon import Muon
    
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(
        adamw_params, 
        lr=config.muon_lr*0.1, 
        weight_decay=config.weight_decay
    )

    return [muon_optimizer, adamw_optimizer]


def evaluate_model(model, val_loader, config, device, max_steps=None):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    steps = 0
    
    max_eval_steps = max_steps or config.eval_steps

    with torch.no_grad():
        for x, y in val_loader:
            if steps >= max_eval_steps:
                break
                
            x, y = x.to(device), y.to(device)
            
            if config.use_amp:
                with autocast('cuda', dtype=torch.float16):
                    logits, _ = model(x, return_aux_loss=False)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            else:
                logits, _ = model(x, return_aux_loss=False)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()
            total_tokens += y.numel()
            steps += 1

    model.train()
    
    avg_loss = total_loss / steps
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        'val_perplexity': perplexity
    }


def save_checkpoint(model, optimizers, schedulers, step, config, save_dir, metrics=None):
    """Save model checkpoint"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'scheduler_states': [sched.state_dict() for sched in schedulers],
        'config': config,
        'metrics': metrics
    }
    
    save_path = os.path.join(save_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path}")
    
    # Save latest checkpoint separately
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)


def train_18b_model(config, train_loader, val_loader, save_dir='checkpoints'):
    """Train the 18B MoE model"""
    from experiments.exp4_18b_moe_training.models_18b import MoE18BLLM
    
    # Print configuration
    config.print_stats()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model
    print(f"\n🚀 Initializing 18B MoE model...")
    model = MoE18BLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check actual memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model = model.to(device)
        model_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"📊 Model loaded to GPU: {model_memory:.2f} GB")
        torch.cuda.empty_cache()
    else:
        model = model.to(device)
        print(f"⚠️  Running on CPU (not recommended for 18B model)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count active parameters (excluding inactive experts)
    active_params = 0
    for name, param in model.named_parameters():
        if 'expert' in name:
            # Only count top_k out of num_experts
            active_params += param.numel() * config.expert_top_k / config.num_experts
        else:
            active_params += param.numel()

    print(f"\n📊 Parameter Counts:")
    print(f"   Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Active per forward: {int(active_params):,} ({active_params/1e9:.2f}B)")
    print(f"   Parameter efficiency: {active_params/total_params:.1%}")

    # Setup optimizers
    print(f"\n🔧 Setting up optimizers...")
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule with warmup + cosine decay
    schedulers = []
    warmup_steps = config.max_steps // 20
    
    for optimizer in optimizers:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Training state
    model.train()
    step = 0
    start_time = time.time()
    total_tokens_processed = 0
    tokens_per_step = config.batch_size * config.max_seq_len
    
    # Metrics tracking
    train_losses = []
    eval_steps = []
    eval_losses = []
    eval_accuracies = []
    eval_perplexities = []
    memory_usage = []
    
    pbar = tqdm(total=config.max_steps, desc="Training 18B MoE")

    # Training loop
    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass
            if config.use_amp:
                with autocast('cuda', dtype=torch.float16):
                    logits, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

                    # Combine main loss and auxiliary loss
                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss

                    loss = total_loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits, aux_loss = model(x, return_aux_loss=True)
                ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss

                loss = total_loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Track metrics
            train_losses.append(ce_loss.item())

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = ce_loss.item()
                    perplexity = math.exp(min(current_loss, 20))
                    
                    if torch.cuda.is_available():
                        mem_gb = torch.cuda.max_memory_allocated() / 1e9
                        memory_usage.append(mem_gb)
                    else:
                        mem_gb = 0

                # Format tokens processed
                if total_tokens_processed >= 1e9:
                    tokens_str = f'{total_tokens_processed/1e9:.2f}B'
                elif total_tokens_processed >= 1e6:
                    tokens_str = f'{total_tokens_processed/1e6:.1f}M'
                else:
                    tokens_str = f'{total_tokens_processed/1e3:.1f}K'

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'aux': f'{aux_loss.item() if aux_loss is not None else 0:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'tokens': tokens_str,
                    'mem': f'{mem_gb:.1f}GB'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                print(f"\n📊 Evaluating at step {step}...")
                eval_metrics = evaluate_model(model, val_loader, config, device)
                
                eval_steps.append(step)
                eval_losses.append(eval_metrics['val_loss'])
                eval_accuracies.append(eval_metrics['val_accuracy'])
                eval_perplexities.append(eval_metrics['val_perplexity'])
                
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens_processed / elapsed
                
                print(f"Step {step}/{config.max_steps} | Tokens: {total_tokens_processed:,} ({total_tokens_processed/1e9:.2f}B)")
                print(f"   Val Loss: {eval_metrics['val_loss']:.4f}")
                print(f"   Val Accuracy: {eval_metrics['val_accuracy']:.4f}")
                print(f"   Val Perplexity: {eval_metrics['val_perplexity']:.2f}")
                print(f"   Tokens/sec: {tokens_per_sec:.1f}")
                print(f"   Elapsed: {elapsed/60:.1f} min")

            # Save checkpoint
            if config.save_every > 0 and step % config.save_every == 0 and step > 0:
                metrics = {
                    'train_loss': train_losses[-1] if train_losses else None,
                    'eval_loss': eval_losses[-1] if eval_losses else None,
                }
                save_checkpoint(model, optimizers, schedulers, step, config, save_dir, metrics)

            # Milestone evaluations
            if step in config.log_milestones:
                print(f"\n🎯 Milestone {step}! Tokens: {total_tokens_processed:,} ({total_tokens_processed/1e9:.2f}B)")
                eval_metrics = evaluate_model(model, val_loader, config, device)
                print(f"   Val Loss: {eval_metrics['val_loss']:.4f}")
                print(f"   Val Perplexity: {eval_metrics['val_perplexity']:.2f}")

            step += 1
            total_tokens_processed += tokens_per_step
            if step % 20 == 0:
                pbar.update(20)

    pbar.close()

    # Final evaluation
    print(f"\n🏁 Final evaluation...")
    final_eval = evaluate_model(model, val_loader, config, device)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"{'='*70}")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"🔢 Tokens processed: {total_tokens_processed:,} ({total_tokens_processed/1e9:.2f}B)")
    print(f"⚡ Average throughput: {total_tokens_processed/total_time:.1f} tokens/sec")
    print(f"📊 Final metrics:")
    print(f"   Val Loss: {final_eval['val_loss']:.4f}")
    print(f"   Val Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")
    
    if torch.cuda.is_available():
        peak_mem = max(memory_usage) if memory_usage else torch.cuda.max_memory_allocated() / 1e9
        print(f"   Peak memory: {peak_mem:.1f} GB")
    
    # Save final checkpoint
    save_checkpoint(model, optimizers, schedulers, step, config, save_dir, final_eval)
    
    # Save metrics
    results = {
        'config': {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'd_ff': config.d_ff,
            'num_experts': config.num_experts,
            'expert_top_k': config.expert_top_k,
            'total_params': total_params,
            'active_params': int(active_params),
        },
        'final_metrics': final_eval,
        'training_time_hours': total_time / 3600,
        'total_tokens_processed': total_tokens_processed,
        'average_tokens_per_sec': total_tokens_processed / total_time,
        'eval_history': {
            'steps': eval_steps,
            'losses': eval_losses,
            'accuracies': eval_accuracies,
            'perplexities': eval_perplexities,
        },
        'peak_memory_gb': max(memory_usage) if memory_usage else None,
    }
    
    results_path = os.path.join(save_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📝 Results saved: {results_path}")
    
    # Plot training curves
    plot_training_curves(eval_steps, eval_losses, eval_perplexities, save_dir)
    
    return model, final_eval


def plot_training_curves(steps, losses, perplexities, save_dir):
    """Plot training curves"""
    if len(steps) < 2:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(steps, losses, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss vs Steps')
    ax1.grid(True, alpha=0.3)
    
    # Perplexity curve
    ax2.plot(steps, perplexities, 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Validation Perplexity')
    ax2.set_title('Validation Perplexity vs Steps')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Training curves saved: {save_path}")
    plt.close()

