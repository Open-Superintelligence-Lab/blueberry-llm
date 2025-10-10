"""
Simple Transformer with MoE and Muon Optimizer Training Script

This experiment trains the simplest possible transformer with:
- Standard multi-head attention (no fancy mechanisms)
- Mixture of Experts for efficiency
- Muon optimizer for training
"""
import sys
import os
import torch
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import from global repo
from config import SimpleTransformerConfig
from models.moe_llm import MoEMinimalLLM
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from optimizers.muon import Muon
from utils.helpers import set_seed


def setup_optimizers(model, config):
    """
    Setup hybrid optimizer:
    - Muon for 2D parameters (attention, FFN weights)
    - AdamW for other parameters (embeddings, norms)
    """
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Use Muon for 2D parameters (weights in attention and FFN)
        if param.ndim == 2 and 'token_embedding' not in name and 'norm' not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    
    print(f"\n📊 Optimizer Setup:")
    print(f"   Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"   AdamW parameters: {sum(p.numel() for p in adamw_params):,}")
    
    optimizers = []
    optimizers.append(Muon(
        muon_params,
        lr=config.muon_lr,
        momentum=config.muon_momentum
    ))
    optimizers.append(torch.optim.AdamW(
        adamw_params,
        lr=config.adamw_lr,
        weight_decay=config.weight_decay
    ))
    
    return optimizers


def create_lr_schedule(optimizer, config):
    """Create cosine learning rate schedule with warmup"""
    warmup_steps = config.max_steps // 20
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(model, val_loader, config, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            with autocast('cuda', dtype=torch.float16, enabled=config.use_amp):
                logits = model(x, return_aux_loss=False)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()
    
    model.train()
    
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy,
        'val_perplexity': perplexity
    }


def train(config, train_loader, val_loader, device):
    """Main training loop"""
    print("\n" + "="*70)
    print("🚀 TRAINING SIMPLE TRANSFORMER WITH MOE AND MUON")
    print("="*70)
    
    # Initialize model
    set_seed(42)
    # Convert SimpleTransformerConfig to MoEModelConfig for compatibility with global model
    moe_config = config.to_moe_config()
    model = MoEMinimalLLM(moe_config).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = model.token_embedding.weight.numel()
    non_embed_params = total_params - embedding_params
    
    # Calculate active parameters (excluding MoE experts)
    active_params = 0
    expert_params = 0
    for name, param in model.named_parameters():
        if 'experts' in name:
            expert_params += param.numel()
        else:
            active_params += param.numel()
    
    print(f"\n📊 Model Architecture:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Non-embedding parameters: {non_embed_params:,}")
    print(f"   Active per forward pass: {active_params:,}")
    print(f"   Expert parameters: {expert_params:,}")
    print(f"   Efficiency: {active_params/total_params:.1%} active")
    print(f"\n   Layers: {config.n_layers}")
    print(f"   Model dimension: {config.d_model}")
    print(f"   Attention heads: {config.n_heads}")
    print(f"   FFN dimension: {config.d_ff}")
    print(f"   Experts: {config.num_experts} (top-{config.expert_top_k})")
    
    # Setup optimizers
    optimizers = setup_optimizers(model, config)
    schedulers = [create_lr_schedule(opt, config) for opt in optimizers]
    
    scaler = GradScaler() if config.use_amp else None
    
    # Training loop
    print(f"\n🎯 Training Configuration:")
    print(f"   Max steps: {config.max_steps:,}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"   Sequence length: {config.max_seq_len}")
    print(f"   Tokens per step: {config.batch_size * config.max_seq_len:,}")
    print(f"   Muon LR: {config.muon_lr}")
    print(f"   AdamW LR: {config.adamw_lr}")
    print(f"   Mixed precision: {config.use_amp}")
    
    model.train()
    step = 0
    pbar = tqdm(total=config.max_steps, desc="Training")
    
    # Tracking metrics
    train_losses = []
    eval_steps = []
    eval_losses = []
    eval_accs = []
    eval_ppls = []
    tokens_per_sec_history = []
    
    start_time = time.time()
    total_tokens_processed = 0
    
    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            # Track tokens processed
            batch_tokens = x.numel()
            total_tokens_processed += batch_tokens
            
            # Forward pass
            if config.use_amp:
                with autocast('cuda', dtype=torch.float16):
                    logits, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    
                    # Total loss includes load balancing
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
                    # Unscale and clip gradients
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    
                    # Update parameters
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
            
            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = ce_loss.item()
                    perplexity = math.exp(min(current_loss, 20))
                    
                    # Calculate tokens per second
                    elapsed_time = time.time() - start_time
                    tokens_per_sec = total_tokens_processed / elapsed_time if elapsed_time > 0 else 0
                    
                    train_losses.append({
                        'step': step,
                        'loss': current_loss,
                        'aux_loss': aux_loss.item() if aux_loss is not None else 0.0,
                        'accuracy': accuracy,
                        'perplexity': perplexity,
                        'tokens_per_sec': tokens_per_sec
                    })
                    
                    tokens_per_sec_history.append({
                        'step': step,
                        'tokens_per_sec': tokens_per_sec
                    })
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'aux': f'{aux_loss.item() if aux_loss is not None else 0:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'tok/s': f'{tokens_per_sec:.0f}'
                })
            
            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config, device)
                
                eval_steps.append(step)
                eval_losses.append(eval_metrics['val_loss'])
                eval_accs.append(eval_metrics['val_accuracy'])
                eval_ppls.append(eval_metrics['val_perplexity'])
                
                # Calculate current tokens/sec
                elapsed_time = time.time() - start_time
                current_tokens_per_sec = total_tokens_processed / elapsed_time if elapsed_time > 0 else 0
                
                print(f"\n📊 Step {step}: "
                      f"Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}, "
                      f"Tokens/sec: {current_tokens_per_sec:,.0f}")
            
            # Milestone evaluations
            if step in config.log_milestones:
                eval_metrics = evaluate_model(model, val_loader, config, device)
                elapsed_time = time.time() - start_time
                current_tokens_per_sec = total_tokens_processed / elapsed_time if elapsed_time > 0 else 0
                print(f"\n🎯 Milestone {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Tokens/sec: {current_tokens_per_sec:,.0f}")
            
            step += 1
            if step % 20 == 0:
                pbar.update(20)
    
    pbar.close()
    
    # Final evaluation
    print("\n" + "="*70)
    print("📊 FINAL EVALUATION")
    print("="*70)
    
    final_eval = evaluate_model(model, val_loader, config, device)
    eval_steps.append(config.max_steps)
    eval_losses.append(final_eval['val_loss'])
    eval_accs.append(final_eval['val_accuracy'])
    eval_ppls.append(final_eval['val_perplexity'])
    
    total_time = time.time() - start_time
    avg_tokens_per_sec = total_tokens_processed / total_time if total_time > 0 else 0
    
    print(f"\n🏆 Final Results:")
    print(f"   Validation Loss: {final_eval['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_eval['val_perplexity']:.2f}")
    print(f"\n⚡ Performance Metrics:")
    print(f"   Total Tokens Processed: {total_tokens_processed:,}")
    print(f"   Average Tokens/sec: {avg_tokens_per_sec:,.0f}")
    print(f"   Total Training Time: {total_time/60:.1f} minutes")
    print(f"   Avg Time per Step: {total_time/config.max_steps:.2f} seconds")
    
    return {
        'model': model,
        'final_metrics': final_eval,
        'train_losses': train_losses,
        'eval_steps': eval_steps,
        'eval_losses': eval_losses,
        'eval_accs': eval_accs,
        'eval_ppls': eval_ppls,
        'total_time': total_time,
        'total_tokens_processed': total_tokens_processed,
        'avg_tokens_per_sec': avg_tokens_per_sec,
        'tokens_per_sec_history': tokens_per_sec_history
    }


def plot_results(results, save_dir='results'):
    """Plot training and evaluation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Training loss over time
    train_steps = [t['step'] for t in results['train_losses']]
    train_loss_values = [t['loss'] for t in results['train_losses']]
    
    # Create 2x3 grid for 6 plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    # Training loss
    axes[0, 0].plot(train_steps, train_loss_values, alpha=0.6, linewidth=1)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss
    axes[0, 1].plot(results['eval_steps'], results['eval_losses'], 'o-', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[0, 2].plot(results['eval_steps'], results['eval_accs'], 'o-', linewidth=2, markersize=6, color='green')
    axes[0, 2].set_xlabel('Training Step')
    axes[0, 2].set_ylabel('Validation Accuracy')
    axes[0, 2].set_title('Validation Accuracy')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Validation perplexity
    axes[1, 0].plot(results['eval_steps'], results['eval_ppls'], 'o-', linewidth=2, markersize=6, color='red')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Validation Perplexity')
    axes[1, 0].set_title('Validation Perplexity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Tokens per second over time
    if results['tokens_per_sec_history']:
        tok_steps = [t['step'] for t in results['tokens_per_sec_history']]
        tok_per_sec = [t['tokens_per_sec'] for t in results['tokens_per_sec_history']]
        axes[1, 1].plot(tok_steps, tok_per_sec, linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Tokens/sec')
        axes[1, 1].set_title('Training Throughput (Tokens/sec)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].ticklabel_format(style='plain', axis='y')
    
    # Training accuracy over time (from train_losses)
    train_accs = [t.get('accuracy', 0) for t in results['train_losses']]
    axes[1, 2].plot(train_steps, train_accs, alpha=0.6, linewidth=1, color='orange')
    axes[1, 2].set_xlabel('Training Step')
    axes[1, 2].set_ylabel('Training Accuracy')
    axes[1, 2].set_title('Training Accuracy Over Time')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Training curves saved to {save_dir}/training_curves.png")
    
    plt.close()


def save_results(results, config, save_dir='results'):
    """Save results to JSON"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'config': {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'd_ff': config.d_ff,
            'num_experts': config.num_experts,
            'expert_top_k': config.expert_top_k,
            'max_steps': config.max_steps,
            'batch_size': config.batch_size,
            'max_seq_len': config.max_seq_len,
            'muon_lr': config.muon_lr,
            'adamw_lr': config.adamw_lr,
        },
        'final_metrics': results['final_metrics'],
        'train_losses': results['train_losses'],
        'eval_metrics': {
            'steps': results['eval_steps'],
            'losses': results['eval_losses'],
            'accuracies': results['eval_accs'],
            'perplexities': results['eval_ppls'],
        },
        'performance_metrics': {
            'total_tokens_processed': results['total_tokens_processed'],
            'avg_tokens_per_sec': results['avg_tokens_per_sec'],
            'tokens_per_sec_history': results['tokens_per_sec_history'],
            'total_time_seconds': results['total_time'],
            'total_time_minutes': results['total_time'] / 60,
        },
    }
    
    with open(f'{save_dir}/results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Results saved to {save_dir}/results.json")


def main():
    """Main experiment entry point"""
    print("\n" + "="*70)
    print("🔬 EXPERIMENT 5: SIMPLE TRANSFORMER WITH MOE AND MUON")
    print("="*70)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔍 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\n📚 Loading data...")
    temp_config = SimpleTransformerConfig()
    texts, tokenizer, tokens = load_and_cache_data(temp_config)
    
    # Create config with vocab_size
    config = SimpleTransformerConfig(vocab_size=temp_config.vocab_size)
    
    # Create datasets
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    print(f"   Vocabulary size: {config.vocab_size:,}")
    
    # Train model
    results = train(config, train_loader, val_loader, device)
    
    # Save results
    save_results(results, config)
    plot_results(results)
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

