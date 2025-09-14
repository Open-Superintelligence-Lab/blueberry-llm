#!/usr/bin/env python3
"""
Training script with LIVE Advanced GPU Profiler integration
This shows real-time profiling dashboard during training.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader, random_split
from auto_config import auto_configure
from llm import load_and_cache_data, TextTokenDataset, MoEModelConfig, MoEMinimalLLM, setup_muon_optimizer, evaluate_model
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
import math
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler import AdvancedGPUProfiler, ProfilerContext, patch_model_for_profiling
from llm import train_moe_model, load_and_cache_data, TextTokenDataset, MoEModelConfig, MoEMinimalLLM, setup_muon_optimizer, evaluate_model
from auto_config import auto_configure

def train_with_live_profiler():
    """Train MoE model with LIVE profiling dashboard"""
    print("🫐 Blueberry LLM Training with LIVE GPU Profiler")
    print("=" * 60)
    
    # Auto-configure everything
    configurator = auto_configure()
    configurator.print_config()
    
    # Get model configuration
    model_config = configurator.get_model_config()
    
    # Auto-size dataset based on hardware
    if configurator.config.num_gpus == 0:
        model_config.num_documents = 500
        model_config.max_tokens = 50000
    elif configurator.config.gpu_memory_gb < 16:
        model_config.num_documents = 1000
        model_config.max_tokens = 100000
    elif configurator.config.num_gpus <= 2:
        model_config.num_documents = 2000
        model_config.max_tokens = 250000
    else:
        model_config.num_documents = 5000
        model_config.max_tokens = 500000
    
    print(f"\n📊 Loading {model_config.num_documents} documents, {model_config.max_tokens:,} tokens...")
    
    # Load data
    texts, tokenizer, tokens = load_and_cache_data(model_config)
    dataset = TextTokenDataset(tokens, model_config.max_seq_len)
    
    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"   Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Create profiler
    profiler = AdvancedGPUProfiler(
        num_experts=model_config.num_experts, 
        enable_profiling=True,
        output_dir="profiler_output"
    )
    
    print(f"\n🔍 LIVE Advanced GPU Profiler enabled")
    print(f"   Experts: {model_config.num_experts}")
    print(f"   Dashboard will show every 100 steps")
    print(f"   Output directory: profiler_output/")
    
    # Train with LIVE profiling
    print(f"\n🚀 Starting training with LIVE profiling...")
    
    # Start profiling
    profiler.start_profiling()
    
    # Initialize model
    model = MoEMinimalLLM(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Patch model for automatic profiling
    patched_model = patch_model_for_profiling(model, profiler)
    
    # Count parameters
    total_params = sum(param.numel() for param in model.parameters())
    active_params = sum(param.numel() for name, param in model.named_parameters()
                       if 'expert' not in name)
    expert_params = total_params - active_params
    
    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  📊 Active parameters: {active_params:,}")
    print(f"  📊 Expert parameters: {expert_params:,}")
    print(f"  📊 Parameter efficiency: {active_params/total_params:.1%} active per forward pass")
    
    # Setup optimizers
    optimizers = setup_muon_optimizer(model, model_config)
    
    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = model_config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (model_config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)
    
    scaler = GradScaler() if model_config.use_amp else None
    
    # Training loop with LIVE profiling
    patched_model.train()
    step = 0
    pbar = tqdm(total=model_config.max_steps, desc="Training MoE with LIVE Profiler")
    
    while step < model_config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= model_config.max_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            if model_config.use_amp:
                with autocast():
                    logits, aux_loss = patched_model(x, return_aux_loss=True)
                    ce_loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), y.view(-1))
                    
                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss
                    
                    loss = total_loss / model_config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits, aux_loss = patched_model(x, return_aux_loss=True)
                ce_loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), y.view(-1))
                
                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss
                
                loss = total_loss / model_config.gradient_accumulation_steps
                loss.backward()
            
            # Optimizer step
            if (step + 1) % model_config.gradient_accumulation_steps == 0:
                if model_config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.grad_clip)
                    
                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
            
            # LIVE Profiler Dashboard every 100 steps
            if step % 100 == 0 and step > 0:
                print(f"\n{'='*60}")
                print(f"📊 LIVE PROFILER DASHBOARD - Step {step}")
                print(f"{'='*60}")
                profiler.print_dashboard()
                print(f"{'='*60}")
            
            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = ce_loss.item()
                    perplexity = math.exp(min(current_loss, 20))
                
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'aux': f'{aux_loss.item() if aux_loss is not None else 0:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}'
                })
            
            # Evaluation
            if step % model_config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, model_config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")
            
            step += 1
            if step % 20 == 0:
                pbar.update(20)
    
    pbar.close()
    
    # Final evaluation
    final_eval = evaluate_model(model, val_loader, model_config)
    print(f"\n📊 Final Results:")
    print(f"   Val Loss: {final_eval['val_loss']:.4f}")
    print(f"   Val Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")
    
    # Final profiling dashboard
    print(f"\n📊 FINAL PROFILER DASHBOARD:")
    profiler.print_dashboard()
    
    # Stop profiling
    profiler.stop_profiling()
    
    # Save model with profiling metadata
    print(f"\n💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'auto_config': configurator.config,
        'tokenizer': tokenizer,
        'final_metrics': final_eval,
        'profiler_stats': profiler.get_current_stats()
    }, 'blueberry_model_with_live_profiling.pt')
    
    print("✅ Training with LIVE profiling complete!")
    print(f"   Final validation loss: {final_eval['val_loss']:.4f}")
    print(f"   Final validation accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Model saved as: blueberry_model_with_live_profiling.pt")
    print(f"   Profiling reports saved in: profiler_output/")

if __name__ == "__main__":
    try:
        train_with_live_profiler()
        print("\n🎉 LIVE profiling training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
