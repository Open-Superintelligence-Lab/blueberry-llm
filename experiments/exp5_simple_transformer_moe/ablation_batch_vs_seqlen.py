"""
Ablation Study: Batch Size vs Sequence Length

Research Question: What's better to fill GPU memory with?
- Variant A: Large batch size + short sequences
- Variant B: Small batch size + long sequences

Both variants use same total tokens and similar GPU memory.
"""

import sys
import os
import torch
import torch.nn.functional as F
import math
import time
import json
import argparse
from dataclasses import dataclass, replace
from typing import Dict, List
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root and experiment directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
exp_dir = os.path.dirname(__file__)
sys.path.insert(0, project_root)
sys.path.insert(0, exp_dir)

# Import from global repo
from config import SimpleTransformerConfig
from models.moe_llm import MoEMinimalLLM
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from optimizers.muon import Muon
from utils.helpers import set_seed


@dataclass
class AblationConfig:
    """Configuration for batch vs seqlen ablation"""
    name: str
    batch_size: int
    seq_len: int
    lr: float
    grad_accum: int = 1
    
    @property
    def effective_batch(self):
        return self.batch_size * self.grad_accum
    
    @property
    def tokens_per_step(self):
        return self.batch_size * self.seq_len * self.grad_accum


def estimate_memory_gb(config, model_params_m=50):
    """Estimate GPU memory usage"""
    # Model weights (FP16)
    model_mem = model_params_m * 2
    
    # Optimizer states (Muon + AdamW)
    opt_mem = model_params_m * 0.8 * 2 + model_params_m * 0.2 * 4
    
    # Activations (rough estimate: 12 bytes per token per MB of model)
    act_mem = config.batch_size * config.seq_len * model_params_m * 12 / 1000
    
    # Gradients
    grad_mem = model_params_m * 2
    
    total = model_mem + opt_mem + act_mem + grad_mem
    return total / 1000  # Convert to GB


def create_ablation_configs():
    """Create configurations for ablation study"""
    
    # Base model config
    base_config = SimpleTransformerConfig()
    
    # Calculate memory for different batch/seqlen combinations
    # Target: ~8-10GB memory usage (fits on most GPUs)
    
    configs = []
    
    # STRATEGY A: Large Batch + Short Sequences
    # Maximize batch size with short sequences
    for lr in [0.005, 0.01, 0.02]:
        configs.append(AblationConfig(
            name=f"large_batch_lr{lr}",
            batch_size=64,  # 4x larger batch
            seq_len=256,    # 2x shorter sequence
            lr=lr,
            grad_accum=2
        ))
    
    # STRATEGY B: Small Batch + Long Sequences  
    # Maximize sequence length with small batches
    for lr in [0.005, 0.01, 0.02]:
        configs.append(AblationConfig(
            name=f"long_seq_lr{lr}",
            batch_size=8,    # 4x smaller batch
            seq_len=1024,    # 2x longer sequence
            lr=lr,
            grad_accum=8
        ))
    
    # BASELINE: Balanced (from original config)
    for lr in [0.005, 0.01, 0.02]:
        configs.append(AblationConfig(
            name=f"balanced_lr{lr}",
            batch_size=24,
            seq_len=512,
            lr=lr,
            grad_accum=4
        ))
    
    # Print memory estimates
    print("\n" + "="*80)
    print("📊 ABLATION CONFIGURATIONS")
    print("="*80)
    print(f"\n{'Name':<20} {'Batch':<8} {'SeqLen':<8} {'EffBatch':<10} {'Tokens/Step':<12} {'Memory':<8} {'LR':<8}")
    print("-"*80)
    
    for cfg in configs:
        mem = estimate_memory_gb(cfg)
        print(f"{cfg.name:<20} {cfg.batch_size:<8} {cfg.seq_len:<8} {cfg.effective_batch:<10} "
              f"{cfg.tokens_per_step:<12} {mem:<8.1f}GB {cfg.lr:<8.3f}")
    
    return configs


def train_single_config(ablation_config: AblationConfig, base_config: SimpleTransformerConfig,
                       train_loader, val_loader, device, max_steps=5000):
    """Train a single configuration"""
    
    # Update base config with ablation settings
    config = replace(base_config,
                    batch_size=ablation_config.batch_size,
                    max_seq_len=ablation_config.seq_len,
                    gradient_accumulation_steps=ablation_config.grad_accum,
                    muon_lr=ablation_config.lr,
                    max_steps=max_steps,
                    eval_every=10)
    
    print(f"\n{'='*80}")
    print(f"🚀 Training: {ablation_config.name}")
    print(f"{'='*80}")
    print(f"   Batch: {ablation_config.batch_size}, SeqLen: {ablation_config.seq_len}, "
          f"EffBatch: {ablation_config.effective_batch}, LR: {ablation_config.lr}")
    
    # Initialize model
    set_seed(42)
    # Convert SimpleTransformerConfig to MoEModelConfig for compatibility with global model
    moe_config = config.to_moe_config()
    model = MoEMinimalLLM(moe_config).to(device)
    
    # Setup optimizers
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and 'token_embedding' not in name and 'norm' not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    
    optimizers = [
        Muon(muon_params, lr=config.muon_lr, momentum=config.muon_momentum),
        torch.optim.AdamW(adamw_params, lr=config.adamw_lr, weight_decay=config.weight_decay)
    ]
    
    # Simple linear warmup
    def lr_lambda(step):
        warmup = max_steps // 20
        return min(1.0, step / warmup) if warmup > 0 else 1.0
    
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda) for opt in optimizers]
    scaler = GradScaler() if config.use_amp else None
    
    # Training loop
    model.train()
    step = 0
    train_losses = []
    val_metrics = []
    tokens_per_sec_history = []
    
    start_time = time.time()
    total_tokens = 0
    
    pbar = tqdm(total=max_steps, desc=ablation_config.name)
    
    train_iter = iter(train_loader)
    
    while step < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        
        # Ensure correct sequence length
        if x.size(1) != config.max_seq_len:
            if x.size(1) < config.max_seq_len:
                # Pad if too short
                pad_len = config.max_seq_len - x.size(1)
                x = F.pad(x, (0, pad_len), value=0)
                y = F.pad(y, (0, pad_len), value=0)
            else:
                # Truncate if too long
                x = x[:, :config.max_seq_len]
                y = y[:, :config.max_seq_len]
        
        batch_tokens = x.numel()
        total_tokens += batch_tokens
        
        # Forward pass
        if config.use_amp:
            with autocast('cuda', dtype=torch.float16):
                logits, aux_loss = model(x, return_aux_loss=True)
                ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                total_loss = ce_loss + (aux_loss if aux_loss is not None else 0)
                loss = total_loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits, aux_loss = model(x, return_aux_loss=True)
            ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            total_loss = ce_loss + (aux_loss if aux_loss is not None else 0)
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
        
        # Logging
        if step % 100 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
            
            train_losses.append({
                'step': step,
                'loss': ce_loss.item(),
                'accuracy': acc,
                'tokens_per_sec': tokens_per_sec
            })
            
            tokens_per_sec_history.append(tokens_per_sec)
            
            pbar.set_postfix({
                'loss': f'{ce_loss.item():.4f}',
                'acc': f'{acc:.3f}',
                'tok/s': f'{tokens_per_sec:.0f}'
            })
        
        # Validation
        if step % config.eval_every == 0 and step > 0:
            model.eval()
            val_loss = 0
            val_acc = 0
            val_count = 0
            
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_loader):
                    if i >= 50:
                        break
                    
                    vx, vy = vx.to(device), vy.to(device)
                    
                    # Adjust sequence length
                    if vx.size(1) != config.max_seq_len:
                        if vx.size(1) < config.max_seq_len:
                            pad_len = config.max_seq_len - vx.size(1)
                            vx = F.pad(vx, (0, pad_len), value=0)
                            vy = F.pad(vy, (0, pad_len), value=0)
                        else:
                            vx = vx[:, :config.max_seq_len]
                            vy = vy[:, :config.max_seq_len]
                    
                    with autocast('cuda', dtype=torch.float16, enabled=config.use_amp):
                        vlogits = model(vx, return_aux_loss=False)
                        vloss = F.cross_entropy(vlogits.view(-1, config.vocab_size), vy.view(-1))
                    
                    val_loss += vloss.item()
                    val_acc += (vlogits.argmax(-1) == vy).float().mean().item()
                    val_count += 1
            
            val_loss /= val_count
            val_acc /= val_count
            
            val_metrics.append({
                'step': step,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            })
            
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            
            print(f"\n   Step {step}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Tok/s={tokens_per_sec:,.0f}")
            model.train()
        
        step += 1
        pbar.update(1)
    
    pbar.close()
    
    total_time = time.time() - start_time
    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    
    # Final evaluation
    model.eval()
    final_val_loss = 0
    final_val_acc = 0
    val_count = 0
    
    with torch.no_grad():
        for i, (vx, vy) in enumerate(val_loader):
            if i >= 100:
                break
            
            vx, vy = vx.to(device), vy.to(device)
            
            if vx.size(1) != config.max_seq_len:
                if vx.size(1) < config.max_seq_len:
                    pad_len = config.max_seq_len - vx.size(1)
                    vx = F.pad(vx, (0, pad_len), value=0)
                    vy = F.pad(vy, (0, pad_len), value=0)
                else:
                    vx = vx[:, :config.max_seq_len]
                    vy = vy[:, :config.max_seq_len]
            
            with autocast('cuda', dtype=torch.float16, enabled=config.use_amp):
                vlogits = model(vx, return_aux_loss=False)
                vloss = F.cross_entropy(vlogits.view(-1, config.vocab_size), vy.view(-1))
            
            final_val_loss += vloss.item()
            final_val_acc += (vlogits.argmax(-1) == vy).float().mean().item()
            val_count += 1
    
    final_val_loss /= val_count
    final_val_acc /= val_count
    
    print(f"\n✅ Final: Val Loss={final_val_loss:.4f}, Val Acc={final_val_acc:.4f}, "
          f"Avg Tok/s={avg_tokens_per_sec:,.0f}, Time={total_time/60:.1f}min")
    
    return {
        'config': ablation_config,
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'final_val_loss': final_val_loss,
        'final_val_acc': final_val_acc,
        'total_tokens': total_tokens,
        'total_time': total_time,
        'avg_tokens_per_sec': avg_tokens_per_sec,
        'tokens_per_sec_history': tokens_per_sec_history
    }


def main():
    parser = argparse.ArgumentParser(description='MoE Ablation: Batch vs SeqLen')
    parser.add_argument('--batch', type=int, help='Batch size (e.g., 64)')
    parser.add_argument('--seqlen', type=int, help='Sequence length (e.g., 256)')
    parser.add_argument('--lr', type=float, help='Learning rate (e.g., 0.01)')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--steps', type=int, default=20, help='Training steps')
    parser.add_argument('--name', type=str, default='custom', help='Config name')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔬 ABLATION STUDY: BATCH SIZE vs SEQUENCE LENGTH")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔍 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\n📚 Loading data...")
    temp_config = SimpleTransformerConfig()
    # Convert to MoEModelConfig for data loading
    temp_moe_config = temp_config.to_moe_config()
    texts, tokenizer, tokens = load_and_cache_data(temp_moe_config, cache_dir="data_cache")
    
    base_config = SimpleTransformerConfig(vocab_size=temp_moe_config.vocab_size)
    
    # Create ablation configurations
    if args.batch and args.seqlen and args.lr:
        # Single custom config
        ablation_configs = [AblationConfig(
            name=args.name, batch_size=args.batch, seq_len=args.seqlen, 
            lr=args.lr, grad_accum=args.grad_accum
        )]
        print(f"\n🎯 Running custom config: {args.name}")
        print(f"   Batch={args.batch}, SeqLen={args.seqlen}, LR={args.lr}, GradAccum={args.grad_accum}")
    else:
        ablation_configs = create_ablation_configs()
    
    # We'll create separate dataloaders for each config
    all_results = []
    
    for abl_config in ablation_configs:
        # Create dataset with appropriate sequence length
        dataset = TextTokenDataset(tokens, abl_config.seq_len)
        
        val_size = len(dataset) // 10
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=abl_config.batch_size,
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=abl_config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Train this configuration
        result = train_single_config(
            abl_config, base_config, train_loader, val_loader, device, max_steps=args.steps
        )
        all_results.append(result)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Analyze results
    print("\n" + "="*80)
    print("📊 FINAL RESULTS COMPARISON")
    print("="*80)
    
    print(f"\n{'Configuration':<25} {'Val Loss':<12} {'Val Acc':<12} {'Avg Tok/s':<15} {'Time (min)':<12}")
    print("-"*80)
    
    for result in all_results:
        cfg = result['config']
        print(f"{cfg.name:<25} {result['final_val_loss']:<12.4f} {result['final_val_acc']:<12.4f} "
              f"{result['avg_tokens_per_sec']:<15,.0f} {result['total_time']/60:<12.1f}")
    
    # Group by strategy
    large_batch = [r for r in all_results if 'large_batch' in r['config'].name]
    long_seq = [r for r in all_results if 'long_seq' in r['config'].name]
    balanced = [r for r in all_results if 'balanced' in r['config'].name]
    
    print(f"\n{'='*80}")
    print("📈 STRATEGY COMPARISON (AVERAGES)")
    print("="*80)
    
    for strategy_name, strategy_results in [
        ('Large Batch (64x256)', large_batch),
        ('Long Sequence (8x1024)', long_seq),
        ('Balanced (24x512)', balanced)
    ]:
        if strategy_results:
            avg_loss = sum(r['final_val_loss'] for r in strategy_results) / len(strategy_results)
            avg_acc = sum(r['final_val_acc'] for r in strategy_results) / len(strategy_results)
            avg_throughput = sum(r['avg_tokens_per_sec'] for r in strategy_results) / len(strategy_results)
            
            print(f"\n{strategy_name}:")
            print(f"   Avg Val Loss: {avg_loss:.4f}")
            print(f"   Avg Val Acc: {avg_acc:.4f}")
            print(f"   Avg Throughput: {avg_throughput:,.0f} tok/s")
    
    # Save results
    os.makedirs('results/ablation_batch_seqlen', exist_ok=True)
    
    # Save detailed results
    save_data = {
        'ablation_configs': [
            {
                'name': r['config'].name,
                'batch_size': r['config'].batch_size,
                'seq_len': r['config'].seq_len,
                'lr': r['config'].lr,
                'effective_batch': r['config'].effective_batch,
                'tokens_per_step': r['config'].tokens_per_step
            } for r in all_results
        ],
        'results': [
            {
                'config_name': r['config'].name,
                'final_val_loss': r['final_val_loss'],
                'final_val_acc': r['final_val_acc'],
                'avg_tokens_per_sec': r['avg_tokens_per_sec'],
                'total_time': r['total_time'],
                'train_losses': r['train_losses'],
                'val_metrics': r['val_metrics']
            } for r in all_results
        ]
    }
    
    with open('results/ablation_batch_seqlen/results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n💾 Results saved to results/ablation_batch_seqlen/results.json")
    
    # Determine winner
    print(f"\n{'='*80}")
    print("🏆 CONCLUSION")
    print("="*80)
    
    best_result = min(all_results, key=lambda r: r['final_val_loss'])
    fastest_result = max(all_results, key=lambda r: r['avg_tokens_per_sec'])
    
    print(f"\n✅ Best Performance: {best_result['config'].name}")
    print(f"   Val Loss: {best_result['final_val_loss']:.4f}")
    print(f"   Val Acc: {best_result['final_val_acc']:.4f}")
    
    print(f"\n⚡ Fastest Training: {fastest_result['config'].name}")
    print(f"   Throughput: {fastest_result['avg_tokens_per_sec']:,.0f} tok/s")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

