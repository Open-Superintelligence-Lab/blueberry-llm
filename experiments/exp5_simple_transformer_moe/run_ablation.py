#!/usr/bin/env python3
"""
Simple runner for ablation configs.
Usage: python run_ablation.py <config_name>
"""

import sys
import os
import torch
import torch.nn.functional as F
from dataclasses import replace
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import json

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

from configs_ablation import CONFIGS
from config import SimpleTransformerConfig
from models.moe_llm import MoEMinimalLLM
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from optimizers.muon import Muon
from utils.helpers import set_seed

# ============= CONFIGURABLE PARAMETERS =============
VALIDATION_INTERVAL = 25  # Run validation every N steps (easy to change!)
# ===================================================


def train(config_name):
    if config_name not in CONFIGS:
        print(f"❌ Config '{config_name}' not found!")
        print(f"Available configs: {', '.join(CONFIGS.keys())}")
        return
    
    abl_config = CONFIGS[config_name]
    
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING: {abl_config.name}")
    print(f"{'='*80}")
    print(f"   Batch: {abl_config.batch_size} × SeqLen: {abl_config.seq_len}")
    print(f"   LR: {abl_config.lr}, GradAccum: {abl_config.grad_accum}, Steps: {abl_config.max_steps}")
    print(f"   Effective Batch: {abl_config.effective_batch}")
    print(f"   Tokens/Step: {abl_config.tokens_per_step:,}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"\n   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\n📚 Loading data...")
    temp_config = SimpleTransformerConfig()
    temp_moe_config = temp_config.to_moe_config()
    texts, tokenizer, tokens = load_and_cache_data(temp_moe_config, cache_dir="data_cache")
    
    base_config = SimpleTransformerConfig(vocab_size=temp_moe_config.vocab_size)
    
    # Update config with ablation settings
    config = replace(base_config,
                    batch_size=abl_config.batch_size,
                    max_seq_len=abl_config.seq_len,
                    gradient_accumulation_steps=abl_config.grad_accum,
                    muon_lr=abl_config.lr,
                    max_steps=abl_config.max_steps)
    
    # Create dataset
    dataset = TextTokenDataset(tokens, abl_config.seq_len)
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=abl_config.batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=abl_config.batch_size,
                           shuffle=False, num_workers=2)
    
    # Initialize model
    set_seed(42)
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
    
    def lr_lambda(step):
        warmup = abl_config.max_steps // 20
        return min(1.0, step / warmup) if warmup > 0 else 1.0
    
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda) for opt in optimizers]
    scaler = GradScaler() if config.use_amp else None
    
    # Training
    model.train()
    step = 0
    import time
    start_time = time.time()
    total_tokens = 0
    
    # Track training curves
    train_history = []
    val_history = []  # Track validation throughout training
    
    # Validation function
    def run_validation():
        model.eval()
        val_loss_sum = 0
        val_acc_sum = 0
        val_count = 0
        
        with torch.no_grad():
            for i, (vx, vy) in enumerate(val_loader):
                if i >= 50:  # Limit validation batches for speed
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
                
                val_loss_sum += vloss.item()
                val_acc_sum += (vlogits.argmax(-1) == vy).float().mean().item()
                val_count += 1
        
        model.train()
        return val_loss_sum / val_count if val_count > 0 else 0, val_acc_sum / val_count if val_count > 0 else 0
    
    pbar = tqdm(total=abl_config.max_steps, desc=abl_config.name)
    train_iter = iter(train_loader)
    
    best_loss = float('inf')
    
    while step < abl_config.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        
        if x.size(1) != config.max_seq_len:
            if x.size(1) < config.max_seq_len:
                pad_len = config.max_seq_len - x.size(1)
                x = F.pad(x, (0, pad_len), value=0)
                y = F.pad(y, (0, pad_len), value=0)
            else:
                x = x[:, :config.max_seq_len]
                y = y[:, :config.max_seq_len]
        
        total_tokens += x.numel()
        
        if config.use_amp:
            with autocast('cuda', dtype=torch.float16):
                logits, aux_loss = model(x, return_aux_loss=True)
                ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = (ce_loss + (aux_loss if aux_loss is not None else 0)) / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits, aux_loss = model(x, return_aux_loss=True)
            ce_loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            loss = (ce_loss + (aux_loss if aux_loss is not None else 0)) / config.gradient_accumulation_steps
            loss.backward()
        
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
        
        elapsed = time.time() - start_time
        tok_s = total_tokens / elapsed if elapsed > 0 else 0
        
        with torch.no_grad():
            acc = (logits.argmax(-1) == y).float().mean().item()
        
        # Save training point
        train_history.append({
            'step': step,
            'time': elapsed,
            'tokens': total_tokens,
            'loss': ce_loss.item(),
            'acc': acc
        })
        
        pbar.set_postfix({'loss': f'{ce_loss.item():.4f}', 'acc': f'{acc:.3f}', 'tok/s': f'{tok_s:.0f}'})
        
        if ce_loss.item() < best_loss:
            best_loss = ce_loss.item()
        
        # Periodic validation
        if (step + 1) % VALIDATION_INTERVAL == 0 or step == 0:
            val_loss, val_acc = run_validation()
            val_history.append({
                'step': step,
                'time': elapsed,
                'tokens': total_tokens,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            pbar.write(f"   Step {step}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final validation
    model.eval()
    val_loss = 0
    val_acc = 0
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
            
            val_loss += vloss.item()
            val_acc += (vlogits.argmax(-1) == vy).float().mean().item()
            val_count += 1
    
    val_loss /= val_count
    val_acc /= val_count
    total_time = time.time() - start_time
    avg_tok_s = total_tokens / total_time
    
    print(f"\n{'='*80}")
    print(f"✅ RESULTS: {abl_config.name}")
    print(f"{'='*80}")
    print(f"   Val Loss: {val_loss:.4f}")
    print(f"   Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"   Throughput: {avg_tok_s:,.0f} tokens/sec")
    print(f"   Time: {total_time:.1f}s ({total_time/60:.2f} min)")
    if torch.cuda.is_available():
        print(f"   Peak Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"{'='*80}\n")
    
    # Save result
    os.makedirs('results/ablation_batch_seqlen', exist_ok=True)
    result = {
        'config_name': abl_config.name,
        'batch_size': abl_config.batch_size,
        'seq_len': abl_config.seq_len,
        'lr': abl_config.lr,
        'grad_accum': abl_config.grad_accum,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'throughput': avg_tok_s,
        'time': total_time,
        'peak_memory_gb': torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0,
        'train_history': train_history,  # Training curves
        'val_history': val_history  # Validation curves throughout training
    }
    
    with open(f'results/ablation_batch_seqlen/{abl_config.name}_result.json', 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n📋 Available configs:")
        for name, cfg in CONFIGS.items():
            print(f"  {name:<12} - batch={cfg.batch_size}, seqlen={cfg.seq_len}, lr={cfg.lr}")
        print(f"\nUsage: python run_ablation.py <config_name>")
        print(f"Example: python run_ablation.py custom")
    else:
        train(sys.argv[1])

