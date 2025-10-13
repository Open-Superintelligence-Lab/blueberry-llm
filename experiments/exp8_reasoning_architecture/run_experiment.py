"""
Training script for Reasoning Architecture Experiment
Experiment 8: Based on exp7's winning Hybrid Sparse 17% architecture

Supports two modes:
1. Baseline: Exp7 winner architecture (Hybrid Sparse 17%)
2. Recursive: Adds hierarchical reasoning with ACT halting

Usage:
    # Train baseline only (default)
    python run_experiment.py
    
    # Train recursive only
    python run_experiment.py --model-type recursive
    
    # Train and compare both models
    python run_experiment.py --compare
    
    # Extended training with comparison
    python run_experiment.py --experiment extended --compare
    
    # Resume from checkpoint
    python run_experiment.py --resume checkpoints_baseline/best_model.pt
    
    # Resume and extend training
    python run_experiment.py --resume checkpoints_baseline/best_model.pt --extend-steps 5000
"""

import torch
import torch.nn as nn
import sys
import os
import time
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Fix tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp8_reasoning_architecture.config import (
    ExperimentConfig,
    get_base_reasoning_config,
    get_extended_reasoning_config,
    get_recursive_reasoning_config,
)
from experiments.exp8_reasoning_architecture.models import (
    ReasoningModelWrapper,
    count_parameters,
)
from experiments.exp8_reasoning_architecture.recursive_reasoning import RecursiveCarryState
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from utils.helpers import set_seed
from torch.utils.data import DataLoader


class Trainer:
    """Training manager for Reasoning Model"""
    
    def __init__(self, model, config, train_loader, val_loader, device, save_dir=None, use_recursive=False):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("checkpoints")
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.use_recursive = use_recursive
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None
        
        # History
        self.train_history = []
        self.val_history = []
        
        # Carry state for recursive reasoning (if enabled)
        self.carry = None
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return max(0.1, (self.config.max_steps - step) / (self.config.max_steps - self.config.warmup_steps))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch):
        """Single training step with optional carry state for recursive reasoning"""
        self.model.train()
        
        # Handle tuple output from dataset
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
        else:
            input_ids = batch.to(self.device)
        
        labels = input_ids.clone()
        
        # Forward pass (with carry state if recursive)
        if self.use_recursive:
            outputs = self.model(input_ids=input_ids, labels=labels, carry=self.carry)
            # Update carry for next iteration
            if 'carry' in outputs:
                self.carry = outputs['carry']
        else:
            outputs = self.model(input_ids=input_ids, labels=labels)
        
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Extract metrics for recursive model
        metrics = outputs.get('metrics', {}) if isinstance(outputs, dict) else {}
        
        return loss.item(), metrics
    
    @torch.no_grad()
    def evaluate(self, max_batches=None):
        """Evaluate on validation set with optional carry state"""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        eval_carry = None
        
        for i, batch in enumerate(self.val_loader):
            if max_batches and i >= max_batches:
                break
            
            # Handle tuple output from dataset
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
            else:
                input_ids = batch.to(self.device)
            
            labels = input_ids.clone()
            
            # Forward with carry if recursive
            if self.use_recursive:
                outputs = self.model(input_ids=input_ids, labels=labels, carry=eval_carry)
                if 'carry' in outputs:
                    eval_carry = outputs['carry']
            else:
                outputs = self.model(input_ids=input_ids, labels=labels)
            
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            shift_preds = predictions[..., :-1].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            correct = (shift_preds == shift_labels).sum().item()
            
            total_loss += loss.item() * input_ids.numel()
            total_correct += correct
            total_tokens += shift_labels.numel()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
        }
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("Starting Training - Reasoning Architecture (Exp8)")
        print("="*70)
        
        start_time = time.time()
        running_loss = 0
        steps_since_log = 0
        running_metrics = {}
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                
                # Training step
                loss, metrics = self.train_step(batch)
                running_loss += loss
                steps_since_log += 1
                self.global_step += 1
                
                # Accumulate metrics
                for k, v in metrics.items():
                    if k not in running_metrics:
                        running_metrics[k] = 0
                    running_metrics[k] += v
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = running_loss / steps_since_log
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed
                    
                    log_msg = (f"Step {self.global_step}/{self.config.max_steps} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {lr:.6f} | "
                              f"Speed: {steps_per_sec:.2f} steps/s")
                    
                    # Add recursive metrics if available
                    if self.use_recursive and running_metrics:
                        avg_metrics = {k: v / steps_since_log for k, v in running_metrics.items()}
                        log_msg += f" | Steps: {avg_metrics.get('reasoning_steps', 0):.1f} | Halt: {avg_metrics.get('halt_rate', 0):.2%}"
                    
                    print(log_msg)
                    
                    history_entry = {
                        'step': self.global_step,
                        'loss': avg_loss,
                        'lr': lr,
                    }
                    
                    # Add avg metrics to history
                    if running_metrics:
                        for k, v in running_metrics.items():
                            history_entry[k] = v / steps_since_log
                    
                    self.train_history.append(history_entry)
                    
                    running_loss = 0
                    steps_since_log = 0
                    running_metrics = {}
                
                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    val_metrics = self.evaluate(max_batches=self.config.eval_batches)
                    
                    print(f"\n{'='*70}")
                    print(f"Evaluation at step {self.global_step}")
                    print(f"{'='*70}")
                    print(f"Val Loss: {val_metrics['loss']:.4f}")
                    print(f"Val Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
                    print(f"Val Perplexity: {val_metrics['perplexity']:.2f}")
                    print(f"{'='*70}\n")
                    
                    self.val_history.append({
                        'step': self.global_step,
                        **val_metrics
                    })
                    
                    # Save best model
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint('best_model.pt', is_best=True)
                        print(f"✓ New best validation loss: {self.best_val_loss:.4f} (saved)")
                
                # Periodic checkpoint saving
                if self.global_step % self.config.save_interval == 0:
                    checkpoint_path = self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt', is_best=False)
                    print(f"💾 Checkpoint saved at step {self.global_step}: {checkpoint_path}")
            
            self.epoch += 1
        
        total_time = time.time() - start_time
        
        # Save final model
        final_checkpoint = self.save_checkpoint('final_model.pt', is_best=False)
        
        print(f"\n{'='*70}")
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"\n💾 Models saved:")
        print(f"  Best model: {self.best_checkpoint_path}")
        print(f"  Final model: {final_checkpoint}")
        print(f"{'='*70}\n")
        
        return {
            'total_time': total_time,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_checkpoint': str(self.best_checkpoint_path),
            'final_checkpoint': str(final_checkpoint),
        }
    
    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint for training resumption"""
        checkpoint_path = self.save_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            self.best_checkpoint_path = checkpoint_path
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state"""
        print(f"\n{'='*70}")
        print(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"{'='*70}")
        
        torch.serialization.add_safe_globals([ExperimentConfig])
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        old_config = checkpoint['config']
        max_steps_changed = old_config.max_steps != self.config.max_steps
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint['best_val_loss']
        
        if max_steps_changed:
            print(f"  ⚠ max_steps changed ({old_config.max_steps} → {self.config.max_steps})")
            print(f"  Recreating LR scheduler with new schedule...")
            self.scheduler = self._create_scheduler()
            for _ in range(self.global_step):
                self.scheduler.step()
            print(f"  ✓ Scheduler recreated and fast-forwarded to step {self.global_step}")
        else:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        print(f"✓ Checkpoint loaded successfully!")
        print(f"  Resuming from step: {self.global_step}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Best val loss so far: {self.best_val_loss:.4f}")
        print(f"  Current LR: {self.scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*70}\n")
        
        return checkpoint


def plot_training_curves(train_history, val_history, save_path):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    if train_history:
        steps = [h['step'] for h in train_history]
        losses = [h['loss'] for h in train_history]
        axes[0, 0].plot(steps, losses, 'b-', linewidth=2, label='Train Loss')
        axes[0, 0].set_xlabel('Step', fontweight='bold')
        axes[0, 0].set_ylabel('Loss', fontweight='bold')
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # Validation loss
    if val_history:
        steps = [h['step'] for h in val_history]
        losses = [h['loss'] for h in val_history]
        axes[0, 1].plot(steps, losses, 'r-', linewidth=2, marker='o', label='Val Loss')
        axes[0, 1].set_xlabel('Step', fontweight='bold')
        axes[0, 1].set_ylabel('Loss', fontweight='bold')
        axes[0, 1].set_title('Validation Loss', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Validation accuracy
    if val_history:
        steps = [h['step'] for h in val_history]
        accs = [h['accuracy'] * 100 for h in val_history]
        axes[1, 0].plot(steps, accs, 'g-', linewidth=2, marker='s', label='Val Accuracy')
        axes[1, 0].set_xlabel('Step', fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[1, 0].set_title('Validation Accuracy', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Validation perplexity
    if val_history:
        steps = [h['step'] for h in val_history]
        ppls = [h['perplexity'] for h in val_history]
        axes[1, 1].plot(steps, ppls, 'm-', linewidth=2, marker='^', label='Val Perplexity')
        axes[1, 1].set_xlabel('Step', fontweight='bold')
        axes[1, 1].set_ylabel('Perplexity', fontweight='bold')
        axes[1, 1].set_title('Validation Perplexity', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")


def train_single_model(config, use_recursive, model_type, train_loader, val_loader, device, exp_base_dir):
    """Train a single model (baseline or recursive)"""
    
    print("\n" + "="*70)
    print(f"Training {model_type.upper()} Model")
    print("="*70)
    
    # Create model
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    if dtype == torch.float32:
        print("⚠ Warning: bfloat16 not supported, FLA may have issues with float32")
    
    model = ReasoningModelWrapper(config, use_recursive=use_recursive)
    model.print_info()
    model = model.to(device=device, dtype=dtype)
    print(f"\nUsing dtype: {dtype}")
    
    # Setup directories
    checkpoints_dir = exp_base_dir / f"checkpoints_{model_type}"
    results_dir = exp_base_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📁 Output directories:")
    print(f"   Checkpoints: {checkpoints_dir}")
    print(f"   Results: {results_dir}")
    
    # Train
    trainer = Trainer(model, config, train_loader, val_loader, device, 
                     save_dir=checkpoints_dir, use_recursive=use_recursive)
    results = trainer.train()
    
    # Save results
    results_summary = {
        'experiment_name': f'Reasoning Architecture ({model_type.capitalize()})',
        'model_type': model_type,
        'base_architecture': 'Exp7 Hybrid Sparse 17%',
        'use_recursive': use_recursive,
        'attention_layers': config.attn_config.get('layers', []) if config.attn_config else [],
        'config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_attention_heads,
            'max_seq_len': config.max_seq_len,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'max_steps': config.max_steps,
        },
        'model_info': model.get_info(),
        'results': {
            'total_time': results['total_time'],
            'best_val_loss': results['best_val_loss'],
            'final_train_loss': results['train_history'][-1]['loss'] if results['train_history'] else None,
            'final_val_metrics': results['val_history'][-1] if results['val_history'] else None,
        },
    }
    
    # Add recursive-specific config if applicable
    if use_recursive:
        results_summary['recursive_config'] = getattr(config, 'recursive', {})
    
    # Save with model_type suffix
    results_file = results_dir / f'training_results_{model_type}.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Plot training curves
    curves_file = results_dir / f'training_curves_{model_type}.png'
    plot_training_curves(
        results['train_history'],
        results['val_history'],
        curves_file
    )
    
    print("\n" + "="*70)
    print(f"✅ {model_type.upper()} TRAINING COMPLETED!")
    print("="*70)
    print(f"   Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"   Training Time: {results['total_time']:.1f}s")
    print("="*70)
    
    return results_summary


def main():
    """Main experiment function with dual training support"""
    parser = argparse.ArgumentParser(description='Train Reasoning Architecture (Exp8)')
    parser.add_argument('--experiment', type=str, default='base',
                        choices=['base', 'extended', 'recursive'],
                        help='Experiment variant (default: base)')
    parser.add_argument('--compare', action='store_true', default=False,
                        help='Train both baseline and recursive models for comparison')
    parser.add_argument('--model-type', type=str, default=None,
                        choices=['baseline', 'recursive'],
                        help='Train specific model type (overrides --compare)')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume from')
    parser.add_argument('--extend-steps', type=int, default=None,
                        help='Extend training to this many total steps')
    args = parser.parse_args()
    
    # Experiment configuration mapping
    EXPERIMENTS = {
        'base': ('Reasoning Architecture (Base)', get_base_reasoning_config),
        'extended': ('Reasoning Architecture (Extended)', get_extended_reasoning_config),
        'recursive': ('Reasoning Architecture (Recursive)', get_recursive_reasoning_config),
    }
    
    exp_name, get_config_fn = EXPERIMENTS[args.experiment]
    config = get_config_fn()
    
    print("="*70)
    print(f"EXPERIMENT 8: {exp_name}")
    print("Based on Exp7 Winner: Hybrid Sparse 17%")
    if args.compare:
        print("Mode: COMPARISON (Training both Baseline and Recursive)")
    elif args.model_type:
        print(f"Mode: Single Model ({args.model_type.upper()})")
    print("="*70)
    
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # Extend training steps if requested
    if args.extend_steps:
        original_max_steps = config.max_steps
        config.max_steps = args.extend_steps
        print(f"\n📈 Extending training: {original_max_steps} → {args.extend_steps} steps")
    
    print(f"\nUsing device: {device}")
    print(f"Configuration: {config.hidden_size}d, {config.num_hidden_layers} layers")
    print(f"Winner Architecture: Attention at layers {config.attn_config.get('layers', [])} (17%)")
    
    # Load data
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    
    from dataclasses import dataclass
    @dataclass
    class DataConfig:
        num_documents: int = config.num_documents
        max_tokens: int = config.max_tokens
        vocab_size: int = config.vocab_size
    
    data_config = DataConfig()
    texts, tokenizer, tokens = load_and_cache_data(data_config)
    config.vocab_size = len(tokenizer)
    
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Total tokens: {len(tokens):,}")
    
    # Split tokens
    val_split_ratio = 0.1
    val_token_start = int(len(tokens) * (1 - val_split_ratio))
    
    train_tokens = tokens[:val_token_start]
    val_tokens = tokens[val_token_start:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Create progressive data loaders
    from data.streaming_dataset import create_progressive_loaders
    
    train_loader, val_loader = create_progressive_loaders(
        train_tokens, val_tokens,
        config.max_seq_len, config.batch_size,
        None, None
    )
    
    print(f"Train windows available: {len(train_loader):,}")
    print(f"Val windows available: {len(val_loader):,}")
    
    exp_base_dir = Path(__file__).parent
    
    # Determine which models to train
    if args.model_type:
        # Train specific model type
        use_recursive = args.model_type == 'recursive'
        results_summary = train_single_model(
            config, use_recursive, args.model_type,
            train_loader, val_loader, device, exp_base_dir
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"   Model Type: {args.model_type}")
        print(f"   Best Val Loss: {results_summary['results']['best_val_loss']:.4f}")
        print("="*70)
        
    elif args.compare:
        # Train both baseline and recursive models
        print("\n" + "="*70)
        print("DUAL TRAINING: Baseline + Recursive")
        print("="*70)
        
        # Train baseline first
        baseline_summary = train_single_model(
            config, False, 'baseline',
            train_loader, val_loader, device, exp_base_dir
        )
        
        # Get recursive config if not already using it
        if not hasattr(config, 'recursive'):
            recursive_config = get_recursive_reasoning_config()
        else:
            recursive_config = config
        
        # Train recursive
        recursive_summary = train_single_model(
            recursive_config, True, 'recursive',
            train_loader, val_loader, device, exp_base_dir
        )
        
        # Run comparison script
        print("\n" + "="*70)
        print("Running Comparison Analysis...")
        print("="*70)
        
        try:
            import subprocess
            comparison_script = exp_base_dir / "compare_baseline_vs_recursive.py"
            if comparison_script.exists():
                result = subprocess.run(
                    [sys.executable, str(comparison_script)],
                    cwd=str(exp_base_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print(f"⚠ Comparison script failed: {result.stderr}")
            else:
                print(f"⚠ Comparison script not found: {comparison_script}")
                print("Run manually: python compare_baseline_vs_recursive.py")
        except Exception as e:
            print(f"⚠ Could not run comparison script: {e}")
            print("Run manually: python compare_baseline_vs_recursive.py")
        
        print("\n" + "="*70)
        print("✅ DUAL TRAINING COMPLETED!")
        print("="*70)
        print(f"Baseline Val Loss: {baseline_summary['results']['best_val_loss']:.4f}")
        print(f"Recursive Val Loss: {recursive_summary['results']['best_val_loss']:.4f}")
        improvement = (baseline_summary['results']['best_val_loss'] - recursive_summary['results']['best_val_loss']) / baseline_summary['results']['best_val_loss'] * 100
        print(f"Improvement: {improvement:+.2f}%")
        print("="*70)
        
    else:
        # Default: train baseline only
        results_summary = train_single_model(
            config, False, 'baseline',
            train_loader, val_loader, device, exp_base_dir
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"   Best Val Loss: {results_summary['results']['best_val_loss']:.4f}")
        print("="*70)


if __name__ == "__main__":
    main()

