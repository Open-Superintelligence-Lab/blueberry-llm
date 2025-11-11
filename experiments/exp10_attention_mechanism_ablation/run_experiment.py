"""
Main training script for Experiment 10: Attention Mechanism Ablation

Trains and compares different attention mechanisms in isolation.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp10_attention_mechanism_ablation.config import (
    AttentionAblationConfig,
    ALL_CONFIGS
)
from experiments.exp10_attention_mechanism_ablation.models import ModelWrapper


class Trainer:
    """Training manager for attention ablation experiments"""
    
    def __init__(self, config: AttentionAblationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Setup
        self.setup_tokenizer()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        
        # Metrics
        self.metrics = {
            "train_loss": [],
            "train_steps": [],
            "eval_loss": [],
            "eval_steps": [],
            "eval_perplexity": [],
            "train_time": [],
        }
    
    def setup_tokenizer(self):
        """Setup tokenizer"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def setup_data(self):
        """Setup datasets"""
        print(f"Loading dataset: {self.config.dataset_name}...")
        
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split="train"
        )
        
        # Split train/val
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        self.train_dataset = dataset["train"]
        self.val_dataset = dataset["test"]
        
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")
    
    def setup_model(self):
        """Setup model"""
        print(f"\nCreating model with {self.config.attention_type} attention...")
        self.model = ModelWrapper(self.config)
        self.model.to(self.device)
        
        # Print model info
        self.model.print_model_info()
        
        # Optionally compile model
        if self.config.compile_model and hasattr(torch, 'compile'):
            print("Compiling model...")
            self.model = torch.compile(self.model)
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Cosine schedule with warmup
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def collate_fn(self, examples):
        """Collate function for dataloader"""
        texts = [ex["text"] for ex in examples if len(ex["text"].strip()) > 0]
        if not texts:
            # Return dummy batch if no valid texts
            return {
                "input_ids": torch.zeros((1, 128), dtype=torch.long),
                "attention_mask": torch.zeros((1, 128), dtype=torch.long)
            }
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_seq_len,
            return_tensors="pt"
        )
        return tokenized
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs["loss"]
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False
        )
        
        total_loss = 0
        total_tokens = 0
        
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs["loss"]
            
            # Count tokens (excluding padding)
            mask = input_ids != self.tokenizer.pad_token_id
            num_tokens = mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*80}")
        print(f"Starting training: {self.config.experiment_name}")
        print(f"{'='*80}\n")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True
        )
        
        step = 0
        start_time = time.time()
        
        while step < self.config.max_steps:
            for batch in train_loader:
                if step >= self.config.max_steps:
                    break
                
                # Train step
                loss = self.train_step(batch)
                
                # Log
                if step % self.config.log_interval == 0:
                    elapsed = time.time() - start_time
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"Step {step:4d}/{self.config.max_steps} | "
                        f"Loss: {loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {elapsed:.1f}s"
                    )
                    
                    self.metrics["train_loss"].append(loss)
                    self.metrics["train_steps"].append(step)
                    self.metrics["train_time"].append(elapsed)
                
                # Evaluate
                if step % self.config.eval_interval == 0 and step > 0:
                    eval_loss, perplexity = self.evaluate()
                    print(f"  → Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")
                    
                    self.metrics["eval_loss"].append(eval_loss)
                    self.metrics["eval_steps"].append(step)
                    self.metrics["eval_perplexity"].append(perplexity)
                
                step += 1
        
        # Final evaluation
        print("\nRunning final evaluation...")
        eval_loss, perplexity = self.evaluate()
        print(f"Final Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")
        
        self.metrics["eval_loss"].append(eval_loss)
        self.metrics["eval_steps"].append(step)
        self.metrics["eval_perplexity"].append(perplexity)
        self.metrics["final_eval_loss"] = eval_loss
        self.metrics["final_perplexity"] = perplexity
        self.metrics["total_time"] = time.time() - start_time
        
        print(f"\nTraining complete! Total time: {self.metrics['total_time']:.1f}s")
        
        return self.metrics
    
    def save_results(self):
        """Save training results"""
        save_dir = Path(self.config.save_dir) / self.config.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save config
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith("_")
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Results saved to {save_dir}")


def run_single_experiment(config: AttentionAblationConfig):
    """Run a single experiment"""
    trainer = Trainer(config)
    metrics = trainer.train()
    trainer.save_results()
    return metrics


def run_comprehensive_ablation():
    """Run comprehensive ablation across all attention mechanisms"""
    print("=" * 80)
    print("EXPERIMENT 10: COMPREHENSIVE ATTENTION MECHANISM ABLATION")
    print("=" * 80)
    print(f"\nTesting {len(ALL_CONFIGS)} attention mechanisms:\n")
    
    for name in ALL_CONFIGS:
        print(f"  • {name}")
    
    print("\n" + "=" * 80 + "\n")
    
    all_results = {}
    
    for name, config_fn in ALL_CONFIGS.items():
        print(f"\n{'#'*80}")
        print(f"# Running: {name}")
        print(f"{'#'*80}\n")
        
        try:
            config = config_fn()
            metrics = run_single_experiment(config)
            all_results[name] = {
                "final_loss": metrics["final_eval_loss"],
                "final_perplexity": metrics["final_perplexity"],
                "total_time": metrics["total_time"],
                "attention_type": config.attention_type,
            }
        except Exception as e:
            print(f"ERROR: Failed to run {name}: {e}")
            all_results[name] = {"error": str(e)}
            continue
    
    # Save comprehensive results
    results_dir = Path("experiments/exp10_attention_mechanism_ablation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "comprehensive_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80 + "\n")
    
    # Sort by final loss
    sorted_results = sorted(
        [(name, res) for name, res in all_results.items() if "error" not in res],
        key=lambda x: x[1]["final_loss"]
    )
    
    print(f"{'Rank':<6} {'Mechanism':<20} {'Loss':<10} {'Perplexity':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for rank, (name, res) in enumerate(sorted_results, 1):
        print(
            f"{rank:<6} {name:<20} {res['final_loss']:<10.4f} "
            f"{res['final_perplexity']:<12.2f} {res['total_time']:<10.1f}"
        )
    
    print("\n" + "=" * 80)
    
    # Create comparison plot
    plot_results(all_results)
    
    return all_results


def plot_results(results: Dict):
    """Create comparison plots"""
    # Filter out errors
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Sort by loss
    sorted_items = sorted(valid_results.items(), key=lambda x: x[1]["final_loss"])
    names = [item[0] for item in sorted_items]
    losses = [item[1]["final_loss"] for item in sorted_items]
    perplexities = [item[1]["final_perplexity"] for item in sorted_items]
    times = [item[1]["total_time"] for item in sorted_items]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss comparison
    axes[0].barh(names, losses, color='steelblue')
    axes[0].set_xlabel('Final Validation Loss')
    axes[0].set_title('Validation Loss by Attention Mechanism')
    axes[0].invert_yaxis()
    
    # Perplexity comparison
    axes[1].barh(names, perplexities, color='coral')
    axes[1].set_xlabel('Final Perplexity')
    axes[1].set_title('Perplexity by Attention Mechanism')
    axes[1].invert_yaxis()
    
    # Training time comparison
    axes[2].barh(names, times, color='seagreen')
    axes[2].set_xlabel('Training Time (seconds)')
    axes[2].set_title('Training Time by Attention Mechanism')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    
    save_path = Path("experiments/exp10_attention_mechanism_ablation/results/comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 10: Attention Mechanism Ablation"
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        default=None,
        choices=list(ALL_CONFIGS.keys()) + ["all"],
        help="Attention mechanism to test (default: all)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    
    args = parser.parse_args()
    
    if args.mechanism == "all" or args.mechanism is None:
        # Run comprehensive ablation
        run_comprehensive_ablation()
    else:
        # Run single mechanism
        config = ALL_CONFIGS[args.mechanism]()
        config.max_steps = args.steps
        config.batch_size = args.batch_size
        run_single_experiment(config)


if __name__ == "__main__":
    main()

