"""
Experiment: Compare xformers memory-efficient attention vs standard PyTorch attention
Uses the current setup with smollm dataset
"""
import time
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader

from configs.moe_config import MoEModelConfig
from configs.dataset_config import DataConfig
from data.loader import prepare_lm_dataset
from training.trainer import train_moe_model
from utils.helpers import set_seed
from utils.logger import setup_logging


def print_system_info():
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}\n")


def run_single_experiment(use_xformers: bool, exp_name: str, train_loader, val_loader, config):
    """Run a single training experiment"""
    print("\n" + "=" * 80)
    print(f"Running experiment: {exp_name}")
    print(f"Using xformers: {use_xformers}")
    print("=" * 80)
    
    # Update config for this experiment
    config.use_mem_efficient_attention = use_xformers
    
    print("\nModel configuration")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"experts: {config.num_experts}, top-k: {config.expert_top_k}")
    print(f"steps: {config.max_steps}, batch size: {config.batch_size}")
    print(f"use_mem_efficient_attention: {config.use_mem_efficient_attention}")
    print(f"vocab size: {config.vocab_size}\n")
    
    print("Starting training...")
    print("-" * 70)
    start = time.time()
    
    model, metrics = train_moe_model(config, train_loader, val_loader)
    
    elapsed = (time.time() - start)
    
    print("\nResults")
    print("-" * 70)
    print(f"Training time: {elapsed:.2f} sec ({elapsed/60:.2f} min)")
    print(f"Val loss:       {metrics['val_loss']:.4f}")
    print(f"Val accuracy:   {metrics['val_accuracy']:.4f}")
    print(f"Val perplexity: {metrics['val_perplexity']:.2f}")
    
    # Save results
    results = {
        "experiment": exp_name,
        "use_xformers": use_xformers,
        "training_time_sec": elapsed,
        "training_time_min": elapsed / 60,
        "val_loss": float(metrics['val_loss']),
        "val_accuracy": float(metrics['val_accuracy']),
        "val_perplexity": float(metrics['val_perplexity']),
        "config": {
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "d_ff": config.d_ff,
            "num_experts": config.num_experts,
            "expert_top_k": config.expert_top_k,
            "max_steps": config.max_steps,
            "batch_size": config.batch_size,
            "max_seq_len": config.max_seq_len,
        }
    }
    
    return model, results


def main():
    # Setup
    results_dir = Path("./experiments/exp_xformers_comparison/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(log_dir="./experiments/exp_xformers_comparison/logs")
    logger.info("Starting xformers comparison experiment")
    
    print_system_info()
    set_seed(42)
    
    # Load dataset once (same for both experiments)
    print("Loading dataset with Hugging Face Datasets API...")
    config = MoEModelConfig()
    data_cfg = DataConfig(
        dataset_path="HuggingFaceTB/smollm-corpus",
        dataset_name="cosmopedia-v2",
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        seq_length=config.max_seq_len,
        num_samples=config.num_documents,
        cache_dir="./hf_cache",
    )
    
    dataset, tokenizer = prepare_lm_dataset(data_cfg)
    config.vocab_size = tokenizer.vocab_size
    logger.info(f"Loaded dataset with {len(dataset):,} sequences")
    
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = splits["train"], splits["test"]
    logger.info(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")
    
    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    
    # Store all results
    all_results = []
    
    # Experiment 1: Standard PyTorch attention
    set_seed(42)  # Reset seed for fair comparison
    model_std, results_std = run_single_experiment(
        use_xformers=False,
        exp_name="standard_attention",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    all_results.append(results_std)
    
    # Save checkpoint
    ckpt_path_std = results_dir / "model_standard_attention.pt"
    torch.save({
        "model_state_dict": model_std.state_dict(),
        "config": config,
        "metrics": results_std,
    }, ckpt_path_std)
    logger.info(f"Standard attention model saved to {ckpt_path_std}")
    
    # Experiment 2: XFormers memory-efficient attention
    set_seed(42)  # Reset seed for fair comparison
    model_xformers, results_xformers = run_single_experiment(
        use_xformers=True,
        exp_name="xformers_attention",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    all_results.append(results_xformers)
    
    # Save checkpoint
    ckpt_path_xformers = results_dir / "model_xformers_attention.pt"
    torch.save({
        "model_state_dict": model_xformers.state_dict(),
        "config": config,
        "metrics": results_xformers,
    }, ckpt_path_xformers)
    logger.info(f"XFormers attention model saved to {ckpt_path_xformers}")
    
    # Save all results
    results_file = results_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Standard':>15} {'XFormers':>15} {'Diff':>15}")
    print("-" * 80)
    
    time_std = results_std['training_time_sec']
    time_xformers = results_xformers['training_time_sec']
    speedup = (time_std - time_xformers) / time_std * 100
    
    print(f"{'Training Time (sec)':<30} {time_std:>15.2f} {time_xformers:>15.2f} {speedup:>14.1f}%")
    print(f"{'Val Loss':<30} {results_std['val_loss']:>15.4f} {results_xformers['val_loss']:>15.4f} {(results_xformers['val_loss'] - results_std['val_loss']):>15.4f}")
    print(f"{'Val Accuracy':<30} {results_std['val_accuracy']:>15.4f} {results_xformers['val_accuracy']:>15.4f} {(results_xformers['val_accuracy'] - results_std['val_accuracy']):>15.4f}")
    print(f"{'Val Perplexity':<30} {results_std['val_perplexity']:>15.2f} {results_xformers['val_perplexity']:>15.2f} {(results_xformers['val_perplexity'] - results_std['val_perplexity']):>15.2f}")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {results_file}")
    print("=" * 80)
    
    logger.info("Experiment complete")
    logger.info(f"Results: {all_results}")


if __name__ == "__main__":
    main()
