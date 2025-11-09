import time
import os
import torch
import logging
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


def main():
    logger = setup_logging(log_dir="./logs")
    logger.info("Starting MoE training")

    print_system_info()
    set_seed(42)
    config = MoEModelConfig()

    print("Loading dataset with Hugging Face Datasets API...")
    data_cfg = DataConfig(
        dataset_path="HuggingFaceTB/smollm-corpus",
        dataset_name="cosmopedia-v2",
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        seq_length=config.max_seq_len,
        num_samples=config.num_documents,
        cache_dir="./hf_cache",
    )

    # Split documents BEFORE tokenization to prevent data leakage
    from datasets import load_dataset
    print("Loading raw dataset and splitting documents...")
    raw_dataset = load_dataset(
        data_cfg.dataset_path,
        data_cfg.dataset_name,
        split=data_cfg.split,
        cache_dir=data_cfg.cache_dir,
        streaming=True,
    )
    
    # Take samples and split into train/val
    raw_samples = list(raw_dataset.take(data_cfg.num_samples))
    num_val = int(len(raw_samples) * 0.1)
    num_train = len(raw_samples) - num_val
    
    from datasets import Dataset
    raw_train = Dataset.from_list(raw_samples[:num_train])
    raw_val = Dataset.from_list(raw_samples[num_train:])
    logger.info(f"Split into {len(raw_train):,} train docs and {len(raw_val):,} val docs")
    
    # Now tokenize each split separately
    from data.loader import setup_tokenizer, tokenize_and_chunk, finalize_dataset
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size
    
    print("Tokenizing train set...")
    train_ds = tokenize_and_chunk(raw_train, tokenizer, data_cfg)
    train_ds = finalize_dataset(train_ds, data_cfg)
    
    print("Tokenizing validation set...")
    val_ds = tokenize_and_chunk(raw_val, tokenizer, data_cfg)
    val_ds = finalize_dataset(val_ds, data_cfg)
    
    logger.info(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    print("\nModel configuration")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"experts: {config.num_experts}, top‑k: {config.expert_top_k}")
    print(f"steps: {config.max_steps}, batch size: {config.batch_size}")
    print(f"vocab size: {config.vocab_size}\n")
    logger.info(f"Model configuration: {vars(config)}")

    print("Starting training...")
    print("-" * 70)
    start = time.time()

    model, metrics = train_moe_model(config, train_loader, val_loader)
    elapsed = (time.time() - start) / 60
    logger.info("Training complete")

    print("\nResults")
    print("-" * 70)
    print(f"Training time: {elapsed:.2f} min")
    print(f"Val loss:       {metrics['val_loss']:.4f}")
    print(f"Val accuracy:   {metrics['val_accuracy']:.4f}")
    print(f"Val perplexity: {metrics['val_perplexity']:.2f}")
    logger.info(f"Final metrics: {metrics}")

    ckpt_path = "./checkpoints/final_model.pt"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(),
         "config": config,
         "metrics": metrics},
        ckpt_path,
    )
    print(f"Model checkpoint saved to {ckpt_path}")
    logger.info(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
