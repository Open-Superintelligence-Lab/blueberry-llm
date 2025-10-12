import time
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
