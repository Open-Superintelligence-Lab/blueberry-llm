from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Union, Tuple
from datasets import load_dataset, Dataset, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from configs.dataset_config import DataConfig
import logging

logger = logging.getLogger(__name__)


def setup_tokenizer(config: DataConfig) -> PreTrainedTokenizer:
    """Load the SmolLM tokenizer with caching."""
    logger.info(f"Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        use_fast=config.use_fast,
        trust_remote_code=config.trust_remote_code,
        cache_dir=config.cache_dir,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    return tokenizer


def load_smollm_corpus(config: DataConfig) -> Union[Dataset, IterableDataset]:
    """
    Load HuggingFaceTB/smollm-corpus with streaming and caching support.
    
    - Streaming mode: Data is streamed and not loaded all at once
    - Cache: Downloaded data is cached to disk (cache_dir)
    """
    streaming_mode = "streaming" if config.streaming else "regular"
    logger.info(
        f"Loading HuggingFaceTB/smollm-corpus (subset: {config.dataset_name}) "
        f"in {streaming_mode} mode with caching"
    )
    
    dataset = load_dataset(
        config.dataset_path,
        config.dataset_name,
        split=config.split,
        cache_dir=config.cache_dir,  # Cache downloaded data
        streaming=config.streaming,   # Stream to avoid loading all at once
    )
    
    if config.text_column not in dataset.column_names:
        raise ValueError(
            f"Text column '{config.text_column}' not found. "
            f"Available: {dataset.column_names}"
        )
    
    return dataset


def tokenize_and_chunk(
    dataset: Union[Dataset, IterableDataset],
    tokenizer: PreTrainedTokenizer,
    config: DataConfig
) -> Dataset:
    """Tokenize text and group into fixed-length chunks."""
    is_streaming = isinstance(dataset, IterableDataset)
    
    def tokenize_function(examples):
        return tokenizer(
            examples[config.text_column],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
    
    logger.info("Tokenizing dataset...")
    if is_streaming:
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
    else:
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=config.num_proc,
            desc="Tokenizing",
        )
    
    block_size = config.seq_length
    stride = config.stride if config.stride is not None else block_size
    
    def group_texts(examples):
        # Concatenate all sequences in this batch
        arrays = {k: np.concatenate(examples[k]) for k in examples.keys()}
        total = arrays["input_ids"].shape[0]
        
        # Skip batches that are too small
        if total < block_size:
            logger.warning(
                f"Batch only has {total} tokens but block_size is {block_size}. "
                f"Skipping this batch."
            )
            return {k: np.empty((0, block_size), dtype=arrays[k].dtype) for k in arrays.keys()}
        
        # Drop partial block at the end
        trunc = (total // block_size) * block_size
        
        if stride == block_size:
            # Non-overlapping windows (fast path)
            n = trunc // block_size
            return {k: v[:trunc].reshape(n, block_size) for k, v in arrays.items()}
        
        # Overlapping windows using strided views
        n_windows = (trunc - block_size) // stride + 1
        return {
            k: as_strided(
                v, 
                shape=(n_windows, block_size), 
                strides=(stride * v.strides[0], v.strides[0])
            )
            for k, v in arrays.items()
        }

    logger.info(f"Grouping texts into blocks of size {block_size} with stride {stride}")
    if is_streaming:
        lm_dataset = tokenized.map(
            group_texts,
            batched=True,
        )
    else:
        lm_dataset = tokenized.map(
            group_texts,
            batched=True,
            num_proc=config.num_proc,
            desc="Grouping texts",
        )
        
        if len(lm_dataset) == 0:
            logger.warning("The resulting dataset is empty!")
    
    return lm_dataset


def finalize_dataset(dataset: Union[Dataset, IterableDataset], config: DataConfig) -> Dataset:
    """Add labels and set format for PyTorch."""
    is_streaming = isinstance(dataset, IterableDataset)
    
    def create_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    dataset = dataset.map(create_labels, batched=True)
    
    # Materialize streaming dataset if needed
    if is_streaming:
        logger.info("Materializing streaming dataset for training...")
        dataset = Dataset.from_generator(
            lambda: (example for example in dataset),
            features=dataset.features
        )
    
    # Set PyTorch format
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    if config.save_to_disk:
        logger.info(f"Saving preprocessed dataset to {config.save_to_disk}")
        dataset.save_to_disk(config.save_to_disk)
    
    return dataset
