from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import Optional, Union, Tuple
from datasets import load_dataset, Dataset, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from configs.dataset_config import DataConfig
import logging

logger = logging.getLogger(__name__)


def setup_tokenizer(config: DataConfig) -> PreTrainedTokenizer:
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


def load_raw_dataset(config: DataConfig) -> Union[Dataset, IterableDataset]:
    if config.load_from_disk:
        logger.info(f"Loading preprocessed dataset from {config.load_from_disk}")
        dataset = Dataset.load_from_disk(config.load_from_disk)
        logger.info(f"Loaded {len(dataset)} sequences of length {config.seq_length}")
        return dataset
    
    streaming_mode = "streaming" if config.streaming else "regular"
    logger.info(f"Loading dataset '{config.dataset_path}' with name '{config.dataset_name}' ({streaming_mode} mode)")
    dataset = load_dataset(
        config.dataset_path,
        config.dataset_name,
        split=config.split,
        cache_dir=config.cache_dir,
        streaming=config.streaming,
    )
    
    if config.text_column not in dataset.column_names:
        raise ValueError(
            f"Text column '{config.text_column}' not found in dataset. "
            f"Available columns: {dataset.column_names}"
        )
    
    return dataset


def apply_sampling_and_filters(dataset: Union[Dataset, IterableDataset], config: DataConfig) -> Union[Dataset, IterableDataset]:
    is_streaming = isinstance(dataset, IterableDataset)
    
    # Sampling
    if config.num_samples:
        if is_streaming:
            logger.info(f"Taking first {config.num_samples} samples from stream")
            dataset = dataset.take(config.num_samples)
        else:
            actual_num_samples = min(config.num_samples, len(dataset))
            if actual_num_samples < config.num_samples:
                logger.warning(
                    f"Requested {config.num_samples} samples, but only {actual_num_samples} "
                    f"are available in the split '{config.split}'."
                )
            dataset = dataset.select(range(actual_num_samples))
    
    # Preprocessing
    if config.preprocessing_fn:
        logger.info("Applying preprocessing function...")
        # Streaming datasets don't support num_proc
        if is_streaming:
            dataset = dataset.map(config.preprocessing_fn)
        else:
            dataset = dataset.map(config.preprocessing_fn, num_proc=config.num_proc)
    
    # Filtering
    if config.filter_fn:
        logger.info("Applying filter function...")
        if is_streaming:
            dataset = dataset.filter(config.filter_fn)
        else:
            dataset = dataset.filter(config.filter_fn, num_proc=config.num_proc)
    
    return dataset


def tokenize_and_chunk(
    dataset: Union[Dataset, IterableDataset],
    tokenizer: PreTrainedTokenizer,
    config: DataConfig
) -> Dataset:
    is_streaming = isinstance(dataset, IterableDataset)
    
    # Note: truncation=False is intentional. concat text -> fixed size blocks. 
    # avoids tokens at the end of documents
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
    
    # group into fixed len chunks w/ stride
    block_size = config.seq_length
    stride = config.stride if config.stride is not None else block_size
    
    # note: implicit closure over stride
    def group_texts(examples):
        
        # concat all sequences in this batch into single arrays per key
        arrays = {k: np.concatenate(examples[k]) for k in examples.keys()}
        total = arrays["input_ids"].shape[0]
        
        # batch didn't have enough tokens for even one block
        if total < block_size:
            logger.warning(
                f"Batch only has {total} tokens but block_size is {block_size}. "
                f"Skipping this batch."
            )
            return {k: np.empty((0, block_size), dtype=arrays[k].dtype) for k in arrays.keys()}
        
        # drop partial block at the end could pad if needed
        trunc = (total // block_size) * block_size
        
        if stride == block_size:
            # fast path: non-overlapping windows via reshape (zero-copy, O(1))
            n = trunc // block_size
            return {k: v[:trunc].reshape(n, block_size) for k, v in arrays.items()}
        
        # overlapping windows: use strided views (also zero-copy)
        # this creates sliding windows w/o allocating new memory
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
    is_streaming = isinstance(dataset, IterableDataset)
    
    # labels for causal LM
    def create_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    dataset = dataset.map(create_labels, batched=True)
    
    # If streaming, we need to materialize it at some point for training
    # The DataLoader will handle iteration, but we need a concrete dataset
    if is_streaming:
        logger.info("Materializing streaming dataset for training...")
        # Convert IterableDataset to Dataset by collecting all examples
        # This happens lazily as data is streamed
        dataset = Dataset.from_generator(
            lambda: (example for example in dataset),
            features=dataset.features
        )
    
    # set format for PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    if config.save_to_disk:
        logger.info(f"Saving preprocessed dataset to {config.save_to_disk}")
        dataset.save_to_disk(config.save_to_disk)
    
    return dataset


def prepare_lm_dataset(
    config: DataConfig,
    return_tokenizer: bool = True,
) -> Union[Tuple[Dataset, PreTrainedTokenizer], Dataset]:
    
    tokenizer = setup_tokenizer(config)
    dataset = load_raw_dataset(config)
    
    if not config.load_from_disk:
        dataset = apply_sampling_and_filters(dataset, config)
        dataset = tokenize_and_chunk(dataset, tokenizer, config)
        dataset = finalize_dataset(dataset, config)
    
    dataset_info = f"{len(dataset)} sequences" if hasattr(dataset, '__len__') else "streaming sequences"
    logger.info(f"Dataset prepared: {dataset_info} of length {config.seq_length}")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    
    return (dataset, tokenizer) if return_tokenizer else dataset


# presets for common use cases
def quick_dataset(
    preset: str = "cosmopedia",
    seq_length: int = 512,
    num_samples: int = 10000,
    **kwargs
) -> Tuple[Dataset, PreTrainedTokenizer]:
    
    presets = {
        "cosmopedia": DataConfig(
            dataset_path="HuggingFaceTB/smollm-corpus",
            dataset_name="cosmopedia-v2",
            tokenizer_name="HuggingFaceTB/SmolLM-135M",
            seq_length=seq_length,
            num_samples=num_samples,
            streaming=True,
            **kwargs
        ),
        "wikipedia": DataConfig(
            dataset_path="wikipedia",
            dataset_name="20220301.en",
            seq_length=seq_length,
            num_samples=num_samples,
            streaming=True,
            **kwargs
        ),
        "wikitext": DataConfig(
            dataset_path="wikitext",
            dataset_name="wikitext-103-v1",
            seq_length=seq_length,
            num_samples=num_samples,
            streaming=True,
            **kwargs
        ),
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")
    
    return prepare_lm_dataset(presets[preset])
