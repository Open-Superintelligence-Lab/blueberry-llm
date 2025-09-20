"""
Data loading and caching utilities.

This module provides functions for loading and caching tokenized data
to avoid reprocessing on subsequent runs.
"""

import os
import pickle
from typing import List, Tuple, Any
from configs import AdaptiveMoEModelConfig

# Optional imports - will be imported when needed
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, desc="": x  # Fallback tqdm

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = Any  # Fallback for type hints


def load_and_cache_data(
    config: AdaptiveMoEModelConfig, 
    cache_dir: str = "data_cache"
) -> Tuple[List[str], AutoTokenizer, List[int]]:
    """
    Load and cache tokenized data to avoid reprocessing.
    
    Args:
        config: Model configuration containing data parameters
        cache_dir: Directory to store cached data
        
    Returns:
        Tuple of (texts, tokenizer, tokens)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = _load_tokenizer()
    
    # Load and process documents
    texts = _load_documents(config.num_documents)
    
    # Tokenize texts
    tokens = _tokenize_texts(texts, tokenizer, config.max_tokens)
    
    # Update config with vocab size
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens


def _load_tokenizer():
    """Load and configure the tokenizer."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers package is required for tokenizer loading. Install with: pip install transformers")
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_documents(num_documents: int, max_chars_per_doc: int = 3000) -> List[str]:
    """
    Load documents from the dataset.
    
    Args:
        num_documents: Number of documents to load
        max_chars_per_doc: Maximum characters per document
        
    Returns:
        List of text documents
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets package is required for data loading. Install with: pip install datasets")
    
    print(f"📚 Loading {num_documents} documents...")
    
    # Load dataset
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", 
        "cosmopedia-v2", 
        split="train", 
        streaming=True, 
        token=False
    )

    texts = []
    for i, item in enumerate(dataset):
        if i >= num_documents:
            break
        # Truncate documents to reasonable length
        text = item["text"][:max_chars_per_doc]
        texts.append(text)

    print(f"✅ Loaded {len(texts)} documents")
    return texts


def _tokenize_texts(texts: List[str], tokenizer: AutoTokenizer, max_tokens: int) -> List[int]:
    """
    Tokenize texts and return a flat list of tokens.
    
    Args:
        texts: List of text documents
        tokenizer: Tokenizer to use
        max_tokens: Maximum number of tokens to return
        
    Returns:
        Flat list of token IDs
    """
    print("🔤 Tokenizing texts...")
    
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        
        # Stop if we have enough tokens
        if len(all_tokens) >= max_tokens:
            break

    tokens = all_tokens[:max_tokens]
    print(f"✅ Using {len(tokens):,} tokens")
    return tokens


def load_custom_dataset(
    dataset_name: str,
    subset: str = None,
    num_documents: int = 1000,
    max_tokens: int = 500000,
    cache_dir: str = "data_cache"
) -> Tuple[List[str], AutoTokenizer, List[int]]:
    """
    Load a custom dataset with the same interface as load_and_cache_data.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        subset: Subset/configuration of the dataset
        num_documents: Number of documents to load
        max_tokens: Maximum number of tokens
        cache_dir: Directory to store cached data
        
    Returns:
        Tuple of (texts, tokenizer, tokens)
    """
    cache_file = f"{cache_dir}/custom_{dataset_name}_{subset}_{num_documents}_{max_tokens}.pkl"
    
    # Check cache first
    if os.path.exists(cache_file):
        print(f"📦 Loading cached custom data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        return cached_data['texts'], cached_data['tokenizer'], cached_data['tokens']
    
    print(f"🔄 Loading custom dataset: {dataset_name}")
    
    # Load tokenizer
    tokenizer = _load_tokenizer()
    
    # Load custom dataset
    if subset:
        dataset = load_dataset(dataset_name, subset, split="train", streaming=True)
    else:
        dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    # Extract texts (assuming 'text' field exists)
    texts = []
    for i, item in enumerate(dataset):
        if i >= num_documents:
            break
        
        # Try different common field names
        text = item.get('text') or item.get('content') or item.get('article') or str(item)
        texts.append(text[:3000])  # Truncate
    
    # Tokenize
    tokens = _tokenize_texts(texts, tokenizer, max_tokens)
    
    # Cache
    os.makedirs(cache_dir, exist_ok=True)
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    
    print(f"💾 Cached custom data to {cache_file}")
    return texts, tokenizer, tokens
