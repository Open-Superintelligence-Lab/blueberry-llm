from dataclasses import dataclass
from typing import Optional, Callable, Union
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    # Dataset source (supports HF dataset path format)
    dataset_path: str  # e.g., "wikipedia/20231101.en" or "HuggingFaceTB/smollm-corpus"
    dataset_name: Optional[str] = None  # e.g., "cosmopedia-v2"
    split: str = "train"
    
    # Tokenizer
    tokenizer_name: str = "gpt2"
    use_fast: bool = True
    trust_remote_code: bool = False
    
    # Sequence processing
    seq_length: int = 512
    stride: Optional[int] = None  # For overlapping windows, None = seq_length
    
    # Limits
    num_samples: Optional[int] = None  # Limit number of documents
    
    # Columns and preprocessing
    text_column: str = "text"
    preprocessing_fn: Optional[Callable[[dict], dict]] = None
    filter_fn: Optional[Callable[[dict], bool]] = None
    
    # Caching
    cache_dir: Optional[str] = "./hf_cache"
    num_proc: Optional[int] = None  # Parallel processing for .map()

    # Persistence
    save_to_disk: Optional[str] = None  # path to save preprocessed dataset
    load_from_disk: Optional[str] = None  # path to load preprocessed dataset

    def __post_init__(self) -> None:
        # Validate dataset_path (only required if not loading from disk)
        if not self.load_from_disk:
            if not self.dataset_path or not isinstance(self.dataset_path, str):
                raise ValueError("dataset_path must be a non-empty string when not loading from disk")
            if not self.dataset_path.strip():
                raise ValueError("dataset_path cannot be empty or whitespace")
        
        # Validate tokenizer_name
        if not self.tokenizer_name or not isinstance(self.tokenizer_name, str):
            raise ValueError("tokenizer_name must be a non-empty string")
        if not self.tokenizer_name.strip():
            raise ValueError("tokenizer_name cannot be empty or whitespace")
        
        # Validate split
        if not self.split or not isinstance(self.split, str):
            raise ValueError("split must be a non-empty string")
        if not self.split.strip():
            raise ValueError("split cannot be empty or whitespace")
        
        # Validate seq_length
        if not isinstance(self.seq_length, int):
            raise TypeError(f"seq_length must be an integer, got {type(self.seq_length).__name__}")
        if self.seq_length <= 0:
            raise ValueError(f"seq_length must be positive, got {self.seq_length}")
        
        # Validate stride
        if self.stride is not None:
            if not isinstance(self.stride, int):
                raise TypeError(f"stride must be an integer, got {type(self.stride).__name__}")
            if self.stride <= 0:
                raise ValueError(f"stride must be positive, got {self.stride}")
            if self.stride > self.seq_length:
                raise ValueError(
                    f"stride ({self.stride}) cannot be greater than seq_length ({self.seq_length})"
                )
        
        # Validate num_samples
        if self.num_samples is not None:
            if not isinstance(self.num_samples, int):
                raise TypeError(f"num_samples must be an integer, got {type(self.num_samples).__name__}")
            if self.num_samples <= 0:
                raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        
        # Validate text_column
        if not self.text_column or not isinstance(self.text_column, str):
            raise ValueError("text_column must be a non-empty string")
        if not self.text_column.strip():
            raise ValueError("text_column cannot be empty or whitespace")
        
        # Validate num_proc
        if self.num_proc is not None:
            if not isinstance(self.num_proc, int):
                raise TypeError(f"num_proc must be an integer, got {type(self.num_proc).__name__}")
            if self.num_proc <= 0:
                raise ValueError(f"num_proc must be positive, got {self.num_proc}")
        
        # Warn if both save_to_disk and load_from_disk are set
        if self.save_to_disk and self.load_from_disk:
            logger.warning(
                "Both save_to_disk and load_from_disk are set. "
                "The dataset will be loaded from disk, ignoring save_to_disk."
            )
