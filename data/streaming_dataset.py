"""
Streaming dataset that loads fresh data and never repeats.
Tracks what data has been seen across training runs.
"""
import torch
from torch.utils.data import IterableDataset
from typing import List
import random


class StreamingTokenDataset(IterableDataset):
    """
    Streaming dataset that yields non-overlapping windows from a token stream.
    Supports resumption without repeating data.
    
    Args:
        tokens: Full token array
        seq_len: Sequence length
        start_token_idx: Starting position in token array (for resumption)
        shuffle_windows: Whether to shuffle windows before yielding
    """
    
    def __init__(self, tokens: List[int], seq_len: int, start_token_idx: int = 0, shuffle_windows: bool = True):
        self.tokens = tokens
        self.seq_len = seq_len
        self.start_token_idx = start_token_idx
        self.shuffle_windows = shuffle_windows
        
        # Calculate how many complete windows we can make
        available_tokens = len(tokens) - start_token_idx
        self.num_windows = max(0, available_tokens // seq_len)
        
        # Pre-calculate all window start positions
        self.window_starts = [
            start_token_idx + i * seq_len 
            for i in range(self.num_windows)
        ]
        
        if shuffle_windows:
            random.shuffle(self.window_starts)
    
    def __iter__(self):
        """Yield non-overlapping windows"""
        for start_idx in self.window_starts:
            # Input sequence
            x = torch.tensor(
                self.tokens[start_idx:start_idx + self.seq_len], 
                dtype=torch.long
            )
            # Target sequence (shifted by 1)
            y = torch.tensor(
                self.tokens[start_idx + 1:start_idx + self.seq_len + 1], 
                dtype=torch.long
            )
            yield x, y
    
    def __len__(self):
        return self.num_windows
    
    def get_end_token_idx(self):
        """Get the token index where this dataset ends (for continuation)"""
        return self.start_token_idx + (self.num_windows * self.seq_len)


class ProgressiveDataLoader:
    """
    Manages progressive data loading across training runs.
    Tracks what data has been seen and loads new data on resume.
    """
    
    def __init__(self, all_tokens: List[int], seq_len: int, batch_size: int, 
                 start_token_idx: int = 0, shuffle_windows: bool = True):
        self.all_tokens = all_tokens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.start_token_idx = start_token_idx
        self.shuffle_windows = shuffle_windows
        
        # Create dataset starting from the specified position
        self.dataset = StreamingTokenDataset(
            all_tokens, seq_len, start_token_idx, shuffle_windows
        )
        
        # Track progress
        self.windows_consumed = 0
        self.tokens_consumed = start_token_idx
    
    def get_state(self):
        """Get state for checkpoint saving"""
        return {
            'start_token_idx': self.start_token_idx,
            'tokens_consumed': self.tokens_consumed,
            'windows_consumed': self.windows_consumed,
            'dataset_end_idx': self.dataset.get_end_token_idx(),
        }
    
    @staticmethod
    def from_state(all_tokens: List[int], seq_len: int, batch_size: int, 
                   state: dict, shuffle_windows: bool = True):
        """Resume from saved state"""
        # Start from where we left off
        start_idx = state['dataset_end_idx']
        return ProgressiveDataLoader(
            all_tokens, seq_len, batch_size, start_idx, shuffle_windows
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        """Iterate over dataset"""
        batch_x = []
        batch_y = []
        
        for x, y in self.dataset:
            batch_x.append(x)
            batch_y.append(y)
            
            if len(batch_x) == self.batch_size:
                self.windows_consumed += self.batch_size
                self.tokens_consumed += self.batch_size * self.seq_len
                yield torch.stack(batch_x), torch.stack(batch_y)
                batch_x = []
                batch_y = []
        
        # Yield final partial batch if exists
        if len(batch_x) > 0:
            self.windows_consumed += len(batch_x)
            self.tokens_consumed += len(batch_x) * self.seq_len
            yield torch.stack(batch_x), torch.stack(batch_y)


def create_progressive_loaders(train_tokens, val_tokens, seq_len, batch_size, 
                                train_state=None, val_state=None):
    """
    Create train and validation loaders that track progress.
    
    Args:
        train_tokens: Training tokens
        val_tokens: Validation tokens
        seq_len: Sequence length
        batch_size: Batch size
        train_state: Previous training state (for resume)
        val_state: Previous validation state (for resume)
    
    Returns:
        train_loader, val_loader
    """
    
    if train_state is not None:
        train_loader = ProgressiveDataLoader.from_state(
            train_tokens, seq_len, batch_size, train_state, shuffle_windows=True
        )
    else:
        train_loader = ProgressiveDataLoader(
            train_tokens, seq_len, batch_size, start_token_idx=0, shuffle_windows=True
        )
    
    if val_state is not None:
        val_loader = ProgressiveDataLoader.from_state(
            val_tokens, seq_len, batch_size, val_state, shuffle_windows=False
        )
    else:
        val_loader = ProgressiveDataLoader(
            val_tokens, seq_len, batch_size, start_token_idx=0, shuffle_windows=False
        )
    
    return train_loader, val_loader

