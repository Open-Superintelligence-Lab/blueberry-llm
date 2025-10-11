import torch
from torch.utils.data import Dataset
from typing import List


class TextTokenDataset(Dataset):
    """
    Token dataset with configurable stride for creating training windows.
    
    Args:
        tokens: List of token IDs
        seq_len: Length of each sequence window
        stride: Step size between windows (default: seq_len for non-overlapping windows)
                - stride=seq_len: No overlap (most efficient)
                - stride=seq_len//2: 50% overlap
                - stride=1: Maximum overlap (1023x more samples, very inefficient)
    """
    def __init__(self, tokens: List[int], seq_len: int = 512, stride: int = None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len  # Default: non-overlapping
        
        # Calculate number of samples based on stride
        self.num_samples = max(0, (len(tokens) - seq_len) // self.stride + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Calculate actual token index based on stride
        start_idx = idx * self.stride
        x = torch.tensor(self.tokens[start_idx:start_idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[start_idx + 1:start_idx + self.seq_len + 1], dtype=torch.long)
        return x, y
