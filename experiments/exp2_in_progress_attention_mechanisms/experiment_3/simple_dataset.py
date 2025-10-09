"""
Simple synthetic dataset for testing Experiment 3.

This provides a minimal dataset implementation for testing the adaptive sparsity experiment.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict


class SimpleSyntheticDataset(Dataset):
    """
    Simple synthetic dataset for testing.
    
    Generates random token sequences for testing the adaptive sparsity models.
    """
    
    def __init__(self, seq_len: int, vocab_size: int, split: str = 'train', num_samples: int = 1000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.split = split
        self.num_samples = num_samples
        
        # Generate synthetic data
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_ids = sequence[:-1]  # All tokens except last
        labels = sequence[1:]      # All tokens except first
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class TinyStoriesDataset(SimpleSyntheticDataset):
    """
    Alias for SimpleSyntheticDataset to match the expected interface.
    """
    pass
