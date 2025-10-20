import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")


def count_parameters(model):
    """Count the number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unpack_batch(batch, device):
    """Always return (input_ids, labels) no matter what the loader gives us."""
    if isinstance(batch, dict):
        return batch["input_ids"].to(device), batch["labels"].to(device)
    elif isinstance(batch, (list, tuple)):
        return batch[0].to(device), batch[1].to(device)
    else:
        raise TypeError("Unknown batch format")
